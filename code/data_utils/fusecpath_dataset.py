"""Patch selection utilities for the FuseCPath baseline.

The original FuseCPath implementation performs multi-view clustering once per
slide and keeps a fixed number of representative patches from every cluster.
This module provides the same data contract as ``MultiEncoderSlideDataset``
while making that preprocessing usable from the repository's offline-baseline
runner.  ``mvlearn`` is used when available; a deterministic MiniBatchKMeans
fallback keeps the baseline runnable in the standard project environment.
"""

from __future__ import annotations

import time
from typing import Dict, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from data_utils.gme_dataset import MultiEncoderSlideDataset, get_dataset_key


def _reduced_view(values: np.ndarray, indices: np.ndarray, dim: int) -> np.ndarray:
    """Normalize a view and retain a deterministic low-dimensional sketch."""
    view = np.nan_to_num(values[indices].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(view, axis=1, keepdims=True)
    view = view / np.maximum(norms, 1e-8)
    if view.shape[1] > dim:
        columns = np.linspace(0, view.shape[1] - 1, num=dim, dtype=np.int64)
        view = view[:, columns]
    return view


def multiview_representative_indices(
    features: Mapping[str, np.ndarray],
    n_clusters: int = 50,
    patches_per_cluster: int = 10,
    max_candidates: int = 4096,
    view_dim: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Select representative patch indices using all encoder views.

    Candidate patches are clustered using normalized sketches from every
    encoder.  The nearest candidates to each cluster centroid are returned in
    cluster order, matching FuseCPath's ``K x N_K`` patch layout.
    """
    names = sorted(features)
    if not names:
        raise ValueError("FuseCPath requires at least one encoder view.")
    n_patches = min(int(features[name].shape[0]) for name in names)
    if n_patches <= 0:
        raise ValueError("FuseCPath received an empty patch bag.")
    target = max(1, int(n_clusters)) * max(1, int(patches_per_cluster))
    if n_patches <= target:
        return np.arange(n_patches, dtype=np.int64)

    candidate_count = min(n_patches, max(int(max_candidates), target))
    candidate_indices = np.linspace(0, n_patches - 1, candidate_count, dtype=np.int64)
    views = [_reduced_view(features[name][:n_patches], candidate_indices, max(int(view_dim), 1)) for name in names]
    n_clusters_eff = min(max(int(n_clusters), 1), candidate_count)

    labels = None
    try:
        from mvlearn.cluster import MultiviewSpectralClustering

        labels = MultiviewSpectralClustering(
            n_clusters=n_clusters_eff,
            affinity="nearest_neighbors",
            random_state=int(seed),
        ).fit_predict(views)
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError, TypeError, AttributeError):
        from sklearn.cluster import MiniBatchKMeans

        stacked = np.concatenate(views, axis=1)
        labels = MiniBatchKMeans(
            n_clusters=n_clusters_eff,
            random_state=int(seed),
            # The fallback is used on Windows where mvlearn is often absent.
            # Keep it bounded: the default sklearn settings (100 iterations
            # and multiple initializations) make a 50-way clustering of a
            # 4,096-patch, multi-view bag unnecessarily expensive.
            n_init=1,
            max_iter=20,
            max_no_improvement=3,
            reassignment_ratio=0.0,
            batch_size=min(512, candidate_count),
        ).fit_predict(stacked)

    stacked = np.concatenate(views, axis=1)
    selected = []
    per_cluster = max(1, int(patches_per_cluster))
    for cluster_id in range(n_clusters_eff):
        members = np.flatnonzero(labels == cluster_id)
        if members.size == 0:
            continue
        centroid = stacked[members].mean(axis=0)
        distances = np.sum((stacked[members] - centroid) ** 2, axis=1)
        order = members[np.argsort(distances, kind="stable")]
        if order.size < per_cluster:
            repeats = np.resize(order, per_cluster)
            order = repeats
        else:
            order = order[:per_cluster]
        selected.extend(candidate_indices[order].tolist())

    if not selected:
        return candidate_indices[: min(target, candidate_count)]
    return np.asarray(selected, dtype=np.int64)


class FuseCPathSlideDataset(MultiEncoderSlideDataset):
    """Dataset that applies cached multi-view representative patch selection."""

    def __init__(
        self,
        *args,
        fusecpath_clusters: int = 50,
        fusecpath_patches_per_cluster: int = 10,
        fusecpath_max_candidates: int = 4096,
        fusecpath_view_dim: int = 64,
        fusecpath_seed: int = 42,
        **kwargs,
    ) -> None:
        # The offline runner passes the selected encoder names to every
        # dataset.  ``gme_dataset`` predates that argument, so consume it here
        # and apply the same subset before patch selection.
        requested_encoders = kwargs.pop("encoder_names", None)
        kwargs["max_patches"] = 0
        super().__init__(*args, **kwargs)
        if requested_encoders is not None:
            requested = [str(name) for name in requested_encoders]
            missing = sorted(set(requested) - set(self.encoder_names))
            if missing:
                raise ValueError(f"FuseCPath encoders are missing from the manifest: {missing}")
            self.encoder_names = requested
            for slide_id, rows in list(self.slide_rows.items()):
                filtered = rows[rows["feature_dir"].astype(str).isin(self.encoder_names)].copy()
                if set(filtered["feature_dir"].astype(str)) == set(self.encoder_names):
                    self.slide_rows[slide_id] = filtered
                else:
                    del self.slide_rows[slide_id]
            self.slide_ids = sorted(self.slide_rows)
            if not self.slide_ids:
                raise RuntimeError("FuseCPath encoder filtering removed all usable slides.")
        self.fusecpath_clusters = int(fusecpath_clusters)
        self.fusecpath_patches_per_cluster = int(fusecpath_patches_per_cluster)
        self.fusecpath_max_candidates = int(fusecpath_max_candidates)
        self.fusecpath_view_dim = int(fusecpath_view_dim)
        self.fusecpath_seed = int(fusecpath_seed)
        self._selection_cache: Dict[str, np.ndarray] = {}

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int, str]:
        slide_id = self.slide_ids[int(idx)]
        rows = self.slide_rows[slide_id]
        # Read only an evenly spaced candidate subset for clustering.  The
        # original implementation materializes every patch; doing this lazily
        # keeps the comparison practical for WSI bags with tens of thousands
        # of patches while preserving the same multi-view selection contract.
        specs = {}
        min_patches = None
        for _, row in rows.iterrows():
            name = str(row["feature_dir"])
            path = str(row["h5_path"])
            key = str(row["dataset_key"]) if "dataset_key" in row and not pd.isna(row["dataset_key"]) else None
            if key is None:
                key = get_dataset_key(Path(path))
            with h5py.File(path, "r") as handle:
                n_patches = int(handle[key].shape[0])
            specs[name] = (path, key, n_patches)
            min_patches = n_patches if min_patches is None else min(min_patches, n_patches)
        if min_patches is None or min_patches <= 0:
            raise ValueError(f"{slide_id}: no usable patch features for FuseCPath")
        target = max(1, self.fusecpath_clusters) * max(1, self.fusecpath_patches_per_cluster)
        candidate_count = min(min_patches, max(self.fusecpath_max_candidates, target))
        candidate_indices = np.linspace(0, min_patches - 1, candidate_count, dtype=np.int64)

        def read_rows(path: str, key: str, indices: np.ndarray) -> np.ndarray:
            unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
            with h5py.File(path, "r") as handle:
                values = np.asarray(handle[key][unique], dtype=np.float32)
            return values[inverse]

        candidate_features = {
            name: read_rows(path, key, candidate_indices)
            for name, (path, key, _) in specs.items()
        }
        if slide_id not in self._selection_cache:
            started = time.perf_counter()
            print(
                f"[FuseCPath] selecting representatives for {slide_id} "
                f"({candidate_count} candidates, {len(specs)} views)...",
                flush=True,
            )
            selected_positions = multiview_representative_indices(
                candidate_features,
                n_clusters=self.fusecpath_clusters,
                patches_per_cluster=self.fusecpath_patches_per_cluster,
                max_candidates=self.fusecpath_max_candidates,
                view_dim=self.fusecpath_view_dim,
                seed=self.fusecpath_seed + sum(ord(c) for c in slide_id),
            )
            self._selection_cache[slide_id] = candidate_indices[selected_positions]
            print(
                f"[FuseCPath] {slide_id}: selected {len(selected_positions)} patches "
                f"in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
        indices = self._selection_cache[slide_id]
        selected = {
            name: torch.from_numpy(read_rows(path, key, indices))
            for name, (path, key, _) in specs.items()
        }
        label = int(self.clinical.loc[slide_id, self.label_col])
        return selected, label, slide_id
