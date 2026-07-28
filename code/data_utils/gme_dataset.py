"""Data loading helpers for aligned multi-encoder GME training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


FEATURE_KEYS = ("feats", "features")


def get_dataset_key(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as handle:
        for key in FEATURE_KEYS:
            if key in handle:
                return key
        candidates = [key for key in handle.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(handle.keys())}")
        return candidates[0]


def read_h5_features(h5_path: Path, dataset_key: str | None = None) -> np.ndarray:
    key = dataset_key or get_dataset_key(h5_path)
    with h5py.File(h5_path, "r") as handle:
        features = np.asarray(handle[key][:], dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected [N, D], got {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def infer_input_dims(manifest: pd.DataFrame, fold: int | None = None) -> Dict[str, int]:
    rows = manifest if fold is None else manifest[manifest["fold"] == fold]
    if rows.empty:
        raise ValueError(f"No manifest rows found for fold={fold}")

    input_dims: Dict[str, int] = {}
    for feature_dir, group in rows.groupby("feature_dir"):
        dims = sorted(group["feature_dim"].astype(int).unique().tolist())
        if len(dims) != 1:
            raise ValueError(f"{feature_dir}: expected one feature_dim, got {dims}")
        input_dims[str(feature_dir)] = int(dims[0])
    return input_dims


def subset_patch_indices(
    n_patches: int, max_patches: int, training: bool
) -> np.ndarray | None:
    if max_patches <= 0 or n_patches <= max_patches:
        return None
    if training:
        return np.sort(np.random.choice(n_patches, size=max_patches, replace=False))
    return np.arange(max_patches)


class MultiEncoderSlideDataset:
    """Slide-level dataset returning aligned raw features for every encoder."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        fold: int,
        split: str,
        clinical_df: pd.DataFrame,
        label_col: str = "d3m3",
        max_patches: int = 0,
        training: bool = True,
    ) -> None:
        rows = manifest[
            (manifest["fold"].astype(int) == int(fold))
            & (manifest["split"].astype(str) == split)
        ].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows found for fold={fold}, split={split}.")

        self.label_col = label_col
        self.max_patches = int(max_patches)
        self.training = bool(training)
        self.clinical = clinical_df.set_index("slide_id")
        self.encoder_names = sorted(rows["feature_dir"].astype(str).unique().tolist())
        self.slide_rows: Dict[str, pd.DataFrame] = {}

        for slide_id, group in rows.groupby("slide_id"):
            sid = str(slide_id)
            if sid not in self.clinical.index:
                continue
            if pd.isna(self.clinical.loc[sid, label_col]):
                continue
            present = set(group["feature_dir"].astype(str))
            if set(self.encoder_names).issubset(present):
                self.slide_rows[sid] = group.copy()

        self.slide_ids = sorted(self.slide_rows)
        if not self.slide_ids:
            raise RuntimeError(
                f"Fold {fold}, split={split}: no usable slides after clinical/feature filtering."
            )

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int, str]:
        slide_id = self.slide_ids[int(idx)]
        rows = self.slide_rows[slide_id]
        features: Dict[str, np.ndarray] = {}

        for _, row in rows.iterrows():
            encoder = str(row["feature_dir"])
            h5_path = Path(row["h5_path"])
            dataset_key = (
                str(row["dataset_key"])
                if pd.notna(row.get("dataset_key", None))
                else None
            )
            features[encoder] = read_h5_features(h5_path, dataset_key=dataset_key)

        min_patches = min(array.shape[0] for array in features.values())
        patch_indices = subset_patch_indices(
            min_patches, self.max_patches, training=self.training
        )
        tensor_features: Dict[str, torch.Tensor] = {}
        for encoder in self.encoder_names:
            array = features[encoder][:min_patches]
            if patch_indices is not None:
                array = array[patch_indices]
            tensor_features[encoder] = torch.from_numpy(array.astype(np.float32))

        label = int(self.clinical.loc[slide_id, self.label_col])
        return tensor_features, label, slide_id
