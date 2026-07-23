"""Compute coordinate-aligned per-WSI Linear CKA across foundation models.

Each H5 file is treated as one WSI patch bag. Patch rows are aligned by their
stored coordinates before deterministic subsampling. For every WSI, the script
saves an encoder-by-encoder Linear CKA matrix and encoder-level mean CKA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Linear_CKA"
FEATURE_KEYS = ("features", "feats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute one coordinate-aligned Linear CKA matrix per WSI."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file. CLI args override config.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-dirs",
        nargs="+",
        default=None,
        help="Feature folders to compare. Default: all feature_dir values in the manifest.",
    )
    parser.add_argument(
        "--slide-ids",
        nargs="+",
        default=None,
        help="Optional WSI ids to process. Default: every complete WSI in the manifest.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=1024,
        help="Deterministically sample this many common patches per WSI. 0 uses all common patches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for Gram matrix multiplication, e.g. cuda, cuda:1, or cpu.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid WSI instead of recording it in errors.csv.",
    )
    # Parse config first, then let explicit CLI values override it.
    config_probe, _ = parser.parse_known_args()
    if config_probe.config is not None:
        config_path = config_probe.config
        if not config_path.exists():
            candidate = PROJECT_ROOT / "code" / "config" / config_path
            if candidate.exists():
                config_path = candidate
        if not config_path.exists():
            parser.error(f"Config file not found: {config_probe.config}")
        if config_path.suffix.lower() == ".json":
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        else:
            try:
                import yaml
            except ImportError as exc:
                parser.error(f"Reading YAML config requires PyYAML: {exc}")
            with open(config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            parser.error("CKA config must be a YAML/JSON mapping.")
        normalized = {str(key).replace("-", "_"): value for key, value in config.items()}
        parser.set_defaults(**normalized)
    args = parser.parse_args()
    if args.max_patches < 0:
        parser.error("--max-patches must be non-negative.")
    return args


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def infer_dataset_key(h5_path: Path, configured_key: str | None) -> str:
    with h5py.File(h5_path, "r") as handle:
        if configured_key and configured_key in handle:
            return configured_key
        for key in FEATURE_KEYS:
            if key in handle:
                return key
        candidates = [key for key in handle.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found; keys={list(handle.keys())}")
        return candidates[0]


def read_coords(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"{h5_path}: missing coords dataset required for patch alignment.")
        coords = np.asarray(handle["coords"][:])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{h5_path}: expected coords [N, 2], got {coords.shape}")
    return coords


def coordinate_indices(coords_by_encoder: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return common coordinates and row indices in a shared coordinate order."""
    names = list(coords_by_encoder)
    first = coords_by_encoder[names[0]]
    if all(np.array_equal(first, coords_by_encoder[name]) for name in names[1:]):
        indices = np.arange(first.shape[0], dtype=np.int64)
        return first, {name: indices.copy() for name in names}

    maps: Dict[str, Dict[Tuple[int, int], int]] = {}
    common = None
    for name in names:
        coords = coords_by_encoder[name]
        mapping = {tuple(map(int, coord)): idx for idx, coord in enumerate(coords)}
        if len(mapping) != len(coords):
            raise ValueError(f"{name}: duplicate patch coordinates prevent unambiguous alignment.")
        maps[name] = mapping
        coord_set = set(mapping)
        common = coord_set if common is None else common.intersection(coord_set)

    if not common:
        raise ValueError("The selected encoders have no common patch coordinates.")
    common_coords = np.asarray(sorted(common), dtype=first.dtype)
    indices_by_encoder = {
        name: np.asarray([maps[name][tuple(map(int, coord))] for coord in common_coords], dtype=np.int64)
        for name in names
    }
    return common_coords, indices_by_encoder


def stable_slide_seed(seed: int, slide_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{slide_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_common_rows(
    common_count: int,
    indices_by_encoder: Mapping[str, np.ndarray],
    max_patches: int,
    seed: int,
    slide_id: str,
) -> Dict[str, np.ndarray]:
    if max_patches == 0 or common_count <= max_patches:
        selected = np.arange(common_count, dtype=np.int64)
    else:
        rng = np.random.default_rng(stable_slide_seed(seed, slide_id))
        selected = np.sort(rng.choice(common_count, size=max_patches, replace=False))
    return {name: indices[selected] for name, indices in indices_by_encoder.items()}


def read_feature_rows(h5_path: Path, dataset_key: str, row_indices: np.ndarray) -> np.ndarray:
    """Read arbitrary H5 rows and restore the requested shared-coordinate order."""
    order = np.argsort(row_indices)
    sorted_indices = row_indices[order]
    if np.unique(sorted_indices).size != sorted_indices.size:
        raise ValueError(f"{h5_path}: sampled row indices are not unique.")
    with h5py.File(h5_path, "r") as handle:
        dataset = handle[dataset_key]
        if dataset.ndim != 2:
            raise ValueError(f"{h5_path}:{dataset_key}: expected [N, D], got {dataset.shape}")
        features_sorted = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse_order = np.argsort(order)
    features = features_sorted[inverse_order]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_linear_gram(features: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.as_tensor(features, device=device, dtype=dtype)
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.transpose(0, 1)
    norm = torch.linalg.vector_norm(gram)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 0.0:
        raise ValueError("Centered feature Gram matrix has zero or non-finite norm.")
    return gram / norm


def linear_cka_matrix(
    features_by_encoder: Mapping[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    names = list(features_by_encoder)
    sample_counts = {features.shape[0] for features in features_by_encoder.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"Aligned encoders have inconsistent sample counts: {sorted(sample_counts)}")
    if next(iter(sample_counts)) < 2:
        raise ValueError("Linear CKA requires at least two aligned patches.")

    grams = {
        name: normalized_linear_gram(features_by_encoder[name], device, dtype)
        for name in names
    }
    matrix = np.eye(len(names), dtype=np.float64)
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            right = names[j]
            value = torch.sum(grams[left] * grams[right])
            cka = float(value.detach().cpu())
            cka = float(np.clip(cka, 0.0, 1.0))
            matrix[i, j] = matrix[j, i] = cka
    return pd.DataFrame(matrix, index=names, columns=names)


def unique_manifest_rows(manifest: pd.DataFrame, feature_dirs: Sequence[str]) -> pd.DataFrame:
    required = {"slide_id", "feature_dir", "h5_path", "dataset_key"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    rows = manifest[manifest["feature_dir"].astype(str).isin(feature_dirs)].copy()
    rows["slide_id"] = rows["slide_id"].astype(str)
    rows["feature_dir"] = rows["feature_dir"].astype(str)

    conflicts = []
    for keys, group in rows.groupby(["slide_id", "feature_dir"]):
        paths = group["h5_path"].astype(str).unique()
        dataset_keys = group["dataset_key"].fillna("").astype(str).unique()
        if len(paths) != 1 or len(dataset_keys) != 1:
            conflicts.append(keys)
    if conflicts:
        raise ValueError(f"Manifest contains conflicting duplicate rows, examples: {conflicts[:5]}")
    return rows.drop_duplicates(["slide_id", "feature_dir"])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    manifest = pd.read_csv(args.manifest)
    available = sorted(manifest["feature_dir"].dropna().astype(str).unique().tolist())
    feature_dirs = list(args.feature_dirs) if args.feature_dirs else available
    missing = sorted(set(feature_dirs) - set(available))
    if missing:
        raise ValueError(f"Requested feature dirs not present in manifest: {missing}")
    if len(feature_dirs) < 2:
        raise ValueError("Linear CKA requires at least two feature encoders.")

    rows = unique_manifest_rows(manifest, feature_dirs)
    complete_slides = [
        slide_id
        for slide_id, group in rows.groupby("slide_id")
        if set(group["feature_dir"]) == set(feature_dirs)
    ]
    if args.slide_ids:
        requested_slides = set(map(str, args.slide_ids))
        missing_slides = sorted(requested_slides - set(complete_slides))
        if missing_slides:
            raise ValueError(f"Requested WSIs are missing one or more selected encoders: {missing_slides}")
        complete_slides = [slide_id for slide_id in complete_slides if slide_id in requested_slides]
    if not complete_slides:
        raise RuntimeError("No WSI has all requested feature encoders.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest": str(args.manifest),
                "feature_dirs": feature_dirs,
                "slide_ids": sorted(complete_slides),
                "max_patches": args.max_patches,
                "seed": args.seed,
                "device": str(device),
                "dtype": args.dtype,
                "estimator": "centered_linear_cka_biased",
                "patch_alignment": "coords_intersection",
            },
            handle,
            indent=2,
        )

    encoder_rows = []
    pair_rows = []
    wsi_rows = []
    errors = []
    for slide_number, slide_id in enumerate(sorted(complete_slides), start=1):
        print(f"[{slide_number}/{len(complete_slides)}] {slide_id}")
        try:
            slide_rows = rows[rows["slide_id"] == slide_id].set_index("feature_dir")
            paths = {name: Path(str(slide_rows.loc[name, "h5_path"])) for name in feature_dirs}
            keys = {
                name: infer_dataset_key(
                    paths[name],
                    None
                    if pd.isna(slide_rows.loc[name, "dataset_key"])
                    else str(slide_rows.loc[name, "dataset_key"]),
                )
                for name in feature_dirs
            }
            coords = {name: read_coords(paths[name]) for name in feature_dirs}
            common_coords, aligned_indices = coordinate_indices(coords)
            sampled_indices = sample_common_rows(
                common_count=len(common_coords),
                indices_by_encoder=aligned_indices,
                max_patches=args.max_patches,
                seed=args.seed,
                slide_id=slide_id,
            )
            features = {
                name: read_feature_rows(paths[name], keys[name], sampled_indices[name])
                for name in feature_dirs
            }
            matrix = linear_cka_matrix(features, device=device, dtype=dtype)
            matrix.to_csv(matrix_dir / f"{slide_id}_linear_cka.csv", encoding="utf-8-sig", float_format="%.6f")

            values = matrix.to_numpy()
            pair_values = []
            for i, left in enumerate(feature_dirs):
                other_values = np.delete(values[i], i)
                encoder_rows.append(
                    {
                        "slide_id": slide_id,
                        "encoder": left,
                        "mean_cka_to_others": float(other_values.mean()),
                        "mean_cka_dissimilarity": float(1.0 - other_values.mean()),
                        "min_cka_to_others": float(other_values.min()),
                        "max_cka_to_others": float(other_values.max()),
                        "common_patches": int(len(common_coords)),
                        "sampled_patches": int(next(iter(features.values())).shape[0]),
                    }
                )
                for j in range(i + 1, len(feature_dirs)):
                    right = feature_dirs[j]
                    cka = float(values[i, j])
                    pair_values.append(cka)
                    pair_rows.append(
                        {
                            "slide_id": slide_id,
                            "encoder_a": left,
                            "encoder_b": right,
                            "linear_cka": cka,
                            "common_patches": int(len(common_coords)),
                            "sampled_patches": int(next(iter(features.values())).shape[0]),
                        }
                    )
            wsi_rows.append(
                {
                    "slide_id": slide_id,
                    "mean_pairwise_cka": float(np.mean(pair_values)),
                    "min_pairwise_cka": float(np.min(pair_values)),
                    "max_pairwise_cka": float(np.max(pair_values)),
                    "common_patches": int(len(common_coords)),
                    "sampled_patches": int(next(iter(features.values())).shape[0]),
                }
            )
        except Exception as exc:
            if args.strict:
                raise
            errors.append({"slide_id": slide_id, "error": str(exc)})
            print(f"[Warning] {slide_id}: {exc}")

    if not wsi_rows:
        raise RuntimeError("Linear CKA failed for every WSI; inspect the console errors.")
    pd.DataFrame(encoder_rows).to_csv(
        output_dir / "encoder_mean_cka_by_wsi.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "pairwise_cka_by_wsi.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    pd.DataFrame(wsi_rows).to_csv(
        output_dir / "wsi_cka_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    encoder_summary = (
        pd.DataFrame(encoder_rows)
        .groupby("encoder")["mean_cka_to_others"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    encoder_summary["mean_cka_dissimilarity"] = 1.0 - encoder_summary["mean"]
    encoder_summary.to_csv(
        output_dir / "encoder_cka_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")

    print(f"Completed WSIs: {len(wsi_rows)} / {len(complete_slides)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
