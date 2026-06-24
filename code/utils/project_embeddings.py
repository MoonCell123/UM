"""
Project patch-level foundation-model embeddings to a shared 512-D dimension.

CV policy:
    - Build 5 folds from the intersection of available slide IDs.
    - For each fold and each foundation-model feature folder, fit the
      projection only on training-slide embeddings.
    - Apply that fitted projection to both train and validation slides.

This avoids using validation-slide feature distributions while still producing
projected embeddings for every split.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.model_selection import KFold


DEFAULT_FEAT_BASE = Path(r"L:\20x_256px_0px_overlap")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "Projected_Embeddings_512"
DEFAULT_FEATURE_DIRS = ["features_hoptimus1", "features_virchow", "features_hoptimus0"]
FEATURE_KEYS = ("feats", "features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV-safe projection of h5 patch embeddings to 512 dimensions."
    )
    parser.add_argument("--feat-base", type=Path, default=DEFAULT_FEAT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dirs", nargs="+", default=DEFAULT_FEATURE_DIRS)
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=["pca", "incremental_pca"],
        default="pca",
        help="Unsupervised projection fitted on training slides only.",
    )
    parser.add_argument(
        "--max-train-patches",
        type=int,
        default=50000,
        help="Maximum sampled training patches for fitting one projector. Use <=0 for all patches.",
    )
    parser.add_argument(
        "--random-patches-per-slide",
        type=int,
        default=1000,
        help="Maximum patches sampled from each training slide for fitting. Use <=0 for all patches.",
    )
    parser.add_argument(
        "--dataset-key",
        default="features",
        help="Dataset key written to projected h5 files.",
    )
    parser.add_argument(
        "--identity-if-same-dim",
        action="store_true",
        help="Use identity projection when input_dim already equals target_dim.",
    )
    return parser.parse_args()


class IdentityProjector:
    """Small sklearn-like identity transformer for already-512-D features."""

    def fit(self, x: np.ndarray) -> "IdentityProjector":
        self.n_features_in_ = int(x.shape[1])
        self.n_components_ = int(x.shape[1])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


def list_h5_slide_ids(feature_dir: Path) -> List[str]:
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    return sorted(path.stem for path in feature_dir.glob("*.h5"))


def get_common_slide_ids(feat_base: Path, feature_dirs: Sequence[str]) -> List[str]:
    id_sets = []
    for feature_dir_name in feature_dirs:
        slide_ids = set(list_h5_slide_ids(feat_base / feature_dir_name))
        if not slide_ids:
            raise RuntimeError(f"No .h5 files found in {feat_base / feature_dir_name}")
        id_sets.append(slide_ids)
    return sorted(set.intersection(*id_sets))


def get_dataset_name(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as f:
        for key in FEATURE_KEYS:
            if key in f:
                return key
        candidates = [key for key in f.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(f.keys())}")
        return candidates[0]


def read_h5_features_and_coords(h5_path: Path) -> Tuple[np.ndarray, np.ndarray | None, str]:
    dataset_name = get_dataset_name(h5_path)
    with h5py.File(h5_path, "r") as f:
        features = np.asarray(f[dataset_name][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:]) if "coords" in f else None
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected feature shape [N, D], got {features.shape}")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, coords, dataset_name


def stable_seed(seed: int, fold: int, feature_dir_name: str) -> int:
    feature_seed = zlib.crc32(feature_dir_name.encode("utf-8")) % 100000
    return int(seed + fold * 1000 + feature_seed)


def sample_training_patches(
    feature_dir: Path,
    slide_ids: Sequence[str],
    rng: np.random.Generator,
    patches_per_slide: int,
    max_total_patches: int,
) -> np.ndarray:
    sampled = []
    for slide_id in slide_ids:
        h5_path = feature_dir / f"{slide_id}.h5"
        features, _, _ = read_h5_features_and_coords(h5_path)
        if patches_per_slide > 0 and features.shape[0] > patches_per_slide:
            idx = rng.choice(features.shape[0], size=patches_per_slide, replace=False)
            features = features[idx]
        sampled.append(features)

    matrix = np.concatenate(sampled, axis=0).astype(np.float32)
    if max_total_patches > 0 and matrix.shape[0] > max_total_patches:
        idx = rng.choice(matrix.shape[0], size=max_total_patches, replace=False)
        matrix = matrix[idx]
    return matrix


def fit_projector(
    train_matrix: np.ndarray,
    target_dim: int,
    method: str,
    identity_if_same_dim: bool,
):
    n_samples, input_dim = train_matrix.shape
    if target_dim > input_dim:
        raise ValueError(f"target_dim={target_dim} cannot exceed input feature dim={input_dim}.")
    if target_dim > n_samples:
        raise ValueError(
            f"target_dim={target_dim} cannot exceed sampled training patches={n_samples}. "
            "Increase --max-train-patches or --random-patches-per-slide."
        )
    if identity_if_same_dim and input_dim == target_dim:
        return IdentityProjector().fit(train_matrix)
    if method == "incremental_pca":
        projector = IncrementalPCA(n_components=target_dim, batch_size=min(4096, n_samples))
    else:
        projector = PCA(n_components=target_dim, svd_solver="randomized", random_state=42)
    return projector.fit(train_matrix)


def write_projected_h5(
    input_h5: Path,
    output_h5: Path,
    projector,
    dataset_key: str,
    fold: int,
    split: str,
    source_dataset_key: str,
) -> Tuple[int, int, int]:
    features, coords, _ = read_h5_features_and_coords(input_h5)
    projected = projector.transform(features).astype(np.float32)

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset(dataset_key, data=projected, compression="gzip")
        if coords is not None:
            f.create_dataset("coords", data=coords, compression="gzip")
        f.attrs["source_h5"] = str(input_h5)
        f.attrs["source_dataset_key"] = source_dataset_key
        f.attrs["fold"] = int(fold)
        f.attrs["split"] = split
        f.attrs["projection_method"] = type(projector).__name__
        f.attrs["input_dim"] = int(features.shape[1])
        f.attrs["target_dim"] = int(projected.shape[1])
    return int(features.shape[0]), int(features.shape[1]), int(projected.shape[1])


def main() -> None:
    args = parse_args()
    common_slide_ids = get_common_slide_ids(args.feat_base, args.feature_dirs)
    if len(common_slide_ids) < args.cv_folds:
        raise ValueError(f"Need at least {args.cv_folds} common slides, got {len(common_slide_ids)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    slide_array = np.asarray(common_slide_ids)
    manifest_rows: List[Dict[str, object]] = []

    print("=" * 80)
    print("Middle-fusion embedding projection to shared 512-D")
    print("=" * 80)
    print(f"Feature base: {args.feat_base}")
    print(f"Feature dirs: {args.feature_dirs}")
    print(f"Common h5 slides: {len(common_slide_ids)}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Target dim: {args.target_dim}")
    print(f"Output dir: {args.output_dir}")
    print()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(slide_array), start=1):
        train_ids = slide_array[train_idx].tolist()
        val_ids = slide_array[val_idx].tolist()
        split_ids = {"train": train_ids, "val": val_ids}

        for feature_dir_name in args.feature_dirs:
            feature_dir = args.feat_base / feature_dir_name
            rng = np.random.default_rng(stable_seed(args.seed, fold, feature_dir_name))

            print(
                f"[fold {fold}] {feature_dir_name}: "
                f"fit on {len(train_ids)} train slides, transform {len(val_ids)} val slides"
            )
            train_matrix = sample_training_patches(
                feature_dir=feature_dir,
                slide_ids=train_ids,
                rng=rng,
                patches_per_slide=args.random_patches_per_slide,
                max_total_patches=args.max_train_patches,
            )
            projector = fit_projector(
                train_matrix=train_matrix,
                target_dim=args.target_dim,
                method=args.method,
                identity_if_same_dim=args.identity_if_same_dim,
            )

            fold_dir = args.output_dir / f"fold_{fold}" / feature_dir_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            with open(fold_dir / "projection_model.pkl", "wb") as f:
                pickle.dump(projector, f)

            for split, slide_ids in split_ids.items():
                for slide_id in slide_ids:
                    input_h5 = feature_dir / f"{slide_id}.h5"
                    output_h5 = fold_dir / split / f"{slide_id}.h5"
                    source_key = get_dataset_name(input_h5)
                    n_patches, input_dim, output_dim = write_projected_h5(
                        input_h5=input_h5,
                        output_h5=output_h5,
                        projector=projector,
                        dataset_key=args.dataset_key,
                        fold=fold,
                        split=split,
                        source_dataset_key=source_key,
                    )
                    manifest_rows.append({
                        "fold": fold,
                        "split": split,
                        "feature_dir": feature_dir_name,
                        "slide_id": slide_id,
                        "input_h5": str(input_h5),
                        "output_h5": str(output_h5),
                        "n_patches": n_patches,
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                    })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "projection_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    config = {
        "feat_base": str(args.feat_base),
        "feature_dirs": args.feature_dirs,
        "target_dim": args.target_dim,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "method": args.method,
        "max_train_patches": args.max_train_patches,
        "random_patches_per_slide": args.random_patches_per_slide,
        "dataset_key": args.dataset_key,
        "identity_if_same_dim": args.identity_if_same_dim,
        "slide_source": "intersection of .h5 names in selected features_* directories",
        "fit_policy": "fit projector on train split only; transform train and validation split",
        "common_slide_count": len(common_slide_ids),
    }
    with open(args.output_dir / "projection_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved config: {args.output_dir / 'projection_config.json'}")


if __name__ == "__main__":
    main()
