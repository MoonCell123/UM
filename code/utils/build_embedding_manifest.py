"""
Create CV manifests for middle-fusion experiments over h5 embeddings.

Projection itself is intentionally not performed here. In the CGME/GME design,
projection should be a trainable network layer optimized by the downstream
classification loss, not an offline reconstruction objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_FEAT_BASE = Path(r"L:\20x_256px_0px_overlap")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "Middle_Fusion_Manifests"
DEFAULT_CLINICAL_PATH = Path(__file__).resolve().parents[2] / "clinical_information.csv"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
FEATURE_KEYS = ("feats", "features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 5-fold train/val manifests for middle-fusion h5 embeddings."
    )
    parser.add_argument("--feat-base", type=Path, default=DEFAULT_FEAT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dirs", nargs="+", default=DEFAULT_FEATURE_DIRS)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clinical-path", type=Path, default=DEFAULT_CLINICAL_PATH)
    parser.add_argument("--label-col", default="d3m3")
    return parser.parse_args()


def patient_id_from_slide_id(slide_id: str) -> str:
    """Return a patient-level identifier from a slide/sample identifier."""
    parts = str(slide_id).split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3])
    return "-".join(parts[:-1]) if len(parts) > 1 else str(slide_id)


def load_labels(clinical_path: Path, label_col: str) -> pd.DataFrame:
    if clinical_path.suffix.lower() == ".csv":
        clinical = pd.read_csv(clinical_path, encoding="utf-8-sig")
    else:
        clinical = pd.read_excel(clinical_path)
    if "slide_id" not in clinical.columns:
        raise ValueError(f"Clinical table must contain slide_id: {clinical_path}")
    if label_col not in clinical.columns:
        scna_col = "SCNA Cluster No."
        if label_col != "d3m3" or scna_col not in clinical.columns:
            raise ValueError(
                f"Clinical table must contain {label_col!r}; could not derive it from {clinical_path}"
            )
        cluster = pd.to_numeric(clinical[scna_col], errors="coerce")
        clinical[label_col] = cluster.map(lambda value: 0 if value in (1, 2) else 1 if value in (3, 4) else np.nan)
    return clinical[["slide_id", label_col]].copy()


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


def inspect_h5(h5_path: Path) -> Dict[str, object]:
    dataset_name = get_dataset_name(h5_path)
    with h5py.File(h5_path, "r") as f:
        shape = tuple(int(v) for v in f[dataset_name].shape)
        has_coords = "coords" in f
    if len(shape) != 2:
        raise ValueError(f"{h5_path}: expected feature shape [N, D], got {shape}")
    return {
        "dataset_key": dataset_name,
        "n_patches": shape[0],
        "feature_dim": shape[1],
        "has_coords": has_coords,
    }


def build_manifest_rows(
    feat_base: Path,
    feature_dirs: Sequence[str],
    fold: int,
    split: str,
    slide_ids: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    for slide_id in slide_ids:
        for feature_dir_name in feature_dirs:
            h5_path = feat_base / feature_dir_name / f"{slide_id}.h5"
            info = inspect_h5(h5_path)
            rows.append({
                "fold": fold,
                "split": split,
                "slide_id": slide_id,
                "feature_dir": feature_dir_name,
                "h5_path": str(h5_path),
                **info,
            })
    return rows


def main() -> None:
    args = parse_args()
    common_slide_ids = get_common_slide_ids(args.feat_base, args.feature_dirs)
    if len(common_slide_ids) < args.cv_folds:
        raise ValueError(f"Need at least {args.cv_folds} common slides, got {len(common_slide_ids)}.")

    labels_df = load_labels(args.clinical_path, args.label_col)
    labels_df["slide_id"] = labels_df["slide_id"].astype(str)
    labels_df = labels_df.dropna(subset=[args.label_col]).drop_duplicates("slide_id")
    label_map = labels_df.set_index("slide_id")[args.label_col].astype(int).to_dict()
    missing_labels = sorted(set(common_slide_ids) - set(label_map))
    if missing_labels:
        raise ValueError(f"Missing labels for {len(missing_labels)} common slides, e.g. {missing_labels[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slide_array = np.asarray(common_slide_ids)
    slide_labels = np.asarray([label_map[str(slide_id)] for slide_id in slide_array], dtype=int)
    patient_groups = np.asarray([patient_id_from_slide_id(str(slide_id)) for slide_id in slide_array])
    splitter = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    all_rows = []

    print("=" * 80)
    print("Middle-fusion h5 manifest creation")
    print("=" * 80)
    print(f"Feature base: {args.feat_base}")
    print(f"Feature dirs: {args.feature_dirs}")
    print(f"Common h5 slides: {len(common_slide_ids)}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Output dir: {args.output_dir}")
    print()

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(slide_array, slide_labels, groups=patient_groups), start=1
    ):
        train_ids = slide_array[train_idx].tolist()
        val_ids = slide_array[val_idx].tolist()
        train_patients = {patient_id_from_slide_id(slide_id) for slide_id in train_ids}
        val_patients = {patient_id_from_slide_id(slide_id) for slide_id in val_ids}
        if train_patients & val_patients:
            raise RuntimeError(f"Patient leakage in fold {fold}: {sorted(train_patients & val_patients)}")
        print(f"fold {fold}: train={len(train_ids)}, val={len(val_ids)}")

        fold_rows = []
        fold_rows.extend(build_manifest_rows(args.feat_base, args.feature_dirs, fold, "train", train_ids))
        fold_rows.extend(build_manifest_rows(args.feat_base, args.feature_dirs, fold, "val", val_ids))
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(args.output_dir / f"fold_{fold}_manifest.csv", index=False, encoding="utf-8-sig")
        all_rows.extend(fold_rows)

    manifest_df = pd.DataFrame(all_rows)
    manifest_path = args.output_dir / "middle_fusion_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    config = {
        "feat_base": str(args.feat_base),
        "feature_dirs": args.feature_dirs,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "label_col": args.label_col,
        "splitter": "StratifiedGroupKFold",
        "patient_id_policy": "TCGA first three fields; otherwise remove final sample field",
        "slide_source": "intersection of .h5 names in selected features_* directories",
        "projection_policy": "trainable ProjectionHead is used inside the model and optimized by classification loss",
        "common_slide_count": len(common_slide_ids),
    }
    config_path = args.output_dir / "middle_fusion_manifest_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
