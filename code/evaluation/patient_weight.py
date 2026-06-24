"""
Compute patient-specific model weights from feature embeddings.

Notes:
    - The MLP is not trained in this script. Pass --checkpoint for meaningful
      learned weights. Without a checkpoint, a fixed random seed is used so the
      output is deterministic and mainly useful for checking the pipeline.
    - h5 keys "feats" and "features" are supported. If neither exists, the
      first non-"coords" dataset is used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEFAULT_FEAT_BASE = r"L:\20x_256px_0px_overlap"

DEFAULT_PATIENT_IDS = []
FEATURE_KEYS = ("feats", "features")


class PatientWeightMLP(nn.Module):
    """Shared MLP that maps one patient-model representation to one logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_patient_name():
    cur_dir = r"L:\20x_256px_0px_overlap\features_conch_v1"
    for filename in os.listdir(cur_dir):
        if filename.endswith(".h5"):
            pure_name = os.path.splitext(filename)[0]
            DEFAULT_PATIENT_IDS.append(pure_name)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_feature_dirs(feat_base: Path, expected_count: int | None = 20) -> List[Path]:
    if not feat_base.is_dir():
        raise FileNotFoundError(f"Feature base directory does not exist: {feat_base}")

    feature_dirs = sorted(
        p for p in feat_base.iterdir()
        if p.is_dir() and p.name.startswith("features_")
    )
    if not feature_dirs:
        raise RuntimeError(f"No features_* directories found under: {feat_base}")

    if expected_count is not None and len(feature_dirs) != expected_count:
        print(
            f"[Warning] Found {len(feature_dirs)} features_* directories; "
            f"expected {expected_count}. Processing all found directories."
        )
    return feature_dirs


def read_h5_features(h5_path: Path) -> np.ndarray:
    if not h5_path.is_file():
        raise FileNotFoundError(str(h5_path))

    with h5py.File(h5_path, "r") as f:
        dataset_name = None
        for key in FEATURE_KEYS:
            if key in f:
                dataset_name = key
                break
        if dataset_name is None:
            candidates = [key for key in f.keys() if key != "coords"]
            if not candidates:
                raise KeyError(
                    f"{h5_path}: no feature dataset found. Available keys: {list(f.keys())}"
                )
            dataset_name = candidates[0]

        features = f[dataset_name][:]

    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected feature shape [N, D], got {features.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{h5_path}: empty feature matrix, got {features.shape}")

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_stats(values: np.ndarray) -> List[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return [0.0] * 8
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.percentile(values, 25)),
        float(np.median(values)),
        float(np.percentile(values, 75)),
        float(np.max(values)),
        float(np.mean(np.abs(values))),
    ]


def build_patient_feature(features: np.ndarray) -> np.ndarray:
    """
    Convert variable-size patch embeddings [N, D] into a fixed-length vector.
    """
    n_patches, feat_dim = features.shape
    pooled_mean = features.mean(axis=0)
    pooled_std = features.std(axis=0)
    patch_l2 = np.linalg.norm(features, axis=1)

    summary = [
        float(np.log1p(n_patches)),
        float(np.log1p(feat_dim)),
    ]
    summary.extend(_safe_stats(features))
    summary.extend(_safe_stats(pooled_mean))
    summary.extend(_safe_stats(pooled_std))
    summary.extend(_safe_stats(patch_l2))
    return np.asarray(summary, dtype=np.float32)


def standardize_patient_features(
    rows: List[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.stack([row["patient_feature"] for row in rows]).astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std, mean.squeeze(0), std.squeeze(0)


def load_checkpoint_if_needed(model: nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def compute_weights(
    rows: List[Dict[str, object]],
    standardized_features: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> List[Dict[str, object]]:
    model.eval()
    rows_by_patient: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        print("Start tackle " + str(row["patient_id"]))
        rows_by_patient.setdefault(str(row["patient_id"]), []).append(idx)

    output_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for patient_id, indices in rows_by_patient.items():
            x = torch.from_numpy(standardized_features[indices]).float().to(device)
            logits = model(x)
            weights = logits.cpu().numpy()
            logits_np = logits.cpu().numpy()

            order = np.argsort(-weights)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)

            for local_idx, row_idx in enumerate(indices):
                row = rows[row_idx].copy()
                row.pop("patient_feature", None)
                row["logit"] = float(logits_np[local_idx])
                row["weight"] = float(weights[local_idx])
                row["rank"] = int(ranks[local_idx])
                output_rows.append(row)

    return output_rows


def collect_patient_model_features(
    feature_dirs: Sequence[Path],
    patient_ids: Iterable[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for feat_dir in feature_dirs:
        model_name = feat_dir.name.removeprefix("features_")
        for patient_id in patient_ids:
            h5_path = feat_dir / f"{patient_id}.h5"
            try:
                features = read_h5_features(h5_path)
                patient_feature = build_patient_feature(features)
            except Exception as exc:
                missing_rows.append(
                    {
                        "patient_id": patient_id,
                        "model_name": model_name,
                        "feature_dir": feat_dir.name,
                        "h5_path": str(h5_path),
                        "error": str(exc),
                    }
                )
                continue

            rows.append(
                {
                    "patient_id": patient_id,
                    "model_name": model_name,
                    "feature_dir": feat_dir.name,
                    "h5_path": str(h5_path),
                    "n_patches": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                    "patient_feature": patient_feature,
                }
            )
    return rows, missing_rows


def save_outputs(
    weighted_rows: List[Dict[str, object]],
    missing_rows: List[Dict[str, object]],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    output_dir: Path,
    config: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame(weighted_rows)
    detail_df = detail_df.sort_values(["patient_id", "rank", "model_name"])
    detail_df.to_csv(output_dir / "patient_model_weights.csv", index=False)

    matrix_df = detail_df.pivot(
        index="patient_id",
        columns="model_name",
        values="weight",
    )
    matrix_df.to_csv(output_dir / "patient_weight_matrix.csv")

    dist_df = (
        detail_df.groupby("model_name")["weight"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    dist_df.to_csv(output_dir / "weight_distribution_by_model.csv", index=False)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(output_dir / "missing_or_failed_files.csv", index=False)

    np.save(output_dir / "standardizer_mean.npy", scaler_mean)
    np.save(output_dir / "standardizer_std.npy", scaler_std)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute patient-specific softmax weights over features_* model folders."
    )
    parser.add_argument("--feat-base", default=DEFAULT_FEAT_BASE, help="Base directory containing features_* folders.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: ./output/patient_weight/<timestamp>")
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENT_IDS), help="Patient/slide IDs without .h5")
    parser.add_argument("--expected-model-count", type=int, default=20, help="Expected number of features_* directories.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout.")
    parser.add_argument("--checkpoint", default=None, help="Optional trained MLP checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when no checkpoint is provided.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for MLP inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    feat_base = Path(args.feat_base)
    project_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "output" / "patient_weight" / timestamp
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    feature_dirs = find_feature_dirs(feat_base, args.expected_model_count)
    rows, missing_rows = collect_patient_model_features(feature_dirs, args.patients)
    if not rows:
        raise RuntimeError("No patient-model features were loaded. Check paths and h5 file names.")

    standardized_features, scaler_mean, scaler_std = standardize_patient_features(rows)
    input_dim = standardized_features.shape[1]

    model = PatientWeightMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    load_checkpoint_if_needed(model, args.checkpoint, device)

    weighted_rows = compute_weights(rows, standardized_features, model, device)
    config = {
        "feat_base": str(feat_base),
        "feature_dirs": [p.name for p in feature_dirs],
        "patients": list(args.patients),
        "expected_model_count": args.expected_model_count,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "device": str(device),
        "input_dim": input_dim,
        "untrained_mlp": args.checkpoint is None,
    }
    save_outputs(weighted_rows, missing_rows, scaler_mean, scaler_std, output_dir, config)

    print(f"Loaded rows: {len(rows)}")
    print(f"Missing/failed rows: {len(missing_rows)}")
    print(f"Output directory: {output_dir}")
    print("Saved: patient_model_weights.csv, patient_weight_matrix.csv, weight_distribution_by_model.csv")


if __name__ == "__main__":
    get_patient_name()
    main()
