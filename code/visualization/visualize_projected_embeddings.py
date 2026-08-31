"""Visualize per-encoder projected embeddings together with the static Beacon.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_DIR, CODE_DIR / "architecture"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projection_head import MultiEncoderProjectionHead


FEATURE_KEYS = ("feats", "features")
ENCODER_LABELS = {
    "features_hoptimus0": "Hoptimus0",
    "features_hoptimus1": "Hoptimus1",
    "features_virchow": "Virchow",
}
ENCODER_COLORS = {
    "features_hoptimus0": "#2f80ed",
    "features_hoptimus1": "#f2994a",
    "features_virchow": "#27ae60",
}
LABEL_COLORS = {
    0: "#3b82f6",
    1: "#ef4444",
}


class MultiEncoderLinearAligner(torch.nn.Module):
    """Fallback loader for checkpoints whose projection is bare Linear(d -> D)."""

    def __init__(self, input_dims: Mapping[str, int], target_dim: int):
        super().__init__()
        self.heads = torch.nn.ModuleDict({
            name: torch.nn.Linear(int(dim), int(target_dim))
            for name, dim in input_dims.items()
        })

    def forward(self, features_by_encoder: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: self.heads[name](features) for name, features in features_by_encoder.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA visualization of projected embeddings for Hoptimus0, Hoptimus1, Virchow, and Beacon."
    )
    parser.add_argument("--fold-dir", type=Path, required=True, help="Fold output directory containing best_gme_model.pt and static_beacon_and_baselines.pt.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint. Default: <fold-dir>/best_gme_model.pt.")
    parser.add_argument("--beacon-file", type=Path, default=None, help="Static Beacon file. Default: <fold-dir>/static_beacon_and_baselines.pt.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "output" / "Manifests" / "Manifests_seed35" / "fusion_manifest.csv")
    parser.add_argument("--clinical-path", type=Path, default=PROJECT_ROOT / "clinical_information.csv")
    parser.add_argument("--fold", type=int, default=None, help="Fold id. Default: infer from --fold-dir or Beacon file.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <fold-dir>/projected_embedding_vis.")
    parser.add_argument("--max-patches-per-slide", type=int, default=300)
    parser.add_argument("--max-points-per-encoder", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_fold(fold_dir: Path, beacon_payload: Mapping[str, object], explicit_fold: int | None) -> int:
    if explicit_fold is not None:
        return int(explicit_fold)
    if isinstance(beacon_payload, dict) and "fold" in beacon_payload:
        return int(beacon_payload["fold"])
    match = re.search(r"fold_(\d+)", str(fold_dir))
    if match:
        return int(match.group(1))
    raise ValueError("Could not infer fold id. Pass --fold explicitly.")


def load_clinical_labels(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    labels: Dict[str, int] = {}
    for row in read_csv_rows(path):
        slide_id = str(row.get("slide_id", "")).strip()
        cluster_text = str(row.get("SCNA Cluster No.", "")).strip()
        if not slide_id or not cluster_text:
            continue
        try:
            cluster = int(float(cluster_text))
        except ValueError:
            continue
        if cluster in (1, 2):
            labels[slide_id] = 0
        elif cluster in (3, 4):
            labels[slide_id] = 1
    return labels


def filter_manifest(rows: Sequence[Mapping[str, str]], fold: int, split: str) -> Dict[str, Dict[str, Mapping[str, str]]]:
    selected: Dict[str, Dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        if int(row["fold"]) != int(fold):
            continue
        if split != "all" and row["split"] != split:
            continue
        selected[row["slide_id"]][row["feature_dir"]] = row
    if not selected:
        raise ValueError(f"No manifest rows found for fold={fold}, split={split}.")
    return dict(selected)


def infer_input_dims(slide_rows: Mapping[str, Mapping[str, Mapping[str, str]]]) -> Dict[str, int]:
    dims: Dict[str, set[int]] = defaultdict(set)
    for enc_rows in slide_rows.values():
        for encoder, row in enc_rows.items():
            dims[encoder].add(int(row["feature_dim"]))
    bad = {name: sorted(values) for name, values in dims.items() if len(values) != 1}
    if bad:
        raise ValueError(f"Expected one feature_dim per encoder, got {bad}")
    return {name: next(iter(values)) for name, values in dims.items()}


def get_dataset_key(h5_path: Path, preferred: str | None) -> str:
    with h5py.File(h5_path, "r") as f:
        if preferred and preferred in f:
            return preferred
        for key in FEATURE_KEYS:
            if key in f:
                return key
        candidates = [key for key in f.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(f.keys())}")
        return candidates[0]


def read_h5_features(row: Mapping[str, str]) -> np.ndarray:
    h5_path = Path(row["h5_path"])
    key = get_dataset_key(h5_path, row.get("dataset_key"))
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f[key][:], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{h5_path}: expected feature shape [N, D], got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def strip_state_dict_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def load_projection_model(
    checkpoint_path: Path,
    input_dims: Mapping[str, int],
    target_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    projection_state = checkpoint.get("projection_heads")
    if projection_state is None and "state_dict" in checkpoint:
        projection_state = strip_state_dict_prefix(checkpoint["state_dict"], "projection_heads.")
    if not projection_state:
        raise KeyError(f"Could not find projection_heads in {checkpoint_path}")

    keys = list(projection_state.keys())
    if any(".proj." in key for key in keys):
        model = MultiEncoderProjectionHead(input_dims=input_dims, target_dim=target_dim, dropout=0.0)
    else:
        model = MultiEncoderLinearAligner(input_dims=input_dims, target_dim=target_dim)

    incompatible = model.load_state_dict(projection_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[Warning] projection load missing={incompatible.missing_keys[:8]}, unexpected={incompatible.unexpected_keys[:8]}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_beacon(beacon_path: Path, device: torch.device) -> tuple[torch.Tensor, Mapping[str, object]]:
    payload = torch.load(beacon_path, map_location=device)
    if not isinstance(payload, dict) or "beacon" not in payload:
        raise KeyError(f"{beacon_path} does not contain a 'beacon' tensor.")
    beacon = payload["beacon"].to(device).float().reshape(1, -1)
    return beacon, payload


def collect_projected_points(
    projection_model: torch.nn.Module,
    slide_rows: Mapping[str, Mapping[str, Mapping[str, str]]],
    encoder_names: Sequence[str],
    labels: Mapping[str, int],
    device: torch.device,
    max_patches_per_slide: int,
    max_points_per_encoder: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    points: List[Dict[str, object]] = []
    per_encoder_counts = {name: 0 for name in encoder_names}

    for slide_id in sorted(slide_rows):
        enc_rows = slide_rows[slide_id]
        missing = [name for name in encoder_names if name not in enc_rows]
        if missing:
            print(f"[Warning] skip {slide_id}: missing encoders {missing}")
            continue

        raw_np = {name: read_h5_features(enc_rows[name]) for name in encoder_names}
        min_patches = min(arr.shape[0] for arr in raw_np.values())
        if min_patches <= 0:
            continue
        patch_count = min_patches if max_patches_per_slide <= 0 else min(min_patches, max_patches_per_slide)
        patch_idx = np.sort(rng.choice(min_patches, size=patch_count, replace=False))

        raw_tensors = {
            name: torch.from_numpy(raw_np[name][patch_idx]).float().to(device)
            for name in encoder_names
        }
        with torch.no_grad():
            projected = projection_model(raw_tensors)

        label = labels.get(slide_id)
        for encoder in encoder_names:
            remaining = max_points_per_encoder - per_encoder_counts[encoder]
            if remaining <= 0:
                continue
            values = projected[encoder].detach().cpu().numpy().astype(np.float32)
            if values.shape[0] > remaining:
                keep = np.sort(rng.choice(values.shape[0], size=remaining, replace=False))
                values = values[keep]
            per_encoder_counts[encoder] += int(values.shape[0])
            for vector in values:
                points.append({
                    "encoder": encoder,
                    "slide_id": slide_id,
                    "label": label,
                    "vector": vector,
                })

        if all(count >= max_points_per_encoder for count in per_encoder_counts.values()):
            break

    if not points:
        raise RuntimeError("No projected points were collected.")
    return points


def fit_pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return mean, components


def transform_pca(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (vectors - mean) @ components


def save_coordinates(
    rows: Sequence[Dict[str, object]],
    beacon_xy: np.ndarray,
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "slide_id", "label", "x", "y", "is_beacon"])
        writer.writeheader()
        for row in rows:
            xy = row["xy"]
            writer.writerow({
                "encoder": row["encoder"],
                "slide_id": row["slide_id"],
                "label": "" if row["label"] is None else row["label"],
                "x": f"{float(xy[0]):.8f}",
                "y": f"{float(xy[1]):.8f}",
                "is_beacon": 0,
            })
        writer.writerow({
            "encoder": "Beacon",
            "slide_id": "Beacon",
            "label": "",
            "x": f"{float(beacon_xy[0]):.8f}",
            "y": f"{float(beacon_xy[1]):.8f}",
            "is_beacon": 1,
        })


def configure_axes(ax, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")


def plot_panels(rows: Sequence[Dict[str, object]], encoder_names: Sequence[str], beacon_xy: np.ndarray, output_path: Path) -> None:
    all_xy = np.vstack([row["xy"] for row in rows] + [beacon_xy.reshape(1, 2)])
    pad = np.maximum((all_xy.max(axis=0) - all_xy.min(axis=0)) * 0.08, 1e-6)
    xlim = (float(all_xy[:, 0].min() - pad[0]), float(all_xy[:, 0].max() + pad[0]))
    ylim = (float(all_xy[:, 1].min() - pad[1]), float(all_xy[:, 1].max() + pad[1]))

    fig, axes = plt.subplots(1, len(encoder_names), figsize=(5.4 * len(encoder_names), 5), constrained_layout=True)
    if len(encoder_names) == 1:
        axes = [axes]

    for ax, encoder in zip(axes, encoder_names):
        subset = [row for row in rows if row["encoder"] == encoder]
        xy = np.vstack([row["xy"] for row in subset])
        labels = [row["label"] for row in subset]
        if any(label is not None for label in labels):
            for label_value, label_name in ((0, "D3"), (1, "M3")):
                idx = np.asarray([label == label_value for label in labels])
                if idx.any():
                    ax.scatter(
                        xy[idx, 0],
                        xy[idx, 1],
                        s=5,
                        alpha=0.22,
                        c=LABEL_COLORS[label_value],
                        linewidths=0,
                        label=label_name,
                    )
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=5, alpha=0.22, c=ENCODER_COLORS.get(encoder, "#4b5563"), linewidths=0)

        ax.scatter(
            [beacon_xy[0]],
            [beacon_xy[1]],
            s=180,
            marker="*",
            c="#111827",
            edgecolors="white",
            linewidths=1.0,
            label="Beacon",
            zorder=5,
        )
        configure_axes(ax, f"{ENCODER_LABELS.get(encoder, encoder)} projected embeddings", xlim, ylim)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)

    fig.suptitle("Projected Patch Embeddings in Shared PCA Space", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_overlay(rows: Sequence[Dict[str, object]], encoder_names: Sequence[str], beacon_xy: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    for encoder in encoder_names:
        subset = [row for row in rows if row["encoder"] == encoder]
        xy = np.vstack([row["xy"] for row in subset])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=5,
            alpha=0.18,
            c=ENCODER_COLORS.get(encoder, "#4b5563"),
            linewidths=0,
            label=ENCODER_LABELS.get(encoder, encoder),
        )
    ax.scatter([beacon_xy[0]], [beacon_xy[1]], s=220, marker="*", c="#111827", edgecolors="white", linewidths=1.0, label="Beacon", zorder=5)
    ax.set_title("Projected Embeddings by Encoder", fontsize=13, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or args.fold_dir / "best_gme_model.pt"
    beacon_path = args.beacon_file or args.fold_dir / "static_beacon_and_baselines.pt"
    output_dir = args.output_dir or args.fold_dir / "projected_embedding_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    beacon, beacon_payload = load_beacon(beacon_path, device)
    fold = infer_fold(args.fold_dir, beacon_payload, args.fold)
    manifest_rows = read_csv_rows(args.manifest)
    slide_rows = filter_manifest(manifest_rows, fold=fold, split=args.split)
    input_dims = infer_input_dims(slide_rows)
    encoder_names = sorted(input_dims.keys())
    target_dim = int(beacon.shape[-1])

    labels = load_clinical_labels(args.clinical_path)
    projection_model = load_projection_model(checkpoint_path, input_dims=input_dims, target_dim=target_dim, device=device)
    points = collect_projected_points(
        projection_model=projection_model,
        slide_rows=slide_rows,
        encoder_names=encoder_names,
        labels=labels,
        device=device,
        max_patches_per_slide=args.max_patches_per_slide,
        max_points_per_encoder=args.max_points_per_encoder,
        seed=args.seed,
    )

    vectors = np.vstack([point["vector"] for point in points]).astype(np.float32)
    mean, components = fit_pca_2d(vectors)
    xy = transform_pca(vectors, mean, components)
    beacon_xy = transform_pca(beacon.detach().cpu().numpy().astype(np.float32), mean, components)[0]
    for point, point_xy in zip(points, xy):
        point["xy"] = point_xy
        del point["vector"]

    panel_path = output_dir / f"fold_{fold}_{args.split}_projected_embeddings_by_encoder.png"
    overlay_path = output_dir / f"fold_{fold}_{args.split}_projected_embeddings_overlay.png"
    csv_path = output_dir / f"fold_{fold}_{args.split}_projected_embedding_pca_coordinates.csv"
    plot_panels(points, encoder_names, beacon_xy, panel_path)
    plot_overlay(points, encoder_names, beacon_xy, overlay_path)
    save_coordinates(points, beacon_xy, csv_path)

    print(f"Fold: {fold}")
    print(f"Split: {args.split}")
    print(f"Encoders: {', '.join(encoder_names)}")
    print(f"Projected points: {len(points)}")
    print(f"Saved panel plot: {panel_path}")
    print(f"Saved overlay plot: {overlay_path}")
    print(f"Saved coordinates: {csv_path}")


if __name__ == "__main__":
    main()
