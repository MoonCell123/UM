"""Cluster foundation models from coordinate-aligned patch embeddings.

Raw foundation embeddings use different dimensions and coordinate systems, so
they must not be concatenated and jointly PCA-reduced. This script instead:

1. samples coordinate-aligned patches from every requested encoder;
2. computes pairwise linear CKA for each slide;
3. averages the CKA matrices across slides;
4. applies PCA to each model's CKA similarity profile;
5. writes a heatmap, hierarchical dendrogram, and PCA cluster plot.

The resulting distances describe representation similarity between foundation
models and can guide encoder coalition selection without assuming that a v2
model is automatically preferable to its v1 predecessor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEAT_BASE = Path(r"L:\20x_256px_0px_overlap")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "foundation_model_clustering"
FEATURE_KEYS = ("features", "feats")
DEFAULT_FEATURE_DIRS = (
    "features_conch_v1",
    "features_conch_v15",
    "features_gigapath",
    "features_hibou_l",
    "features_hoptimus0",
    "features_hoptimus1",
    "features_midnight12k",
    "features_phikon",
    "features_phikon_v2",
    "features_uni_v1",
    "features_uni_v2",
    "features_virchow",
    "features_virchow2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster foundation models by coordinate-aligned linear CKA and plot PCA similarity profiles."
    )
    parser.add_argument("--feat-base", type=Path, default=DEFAULT_FEAT_BASE)
    parser.add_argument("--feature-dirs", nargs="+", default=list(DEFAULT_FEATURE_DIRS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-slides",
        type=int,
        default=0,
        help="Maximum complete slides to use. 0 means all complete slides.",
    )
    parser.add_argument(
        "--max-patches-per-slide",
        type=int,
        default=256,
        help="Deterministic number of coordinate-aligned patches sampled per slide. 0 means all.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of average-linkage clusters shown in the model-level PCA plot.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true", help="Stop instead of skipping an invalid slide.")
    args = parser.parse_args()
    if len(args.feature_dirs) < 2:
        parser.error("At least two feature dirs are required.")
    if args.max_slides < 0 or args.max_patches_per_slide < 0:
        parser.error("Slide and patch limits must be non-negative.")
    if not 2 <= args.n_clusters <= len(args.feature_dirs):
        parser.error("--n-clusters must be between 2 and the number of feature dirs.")
    return args


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def complete_slide_ids(feat_base: Path, feature_dirs: Sequence[str]) -> List[str]:
    slide_sets = []
    for name in feature_dirs:
        directory = feat_base / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Feature directory does not exist: {directory}")
        slide_sets.append({path.stem for path in directory.glob("*.h5")})
    common = set.intersection(*slide_sets)
    if not common:
        raise RuntimeError("No slide has every requested foundation-model embedding.")
    return sorted(common)


def get_feature_key(handle: h5py.File) -> str:
    for key in FEATURE_KEYS:
        if key in handle:
            return key
    candidates = [key for key in handle.keys() if key != "coords"]
    if not candidates:
        raise KeyError(f"No feature dataset found; keys={list(handle.keys())}")
    return candidates[0]


def read_coords(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"{path}: missing coords required for patch alignment")
        coords = np.asarray(handle["coords"][:])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: expected coords [N, 2], got {coords.shape}")
    return coords


def common_coordinate_indices(coords_by_encoder: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    names = list(coords_by_encoder)
    first = coords_by_encoder[names[0]]
    if all(np.array_equal(first, coords_by_encoder[name]) for name in names[1:]):
        indices = np.arange(first.shape[0], dtype=np.int64)
        return {name: indices.copy() for name in names}

    index_maps: Dict[str, Dict[Tuple[int, int], int]] = {}
    common_coords = None
    for name, coords in coords_by_encoder.items():
        mapping = {tuple(map(int, coord)): index for index, coord in enumerate(coords)}
        if len(mapping) != len(coords):
            raise ValueError(f"{name}: duplicate coordinates prevent unambiguous alignment")
        index_maps[name] = mapping
        common_coords = set(mapping) if common_coords is None else common_coords.intersection(mapping)
    if not common_coords:
        raise ValueError("No common patch coordinates across the selected encoders")
    ordered = sorted(common_coords)
    return {
        name: np.asarray([index_maps[name][coord] for coord in ordered], dtype=np.int64)
        for name in names
    }


def stable_seed(seed: int, slide_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{slide_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_indices(indices_by_encoder: Mapping[str, np.ndarray], limit: int, seed: int, slide_id: str) -> Dict[str, np.ndarray]:
    count = len(next(iter(indices_by_encoder.values())))
    if limit == 0 or count <= limit:
        selected = np.arange(count, dtype=np.int64)
    else:
        rng = np.random.default_rng(stable_seed(seed, slide_id))
        selected = np.sort(rng.choice(count, size=limit, replace=False))
    return {name: indices[selected] for name, indices in indices_by_encoder.items()}


def read_feature_rows(path: Path, row_indices: np.ndarray) -> np.ndarray:
    order = np.argsort(row_indices)
    sorted_indices = row_indices[order]
    if np.unique(sorted_indices).size != sorted_indices.size:
        raise ValueError(f"{path}: sampled feature row indices are not unique")
    with h5py.File(path, "r") as handle:
        dataset = handle[get_feature_key(handle)]
        if dataset.ndim != 2:
            raise ValueError(f"{path}: expected feature shape [N, D], got {dataset.shape}")
        sorted_features = np.asarray(dataset[sorted_indices], dtype=np.float32)
    features = sorted_features[np.argsort(order)]
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_gram(features: np.ndarray, device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(features, device=device, dtype=torch.float32)
    values = values - values.mean(dim=0, keepdim=True)
    gram = values @ values.transpose(0, 1)
    norm = torch.linalg.vector_norm(gram)
    if not torch.isfinite(norm) or float(norm.detach().cpu()) <= 0.0:
        raise ValueError("Centered Gram matrix has zero or non-finite norm")
    return gram / norm


def linear_cka_matrix(features_by_encoder: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    names = list(features_by_encoder)
    counts = {values.shape[0] for values in features_by_encoder.values()}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise ValueError("Linear CKA requires at least two aligned patches per encoder")
    grams = {name: normalized_gram(values, device) for name, values in features_by_encoder.items()}
    matrix = np.eye(len(names), dtype=np.float64)
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            value = float(torch.sum(grams[left] * grams[right]).detach().cpu())
            matrix[left_index, right_index] = matrix[right_index, left_index] = np.clip(value, 0.0, 1.0)
    return matrix


def fit_profile_pca(similarity: np.ndarray, components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    profiles = similarity.copy()
    centered = profiles - profiles.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ right_vectors[:components].T
    explained = np.square(singular_values[:components]) / max(float(np.square(singular_values).sum()), 1e-12)
    return scores, explained


def plot_heatmap(similarity: np.ndarray, names: Sequence[str], order: np.ndarray, output_path: Path) -> None:
    ordered_names = [names[index] for index in order]
    ordered_matrix = similarity[np.ix_(order, order)]
    size = max(8.0, len(names) * 0.72)
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    image = ax.imshow(ordered_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(names)), ordered_names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), ordered_names, fontsize=8)
    ax.set_title("Mean Coordinate-Aligned Linear CKA")
    for row in range(len(names)):
        for col in range(len(names)):
            color = "white" if ordered_matrix[row, col] < 0.55 else "black"
            ax.text(col, row, f"{ordered_matrix[row, col]:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(image, ax=ax, label="Linear CKA similarity")
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_dendrogram(linkage_matrix: np.ndarray, names: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10.0, len(names) * 0.82), 6.5), constrained_layout=True)
    dendrogram(linkage_matrix, labels=list(names), leaf_rotation=50, leaf_font_size=9, ax=ax)
    ax.set_title("Foundation Model Clustering from 1 - Linear CKA")
    ax.set_ylabel("Average-linkage distance")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_profile_pca(
    coordinates: np.ndarray,
    names: Sequence[str],
    clusters: np.ndarray,
    explained: np.ndarray,
    output_path: Path,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    legend_handles = []
    for index, name in enumerate(names):
        color = cmap((int(clusters[index]) - 1) % 10)
        marker = ax.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            s=175,
            c=[color],
            edgecolors="white",
            linewidths=1.2,
        )
        ax.text(
            coordinates[index, 0],
            coordinates[index, 1],
            str(index + 1),
            color="white",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
        legend_handles.append((marker, f"{index + 1}. {name}"))
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_title("PCA of Foundation-Model CKA Similarity Profiles")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% profile variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% profile variance)")
    ax.grid(alpha=0.25)
    ax.legend(
        [handle for handle, _ in legend_handles],
        [label for _, label in legend_handles],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=False,
    )
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    feature_dirs = list(dict.fromkeys(args.feature_dirs))
    complete_slides = complete_slide_ids(args.feat_base, feature_dirs)
    if args.max_slides > 0:
        rng = np.random.default_rng(args.seed)
        complete_slides = sorted(rng.choice(complete_slides, size=min(args.max_slides, len(complete_slides)), replace=False).tolist())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "per_slide_cka"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {len(feature_dirs)} encoders across {len(complete_slides)} complete slides on {device}.")

    matrices: List[np.ndarray] = []
    pair_rows: List[Dict[str, object]] = []
    slide_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    for position, slide_id in enumerate(complete_slides, start=1):
        print(f"[{position}/{len(complete_slides)}] {slide_id}")
        try:
            paths = {name: args.feat_base / name / f"{slide_id}.h5" for name in feature_dirs}
            coords = {name: read_coords(path) for name, path in paths.items()}
            aligned = common_coordinate_indices(coords)
            sampled = sample_indices(aligned, args.max_patches_per_slide, args.seed, slide_id)
            features = {name: read_feature_rows(paths[name], sampled[name]) for name in feature_dirs}
            matrix = linear_cka_matrix(features, device)
            matrices.append(matrix)
            pd.DataFrame(matrix, index=feature_dirs, columns=feature_dirs).to_csv(
                matrix_dir / f"{slide_id}_linear_cka.csv", encoding="utf-8-sig", float_format="%.6f"
            )
            upper = matrix[np.triu_indices_from(matrix, k=1)]
            slide_rows.append({
                "slide_id": slide_id,
                "common_patches": int(len(next(iter(aligned.values())))),
                "sampled_patches": int(next(iter(features.values())).shape[0]),
                "mean_pairwise_cka": float(upper.mean()),
                "min_pairwise_cka": float(upper.min()),
                "max_pairwise_cka": float(upper.max()),
            })
            for left_index, left in enumerate(feature_dirs):
                for right in feature_dirs[left_index + 1:]:
                    right_index = feature_dirs.index(right)
                    pair_rows.append({"slide_id": slide_id, "encoder_a": left, "encoder_b": right, "linear_cka": float(matrix[left_index, right_index])})
        except Exception as exc:
            if args.strict:
                raise
            errors.append({"slide_id": slide_id, "error": str(exc)})
            print(f"[Warning] skipped {slide_id}: {exc}")

    if not matrices:
        raise RuntimeError("No slide completed successfully; inspect errors.csv.")
    mean_similarity = np.mean(np.stack(matrices, axis=0), axis=0)
    mean_similarity = (mean_similarity + mean_similarity.T) / 2.0
    np.fill_diagonal(mean_similarity, 1.0)
    distance = np.clip(1.0 - mean_similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    leaf_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    clusters = fcluster(linkage_matrix, t=args.n_clusters, criterion="maxclust")
    pca_coordinates, explained = fit_profile_pca(mean_similarity)

    pd.DataFrame(mean_similarity, index=feature_dirs, columns=feature_dirs).to_csv(
        output_dir / "mean_linear_cka_similarity.csv", encoding="utf-8-sig", float_format="%.6f"
    )
    pd.DataFrame(distance, index=feature_dirs, columns=feature_dirs).to_csv(
        output_dir / "linear_cka_distance.csv", encoding="utf-8-sig", float_format="%.6f"
    )
    pd.DataFrame(pair_rows).to_csv(output_dir / "pairwise_cka_by_slide.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(slide_rows).to_csv(output_dir / "slide_cka_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    cluster_table = pd.DataFrame({
        "encoder": feature_dirs,
        "cluster": clusters,
        "PC1": pca_coordinates[:, 0],
        "PC2": pca_coordinates[:, 1],
        "mean_cka_to_others": (mean_similarity.sum(axis=1) - 1.0) / (len(feature_dirs) - 1),
    }).sort_values(["cluster", "encoder"])
    cluster_table.to_csv(output_dir / "foundation_model_clusters.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({
            "feat_base": str(args.feat_base),
            "feature_dirs": feature_dirs,
            "completed_slides": len(matrices),
            "requested_slides": len(complete_slides),
            "max_patches_per_slide": args.max_patches_per_slide,
            "seed": args.seed,
            "device": str(device),
            "estimator": "mean_per_slide_centered_linear_cka",
            "patch_alignment": "intersection_of_h5_coords",
            "clustering": "average_linkage_on_1_minus_mean_cka",
            "n_clusters": args.n_clusters,
            "pca_profile_explained_variance": explained.tolist(),
        }, handle, indent=2)
    plot_heatmap(mean_similarity, feature_dirs, np.asarray(leaf_order), output_dir / "foundation_model_cka_heatmap.png")
    plot_dendrogram(linkage_matrix, feature_dirs, output_dir / "foundation_model_dendrogram.png")
    plot_profile_pca(pca_coordinates, feature_dirs, clusters, explained, output_dir / "foundation_model_similarity_profile_pca.png")
    print(f"Completed slides: {len(matrices)} / {len(complete_slides)}")
    print(f"Saved outputs: {output_dir}")


if __name__ == "__main__":
    main()
