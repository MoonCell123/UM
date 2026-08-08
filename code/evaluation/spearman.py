"""Feature-level Spearman analysis across all selected embedding pairs.

Each H5 file is treated as one WSI patch bag. Patches from every selected
encoder are coordinate-aligned and deterministically subsampled once per WSI.
For each encoder pair, the script computes the Spearman correlation between all
feature-coordinate pairs across those aligned patches.

Outputs are written below ``output-dir/<timestamp>``:

* ``wsi_spearman_summary.csv``: one row per WSI and encoder pair;
* ``encoder_pair_spearman_summary.csv``: WSI-averaged pair-level statistics;
* ``mean_abs_spearman_heatmap.png`` and ``mean_abs_spearman_matrix.csv``;
* ``cka_vs_mean_abs_spearman.png`` and its CSV summary when ``--cka-file`` is set;
* ``feature_pair_spearman_top.csv``: strongest feature pairs per WSI and encoder pair;
* ``matrices/``: optional complete feature-pair Spearman matrices;
* ``errors.csv``: WSIs skipped in non-strict mode.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

try:
    # Supports execution as ``python code/evaluation/Spearman.py``.
    from compute_linear_cka import (
        coordinate_indices,
        infer_dataset_key,
        read_coords,
        read_feature_rows,
        sample_common_rows,
        unique_manifest_rows,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from .compute_linear_cka import (
        coordinate_indices,
        infer_dataset_key,
        read_coords,
        read_feature_rows,
        sample_common_rows,
        unique_manifest_rows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Spearman"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all-pair feature-level Spearman correlations across selected embeddings."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config file.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-dirs",
        nargs="+",
        metavar="FEATURE_DIR",
        default=None,
        help="Feature_dir names to compare. Default: all encoders in the manifest.",
    )
    parser.add_argument("--slide-ids", nargs="+", default=None, help="Optional WSI ids to process.")
    parser.add_argument(
        "--max-patches",
        type=int,
        default=1024,
        help="Deterministically sample this many patches common to all selected encoders per WSI. 0 uses all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Accepted for workflow compatibility; feature-level Spearman is computed on CPU with NumPy/SciPy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of strongest absolute feature pairs saved per WSI and encoder pair (0 disables the table).",
    )
    parser.add_argument(
        "--high-rho-threshold",
        type=float,
        default=0.5,
        help="Absolute rho threshold used for the high-correlation feature-pair fraction.",
    )
    parser.add_argument(
        "--cka-file",
        type=Path,
        default=None,
        help="Optional pairwise_cka_by_wsi.csv for the CKA-Spearman correspondence scatter plot.",
    )
    parser.add_argument(
        "--save-feature-matrices",
        action="store_true",
        help="Save each complete [features_A, features_B] Spearman matrix as .npy.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid WSI instead of recording it in errors.csv.",
    )

    probe, _ = parser.parse_known_args()
    if probe.config is not None:
        config_path = probe.config
        if not config_path.exists():
            candidate = PROJECT_ROOT / "code" / "config" / config_path
            if candidate.exists():
                config_path = candidate
        if not config_path.exists():
            parser.error(f"Config file not found: {probe.config}")
        try:
            if config_path.suffix.lower() == ".json":
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
            else:
                import yaml

                with open(config_path, "r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle) or {}
        except ImportError as exc:
            parser.error(f"Reading YAML config requires PyYAML: {exc}")
        if not isinstance(config, dict):
            parser.error("Spearman config must be a YAML/JSON mapping.")
        parser.set_defaults(**{str(key).replace("-", "_"): value for key, value in config.items()})

    args = parser.parse_args()
    if args.max_patches < 0:
        parser.error("--max-patches must be non-negative.")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative.")
    if not 0.0 <= args.high_rho_threshold <= 1.0:
        parser.error("--high-rho-threshold must be between 0 and 1.")
    if args.cka_file is not None and not args.cka_file.is_file():
        parser.error(f"CKA file not found: {args.cka_file}")
    return args


def rank_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank, center, and normalize feature coordinates across aligned patches."""
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(f"Expected [N >= 2, D] features, got {features.shape}.")
    ranks = rankdata(np.asarray(features, dtype=np.float64), axis=0, method="average")
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(ranks, axis=0)
    return ranks, norms


def feature_spearman_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Return the cross-feature Spearman matrix for two aligned embeddings."""
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("Spearman analysis requires the same number of aligned patches.")
    ranks_a, norms_a = rank_features(features_a)
    ranks_b, norms_b = rank_features(features_b)
    return spearman_from_ranks(ranks_a, norms_a, ranks_b, norms_b)


def spearman_from_ranks(
    ranks_a: np.ndarray,
    norms_a: np.ndarray,
    ranks_b: np.ndarray,
    norms_b: np.ndarray,
) -> np.ndarray:
    """Calculate a cross-feature Spearman matrix from pre-ranked feature arrays."""
    denominator = norms_a[:, None] * norms_b[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = (ranks_a.T @ ranks_b) / denominator
    matrix[denominator == 0] = np.nan
    return np.clip(matrix, -1.0, 1.0)


def summarize_spearman(matrix: np.ndarray, high_rho_threshold: float) -> Dict[str, float]:
    finite = np.isfinite(matrix)
    if not finite.any():
        raise ValueError("All feature pairs are constant; Spearman correlation is undefined.")
    values = matrix[finite]
    absolute = np.abs(values)
    absolute_matrix = np.where(finite, np.abs(matrix), -np.inf)
    best_a = np.max(absolute_matrix, axis=1)
    best_b = np.max(absolute_matrix, axis=0)
    best_a[~np.isfinite(best_a)] = np.nan
    best_b[~np.isfinite(best_b)] = np.nan
    return {
        "mean_cross_feature_spearman": float(np.mean(values)),
        "median_cross_feature_spearman": float(np.median(values)),
        "mean_abs_cross_feature_spearman": float(np.mean(absolute)),
        "median_abs_cross_feature_spearman": float(np.median(absolute)),
        "max_abs_cross_feature_spearman": float(np.max(absolute)),
        "mean_best_match_abs_for_a": float(np.nanmean(best_a)),
        "mean_best_match_abs_for_b": float(np.nanmean(best_b)),
        "mean_best_match_abs": float(np.nanmean(np.concatenate((best_a, best_b)))),
        "fraction_high_abs_rho": float(np.mean(absolute >= high_rho_threshold)),
    }


def strongest_feature_pairs(
    matrix: np.ndarray,
    encoder_a: str,
    encoder_b: str,
    top_k: int,
) -> list[Dict[str, object]]:
    if top_k <= 0:
        return []
    finite = np.isfinite(matrix)
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        return []
    order = np.argsort(np.abs(matrix.ravel()[candidates]))[::-1][:top_k]
    rows = []
    n_b = matrix.shape[1]
    for position in order:
        flat_index = int(candidates[position])
        i, j = divmod(flat_index, n_b)
        rho = float(matrix[i, j])
        rows.append(
            {
                "feature_a": int(i),
                "feature_b": int(j),
                "encoder_a": encoder_a,
                "encoder_b": encoder_b,
                "spearman_rho": rho,
                "abs_spearman_rho": abs(rho),
            }
        )
    return rows


def canonical_pair(left: object, right: object) -> str:
    return " || ".join(sorted((str(left), str(right))))


def validate_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")


def pair_summary(summary: pd.DataFrame, high_rho_threshold: float) -> pd.DataFrame:
    statistics = [
        "mean_abs_cross_feature_spearman",
        "median_abs_cross_feature_spearman",
        "mean_best_match_abs_for_a",
        "mean_best_match_abs_for_b",
        "mean_best_match_abs",
        "fraction_high_abs_rho",
        "max_abs_cross_feature_spearman",
    ]
    grouped = summary.groupby(["encoder_a", "encoder_b"], as_index=False)
    result = grouped[statistics].mean()
    result = result.rename(
        columns={
            "mean_abs_cross_feature_spearman": "mean_absolute_rho",
            "median_abs_cross_feature_spearman": "mean_wsi_median_absolute_rho",
            "mean_best_match_abs_for_a": "mean_best_match_absolute_rho_a",
            "mean_best_match_abs_for_b": "mean_best_match_absolute_rho_b",
            "mean_best_match_abs": "mean_best_match_absolute_rho",
            "fraction_high_abs_rho": "mean_high_correlation_feature_fraction",
            "max_abs_cross_feature_spearman": "mean_maximum_absolute_rho",
        }
    )
    counts = grouped.size().rename(columns={"size": "n_wsi"})
    result = result.merge(counts, on=["encoder_a", "encoder_b"], validate="one_to_one")
    result["high_rho_threshold"] = float(high_rho_threshold)
    return result.sort_values(["encoder_a", "encoder_b"]).reset_index(drop=True)


def short_name(name: str) -> str:
    return name.removeprefix("features_")


def plot_mean_abs_heatmap(
    summary: pd.DataFrame,
    feature_dirs: Sequence[str],
    output_dir: Path,
) -> None:
    matrix = pd.DataFrame(np.nan, index=feature_dirs, columns=feature_dirs, dtype=float)
    for row in summary.itertuples(index=False):
        matrix.loc[row.encoder_a, row.encoder_b] = row.mean_absolute_rho
        matrix.loc[row.encoder_b, row.encoder_a] = row.mean_absolute_rho
    matrix.to_csv(output_dir / "mean_abs_spearman_matrix.csv", encoding="utf-8-sig", float_format="%.6f")

    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    maximum = float(np.nanmax(matrix.to_numpy(dtype=float)))
    color_maximum = max(0.5, maximum)
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=color_maximum)
    labels = [short_name(name) for name in feature_dirs]
    ax.set_xticks(range(len(feature_dirs)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(feature_dirs)), labels=labels)
    for row_index in range(len(feature_dirs)):
        for column_index in range(len(feature_dirs)):
            value = matrix.iloc[row_index, column_index]
            if np.isfinite(value):
                color = "white" if value < color_maximum * 0.55 else "#111827"
                ax.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
            elif row_index == column_index:
                ax.text(column_index, row_index, "N/A", ha="center", va="center", color="#4b5563", fontsize=8)
    ax.set_title("Mean absolute feature-level Spearman correlation")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("Mean absolute rho")
    fig.savefig(output_dir / "mean_abs_spearman_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_cka_correspondence(
    spearman_summary: pd.DataFrame,
    cka_file: Path,
    output_dir: Path,
) -> None:
    cka = pd.read_csv(cka_file)
    validate_columns(cka, ("encoder_a", "encoder_b", "linear_cka"), cka_file)
    cka = cka.copy()
    cka["pair"] = [canonical_pair(a, b) for a, b in zip(cka.encoder_a, cka.encoder_b)]
    cka["linear_cka"] = pd.to_numeric(cka["linear_cka"], errors="coerce")
    cka_summary = (
        cka.dropna(subset=["pair", "linear_cka"])
        .groupby("pair", as_index=False)
        .agg(mean_linear_cka=("linear_cka", "mean"), std_linear_cka=("linear_cka", "std"), n_wsi_cka=("linear_cka", "count"))
    )
    spearman = spearman_summary.copy()
    spearman["pair"] = [canonical_pair(a, b) for a, b in zip(spearman.encoder_a, spearman.encoder_b)]
    merged = spearman.merge(cka_summary, on="pair", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No encoder pairs overlap between the Spearman and CKA summaries.")
    merged.to_csv(
        output_dir / "cka_spearman_pair_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    x = merged["mean_linear_cka"].to_numpy(dtype=float)
    y = merged["mean_absolute_rho"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(merged) > 1 else float("nan")
    rank_result = spearmanr(x, y) if len(merged) > 1 else None
    rank_correlation = float(rank_result.statistic) if rank_result is not None else float("nan")
    rank_pvalue = float(rank_result.pvalue) if rank_result is not None else float("nan")
    fig, ax = plt.subplots(figsize=(8.2, 6.5), constrained_layout=True)
    ax.scatter(x, y, s=88, color="#0f766e", edgecolors="white", linewidths=1.0, zorder=3)
    for row in merged.itertuples(index=False):
        label = f"{short_name(row.encoder_a)} / {short_name(row.encoder_b)}"
        ax.annotate(label, (row.mean_linear_cka, row.mean_absolute_rho), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if len(merged) > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        line_x = np.linspace(x.min(), x.max(), num=100)
        ax.plot(line_x, slope * line_x + intercept, color="#9a3412", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Mean pairwise Linear CKA")
    ax.set_ylabel("Mean absolute feature-level Spearman rho")
    ax.set_title(
        "CKA and feature-level Spearman correspondence\n"
        f"Pairs={len(merged)} | Pearson r={pearson:.3f} | Spearman rho={rank_correlation:.3f}, p={rank_pvalue:.3g}",
        fontsize=13,
        pad=12,
    )
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "cka_vs_mean_abs_spearman.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    available = sorted(manifest["feature_dir"].dropna().astype(str).unique().tolist())
    feature_dirs = list(args.feature_dirs) if args.feature_dirs else available
    if len(feature_dirs) < 2:
        raise ValueError("Spearman analysis requires at least two feature encoders.")
    if len(feature_dirs) != len(set(feature_dirs)):
        raise ValueError("--feature-dirs must not contain duplicate encoder names.")
    missing = sorted(set(feature_dirs) - set(available))
    if missing:
        raise ValueError(f"Requested feature dirs not present in manifest: {missing}")
    encoder_pairs = list(combinations(feature_dirs, 2))

    rows = unique_manifest_rows(manifest, feature_dirs)
    complete_slides = [
        slide_id
        for slide_id, group in rows.groupby("slide_id")
        if set(group["feature_dir"]) == set(feature_dirs)
    ]
    if args.slide_ids:
        requested = set(map(str, args.slide_ids))
        missing_slides = sorted(requested - set(complete_slides))
        if missing_slides:
            raise ValueError(f"Requested WSIs are missing one or more selected encoders: {missing_slides}")
        complete_slides = [slide_id for slide_id in complete_slides if slide_id in requested]
    if not complete_slides:
        raise RuntimeError("No WSI has every selected feature encoder.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    matrix_dir = output_dir / "matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_feature_matrices:
        matrix_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest": str(args.manifest),
                "feature_dirs": feature_dirs,
                "encoder_pairs": [list(pair) for pair in encoder_pairs],
                "slide_ids": sorted(complete_slides),
                "max_patches": args.max_patches,
                "seed": args.seed,
                "device": str(args.device),
                "top_k": args.top_k,
                "high_rho_threshold": args.high_rho_threshold,
                "cka_file": str(args.cka_file) if args.cka_file else None,
                "save_feature_matrices": bool(args.save_feature_matrices),
                "estimator": "feature_level_spearman_across_aligned_patches",
                "patch_alignment": "coords_intersection_across_all_selected_encoders",
            },
            handle,
            indent=2,
        )

    summary_rows: list[Dict[str, object]] = []
    feature_pair_rows: list[Dict[str, object]] = []
    errors: list[Dict[str, str]] = []
    for slide_number, slide_id in enumerate(sorted(complete_slides), start=1):
        print(f"[{slide_number}/{len(complete_slides)}] {slide_id}")
        try:
            slide_rows = rows[rows["slide_id"] == slide_id].set_index("feature_dir")
            paths = {name: Path(str(slide_rows.loc[name, "h5_path"])) for name in feature_dirs}
            keys = {
                name: infer_dataset_key(
                    paths[name],
                    None if pd.isna(slide_rows.loc[name, "dataset_key"]) else str(slide_rows.loc[name, "dataset_key"]),
                )
                for name in feature_dirs
            }
            coords = {name: read_coords(paths[name]) for name in feature_dirs}
            common_coords, aligned_indices = coordinate_indices(coords)
            sampled_indices = sample_common_rows(
                len(common_coords), aligned_indices, args.max_patches, args.seed, slide_id
            )
            ranked = {}
            for name in feature_dirs:
                features = read_feature_rows(paths[name], keys[name], sampled_indices[name])
                ranked[name] = rank_features(features)

            for encoder_a, encoder_b in encoder_pairs:
                ranks_a, norms_a = ranked[encoder_a]
                ranks_b, norms_b = ranked[encoder_b]
                matrix = spearman_from_ranks(ranks_a, norms_a, ranks_b, norms_b)
                if args.save_feature_matrices:
                    np.save(matrix_dir / f"{slide_id}__{encoder_a}__{encoder_b}_spearman.npy", matrix)
                summary = summarize_spearman(matrix, args.high_rho_threshold)
                summary.update(
                    {
                        "slide_id": slide_id,
                        "encoder_a": encoder_a,
                        "encoder_b": encoder_b,
                        "feature_dim_a": int(matrix.shape[0]),
                        "feature_dim_b": int(matrix.shape[1]),
                        "common_patches": int(len(common_coords)),
                        "sampled_patches": int(ranks_a.shape[0]),
                        "high_rho_threshold": float(args.high_rho_threshold),
                    }
                )
                summary_rows.append(summary)
                for feature_pair in strongest_feature_pairs(matrix, encoder_a, encoder_b, args.top_k):
                    feature_pair.update(
                        {
                            "slide_id": slide_id,
                            "common_patches": int(len(common_coords)),
                            "sampled_patches": int(ranks_a.shape[0]),
                        }
                    )
                    feature_pair_rows.append(feature_pair)
        except Exception as exc:
            if args.strict:
                raise
            errors.append({"slide_id": slide_id, "error": str(exc)})
            print(f"[Warning] {slide_id}: {exc}")

    if not summary_rows:
        raise RuntimeError("Spearman analysis failed for every WSI; inspect the console errors.")
    wsi_summary = pd.DataFrame(summary_rows)
    wsi_summary.to_csv(
        output_dir / "wsi_spearman_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    encoder_pair_summary = pair_summary(wsi_summary, args.high_rho_threshold)
    encoder_pair_summary.to_csv(
        output_dir / "encoder_pair_spearman_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    plot_mean_abs_heatmap(encoder_pair_summary, feature_dirs, output_dir)
    if feature_pair_rows:
        pd.DataFrame(feature_pair_rows).to_csv(
            output_dir / "feature_pair_spearman_top.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
        )
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    if args.cka_file is not None:
        write_cka_correspondence(encoder_pair_summary, args.cka_file, output_dir)
    print(f"Completed WSIs: {wsi_summary['slide_id'].nunique()} / {len(complete_slides)}")
    print(f"Completed encoder pairs: {len(encoder_pair_summary)} / {len(encoder_pairs)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
