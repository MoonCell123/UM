"""Feature-level Spearman analysis for two coordinate-aligned embeddings.

Each H5 file is treated as one WSI patch bag.  Patches are aligned by their
stored coordinates before deterministic subsampling.  For every WSI this
script computes the Spearman correlation between every feature in embedding A
and every feature in embedding B (correlation is taken across aligned patches).
This also works when the two encoders have different feature dimensions.

Outputs are written below ``output-dir/<timestamp>``:

* ``wsi_spearman_summary.csv``: one summary row per WSI;
* ``feature_pair_spearman_top.csv``: strongest feature pairs per WSI;
* ``matrices/<slide_id>_spearman.npy``: optional full feature-pair matrix;
* ``errors.csv``: WSIs skipped in non-strict mode.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
        description="Compute feature-level Spearman correlations between two embeddings."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config file.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--feature-dirs",
        nargs=2,
        metavar=("EMBEDDING_A", "EMBEDDING_B"),
        required=False,
        default=None,
        help="Exactly two feature_dir names to compare. Default: first two in the manifest.",
    )
    parser.add_argument("--slide-ids", nargs="+", default=None, help="Optional WSI ids to process.")
    parser.add_argument(
        "--max-patches",
        type=int,
        default=1024,
        help="Deterministically sample this many common patches per WSI. 0 uses all patches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of strongest absolute feature pairs saved per WSI (0 disables the table).",
    )
    parser.add_argument(
        "--save-feature-matrices",
        action="store_true",
        help="Save the complete [features_A, features_B] Spearman matrix as .npy per WSI.",
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
    return args


def feature_spearman_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Return the cross-feature Spearman matrix.

    Rows are aligned patches and columns are feature coordinates.  Ranking is
    performed independently for each feature across patches, so the output at
    ``[i, j]`` is ``rho(feature_a_i, feature_b_j)``.  Constant features yield
    NaN, because their rank correlation is undefined.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError(f"Expected two [N, D] arrays, got {features_a.shape} and {features_b.shape}.")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("Spearman analysis requires the same number of aligned patches.")
    if features_a.shape[0] < 2:
        raise ValueError("Spearman analysis requires at least two aligned patches.")

    # rankdata handles ties using the conventional average-rank definition.
    ranks_a = rankdata(np.asarray(features_a, dtype=np.float64), axis=0, method="average")
    ranks_b = rankdata(np.asarray(features_b, dtype=np.float64), axis=0, method="average")
    ranks_a -= ranks_a.mean(axis=0, keepdims=True)
    ranks_b -= ranks_b.mean(axis=0, keepdims=True)
    norms_a = np.linalg.norm(ranks_a, axis=0)
    norms_b = np.linalg.norm(ranks_b, axis=0)
    denominator = norms_a[:, None] * norms_b[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = (ranks_a.T @ ranks_b) / denominator
    matrix[denominator == 0] = np.nan
    return np.clip(matrix, -1.0, 1.0)


def summarize_spearman(matrix: np.ndarray) -> Dict[str, float]:
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
    summary = {
        "mean_cross_feature_spearman": float(np.mean(values)),
        "median_cross_feature_spearman": float(np.median(values)),
        "mean_abs_cross_feature_spearman": float(np.mean(absolute)),
        "median_abs_cross_feature_spearman": float(np.median(absolute)),
        "max_abs_cross_feature_spearman": float(np.max(absolute)),
        "mean_best_match_abs_for_a": float(np.nanmean(best_a)),
        "mean_best_match_abs_for_b": float(np.nanmean(best_b)),
        "fraction_abs_rho_ge_0_5": float(np.mean(absolute >= 0.5)),
    }
    if matrix.shape[0] == matrix.shape[1]:
        diagonal = np.diag(matrix)
        diagonal = diagonal[np.isfinite(diagonal)]
        summary["mean_paired_feature_spearman"] = float(np.mean(diagonal)) if diagonal.size else float("nan")
        summary["median_paired_feature_spearman"] = float(np.median(diagonal)) if diagonal.size else float("nan")
    else:
        summary["mean_paired_feature_spearman"] = float("nan")
        summary["median_paired_feature_spearman"] = float("nan")
    return summary


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


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    available = sorted(manifest["feature_dir"].dropna().astype(str).unique().tolist())
    feature_dirs = list(args.feature_dirs) if args.feature_dirs else available[:2]
    if len(feature_dirs) != 2:
        raise ValueError("Spearman analysis requires exactly two feature encoders.")
    missing = sorted(set(feature_dirs) - set(available))
    if missing:
        raise ValueError(f"Requested feature dirs not present in manifest: {missing}")

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
        raise RuntimeError("No WSI has both selected feature encoders.")

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
                "top_k": args.top_k,
                "save_feature_matrices": bool(args.save_feature_matrices),
                "estimator": "feature_level_spearman_across_aligned_patches",
                "patch_alignment": "coords_intersection",
            },
            handle,
            indent=2,
        )

    summary_rows: list[Dict[str, object]] = []
    pair_rows: list[Dict[str, object]] = []
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
            features = {
                name: read_feature_rows(paths[name], keys[name], sampled_indices[name]) for name in feature_dirs
            }
            matrix = feature_spearman_matrix(features[feature_dirs[0]], features[feature_dirs[1]])
            if args.save_feature_matrices:
                np.save(matrix_dir / f"{slide_id}_spearman.npy", matrix)
            summary = summarize_spearman(matrix)
            summary.update(
                {
                    "slide_id": slide_id,
                    "encoder_a": feature_dirs[0],
                    "encoder_b": feature_dirs[1],
                    "feature_dim_a": int(matrix.shape[0]),
                    "feature_dim_b": int(matrix.shape[1]),
                    "common_patches": int(len(common_coords)),
                    "sampled_patches": int(features[feature_dirs[0]].shape[0]),
                }
            )
            summary_rows.append(summary)
            for pair in strongest_feature_pairs(matrix, feature_dirs[0], feature_dirs[1], args.top_k):
                pair["slide_id"] = slide_id
                pair["common_patches"] = int(len(common_coords))
                pair["sampled_patches"] = int(features[feature_dirs[0]].shape[0])
                pair_rows.append(pair)
        except Exception as exc:
            if args.strict:
                raise
            errors.append({"slide_id": slide_id, "error": str(exc)})
            print(f"[Warning] {slide_id}: {exc}")

    if not summary_rows:
        raise RuntimeError("Spearman analysis failed for every WSI; inspect the console errors.")
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "wsi_spearman_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(
            output_dir / "feature_pair_spearman_top.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
        )
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    print(f"Completed WSIs: {len(summary_rows)} / {len(complete_slides)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
