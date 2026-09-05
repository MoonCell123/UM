"""Plot a mean pairwise Linear CKA similarity matrix from existing CKA outputs.

The script consumes ``pairwise_cka_by_wsi.csv`` written by
``compute_linear_cka.py`` and averages the per-WSI Linear CKA values for each
encoder pair.  It is intentionally separate from the embedding/PCA analysis:
no feature files are read and no CKA value is recomputed.

The resulting heatmap follows the visual convention of
``foundation_model_cka_heatmap.svg``: viridis colour map, average-linkage leaf
order, in-cell values, and PNG/PDF/SVG exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a mean pairwise Linear CKA similarity matrix from existing CSV results."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cka-file",
        type=Path,
        help="Per-WSI pairwise CKA CSV with encoder_a, encoder_b, and linear_cka columns.",
    )
    source.add_argument(
        "--matrix-file",
        type=Path,
        help="Existing square mean Linear CKA matrix CSV, with encoder names in its first column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: a pairwise_linear_cka_matrix directory beside the input CSV.",
    )
    parser.add_argument(
        "--output-name",
        default="pairwise_linear_cka_similarity_matrix",
        help="Base name for PNG, PDF, SVG, CSV, and metadata outputs.",
    )
    parser.add_argument(
        "--order",
        choices=("average_linkage", "alphabetical", "input"),
        default="average_linkage",
        help="Encoder ordering in the matrix. Default matches foundation_model_cka_heatmap.svg.",
    )
    parser.add_argument(
        "--encoder-order",
        nargs="+",
        default=None,
        help="Optional explicit encoder order. Overrides --order and must list every encoder exactly once.",
    )
    parser.add_argument("--title", default="Mean Coordinate-Aligned Linear CKA")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()
    if not args.output_name.strip():
        parser.error("--output-name must not be empty.")
    if not np.isfinite(args.vmin) or not np.isfinite(args.vmax) or args.vmin >= args.vmax:
        parser.error("--vmin and --vmax must be finite, with vmin < vmax.")
    return args


def canonical_pair(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def validate_similarity_matrix(matrix: pd.DataFrame, source: Path) -> pd.DataFrame:
    if matrix.empty or matrix.shape[0] < 2:
        raise ValueError(f"{source}: a CKA matrix needs at least two encoders.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{source}: expected a square CKA matrix, got shape {matrix.shape}.")
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError(f"{source}: encoder names must be unique.")
    if set(matrix.index) != set(matrix.columns):
        raise ValueError(f"{source}: row and column encoder names must match.")

    matrix = matrix.loc[list(matrix.index), list(matrix.index)].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: CKA matrix contains non-numeric or non-finite values.")
    if not np.allclose(values, values.T, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{source}: CKA matrix is not symmetric.")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(f"{source}: Linear CKA values must lie in [0, 1].")

    values = np.clip((values + values.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=matrix.index, columns=matrix.index)


def matrix_from_pairwise_cka(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"CKA file not found: {path}")
    frame = pd.read_csv(path)
    required = {"encoder_a", "encoder_b", "linear_cka"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}.")

    frame = frame.loc[:, ["encoder_a", "encoder_b", "linear_cka"]].copy()
    frame["encoder_a"] = frame["encoder_a"].astype(str)
    frame["encoder_b"] = frame["encoder_b"].astype(str)
    frame["linear_cka"] = pd.to_numeric(frame["linear_cka"], errors="coerce")
    frame = frame.dropna(subset=["linear_cka"])
    frame = frame[frame["encoder_a"] != frame["encoder_b"]]
    if frame.empty:
        raise ValueError(f"{path}: no valid between-encoder Linear CKA rows were found.")

    pairs = [canonical_pair(left, right) for left, right in zip(frame.encoder_a, frame.encoder_b)]
    frame["encoder_a"] = [pair[0] for pair in pairs]
    frame["encoder_b"] = [pair[1] for pair in pairs]
    summary = (
        frame.groupby(["encoder_a", "encoder_b"], as_index=False)["linear_cka"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_linear_cka", "std": "std_linear_cka", "count": "n_wsi"})
    )
    names = sorted(set(summary["encoder_a"]).union(summary["encoder_b"]))
    matrix = pd.DataFrame(np.eye(len(names), dtype=np.float64), index=names, columns=names)
    for row in summary.itertuples(index=False):
        matrix.loc[row.encoder_a, row.encoder_b] = row.mean_linear_cka
        matrix.loc[row.encoder_b, row.encoder_a] = row.mean_linear_cka
    if matrix.isna().to_numpy().any():
        missing_pairs = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if pd.isna(matrix.loc[left, right]):
                    missing_pairs.append(f"{left} || {right}")
        raise ValueError(f"{path}: missing CKA values for pair(s): {missing_pairs}")
    return validate_similarity_matrix(matrix, path), summary


def matrix_from_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return validate_similarity_matrix(matrix, path)


def resolve_order(matrix: pd.DataFrame, strategy: str, explicit_order: Sequence[str] | None) -> list[str]:
    names = list(matrix.index)
    if explicit_order is not None:
        ordered = [str(name) for name in explicit_order]
        if len(ordered) != len(names) or len(set(ordered)) != len(ordered) or set(ordered) != set(names):
            raise ValueError("--encoder-order must contain every matrix encoder exactly once.")
        return ordered
    if strategy == "input":
        return names
    if strategy == "alphabetical":
        return sorted(names)

    distance = np.clip(1.0 - matrix.to_numpy(dtype=np.float64), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    leaf_indices = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    return [names[index] for index in leaf_indices]


def plot_heatmap(matrix: pd.DataFrame, order: Sequence[str], args: argparse.Namespace, output_path: Path) -> None:
    ordered = matrix.loc[list(order), list(order)]
    names = list(ordered.index)
    size = max(8.0, len(names) * 0.72)
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    image = ax.imshow(ordered.to_numpy(), vmin=args.vmin, vmax=args.vmax, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title(args.title)
    threshold = (args.vmin + args.vmax) / 2.0
    for row in range(len(names)):
        for col in range(len(names)):
            value = float(ordered.iat[row, col])
            color = "white" if value < threshold else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(image, ax=ax, label="Linear CKA similarity")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = args.cka_file or args.matrix_file
    assert source is not None
    if args.cka_file is not None:
        matrix, pair_summary = matrix_from_pairwise_cka(args.cka_file)
    else:
        matrix = matrix_from_csv(args.matrix_file)
        pair_summary = pd.DataFrame()

    output_dir = args.output_dir or source.parent / "pairwise_linear_cka_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    order = resolve_order(matrix, args.order, args.encoder_order)
    output_path = output_dir / f"{args.output_name}.png"
    plot_heatmap(matrix, order, args, output_path)

    matrix.loc[order, order].to_csv(
        output_dir / f"{args.output_name}.csv", encoding="utf-8-sig", float_format="%.6f"
    )
    if not pair_summary.empty:
        pair_summary.to_csv(
            output_dir / f"{args.output_name}_pair_summary.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
    metadata = {
        "source": str(source),
        "source_type": "pairwise_cka_by_wsi" if args.cka_file is not None else "mean_cka_matrix",
        "estimator": "mean_per_wsi_centered_linear_cka",
        "order": "explicit" if args.encoder_order is not None else args.order,
        "encoder_order": order,
        "colour_range": [args.vmin, args.vmax],
        "outputs": {
            "png": str(output_path),
            "pdf": str(output_path.with_suffix(".pdf")),
            "svg": str(output_path.with_suffix(".svg")),
        },
    }
    (output_dir / f"{args.output_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved CKA heatmap: {output_path}")
    print(f"Encoder order: {', '.join(order)}")


if __name__ == "__main__":
    main()
