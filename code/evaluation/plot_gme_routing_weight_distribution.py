"""Create publication-ready visualizations of GME OOF routing weights.

The script aggregates ``final_routing_weights.csv`` from every completed fold
of one GME run.  Each slide should occur in exactly one fold's validation
output, so the resulting collection is out-of-fold (OOF) and is appropriate
for reporting the sample-specific behavior of the router.

The main figure contains three panels:
1. a slide-by-encoder heatmap, with rows grouped by dominant encoder;
2. encoder-wise violin and jitter distributions, including the mean-fusion
   reference of 1 / n_encoders;
3. a normalized routing-entropy histogram.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "Analysis_Result"
WEIGHT_FILE_NAME = "final_routing_weights.csv"
REQUIRED_COLUMNS = ("slide_id", "encoder", "weight")
ENCODER_COLORS = ("#2563eb", "#d97706", "#16803c", "#7c3aed", "#dc2626", "#0891b2", "#a16207")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OOF GME routing-weight distributions for a completed run."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="GME run directory containing fold_* outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures and summary tables. Default: output/Analysis_Result/<run_name>/routing_weight_distribution.",
    )
    return parser.parse_args()


def fold_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected fold directory named fold_<integer>, got {path}") from exc


def display_encoder(name: str) -> str:
    label = name.removeprefix("features_").replace("_", " ")
    return label[:1].upper() + label[1:]


def validate_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")


def collect_oof_weights(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[Path]]:
    fold_dirs = sorted(
        (path for path in run_dir.glob("fold_*") if path.is_dir()),
        key=fold_number,
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {run_dir}")

    frames: list[pd.DataFrame] = []
    encoder_order: list[str] | None = None
    weight_files: list[Path] = []
    for fold_dir in fold_dirs:
        path = fold_dir / WEIGHT_FILE_NAME
        if not path.is_file():
            raise FileNotFoundError(f"OOF routing weights not found: {path}")
        frame = pd.read_csv(path)
        validate_columns(frame, REQUIRED_COLUMNS, path)
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        if frame[["slide_id", "encoder", "weight"]].isna().any().any():
            raise ValueError(f"{path} contains missing slide_id, encoder, or weight values")
        if frame.duplicated(["slide_id", "encoder"]).any():
            raise ValueError(f"{path} has duplicate slide_id/encoder rows")
        current_encoders = frame["encoder"].drop_duplicates().tolist()
        if encoder_order is None:
            encoder_order = current_encoders
        elif set(current_encoders) != set(encoder_order):
            raise ValueError(
                f"Encoder set differs in {path}. Expected {encoder_order}, got {current_encoders}."
            )
        frame["fold"] = fold_number(fold_dir)
        frames.append(frame)
        weight_files.append(path)

    assert encoder_order is not None
    long_weights = pd.concat(frames, ignore_index=True)
    if long_weights.duplicated(["slide_id", "encoder"]).any():
        duplicated = long_weights.loc[long_weights.duplicated(["slide_id", "encoder"], keep=False), "slide_id"].unique()
        raise ValueError(
            "Slides appear in more than one fold's final routing output; expected OOF predictions. "
            f"Examples: {duplicated[:5].tolist()}"
        )

    wide_weights = long_weights.pivot(index="slide_id", columns="encoder", values="weight")
    wide_weights = wide_weights.reindex(columns=encoder_order)
    if wide_weights.isna().any().any():
        raise ValueError("At least one OOF slide is missing an encoder weight")
    weight_sums = wide_weights.sum(axis=1)
    if not np.allclose(weight_sums.to_numpy(), 1.0, rtol=0.0, atol=1e-3):
        raise ValueError(
            "Routing weights must sum to one per slide. "
            f"Observed range: [{weight_sums.min():.6f}, {weight_sums.max():.6f}]"
        )

    fold_by_slide = long_weights[["slide_id", "fold"]].drop_duplicates().set_index("slide_id")["fold"]
    if fold_by_slide.index.duplicated().any():
        raise ValueError("A slide has inconsistent fold assignments")
    return long_weights, wide_weights, encoder_order, weight_files


def make_slide_summary(wide_weights: pd.DataFrame, fold_by_slide: pd.Series) -> pd.DataFrame:
    n_encoders = wide_weights.shape[1]
    values = wide_weights.to_numpy(dtype=float)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    summary = pd.DataFrame(
        {
            "slide_id": wide_weights.index,
            "fold": fold_by_slide.reindex(wide_weights.index).to_numpy(),
            "dominant_encoder": wide_weights.idxmax(axis=1).to_numpy(),
            "max_weight": wide_weights.max(axis=1).to_numpy(),
            "routing_entropy": entropy,
            "normalized_routing_entropy": entropy / np.log(n_encoders),
        }
    )
    return summary


def sort_for_heatmap(wide_weights: pd.DataFrame, slide_summary: pd.DataFrame, encoder_order: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank = {encoder: index for index, encoder in enumerate(encoder_order)}
    summary = slide_summary.copy()
    summary["dominant_rank"] = summary["dominant_encoder"].map(rank)
    summary = summary.sort_values(
        ["dominant_rank", "max_weight", "normalized_routing_entropy", "slide_id"],
        ascending=[True, False, True, True],
    )
    ordered_weights = wide_weights.loc[summary["slide_id"]]
    return ordered_weights, summary.drop(columns="dominant_rank")


def make_encoder_summary(wide_weights: pd.DataFrame, slide_summary: pd.DataFrame, encoder_order: Sequence[str]) -> pd.DataFrame:
    uniform_weight = 1.0 / len(encoder_order)
    dominant_counts = slide_summary["dominant_encoder"].value_counts()
    records = []
    for encoder in encoder_order:
        weights = wide_weights[encoder]
        records.append(
            {
                "encoder": encoder,
                "n_slides": len(weights),
                "mean_weight": weights.mean(),
                "median_weight": weights.median(),
                "std_weight": weights.std(ddof=1),
                "q25_weight": weights.quantile(0.25),
                "q75_weight": weights.quantile(0.75),
                "min_weight": weights.min(),
                "max_weight": weights.max(),
                "dominant_slide_count": int(dominant_counts.get(encoder, 0)),
                "dominant_slide_fraction": float(dominant_counts.get(encoder, 0) / len(weights)),
                "mean_fusion_reference": uniform_weight,
                "mean_minus_uniform": weights.mean() - uniform_weight,
            }
        )
    return pd.DataFrame(records)


def draw_heatmap(ax: plt.Axes, weights: pd.DataFrame, encoder_order: Sequence[str]) -> None:
    vmax = max(float(weights.to_numpy().max()), 1.0 / len(encoder_order))
    image = ax.imshow(weights.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=vmax)
    ax.set_title("OOF slide-level routing weights", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dominant encoder group")
    ax.set_xticks(np.arange(len(encoder_order)), [display_encoder(encoder) for encoder in encoder_order], rotation=20, ha="right")
    dominant_encoders = weights.idxmax(axis=1).to_numpy()
    boundaries = np.flatnonzero(dominant_encoders[1:] != dominant_encoders[:-1])
    for boundary in boundaries:
        ax.axhline(boundary + 0.5, color="white", linewidth=1.2, alpha=0.9)

    group_starts = np.r_[0, boundaries + 1]
    group_ends = np.r_[boundaries, len(dominant_encoders) - 1]
    group_centers = (group_starts + group_ends) / 2.0
    group_labels = [
        f"{display_encoder(dominant_encoders[start])} (n={end - start + 1})"
        for start, end in zip(group_starts, group_ends)
    ]
    ax.set_yticks(group_centers, labels=group_labels)
    ax.tick_params(axis="y", labelsize=8, length=0, pad=7)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.023, pad=0.015)
    colorbar.set_label("Routing weight")


def draw_violin(ax: plt.Axes, weights: pd.DataFrame, encoder_order: Sequence[str]) -> None:
    positions = np.arange(1, len(encoder_order) + 1)
    violin = ax.violinplot(
        [weights[encoder].to_numpy() for encoder in encoder_order],
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(ENCODER_COLORS[index % len(ENCODER_COLORS)])
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color("#374151")
        violin[key].set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for index, encoder in enumerate(encoder_order):
        jitter = rng.uniform(-0.075, 0.075, size=len(weights))
        ax.scatter(
            np.full(len(weights), positions[index]) + jitter,
            weights[encoder],
            s=11,
            color=ENCODER_COLORS[index % len(ENCODER_COLORS)],
            alpha=0.38,
            linewidths=0,
        )
    uniform_weight = 1.0 / len(encoder_order)
    ax.axhline(uniform_weight, color="#111827", linestyle="--", linewidth=1.1, label=f"Mean fusion = {uniform_weight:.2f}")
    ax.set_title("Encoder-wise OOF weight distributions", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Routing weight")
    ax.set_xticks(positions, [display_encoder(encoder) for encoder in encoder_order], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def draw_entropy(ax: plt.Axes, slide_summary: pd.DataFrame) -> None:
    entropy = slide_summary["normalized_routing_entropy"].to_numpy()
    ax.hist(entropy, bins=min(16, max(6, int(np.sqrt(len(entropy))))), color="#475569", edgecolor="white", alpha=0.9)
    median = float(np.median(entropy))
    ax.axvline(median, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Median = {median:.2f}")
    ax.set_title("Routing concentration across slides", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized routing entropy")
    ax.set_ylabel("Number of OOF slides")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def save_figure(weights: pd.DataFrame, slide_summary: pd.DataFrame, encoder_order: Sequence[str], output_dir: Path) -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.7, 1.0), width_ratios=(1.45, 1.0))
    heatmap_ax = fig.add_subplot(grid[0, :])
    violin_ax = fig.add_subplot(grid[1, 0])
    entropy_ax = fig.add_subplot(grid[1, 1])
    draw_heatmap(heatmap_ax, weights, encoder_order)
    draw_violin(violin_ax, weights, encoder_order)
    draw_entropy(entropy_ax, slide_summary)
    fig.suptitle("Out-of-Fold routing-weight distribution", fontsize=15, fontweight="bold")
    output_path = output_dir / "gme_routing_weight_distribution.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"GME run directory not found: {run_dir}")
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / run_dir.name / "routing_weight_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_weights, wide_weights, encoder_order, weight_files = collect_oof_weights(run_dir)
    fold_by_slide = long_weights[["slide_id", "fold"]].drop_duplicates().set_index("slide_id")["fold"]
    slide_summary = make_slide_summary(wide_weights, fold_by_slide)
    ordered_weights, ordered_slide_summary = sort_for_heatmap(wide_weights, slide_summary, encoder_order)
    encoder_summary = make_encoder_summary(wide_weights, slide_summary, encoder_order)

    long_weights.to_csv(output_dir / "oof_routing_weights_long.csv", index=False, float_format="%.6f")
    ordered_weights.reset_index().to_csv(output_dir / "oof_routing_weights_wide.csv", index=False, float_format="%.6f")
    ordered_slide_summary.to_csv(output_dir / "oof_routing_entropy_by_slide.csv", index=False, float_format="%.6f")
    encoder_summary.to_csv(output_dir / "routing_weight_summary_by_encoder.csv", index=False, float_format="%.6f")
    save_figure(ordered_weights, ordered_slide_summary, encoder_order, output_dir)

    metadata = {
        "run_dir": str(run_dir),
        "weight_files": [str(path) for path in weight_files],
        "n_folds": len(weight_files),
        "n_oof_slides": int(len(wide_weights)),
        "encoders": encoder_order,
        "row_order": "dominant encoder, then descending maximum weight, then ascending normalized entropy",
        "weight_reference": float(1.0 / len(encoder_order)),
        "entropy": "Shannon entropy normalized by log(n_encoders)",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Fold weight files: {len(weight_files)}")
    print(f"OOF slides: {len(wide_weights)}")
    print(f"Encoders: {len(encoder_order)}")
    print(f"Saved outputs: {output_dir}")


if __name__ == "__main__":
    main()
