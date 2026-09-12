"""Compare Router-only and full CoR-Fuse runs.

The comparison is deliberately mechanism-aware.  It uses out-of-fold routing
weights to quantify encoder collapse and coverage, fold-level metrics for the
predictive comparison, and stage-2 histories to quantify optimization
stability.  It does not treat a change in routing entropy as evidence of a
predictive improvement by itself.

Example
-------
python compare_router_only_cor_fuse.py \
    --router-only output/GME/BLCA/Router_Only \
    --cor-fuse output/GME/BLCA/All \
    --output-dir output/Analysis_Result/router_vs_cor_fuse/BLCA \
    --dataset-name BLCA
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROUTER = "Router-only"
METHOD_FULL = "CoR-Fuse"
METHODS = (METHOD_ROUTER, METHOD_FULL)
WEIGHT_COLUMNS = ("slide_id", "encoder", "weight")
PERFORMANCE_COLUMNS = ("auc", "auprc", "accuracy", "f1", "precision", "recall")
ROUTING_METRICS = (
    "normalized_entropy",
    "effective_encoder_count",
    "max_weight",
    "top1_margin",
    "total_variation_from_uniform",
)
METHOD_COLORS = {METHOD_ROUTER: "#64748b", METHOD_FULL: "#0f766e"}
ENCODER_COLORS = (
    "#2563eb",
    "#d97706",
    "#16803c",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
    "#a16207",
    "#be185d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare out-of-fold routing behavior, fold performance, and "
            "stage-2 training stability of Router-only and CoR-Fuse runs."
        )
    )
    parser.add_argument("--router-only", type=Path, required=True, help="Router-only run directory.")
    parser.add_argument("--cor-fuse", type=Path, required=True, help="Full CoR-Fuse run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV summaries, JSON metadata, and figures.",
    )
    parser.add_argument("--dataset-name", default="comparison", help="Dataset label used in figure titles.")
    return parser.parse_args()


def fold_number(path: Path) -> int:
    match = re.fullmatch(r"fold_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Expected a fold_<integer> directory, got {path}")
    return int(match.group(1))


def fold_dirs(run_dir: Path) -> list[Path]:
    paths = [path for path in run_dir.iterdir() if path.is_dir() and re.fullmatch(r"fold_\d+", path.name)]
    return sorted(paths, key=fold_number)


def display_encoder(name: str) -> str:
    label = str(name).removeprefix("features_").replace("_", " ")
    return label[:1].upper() + label[1:]


def validate_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")


def collect_weights(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    """Read and validate OOF routing weights from all completed folds."""

    directories = fold_dirs(run_dir)
    if not directories:
        raise FileNotFoundError(f"No fold_* directories found under {run_dir}")

    frames: list[pd.DataFrame] = []
    encoder_order: list[str] = []
    for fold_dir in directories:
        path = fold_dir / "final_routing_weights.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing final routing weights: {path}")
        frame = pd.read_csv(path)
        validate_columns(frame, WEIGHT_COLUMNS, path)
        frame = frame.loc[:, list(WEIGHT_COLUMNS)].copy()
        frame["slide_id"] = frame["slide_id"].astype(str)
        frame["encoder"] = frame["encoder"].astype(str)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
        if frame[list(WEIGHT_COLUMNS)].isna().any().any():
            raise ValueError(f"{path} contains missing routing-weight values")
        if frame.duplicated(["slide_id", "encoder"]).any():
            raise ValueError(f"{path} contains duplicate slide_id/encoder rows")

        current_order = frame["encoder"].drop_duplicates().tolist()
        if not encoder_order:
            encoder_order = current_order
        elif set(current_order) != set(encoder_order):
            raise ValueError(
                f"Encoder set differs in {path}: expected {encoder_order}, got {current_order}"
            )

        counts = frame.groupby("slide_id")["encoder"].nunique()
        if not (counts == len(encoder_order)).all():
            bad = counts[counts != len(encoder_order)]
            raise ValueError(f"{path} has incomplete encoder rows for slides: {bad.index[:5].tolist()}")
        sums = frame.groupby("slide_id")["weight"].sum()
        if not np.allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-3):
            raise ValueError(
                f"{path}: routing weights must sum to one; observed "
                f"[{sums.min():.6f}, {sums.max():.6f}]"
            )
        frame["fold"] = fold_number(fold_dir)
        frames.append(frame)

    long_weights = pd.concat(frames, ignore_index=True)
    duplicate_slides = long_weights.duplicated(["slide_id", "encoder"], keep=False)
    if duplicate_slides.any():
        examples = long_weights.loc[duplicate_slides, "slide_id"].unique()[:5].tolist()
        raise ValueError(f"A slide occurs in multiple OOF folds: {examples}")

    wide_weights = long_weights.pivot(index="slide_id", columns="encoder", values="weight")
    wide_weights = wide_weights.reindex(columns=encoder_order)
    if wide_weights.isna().any().any():
        raise ValueError("At least one OOF slide is missing an encoder weight")
    fold_by_slide = long_weights[["slide_id", "fold"]].drop_duplicates().set_index("slide_id")["fold"]
    return long_weights, wide_weights, encoder_order, fold_by_slide


def compute_slide_metrics(
    wide_weights: pd.DataFrame,
    fold_by_slide: pd.Series,
    encoder_order: Sequence[str],
    method: str,
) -> pd.DataFrame:
    values = wide_weights.loc[:, list(encoder_order)].to_numpy(dtype=float)
    n_encoders = values.shape[1]
    uniform = 1.0 / n_encoders
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    sorted_values = np.sort(values, axis=1)
    second = sorted_values[:, -2] if n_encoders > 1 else np.zeros(len(values))
    metrics = pd.DataFrame(
        {
            "slide_id": wide_weights.index.astype(str),
            "fold": fold_by_slide.reindex(wide_weights.index).to_numpy(),
            "method": method,
            "dominant_encoder": wide_weights.idxmax(axis=1).to_numpy(),
            "max_weight": values.max(axis=1),
            "top1_margin": values.max(axis=1) - second,
            "routing_entropy": entropy,
            "normalized_entropy": entropy / math.log(n_encoders) if n_encoders > 1 else 1.0,
            "effective_encoder_count": np.exp(entropy),
            "total_variation_from_uniform": 0.5 * np.abs(values - uniform).sum(axis=1),
            "l2_from_uniform": np.sqrt(((values - uniform) ** 2).sum(axis=1)),
        }
    )
    return metrics


def read_fold_metrics(run_dir: Path, method: str) -> pd.DataFrame:
    """Read fold-level final metrics, with a fallback for older runs."""

    path = run_dir / "fold_metrics.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        validate_columns(frame, ("fold",), path)
    else:
        rows: list[dict[str, object]] = []
        for fold_dir in fold_dirs(run_dir):
            final_path = fold_dir / "final_metrics.csv"
            if not final_path.is_file():
                continue
            current = pd.read_csv(final_path)
            if current.empty:
                continue
            row = current.iloc[0].to_dict()
            row["fold"] = fold_number(fold_dir)
            rows.append(row)
        if not rows:
            raise FileNotFoundError(f"No fold metrics found under {run_dir}")
        frame = pd.DataFrame(rows)

    frame = frame.copy()
    frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
    frame = frame.loc[frame["fold"].notna()].copy()
    frame["fold"] = frame["fold"].astype(int)
    for column in PERFORMANCE_COLUMNS + ("parameters", "flops", "inference_time"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["method"] = method
    return frame.sort_values("fold").reset_index(drop=True)


def read_training_history(run_dir: Path, method: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold_dir in fold_dirs(run_dir):
        path = fold_dir / "stage2_training_history.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "epoch" not in frame.columns:
            continue
        frame = frame.copy()
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        frame = frame.loc[frame["epoch"].notna()].copy()
        frame["epoch"] = frame["epoch"].astype(int)
        if "train_loss" not in frame.columns and "train_cls_loss" in frame.columns:
            frame["train_loss"] = frame["train_cls_loss"]
        for column in ("train_loss", "inner_auc", "inner_auprc"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["fold"] = fold_number(fold_dir)
        frame["method"] = method
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["epoch", "fold", "method", "train_loss", "inner_auc", "inner_auprc"])
    return pd.concat(rows, ignore_index=True)


def read_config(run_dir: Path) -> Mapping[str, object]:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def describe_encoder_weights(
    wide_weights: pd.DataFrame,
    slide_metrics: pd.DataFrame,
    encoder_order: Sequence[str],
    method: str,
) -> pd.DataFrame:
    dominant = slide_metrics["dominant_encoder"].value_counts()
    records: list[dict[str, object]] = []
    for encoder in encoder_order:
        values = wide_weights[encoder]
        records.append(
            {
                "method": method,
                "encoder": encoder,
                "n_slides": int(len(values)),
                "mean_weight": float(values.mean()),
                "median_weight": float(values.median()),
                "std_weight": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "q25_weight": float(values.quantile(0.25)),
                "q75_weight": float(values.quantile(0.75)),
                "min_weight": float(values.min()),
                "max_weight": float(values.max()),
                "dominant_slide_count": int(dominant.get(encoder, 0)),
                "dominant_slide_fraction": float(dominant.get(encoder, 0) / len(values)),
                "uniform_reference": 1.0 / len(encoder_order),
            }
        )
    return pd.DataFrame(records)


def summary_row(dataset: str, method: str, metric: str, values: pd.Series | np.ndarray) -> dict[str, object]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return {
            "dataset": dataset,
            "method": method,
            "level": "summary",
            "metric": metric,
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "dataset": dataset,
        "method": method,
        "level": "summary",
        "metric": metric,
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_summary_tables(
    dataset: str,
    run_data: Mapping[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    encoder_frames: list[pd.DataFrame] = []
    slide_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []
    for method in METHODS:
        data = run_data[method]
        slides = data["slide_metrics"]
        metrics = data["fold_metrics"]
        assert isinstance(slides, pd.DataFrame)
        assert isinstance(metrics, pd.DataFrame)
        slide_frames.append(slides)
        encoder_frames.append(data["encoder_summary"])
        performance_frames.append(metrics)
        for metric in ROUTING_METRICS:
            summary_rows.append(summary_row(dataset, method, metric, slides[metric]))
        for metric in PERFORMANCE_COLUMNS + ("parameters", "flops", "inference_time"):
            if metric in metrics.columns:
                summary_rows.append(summary_row(dataset, method, metric, metrics[metric]))

    summary = pd.DataFrame(summary_rows)
    encoder_summary = pd.concat(encoder_frames, ignore_index=True)
    all_slides = pd.concat(slide_frames, ignore_index=True)
    all_performance = pd.concat(performance_frames, ignore_index=True)

    router_slides = run_data[METHOD_ROUTER]["slide_metrics"]
    full_slides = run_data[METHOD_FULL]["slide_metrics"]
    slide_columns = [
        "slide_id",
        "fold",
        *ROUTING_METRICS,
        "routing_entropy",
        "dominant_encoder",
    ]
    paired_slides = router_slides[slide_columns].merge(
        full_slides[slide_columns],
        on="slide_id",
        how="inner",
        suffixes=("_router_only", "_cor_fuse"),
    )
    for metric in ROUTING_METRICS:
        paired_slides[f"delta_{metric}"] = (
            paired_slides[f"{metric}_cor_fuse"] - paired_slides[f"{metric}_router_only"]
        )
    paired_slides["dominant_encoder_changed"] = (
        paired_slides["dominant_encoder_router_only"] != paired_slides["dominant_encoder_cor_fuse"]
    )

    router_perf = run_data[METHOD_ROUTER]["fold_metrics"]
    full_perf = run_data[METHOD_FULL]["fold_metrics"]
    performance_columns = ["fold", *[metric for metric in PERFORMANCE_COLUMNS if metric in router_perf.columns and metric in full_perf.columns]]
    paired_performance = router_perf[performance_columns].merge(
        full_perf[performance_columns],
        on="fold",
        how="inner",
        suffixes=("_router_only", "_cor_fuse"),
    )
    for metric in PERFORMANCE_COLUMNS:
        if f"{metric}_router_only" in paired_performance and f"{metric}_cor_fuse" in paired_performance:
            paired_performance[f"delta_{metric}"] = (
                paired_performance[f"{metric}_cor_fuse"] - paired_performance[f"{metric}_router_only"]
            )

    return summary, encoder_summary, paired_slides, paired_performance


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(output_dir / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_distribution(ax: plt.Axes, slide_frames: Mapping[str, pd.DataFrame], metric: str, title: str, ylabel: str) -> None:
    values = [slide_frames[method][metric].to_numpy(dtype=float) for method in METHODS]
    positions = np.arange(1, len(METHODS) + 1)
    parts = ax.violinplot(values, positions=positions, widths=0.72, showmeans=False, showmedians=True, showextrema=True)
    for index, body in enumerate(parts["bodies"]):
        body.set_facecolor(METHOD_COLORS[METHODS[index]])
        body.set_edgecolor("white")
        body.set_alpha(0.72)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#1f2937")
        parts[key].set_linewidth(0.9)
    rng = np.random.default_rng(42)
    for index, method in enumerate(METHODS):
        jitter = rng.uniform(-0.09, 0.09, size=len(values[index]))
        ax.scatter(
            np.full(len(values[index]), positions[index]) + jitter,
            values[index],
            s=7,
            color=METHOD_COLORS[method],
            alpha=0.24,
            linewidths=0,
        )
        ax.scatter(
            positions[index],
            np.mean(values[index]),
            s=35,
            color="white",
            edgecolor=METHOD_COLORS[method],
            linewidth=1.4,
            zorder=4,
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions, METHODS)
    ax.grid(axis="y", alpha=0.25)


def draw_encoder_bars(ax: plt.Axes, encoder_summaries: Mapping[str, pd.DataFrame], column: str, title: str, ylabel: str) -> None:
    encoders = encoder_summaries[METHODS[0]]["encoder"].tolist()
    x = np.arange(len(encoders))
    width = 0.36
    for index, method in enumerate(METHODS):
        frame = encoder_summaries[method].set_index("encoder").reindex(encoders)
        ax.bar(
            x + (index - 0.5) * width,
            frame[column].to_numpy(dtype=float),
            width=width,
            label=method,
            color=METHOD_COLORS[method],
            alpha=0.88,
        )
    if column == "mean_weight":
        reference = 1.0 / len(encoders)
        ax.axhline(reference, color="#111827", linestyle="--", linewidth=1.0, label=f"Uniform = {reference:.2f}")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [display_encoder(name) for name in encoders], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def draw_paired_delta(ax: plt.Axes, paired_slides: pd.DataFrame) -> None:
    metrics = ["normalized_entropy", "max_weight", "effective_encoder_count", "top1_margin"]
    labels = ["Entropy", "Max weight", "Effective N", "Top-1 margin"]
    data = [paired_slides[f"delta_{metric}"].dropna().to_numpy(dtype=float) for metric in metrics]
    positions = np.arange(1, len(metrics) + 1)
    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(METHOD_COLORS[METHOD_FULL])
        body.set_edgecolor("white")
        body.set_alpha(0.62)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#1f2937")
        parts[key].set_linewidth(0.9)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_title("Paired change: CoR-Fuse minus Router-only", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("Change in routing metric")
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.25)


def plot_routing_behavior(
    dataset: str,
    output_dir: Path,
    run_data: Mapping[str, dict[str, object]],
    encoder_summaries: pd.DataFrame,
    paired_slides: pd.DataFrame,
) -> None:
    slide_frames = {method: run_data[method]["slide_metrics"] for method in METHODS}
    encoder_frames = {
        method: encoder_summaries.loc[encoder_summaries["method"] == method].copy() for method in METHODS
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), constrained_layout=True)
    draw_distribution(
        axes[0, 0], slide_frames, "normalized_entropy", "Routing entropy", "Normalized entropy (higher = less collapse)"
    )
    draw_distribution(
        axes[0, 1], slide_frames, "max_weight", "Maximum encoder weight", "Maximum weight (lower = less concentration)"
    )
    draw_distribution(
        axes[0, 2], slide_frames, "effective_encoder_count", "Effective encoder count", "Effective number of encoders"
    )
    draw_encoder_bars(axes[1, 0], encoder_frames, "mean_weight", "Mean routing weight by encoder", "Mean OOF weight")
    draw_encoder_bars(
        axes[1, 1], encoder_frames, "dominant_slide_fraction", "Top-1 encoder share", "Fraction of OOF slides"
    )
    draw_paired_delta(axes[1, 2], paired_slides)
    fig.suptitle(f"Routing behavior: Router-only vs CoR-Fuse ({dataset})", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "routing_behavior_comparison")


def plot_training_stability(dataset: str, output_dir: Path, run_data: Mapping[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    metric_specs = (("inner_auc", "Stage-2 inner-validation AUC", "Inner AUC"), ("train_loss", "Stage-2 training loss", "Training loss"))
    plotted_any = False
    for ax, (metric, title, ylabel) in zip(axes, metric_specs):
        for method in METHODS:
            history = run_data[method]["history"]
            if metric not in history.columns or history[metric].notna().sum() == 0:
                continue
            plotted_any = True
            color = METHOD_COLORS[method]
            for fold, current in history.groupby("fold"):
                current = current.sort_values("epoch")
                ax.plot(
                    current["epoch"],
                    current[metric],
                    color=color,
                    alpha=0.20,
                    linewidth=0.8,
                )
            grouped = history.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
            x = grouped["epoch"].to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, color=color, linewidth=2.1, label=method)
            if metric == "inner_auc":
                lower = np.clip(mean - std, 0.0, 1.0)
                upper = np.clip(mean + std, 0.0, 1.0)
            else:
                lower = mean - std
                upper = mean + std
            ax.fill_between(x, lower, upper, color=color, alpha=0.12)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if metric == "inner_auc":
            ax.set_ylim(0.0, 1.02)
        if metric == "train_loss":
            positive = []
            for method in METHODS:
                history = run_data[method]["history"]
                if metric in history.columns:
                    positive.extend(history.loc[history[metric] > 0, metric].tolist())
            if positive:
                ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
    if not plotted_any:
        axes[0].text(0.5, 0.5, "No stage-2 history found", ha="center", va="center")
    fig.suptitle(f"Optimization stability: Router-only vs CoR-Fuse ({dataset})", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "training_stability_comparison")


def plot_fold_performance(dataset: str, output_dir: Path, run_data: Mapping[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), constrained_layout=True)
    for ax, metric, title in zip(axes, ("auc", "auprc"), ("Fold-level AUC", "Fold-level AUPRC")):
        all_folds: list[int] = []
        for method in METHODS:
            frame = run_data[method]["fold_metrics"]
            if metric not in frame.columns:
                continue
            x = frame["fold"].to_numpy(dtype=int)
            y = frame[metric].to_numpy(dtype=float)
            all_folds.extend(x.tolist())
            ax.plot(x, y, marker="o", linewidth=1.8, color=METHOD_COLORS[method], label=method)
            ax.axhline(
                float(np.mean(y)),
                color=METHOD_COLORS[method],
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
                label=f"{method} mean = {np.mean(y):.3f}",
            )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Outer fold")
        ax.set_ylabel(metric.upper())
        if all_folds:
            ax.set_xticks(sorted(set(all_folds)))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Predictive performance: Router-only vs CoR-Fuse ({dataset})", fontsize=15, fontweight="bold")
    save_figure(fig, output_dir, "fold_performance_comparison")


def build_training_summary(dataset: str, run_data: Mapping[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        history = run_data[method]["history"]
        if history.empty:
            continue
        for fold, current in history.groupby("fold"):
            current = current.sort_values("epoch")
            row: dict[str, object] = {
                "dataset": dataset,
                "method": method,
                "fold": int(fold),
                "n_epochs": int(len(current)),
                "first_epoch": int(current["epoch"].iloc[0]),
                "last_epoch": int(current["epoch"].iloc[-1]),
            }
            for metric in ("inner_auc", "inner_auprc", "train_loss"):
                if metric not in current.columns or current[metric].notna().sum() == 0:
                    continue
                values = current[metric].dropna().to_numpy(dtype=float)
                row[f"first_{metric}"] = float(values[0])
                row[f"last_{metric}"] = float(values[-1])
                row[f"best_{metric}"] = float(np.max(values)) if metric != "train_loss" else float(np.min(values))
                best_index = int(np.argmax(values) if metric != "train_loss" else np.argmin(values))
                row[f"best_{metric}_epoch"] = int(current.loc[current[metric].dropna().index[best_index], "epoch"])
            rows.append(row)
    return pd.DataFrame(rows)


def write_text_report(
    path: Path,
    dataset: str,
    run_data: Mapping[str, dict[str, object]],
    paired_slides: pd.DataFrame,
    paired_performance: pd.DataFrame,
) -> None:
    lines = [f"Router-only vs CoR-Fuse analysis: {dataset}", "", "Routing behavior (OOF slides):"]
    for method in METHODS:
        slides = run_data[method]["slide_metrics"]
        performance = run_data[method]["fold_metrics"]
        top1 = slides["dominant_encoder"].value_counts(normalize=True)
        top1_text = ", ".join(
            f"{display_encoder(name)}={fraction:.1%}" for name, fraction in top1.items()
        )
        lines.extend(
            [
                f"  {method}: n={len(slides)}; normalized entropy "
                f"{slides['normalized_entropy'].mean():.3f} +/- {slides['normalized_entropy'].std(ddof=1):.3f}; "
                f"effective N {slides['effective_encoder_count'].mean():.3f}; "
                f"max weight {slides['max_weight'].mean():.3f}; "
                f"TV from uniform {slides['total_variation_from_uniform'].mean():.3f}",
                f"    Top-1 share: {top1_text}",
                f"    AUC {performance['auc'].mean():.3f} +/- {performance['auc'].std(ddof=1):.3f}; "
                f"AUPRC {performance['auprc'].mean():.3f} +/- {performance['auprc'].std(ddof=1):.3f}",
            ]
        )
    if not paired_slides.empty:
        lines.extend(
            [
                "",
                "Paired routing changes (CoR-Fuse minus Router-only):",
                f"  normalized entropy: {paired_slides['delta_normalized_entropy'].mean():+.3f}",
                f"  effective encoder count: {paired_slides['delta_effective_encoder_count'].mean():+.3f}",
                f"  maximum weight: {paired_slides['delta_max_weight'].mean():+.3f}",
                f"  top-1 margin: {paired_slides['delta_top1_margin'].mean():+.3f}",
                f"  dominant encoder changed on {paired_slides['dominant_encoder_changed'].mean():.1%} of paired slides",
            ]
        )
    if not paired_performance.empty:
        lines.extend(["", "Paired fold performance changes (CoR-Fuse minus Router-only):"])
        for metric in PERFORMANCE_COLUMNS:
            column = f"delta_{metric}"
            if column in paired_performance:
                delta = paired_performance[column].dropna()
                lines.append(
                    f"  {metric}: {delta.mean():+.3f} +/- {delta.std(ddof=1):.3f}; "
                    f"CoR-Fuse wins {int((delta > 0).sum())}/{len(delta)} folds"
                )
    lines.extend(
        [
            "",
            "Interpretation: higher entropy/effective N and lower maximum weight or "
            "total variation indicate less encoder collapse. These are routing-behavior "
            "measures, not standalone evidence of better prediction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    router_dir = args.router_only.resolve()
    full_dir = args.cor_fuse.resolve()
    output_dir = args.output_dir.resolve()
    for path in (router_dir, full_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_data: dict[str, dict[str, object]] = {}
    for method, run_dir in ((METHOD_ROUTER, router_dir), (METHOD_FULL, full_dir)):
        long_weights, wide_weights, encoder_order, fold_by_slide = collect_weights(run_dir)
        slide_metrics = compute_slide_metrics(wide_weights, fold_by_slide, encoder_order, method)
        fold_metrics = read_fold_metrics(run_dir, method)
        history = read_training_history(run_dir, method)
        encoder_summary = describe_encoder_weights(wide_weights, slide_metrics, encoder_order, method)
        run_data[method] = {
            "run_dir": run_dir,
            "long_weights": long_weights,
            "wide_weights": wide_weights,
            "encoder_order": encoder_order,
            "slide_metrics": slide_metrics,
            "fold_metrics": fold_metrics,
            "history": history,
            "encoder_summary": encoder_summary,
            "config": read_config(run_dir),
        }

    router_encoders = run_data[METHOD_ROUTER]["encoder_order"]
    full_encoders = run_data[METHOD_FULL]["encoder_order"]
    if set(router_encoders) != set(full_encoders):
        raise ValueError(f"The two runs use different encoder sets: {router_encoders} vs {full_encoders}")

    summary, encoder_summary, paired_slides, paired_performance = build_summary_tables(args.dataset_name, run_data)
    training_summary = build_training_summary(args.dataset_name, run_data)

    summary.to_csv(output_dir / "router_vs_cor_fuse_summary.csv", index=False, float_format="%.6f")
    encoder_summary.to_csv(output_dir / "routing_encoder_summary.csv", index=False, float_format="%.6f")
    paired_slides.to_csv(output_dir / "paired_slide_routing_metrics.csv", index=False, float_format="%.6f")
    paired_performance.to_csv(output_dir / "paired_fold_performance.csv", index=False, float_format="%.6f")
    training_summary.to_csv(output_dir / "training_stability_summary.csv", index=False, float_format="%.6f")
    for method in METHODS:
        run_data[method]["slide_metrics"].to_csv(
            output_dir / f"{method.lower().replace('-', '_')}_slide_routing_metrics.csv",
            index=False,
            float_format="%.6f",
        )

    plot_routing_behavior(args.dataset_name, output_dir, run_data, encoder_summary, paired_slides)
    plot_training_stability(args.dataset_name, output_dir, run_data)
    plot_fold_performance(args.dataset_name, output_dir, run_data)

    metadata = {
        "dataset": args.dataset_name,
        "router_only_run": str(router_dir),
        "cor_fuse_run": str(full_dir),
        "encoders": list(router_encoders),
        "router_only_config": run_data[METHOD_ROUTER]["config"],
        "cor_fuse_config": run_data[METHOD_FULL]["config"],
        "n_oof_slides": {
            METHOD_ROUTER: int(len(run_data[METHOD_ROUTER]["slide_metrics"])),
            METHOD_FULL: int(len(run_data[METHOD_FULL]["slide_metrics"])),
            "paired": int(len(paired_slides)),
        },
        "formulas": {
            "normalized_entropy": "H(w) / log(M), H(w)=-sum_m w_m log(w_m)",
            "effective_encoder_count": "exp(H(w))",
            "total_variation_from_uniform": "0.5 * sum_m |w_m - 1/M|",
            "top1_margin": "largest weight minus second-largest weight",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_text_report(output_dir / "analysis_summary.txt", args.dataset_name, run_data, paired_slides, paired_performance)

    print(f"Dataset: {args.dataset_name}")
    print(f"Router-only OOF slides: {len(run_data[METHOD_ROUTER]['slide_metrics'])}")
    print(f"CoR-Fuse OOF slides: {len(run_data[METHOD_FULL]['slide_metrics'])}")
    print(f"Paired OOF slides: {len(paired_slides)}")
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
