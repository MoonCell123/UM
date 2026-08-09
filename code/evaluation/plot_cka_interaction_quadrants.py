"""Plot pairwise CKA and interaction in a four-quadrant diagram.

Each point represents an encoder pair.  The x coordinate is the mean
coordinate-aligned CKA over slides, and the y coordinate is the mean pairwise
interaction over the five interaction folds.  The script writes a signed
interaction plot and an absolute-interaction plot because signed interactions
can cancel when averaged.

The quadrant boundaries are medians across encoder pairs.  A zero interaction
boundary would not be informative for the current result: all pairwise mean
signed interactions are positive.  The median therefore means "higher or
lower than the other encoder pairs" rather than "positive or negative".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "output" / "Analysis_Result" / "20260802_082439_KL=0"
DEFAULT_CKA_ROOT = DEFAULT_ANALYSIS_ROOT / "cka"
DEFAULT_INTERACTION_ROOT = DEFAULT_ANALYSIS_ROOT / "interactions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Analysis_Result" / "20260802_082439_KL=0" / "cka_interaction_quadrants"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot four quadrants from pairwise CKA and interaction summaries."
    )
    parser.add_argument("--cka-root", type=Path, default=DEFAULT_CKA_ROOT)
    parser.add_argument("--interaction-root", type=Path, default=DEFAULT_INTERACTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cka-file",
        type=Path,
        default=None,
        help="Optional explicit pairwise_cka_by_wsi.csv path.",
    )
    parser.add_argument(
        "--interaction-files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit pairwise_interactions.csv paths.",
    )
    return parser.parse_args()


def resolve_cka_file(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"CKA file not found: {explicit}")
        return explicit
    candidates = sorted(
        root.rglob("pairwise_cka_by_wsi.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No pairwise_cka_by_wsi.csv found under {root}")
    return candidates[0]


def resolve_interaction_files(root: Path, explicit: Sequence[Path] | None) -> List[Path]:
    if explicit:
        missing = [path for path in explicit if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Interaction file(s) not found: {missing}")
        return list(explicit)
    candidates = sorted(root.rglob("pairwise_interactions.csv"))
    if not candidates:
        raise FileNotFoundError(f"No pairwise_interactions.csv found under {root}")
    return candidates


def canonical_pair(left: object, right: object) -> str:
    names = sorted((str(left), str(right)))
    return " || ".join(names)


def validate_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")


def load_pair_summary(cka_file: Path, interaction_files: Sequence[Path]) -> pd.DataFrame:
    cka = pd.read_csv(cka_file)
    validate_columns(cka, ("encoder_a", "encoder_b", "linear_cka"), cka_file)
    cka = cka.copy()
    cka["pair"] = [canonical_pair(a, b) for a, b in zip(cka.encoder_a, cka.encoder_b)]
    cka["linear_cka"] = pd.to_numeric(cka["linear_cka"], errors="coerce")
    cka_summary = (
        cka.dropna(subset=["pair", "linear_cka"])
        .groupby("pair", as_index=False)
        .agg(
            cka=("linear_cka", "mean"),
            cka_median=("linear_cka", "median"),
            cka_std=("linear_cka", "std"),
            cka_n=("linear_cka", "count"),
        )
    )

    interaction_frames = []
    for path in interaction_files:
        frame = pd.read_csv(path)
        validate_columns(frame, ("encoder_i", "encoder_j", "interaction"), path)
        frame = frame.copy()
        frame["pair"] = [canonical_pair(a, b) for a, b in zip(frame.encoder_i, frame.encoder_j)]
        frame["interaction"] = pd.to_numeric(frame["interaction"], errors="coerce")
        if "abs_interaction" not in frame.columns:
            frame["abs_interaction"] = frame["interaction"].abs()
        else:
            frame["abs_interaction"] = pd.to_numeric(frame["abs_interaction"], errors="coerce")
        interaction_frames.append(frame.dropna(subset=["pair", "interaction", "abs_interaction"]))
    interactions = pd.concat(interaction_frames, ignore_index=True)
    interaction_summary = (
        interactions.groupby("pair", as_index=False)
        .agg(
            interaction=("interaction", "mean"),
            interaction_median=("interaction", "median"),
            interaction_std=("interaction", "std"),
            abs_interaction=("abs_interaction", "mean"),
            abs_interaction_median=("abs_interaction", "median"),
            interaction_n=("interaction", "count"),
        )
    )

    summary = cka_summary.merge(interaction_summary, on="pair", how="inner", validate="one_to_one")
    if summary.empty:
        raise ValueError("No encoder pairs are shared between the CKA and interaction files")
    if len(summary) < len(cka_summary) or len(summary) < len(interaction_summary):
        print(
            f"Warning: matched {len(summary)} encoder pairs; "
            f"CKA pairs={len(cka_summary)}, interaction pairs={len(interaction_summary)}."
        )
    summary[["encoder_a", "encoder_b"]] = summary["pair"].str.split(" \\|\\| ", expand=True)
    return summary.sort_values(["encoder_a", "encoder_b"]).reset_index(drop=True)


QUADRANT_COLORS = {
    "High CKA / High interaction": "#16803c",
    "High CKA / Low interaction": "#d97706",
    "Low CKA / High interaction": "#2563eb",
    "Low CKA / Low interaction": "#6b7280",
}


def assign_quadrant(cka: float, interaction: float, cka_threshold: float, interaction_threshold: float) -> str:
    high_cka = cka >= cka_threshold
    high_interaction = interaction >= interaction_threshold
    if high_cka and high_interaction:
        return "High CKA / High interaction"
    if high_cka:
        return "High CKA / Low interaction"
    if high_interaction:
        return "Low CKA / High interaction"
    return "Low CKA / Low interaction"


def plot_quadrants(
    summary: pd.DataFrame,
    y_column: str,
    y_label: str,
    title_suffix: str,
    output_path: Path,
) -> tuple[float, float]:
    cka_threshold = float(summary["cka"].median())
    interaction_threshold = float(summary[y_column].median())
    quadrants = [
        assign_quadrant(value, y, cka_threshold, interaction_threshold)
        for value, y in zip(summary["cka"], summary[y_column])
    ]

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    handles = []
    for index, (pair, x_value, y_value, quadrant) in enumerate(
        zip(summary["pair"], summary["cka"], summary[y_column], quadrants), start=1
    ):
        color = QUADRANT_COLORS[quadrant]
        ax.scatter(
            x_value,
            y_value,
            s=190,
            color=color,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        ax.text(x_value, y_value, str(index), color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=f"{index}. {pair}"))

    ax.axvline(cka_threshold, color="#374151", linestyle="--", linewidth=1.0)
    ax.axhline(interaction_threshold, color="#374151", linestyle="--", linewidth=1.0)
    ax.text(0.015, 0.985, "Low CKA / High interaction", transform=ax.transAxes, va="top", color=QUADRANT_COLORS["Low CKA / High interaction"], fontsize=10, fontweight="bold")
    ax.text(0.985, 0.985, "High CKA / High interaction", transform=ax.transAxes, ha="right", va="top", color=QUADRANT_COLORS["High CKA / High interaction"], fontsize=10, fontweight="bold")
    ax.text(0.015, 0.015, "Low CKA / Low interaction", transform=ax.transAxes, va="bottom", color=QUADRANT_COLORS["Low CKA / Low interaction"], fontsize=10, fontweight="bold")
    ax.text(0.985, 0.015, "High CKA / Low interaction", transform=ax.transAxes, ha="right", va="bottom", color=QUADRANT_COLORS["High CKA / Low interaction"], fontsize=10, fontweight="bold")
    ax.set_title(f"CKA–Interaction Quadrants ({title_suffix})")
    ax.set_xlabel("Mean pairwise Linear CKA")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    ax.text(
        1.30,
        1.02,
        f"Median CKA={cka_threshold:.4f} | Median interaction={interaction_threshold:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    return cka_threshold, interaction_threshold


def main() -> None:
    args = parse_args()
    cka_file = resolve_cka_file(args.cka_root, args.cka_file)
    interaction_files = resolve_interaction_files(args.interaction_root, args.interaction_files)
    summary = load_pair_summary(cka_file, interaction_files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signed_thresholds = plot_quadrants(
        summary,
        y_column="interaction",
        y_label="Mean signed interaction",
        title_suffix="mean signed interaction",
        output_path=args.output_dir / "cka_interaction_quadrants.png",
    )
    abs_thresholds = plot_quadrants(
        summary,
        y_column="abs_interaction",
        y_label="Mean absolute interaction",
        title_suffix="mean absolute interaction",
        output_path=args.output_dir / "cka_abs_interaction_quadrants.png",
    )

    summary = summary.copy()
    summary["signed_quadrant"] = [
        assign_quadrant(value, y, signed_thresholds[0], signed_thresholds[1])
        for value, y in zip(summary["cka"], summary["interaction"])
    ]
    summary["absolute_quadrant"] = [
        assign_quadrant(value, y, abs_thresholds[0], abs_thresholds[1])
        for value, y in zip(summary["cka"], summary["abs_interaction"])
    ]
    summary.to_csv(args.output_dir / "cka_interaction_pair_summary.csv", index=False, float_format="%.6f")
    metadata = {
        "cka_file": str(cka_file),
        "interaction_files": [str(path) for path in interaction_files],
        "n_encoder_pairs": int(len(summary)),
        "cka_median_threshold": signed_thresholds[0],
        "signed_interaction_median_threshold": signed_thresholds[1],
        "absolute_interaction_median_threshold": abs_thresholds[1],
        "quadrant_boundary": "median across matched encoder pairs",
        "signed_interaction_note": "All pairwise mean signed interactions are positive in this run; zero would collapse the four-quadrant comparison.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"CKA file: {cka_file}")
    print(f"Interaction files: {len(interaction_files)}")
    print(f"Matched encoder pairs: {len(summary)}")
    print(f"Saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
