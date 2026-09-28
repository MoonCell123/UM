"""Plot the encoder-attribution distribution used in the appendix figure.

The input can be either a CSV file or a GME run directory.  For a run
directory the script first looks for ``fold_*/teacher_attribution_targets.csv``
and falls back to ``fold_*/final_routing_weights.csv``.

Examples
--------
    I:\Anaconda\anaconda3\envs\Pytorch\python.exe code/visualization/plot_attribution_distribution.py \
        --input output/GME/UVM/20260807_064832_n5_High_redundancy

    python code/visualization/plot_attribution_distribution.py \
        --input output/GME/UVM/20260807_064832_n5_High_redundancy \
        --value-column routing_score \
        --output-prefix paper/figs/um_fm/attribution_distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT = Path("paper/figs/um_fm/attribution_distribution")
DEFAULT_ORDER = ["UNI", "GigaPath", "H-Optimus-0", "H-Optimus-1", "Virchow"]
COLORS = {
    "UNI": "#4472a8",
    "GigaPath": "#5b8e7d",
    "H-Optimus-0": "#c9824a",
    "H-Optimus-1": "#9a6fb0",
    "Virchow": "#c45b5b",
}


def display_encoder(name: str) -> str:
    """Convert feature-directory names into paper-facing encoder labels."""

    key = str(name).strip()
    aliases = {
        "features_uni_v1": "UNI",
        "features_uni_v2": "UNI",
        "uni_v1": "UNI",
        "uni_v2": "UNI",
        "features_gigapath": "GigaPath",
        "gigapath": "GigaPath",
        "features_hoptimus0": "H-Optimus-0",
        "hoptimus0": "H-Optimus-0",
        "features_hoptimus1": "H-Optimus-1",
        "hoptimus1": "H-Optimus-1",
        "features_virchow": "Virchow",
        "features_virchow2": "Virchow",
        "virchow": "Virchow",
        "virchow2": "Virchow",
        "features_conch_v1": "CONCH-v1",
        "features_conch_v15": "CONCH-v1.5",
        "features_phikon": "Phikon",
        "features_phikon_v2": "Phikon-v2",
    }
    return aliases.get(key.lower(), key.removeprefix("features_"))


def find_csv_files(input_path: Path) -> tuple[list[Path], str]:
    """Find attribution CSV files and infer the most appropriate value column."""

    if input_path.is_file():
        return [input_path], "auto"
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    target_files = sorted(input_path.rglob("teacher_attribution_targets.csv"))
    if target_files:
        return target_files, "auto"

    target_files = sorted(input_path.rglob("final_routing_weights.csv"))
    if target_files:
        return target_files, "auto"

    raise FileNotFoundError(
        "No teacher_attribution_targets.csv or final_routing_weights.csv found "
        f"below {input_path}"
    )


def load_attributions(
    csv_files: Iterable[Path], value_column: str = "auto"
) -> tuple[pd.DataFrame, str]:
    """Load and validate the long-form encoder attribution table."""

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        frame["_source_file"] = str(csv_file)
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files were provided")

    data = pd.concat(frames, ignore_index=True)
    if "encoder" not in data.columns:
        raise ValueError("Attribution CSV must contain an 'encoder' column")

    if value_column == "auto":
        candidates = ["normalized_score", "attribution", "routing_score", "loo_true_margin", "weight"]
        available = [column for column in candidates if column in data.columns]
        # Prefer an actual varying attribution signal. Some older routing CSVs
        # contain an all-zero placeholder column alongside routing_score.
        value_column = next(
            (
                column
                for column in available
                if pd.to_numeric(data[column], errors="coerce").nunique(dropna=True) > 1
            ),
            available[0] if available else "",
        )
    if value_column not in data.columns:
        raise ValueError(
            f"Value column '{value_column}' is not present. Available columns: "
            f"{', '.join(data.columns)}"
        )

    data["value"] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna(subset=["encoder", "value"]).copy()
    data["encoder_label"] = data["encoder"].map(display_encoder)
    if data.empty:
        raise ValueError("No finite attribution values remain after loading the CSV files")
    return data, value_column


def encoder_order(data: pd.DataFrame, requested: str | None) -> list[str]:
    if requested:
        requested_order = [item.strip() for item in requested.split(",") if item.strip()]
        present = set(data["encoder_label"])
        return [item for item in requested_order if item in present] + sorted(
            present.difference(requested_order)
        )

    present = set(data["encoder_label"])
    return [item for item in DEFAULT_ORDER if item in present] + sorted(
        present.difference(DEFAULT_ORDER)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A CSV file or a GME run directory containing fold CSV files.",
    )
    parser.add_argument(
        "--value-column",
        default="auto",
        help="Column to plot; auto selects normalized_score, attribution, routing_score, or loo_true_margin.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension. Both PDF and SVG are written.",
    )
    parser.add_argument(
        "--order",
        default=None,
        help="Comma-separated encoder display order, e.g. UNI,GigaPath,H-Optimus-0.",
    )
    parser.add_argument("--title", default="Distribution of encoder attributions across outer LOO folds")
    parser.add_argument("--subtitle", default="auto", help="Use 'none' to hide the subtitle.")
    parser.add_argument(
        "--note",
        default="Positive values increase routing preference; negative values suppress it.",
        help="Small explanatory note below the axes; use 'none' to hide it.",
    )
    parser.add_argument("--ylabel", default="LOO attribution (routing score)")
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional fixed y-axis limits, for example --ylim -4 6.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files, inferred_column = find_csv_files(args.input)
    value_column = inferred_column if args.value_column == "auto" else args.value_column
    data, value_column = load_attributions(csv_files, value_column)
    order = encoder_order(data, args.order)
    if not order:
        raise ValueError("None of the requested encoders are present in the input data")

    groups = [data.loc[data["encoder_label"] == label, "value"].to_numpy() for label in order]
    if any(len(values) == 0 for values in groups):
        raise ValueError("An encoder in the requested order has no usable attribution values")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.3,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.05, 3.77), dpi=180)
    positions = np.arange(1, len(order) + 1)
    colors = [COLORS.get(label, "#6f7782") for label in order]

    violin = ax.violinplot(
        groups,
        positions=positions,
        widths=0.76,
        points=200,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        body.set_linewidth(0.7)

    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "#1e242b", "linewidth": 1.2},
        whiskerprops={"color": "#38434d", "linewidth": 0.8},
        capprops={"color": "#38434d", "linewidth": 0.8},
        boxprops={"edgecolor": "#38434d", "linewidth": 0.75},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.axhline(0, color="#818890", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.yaxis.grid(True, color="#e3e6e9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(0.35, len(order) + 0.65)
    if args.ylim:
        ax.set_ylim(args.ylim)
    else:
        low = float(data["value"].min())
        high = float(data["value"].max())
        span = max(high - low, 1e-6)
        ax.set_ylim(low - 0.08 * span, high + 0.08 * span)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4d5964")
    ax.spines["bottom"].set_color("#4d5964")
    ax.tick_params(axis="both", colors="#303942", length=3, width=0.65)
    ax.set_title(args.title, pad=22, fontweight="semibold", color="#1f2933")

    if args.subtitle.lower() != "none":
        if args.subtitle.lower() == "auto":
            if "slide_id" in data.columns:
                n_slides = int(data["slide_id"].nunique())
            else:
                n_slides = int(min(len(group) for group in groups))
            subtitle = f"n = {n_slides} slides per encoder"
        else:
            subtitle = args.subtitle
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#59636e",
        )
    if args.note.lower() != "none":
        fig.text(
            0.99,
            0.012,
            args.note,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#59636e",
        )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.78)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_prefix.with_suffix(".pdf")
    svg_path = args.output_prefix.with_suffix(".svg")
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")
    plt.close(fig)

    counts = data.groupby("encoder_label").size().reindex(order).to_dict()
    print(f"Loaded {len(csv_files)} CSV file(s), value column: {value_column}")
    print(f"Attribution counts: {counts}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
