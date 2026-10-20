"""Analyze exact four-state logits exported by compute_pairwise_interactions.py.

Bits are (encoder_i, encoder_j): 1=retained, 0=baseline replacement. Other
encoders and the frozen mean-fusion reference model remain unchanged.
Default margins explain the TRUE label, not the original fixed class.
This is descriptive post-hoc analysis, never an online routing signal.
Requires numpy/pandas; matplotlib is imported only when plotting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd

STATES = ("00", "10", "01", "11")
REQUIRED = {"schema_version", "slide_id", "label", "fold", "split",
            "encoder_i", "encoder_j", "target_class", "n_classes",
            "interaction", "checkpoint", "baseline_file",
            "replacement_strategy", "reference_forward", "gaussian_std_scale"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-files", nargs="+", type=Path,
                        help="Explicit pairwise_state_logits.csv files from the same experiment.")
    source.add_argument("--input-root", type=Path,
                        help="Recursively discover pairwise_state_logits.csv; repeated runs are rejected.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory.")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("ENCODER_I", "ENCODER_J"),
                        help="Exact encoder names, either order; repeat to select multiple pairs.")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--margin-target", choices=["true", "exported"], default="true",
                        help="true=ground-truth class; exported=original target_class.")
    parser.add_argument("--effect-epsilon", type=float, default=1e-6,
                        help="Near-zero effect tolerance in margin units; NOT a significance threshold.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"],
                        default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not np.isfinite(args.effect_epsilon) or args.effect_epsilon < 0:
        parser.error("--effect-epsilon must be finite and non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def load_states(paths: list[Path], split: str = "val", pairs=None) -> pd.DataFrame:
    """Validate inputs and align pair orientation before combining folds."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"slide_id": str, "encoder_i": str, "encoder_j": str},
                            keep_default_na=False)
        missing = REQUIRED - set(frame)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}. Rerun the updated "
                             "compute_pairwise_interactions.py; old interaction-only CSVs cannot recover logits.")
        if frame.empty:
            raise ValueError(f"{path}: empty state export")
        for column in ("schema_version", "label", "fold", "target_class", "n_classes"):
            numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
                raise ValueError(f"{path}: {column} must contain finite integers")
            frame[column] = numeric.astype(int)
        if (frame.schema_version != 1).any():
            raise ValueError(f"{path}: unsupported schema_version")
        if frame.n_classes.nunique() != 1 or int(frame.n_classes.iloc[0]) < 2:
            raise ValueError(f"{path}: inconsistent n_classes")
        n_classes = int(frame.n_classes.iloc[0])
        expected = {f"logit_{state}_class_{c}" for state in STATES for c in range(n_classes)}
        actual = {column for column in frame if column.startswith("logit_")}
        if actual != expected:
            raise ValueError(f"{path}: incomplete or unexpected state/class logit columns: "
                             f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        for column in sorted(expected) + ["interaction", "gaussian_std_scale"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[sorted(expected) + ["interaction", "gaussian_std_scale"]].to_numpy()).all():
            raise ValueError(f"{path}: non-finite logits/interaction/settings")
        for column in ("label", "target_class"):
            if not frame[column].between(0, n_classes - 1).all():
                raise ValueError(f"{path}: {column} outside class range")
        if not frame.split.isin(["train", "val"]).all():
            raise ValueError(f"{path}: unknown split")
        if not frame.reference_forward.eq("mean_fusion").all():
            raise ValueError(f"{path}: expected frozen mean_fusion reference")
        if not frame.replacement_strategy.isin(["mean", "zero", "gaussian"]).all():
            raise ValueError(f"{path}: unknown replacement strategy")
        for column in ("slide_id", "encoder_i", "encoder_j", "checkpoint", "baseline_file"):
            if frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{path}: empty {column}")
        if frame.encoder_i.eq(frame.encoder_j).any():
            raise ValueError(f"{path}: self-pairs are invalid")
        swap = frame.encoder_i > frame.encoder_j
        left = frame.loc[swap, "encoder_i"].copy()
        frame.loc[swap, "encoder_i"] = frame.loc[swap, "encoder_j"].to_numpy()
        frame.loc[swap, "encoder_j"] = left.to_numpy()
        for c in range(n_classes):
            a, b = f"logit_10_class_{c}", f"logit_01_class_{c}"
            old = frame.loc[swap, a].copy()
            frame.loc[swap, a] = frame.loc[swap, b].to_numpy()
            frame.loc[swap, b] = old.to_numpy()
        frame["source_file"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("No pairwise_state_logits.csv files found")
    combined = pd.concat(frames, ignore_index=True)
    if split != "all":
        combined = combined.loc[combined.split.eq(split)].copy()
    if pairs:
        requested = {tuple(sorted(pair)) for pair in pairs}
        available = set(zip(combined.encoder_i, combined.encoder_j))
        if requested - available:
            raise ValueError(f"Requested pairs not found for split={split}: {sorted(requested-available)}")
        combined = combined.loc[[pair in requested for pair in zip(combined.encoder_i, combined.encoder_j)]].copy()
    if combined.empty:
        raise ValueError(f"No observations remain for split={split}")
    for column in ("n_classes", "target_class", "reference_forward",
                   "replacement_strategy", "gaussian_std_scale"):
        if combined[column].nunique() != 1:
            raise ValueError(f"Mixed {column} settings; analyze separate experiments separately")
    for column in ("checkpoint", "baseline_file"):
        if combined.groupby("fold")[column].nunique().gt(1).any():
            raise ValueError(f"Multiple {column} values within a fold; choose one run")
    # A WSI must not be counted repeatedly by recursively collecting old runs.
    keys = ["split", "slide_id", "encoder_i", "encoder_j"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate WSI/pair within a split (possibly repeated runs/folds). "
                         "Use explicit --input-files for one held-out evaluation.")
    if combined.groupby("slide_id").label.nunique().gt(1).any():
        raise ValueError("Conflicting labels for the same slide_id")
    return combined.sort_values(["split", "fold", "slide_id", "encoder_i", "encoder_j"]).reset_index(drop=True)


def logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))


def margin(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    others = logits.copy()
    rows = np.arange(len(logits))
    others[rows, targets] = -np.inf
    return logits[rows, targets] - logsumexp(others)


def effect_sign(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else (-1 if value < -epsilon else 0)


def transition(absent: float, present: float, epsilon: float) -> str:
    """Describe signs and change, without equating interaction with benefit."""
    before, after = effect_sign(absent, epsilon), effect_sign(present, epsilon)
    names = {-1: "negative", 0: "near_zero", 1: "positive"}
    if before != after:
        return f"{names[before]}_to_{names[after]}"
    change = effect_sign(present - absent, epsilon)
    if before == 0 or change == 0:
        return f"{names[before]}_unchanged"
    if before > 0:
        return "positive_enhanced" if change > 0 else "positive_diminished"
    return "negative_attenuated" if change > 0 else "negative_amplified"


def analyze_states(frame: pd.DataFrame, target: str = "true", epsilon: float = 1e-6) -> pd.DataFrame:
    if target not in {"true", "exported"}:
        raise ValueError("Unknown margin target")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    data = frame.copy()
    classes = int(data.n_classes.iloc[0])
    truth = data.label.to_numpy(dtype=int)
    exported = data.target_class.to_numpy(dtype=int)
    targets = truth if target == "true" else exported
    data["analysis_target_class"] = targets
    data["margin_target_mode"] = target
    original_margins = {}
    logit_scale = np.zeros(len(data))
    for state in STATES:
        logits = data[[f"logit_{state}_class_{c}" for c in range(classes)]].to_numpy(dtype=float)
        original_margins[state] = margin(logits, exported)
        data[f"margin_{state}"] = margin(logits, targets)
        ce = logsumexp(logits) - logits[np.arange(len(data)), truth]
        data[f"ce_true_{state}"] = ce
        data[f"p_true_{state}"] = np.exp(-ce)
        data[f"pred_class_{state}"] = logits.argmax(axis=1)
        logit_scale += np.max(np.abs(logits), axis=1)
    reconstructed = (original_margins["11"] - original_margins["10"]
                     - original_margins["01"] + original_margins["00"])
    error = np.abs(reconstructed - data.interaction.to_numpy())
    # Original GPU operations may be float32; exports are recomputed in float64.
    tolerance = 1e-5 + 4 * np.finfo(np.float32).eps * logit_scale
    if np.any(error > tolerance):
        bad = data.loc[error > tolerance, ["slide_id", "encoder_i", "encoder_j"]].head().to_dict("records")
        raise ValueError(f"Saved interaction disagrees with four-state logits: {bad}")
    data["interaction_export_recomputed"] = reconstructed
    data["interaction_export_abs_error"] = error
    m = {state: data[f"margin_{state}"].to_numpy() for state in STATES}
    data["delta_i_partner_replaced"] = m["10"] - m["00"]
    data["delta_i_partner_retained"] = m["11"] - m["01"]
    data["delta_j_partner_replaced"] = m["01"] - m["00"]
    data["delta_j_partner_retained"] = m["11"] - m["10"]
    data["interaction_analysis"] = m["11"] - m["10"] - m["01"] + m["00"]
    data["abs_interaction_analysis"] = data.interaction_analysis.abs()
    data["joint_gain_vs_best_single"] = m["11"] - np.maximum(m["10"], m["01"])
    data["pair_gain_vs_both_replaced"] = m["11"] - m["00"]
    data["joint_gain_status"] = ["positive" if value > epsilon else "negative" if value < -epsilon else "near_zero"
                                 for value in data.joint_gain_vs_best_single]
    for side in ("i", "j"):
        before = data[f"delta_{side}_partner_replaced"]
        after = data[f"delta_{side}_partner_retained"]
        data[f"transition_{side}"] = [transition(a, b, epsilon) for a, b in zip(before, after)]
        data[f"sign_reversal_{side}"] = ((before > epsilon) & (after < -epsilon)) | ((before < -epsilon) & (after > epsilon))
    return data


def summarize(data: pd.DataFrame, keys: list[str], epsilon: float) -> pd.DataFrame:
    rows = []
    metrics = ["interaction_analysis", "abs_interaction_analysis",
               "joint_gain_vs_best_single", "pair_gain_vs_both_replaced",
               "delta_i_partner_replaced", "delta_i_partner_retained",
               "delta_j_partner_replaced", "delta_j_partner_retained"]
    for values, group in data.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(n_rows=len(group), n_slides=group.slide_id.nunique(), n_folds=group.fold.nunique())
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
        for metric in ("interaction_analysis", "joint_gain_vs_best_single"):
            for sign, mask in (("positive", group[metric] > epsilon),
                               ("negative", group[metric] < -epsilon),
                               ("near_zero", group[metric].abs() <= epsilon)):
                row[f"{metric}_{sign}_fraction"] = float(mask.mean())
        for side in ("i", "j"):
            row[f"sign_reversal_{side}_fraction"] = float(group[f"sign_reversal_{side}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def transition_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (left, right, split), group in data.groupby(["encoder_i", "encoder_j", "split"]):
        for side, encoder in (("i", left), ("j", right)):
            for name, count in group[f"transition_{side}"].value_counts().sort_index().items():
                rows.append(dict(encoder_i=left, encoder_j=right, split=split,
                                 focal_encoder=encoder, transition=name, count=int(count),
                                 n_rows=len(group), fraction=float(count / len(group))))
    return pd.DataFrame(rows)


def plot_pair(group: pd.DataFrame, directory: Path, formats: list[str], dpi: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    left, right, split = group.iloc[0][["encoder_i", "encoder_j", "split"]]
    short = lambda name: str(name).removeprefix("features_")
    pair = f"{short(left)} + {short(right)}"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555"]
    classes = sorted(group.label.unique())
    palette = {label: colors[index % len(colors)] for index, label in enumerate(classes)}
    limit = max(float(group[[f"delta_{s}_partner_{p}" for s in ("i", "j")
                            for p in ("replaced", "retained")]].abs().to_numpy().max()) * 1.12, 1e-3)
    for ax, side, encoder, letter in zip(axes[0], ("i", "j"), (left, right), ("a", "b")):
        for label in classes:
            part = group.loc[group.label.eq(label)]
            ax.scatter(part[f"delta_{side}_partner_replaced"], part[f"delta_{side}_partner_retained"],
                       s=15, alpha=0.65, color=palette[label], linewidths=0)
        ax.plot([-limit, limit], [-limit, limit], color="#777777", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="#BBBBBB", linewidth=0.6)
        ax.axvline(0, color="#BBBBBB", linewidth=0.6)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit),
               xlabel="Partner replaced: margin change", ylabel="Partner retained: margin change")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{letter}  " + textwrap.fill(short(encoder), 28), loc="left", fontweight="bold")
    bins = min(25, max(1, int(np.sqrt(len(group)))))
    for ax, column, letter, label in (
        (axes[1, 0], "interaction_analysis", "c", "Signed interaction"),
        (axes[1, 1], "joint_gain_vs_best_single", "d", "Joint retention gain G"),
    ):
        ax.hist(group[column], bins=bins, color="#708B9B", edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set(xlabel=f"{label} (margin units)", ylabel="WSI count")
        ax.set_title(f"{letter}  {label}", loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=palette[c],
                      markersize=4, label=f"True class {c} (n={int(group.label.eq(c).sum())})") for c in classes]
    axes[0, 0].legend(handles=handles, fontsize=6, loc="best", frameon=False)
    mode = str(group.margin_target_mode.iloc[0])
    target = "True-class margin" if mode == "true" else f"Class-{int(group.target_class.iloc[0])} margin"
    fig.suptitle(textwrap.fill(pair, 68) + f"\n{target} | {split} | n={len(group)} WSI | frozen mean fusion", fontsize=9)
    fig.supxlabel("0 = baseline replacement; other encoders unchanged. G is not an AUC gain.", fontsize=7)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{left}__{right}__{split}")[:100]
    digest = hashlib.sha256(f"{left}|{right}|{split}".encode()).hexdigest()[:8]
    outputs = []
    for extension in dict.fromkeys(formats):
        path = directory / f"{stem}_{digest}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    paths = args.input_files if args.input_files else sorted(args.input_root.rglob("pairwise_state_logits.csv"))
    frame = load_states(paths, split=args.split, pairs=args.pair)
    data = analyze_states(frame, target=args.margin_target, epsilon=args.effect_epsilon)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory to avoid mixing analyses")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"pairwise_state_diagnostics.csv": data,
              "pair_summary.csv": summarize(data, ["encoder_i", "encoder_j", "split"], args.effect_epsilon),
              "pair_summary_by_label.csv": summarize(data, ["encoder_i", "encoder_j", "split", "label"], args.effect_epsilon),
              "pair_summary_by_fold.csv": summarize(data, ["encoder_i", "encoder_j", "split", "fold"], args.effect_epsilon),
              "marginal_transition_summary.csv": transition_summary(data)}
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    plots = []
    if not args.no_plots:
        directory = args.output_dir / "figures"
        directory.mkdir()
        for _, group in data.groupby(["encoder_i", "encoder_j", "split"], sort=True):
            plots.extend(plot_pair(group, directory, args.formats, args.dpi))
    metadata = {"input_files": [str(path.resolve()) for path in paths],
                "margin_target": args.margin_target, "effect_epsilon": args.effect_epsilon,
                "n_observations": len(data), "n_slides": int(data.slide_id.nunique()),
                "interaction_check_max_abs_error": float(data.interaction_export_abs_error.max()),
                "reference_forward": "frozen mean_fusion",
                "states": {"00": "both replaced", "10": "i retained, j replaced",
                           "01": "i replaced, j retained", "11": "both retained"},
                "other_encoders": "retained and unchanged in all four states",
                "interaction_formula": "m11-m10-m01+m00",
                "G_formula": "m11-max(m10,m01)",
                "statistics": "Descriptive WSI-level summaries; no significance tests or independence assumptions. Pooled summaries weight WSIs equally.",
                "warnings": ["True labels are used for offline diagnostics only, never online selection.",
                             "Magnitude/sign are reference-model and baseline dependent; not intrinsic FM utility.",
                             "Baseline replacement is not actual encoder removal and retraining.",
                             "G is a margin difference, not a gain in AUC/AUPRC.",
                             "Gaussian exports reuse exact cached draws, which can differ between coalitions.",
                             "Raw margins across folds may have different scales; inspect per-fold tables.",
                             "Effect epsilon is numerical/descriptive, not statistical significance."],
                "figure_contract": "Quantitative diagnostic grid: two directional marginal-effect scatters, signed-interaction distribution, and G distribution. Each dot is one WSI; y=x is additivity.",
                "plots": plots}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyzed {len(data)} WSI/pair rows from {data.slide_id.nunique()} WSIs.")
    print(f"Saved {len(tables)} CSVs and {len(plots)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
