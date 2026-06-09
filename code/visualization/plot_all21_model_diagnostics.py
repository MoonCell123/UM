"""
Generate per-model diagnostic panels for the 21-model UVM benchmark.

Each panel contains:
1. Pooled out-of-fold confusion matrix
2. Pooled out-of-fold ROC curve
3. OS Kaplan-Meier prognosis curve grouped by predicted class
4. PFI Kaplan-Meier prognosis curve grouped by predicted class
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


warnings.filterwarnings("ignore")
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DISPLAY_NAMES = {
    "hoptimus1": "H-optimus-1",
    "hoptimus0": "H-optimus-0",
    "uni_v1": "UNI v1",
    "uni_v2": "UNI v2",
    "conch_v1": "CONCH v1",
    "conch_v15": "CONCH v1.5",
    "virchow": "Virchow",
    "virchow2": "Virchow2",
    "gigapath": "GigaPath",
    "musk": "MUSK",
    "hibou_l": "Hibou-L",
    "phikon": "Phikon",
    "phikon_v2": "Phikon-v2",
    "kaiko-vits8": "Kaiko ViT-S/8",
    "kaiko-vits16": "Kaiko ViT-S/16",
    "kaiko-vitb8": "Kaiko ViT-B/8",
    "kaiko-vitb16": "Kaiko ViT-B/16",
    "kaiko-vitl14": "Kaiko ViT-L/14",
    "lunit-vits8": "LUNIT ViT-S/8",
    "midnight12k": "Midnight-12k",
    "resnet50": "ResNet-50",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_result_dir = script_dir / "benchmark_output" / "20260331_171049"
    parser = argparse.ArgumentParser(
        description="Create confusion matrix, ROC, and OS prognosis panels for all benchmark models."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=default_result_dir,
        help="Benchmark result directory.",
    )
    parser.add_argument(
        "--out-subdir",
        default="all21_model_diagnostics",
        help="Subdirectory name to create under result_dir/figures.",
    )
    parser.add_argument(
        "--clinical-path",
        type=Path,
        default=None,
        help="Override clinical file path. Defaults to config.json clinical_path.",
    )
    return parser.parse_args()


def load_config(result_dir: Path) -> dict:
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_order(result_dir: Path) -> list[tuple[int, str]]:
    summary_path = result_dir / "summary.csv"
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        sdf = sdf.sort_values("auc_macro_mean", ascending=False).reset_index(drop=True)
        return [(idx + 1, str(model)) for idx, model in enumerate(sdf["model"].tolist())]

    models = sorted(p.name for p in result_dir.iterdir() if p.is_dir() and p.name != "figures")
    return [(idx + 1, model) for idx, model in enumerate(models)]


def load_clinical_df(clinical_path: Path) -> pd.DataFrame:
    if clinical_path.suffix.lower() == ".csv":
        clin = pd.read_csv(clinical_path, encoding="utf-8-sig")
    else:
        clin = pd.read_excel(clinical_path)

    if "slide_id" not in clin.columns:
        raise ValueError(f"'slide_id' not found in clinical table: {clinical_path}")

    required_cols = {"OS", "OStime", "PFI", "PFItime"}
    if not required_cols.issubset(clin.columns):
        raise ValueError(
            f"Expected columns {sorted(required_cols)} in clinical table: {clinical_path}"
        )

    clin = clin.copy()
    clin["slide_id"] = clin["slide_id"].astype(str)
    clin["OS"] = pd.to_numeric(clin["OS"], errors="coerce")
    clin["OStime"] = pd.to_numeric(clin["OStime"], errors="coerce")
    clin["PFI"] = pd.to_numeric(clin["PFI"], errors="coerce")
    clin["PFItime"] = pd.to_numeric(clin["PFItime"], errors="coerce")
    clin = clin[["slide_id", "OS", "OStime", "PFI", "PFItime"]].dropna(subset=["slide_id"])
    return clin


def collect_predictions(model_dir: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(model_dir.glob("fold_*")):
        pred_path = fold_dir / "predictions.csv"
        if not pred_path.exists():
            continue
        fold_df = pd.read_csv(pred_path)
        required = {"slide_id", "label", "pred", "prob_class1"}
        if not required.issubset(fold_df.columns):
            continue
        fold_df = fold_df[["slide_id", "label", "pred", "prob_class1"]].copy()
        fold_df["slide_id"] = fold_df["slide_id"].astype(str)
        fold_df["fold"] = fold_dir.name
        frames.append(fold_df)

    if not frames:
        raise FileNotFoundError(f"No prediction files found under {model_dir}")

    preds = pd.concat(frames, ignore_index=True)

    if preds["slide_id"].duplicated().any():
        preds = (
            preds.groupby("slide_id", as_index=False)
            .agg(
                label=("label", lambda s: int(pd.Series.mode(s).iloc[0])),
                prob_class1=("prob_class1", "mean"),
            )
            .assign(pred=lambda df: (df["prob_class1"] >= 0.5).astype(int))
        )
    else:
        preds["label"] = preds["label"].astype(int)
        preds["pred"] = preds["pred"].astype(int)
        preds["prob_class1"] = preds["prob_class1"].astype(float)
        preds = preds[["slide_id", "label", "pred", "prob_class1"]].copy()

    return preds.sort_values("slide_id").reset_index(drop=True)


def format_p_value(p_value: float) -> str:
    if p_value is None or math.isnan(p_value):
        return "N/A"
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def compute_endpoint_stats(prognosis_df: pd.DataFrame, event_col: str, time_col: str) -> dict:
    stats = {
        "n": int(len(prognosis_df)),
        "events": int(prognosis_df[event_col].sum()) if not prognosis_df.empty else 0,
        "pred_d3_n": int((prognosis_df["pred"] == 0).sum()),
        "pred_m3_n": int((prognosis_df["pred"] == 1).sum()),
        "pred_d3_events": int(prognosis_df.loc[prognosis_df["pred"] == 0, event_col].sum()),
        "pred_m3_events": int(prognosis_df.loc[prognosis_df["pred"] == 1, event_col].sum()),
        "logrank_p": float("nan"),
        "hr_pred_m3_vs_d3": float("nan"),
        "hr_ci_lower": float("nan"),
        "hr_ci_upper": float("nan"),
    }

    low = prognosis_df[prognosis_df["pred"] == 0]
    high = prognosis_df[prognosis_df["pred"] == 1]
    if low.empty or high.empty:
        return stats

    try:
        lr = logrank_test(
            high[time_col],
            low[time_col],
            event_observed_A=high[event_col],
            event_observed_B=low[event_col],
        )
        stats["logrank_p"] = float(lr.p_value)
    except Exception:
        pass

    try:
        cox_df = prognosis_df[[time_col, event_col, "pred"]].rename(
            columns={time_col: "T", event_col: "E", "pred": "group"}
        )
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="T", event_col="E", formula="group")
        stats["hr_pred_m3_vs_d3"] = float(cph.hazard_ratios_["group"])
        ci = cph.confidence_intervals_.loc["group"]
        stats["hr_ci_lower"] = float(np.exp(ci.iloc[0]))
        stats["hr_ci_upper"] = float(np.exp(ci.iloc[1]))
    except Exception:
        pass

    return stats


def plot_confusion(ax: plt.Axes, cm: np.ndarray, accuracy: float) -> None:
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred D3", "Pred M3"])
    ax.set_yticks([0, 1], labels=["True D3", "True M3"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix\nAccuracy = {accuracy:.4f}", fontsize=11, fontweight="bold")

    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({row_pct[i, j] * 100:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )


def plot_roc(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_value: float,
    sens: float,
    spec: float,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bcbcbc", linewidth=1.2, label="Random")
    ax.plot(fpr, tpr, color="#d1495b", linewidth=2.5, label=f"OOF ROC (AUC={auc_value:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#d1495b")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.97,
        0.05,
        f"Sensitivity = {sens:.4f}\nSpecificity = {spec:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def plot_prognosis(
    ax: plt.Axes,
    prognosis_df: pd.DataFrame,
    stats: dict,
    event_col: str,
    time_col: str,
    endpoint_label: str,
    time_unit: str,
    title_suffix: str = "",
) -> None:
    kmf = KaplanMeierFitter()
    colors = {0: "#2f6690", 1: "#c1121f"}
    labels = {0: "Pred D3", 1: "Pred M3"}

    plotted = False
    for group in [0, 1]:
        sub = prognosis_df[prognosis_df["pred"] == group]
        if sub.empty:
            continue
        plotted = True
        label = f"{labels[group]} (n={len(sub)}, events={int(sub[event_col].sum())})"
        kmf.fit(sub[time_col], sub[event_col], label=label)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            show_censors=True,
            color=colors[group],
            linewidth=2.2,
        )

    ax.set_xlabel(f"{endpoint_label} time ({time_unit})")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_title(
        f"{endpoint_label} Prognosis by Predicted Class{title_suffix}",
        fontsize=11,
        fontweight="bold",
    )

    if plotted:
        ax.legend(loc="lower left", fontsize=8.8)
        if math.isnan(stats["hr_pred_m3_vs_d3"]):
            stat_text = f"log-rank p = {format_p_value(stats['logrank_p'])}\nHR(M3 vs D3) = N/A"
        else:
            stat_text = (
                f"log-rank p = {format_p_value(stats['logrank_p'])}\n"
                f"HR(M3 vs D3) = {stats['hr_pred_m3_vs_d3']:.2f}"
            )
        ax.text(
            0.97,
            0.95,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for KM plot",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )


def make_panel(
    model: str,
    rank: int,
    preds: pd.DataFrame,
    clinical_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    display_name = DISPLAY_NAMES.get(model, model)
    y_true = preds["label"].to_numpy(dtype=int)
    y_pred = preds["pred"].to_numpy(dtype=int)
    y_score = preds["prob_class1"].to_numpy(dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc_value = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    prognosis_df = preds.merge(clinical_df, on="slide_id", how="left")
    os_df = prognosis_df.dropna(subset=["OS", "OStime"]).copy()
    os_df["OS"] = os_df["OS"].astype(int)
    os_df["OStime"] = os_df["OStime"].astype(float)
    os_stats = compute_endpoint_stats(os_df, "OS", "OStime")

    dfs_df = prognosis_df.dropna(subset=["PFI", "PFItime"]).copy()
    dfs_df["PFI"] = dfs_df["PFI"].astype(int)
    dfs_df["PFItime"] = dfs_df["PFItime"].astype(float)
    dfs_stats = compute_endpoint_stats(dfs_df, "PFI", "PFItime")

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.8))
    plot_confusion(axes[0], cm, accuracy)
    plot_roc(axes[1], y_true, y_score, auc_value, sensitivity, specificity)
    plot_prognosis(
        axes[2],
        os_df,
        os_stats,
        event_col="OS",
        time_col="OStime",
        endpoint_label="OS",
        time_unit="days",
    )
    plot_prognosis(
        axes[3],
        dfs_df,
        dfs_stats,
        event_col="PFI",
        time_col="PFItime",
        endpoint_label="PFI",
        time_unit="days",
        title_suffix="",
    )

    fig.suptitle(
        f"{rank:02d}. {display_name}  |  5-fold out-of-fold predictions (n={len(preds)})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    stem = f"{rank:02d}_{model}_diagnostics"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "rank": rank,
        "model": model,
        "display_name": display_name,
        "n_cases": len(preds),
        "accuracy": accuracy,
        "auc": auc_value,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "os_n": os_stats["n"],
        "os_events": os_stats["events"],
        "os_pred_d3_n": os_stats["pred_d3_n"],
        "os_pred_m3_n": os_stats["pred_m3_n"],
        "os_pred_d3_events": os_stats["pred_d3_events"],
        "os_pred_m3_events": os_stats["pred_m3_events"],
        "os_logrank_p": os_stats["logrank_p"],
        "os_hr_pred_m3_vs_d3": os_stats["hr_pred_m3_vs_d3"],
        "os_hr_ci_lower": os_stats["hr_ci_lower"],
        "os_hr_ci_upper": os_stats["hr_ci_upper"],
        "pfi_n": dfs_stats["n"],
        "pfi_events": dfs_stats["events"],
        "pfi_pred_d3_n": dfs_stats["pred_d3_n"],
        "pfi_pred_m3_n": dfs_stats["pred_m3_n"],
        "pfi_pred_d3_events": dfs_stats["pred_d3_events"],
        "pfi_pred_m3_events": dfs_stats["pred_m3_events"],
        "pfi_logrank_p": dfs_stats["logrank_p"],
        "pfi_hr_pred_m3_vs_d3": dfs_stats["hr_pred_m3_vs_d3"],
        "pfi_hr_ci_lower": dfs_stats["hr_ci_lower"],
        "pfi_hr_ci_upper": dfs_stats["hr_ci_upper"],
        "png_file": png_path.name,
        "pdf_file": pdf_path.name,
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    config = load_config(result_dir)

    clinical_path = args.clinical_path or Path(config["clinical_path"])
    clinical_df = load_clinical_df(clinical_path)

    out_dir = result_dir / "figures" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = load_summary_order(result_dir)
    rows = []

    for rank, model in ranking:
        model_dir = result_dir / model
        if not model_dir.is_dir():
            print(f"[skip] missing model directory: {model_dir}")
            continue

        print(f"[{rank:02d}/{len(ranking):02d}] {model}")
        preds = collect_predictions(model_dir)
        row = make_panel(
            model=model,
            rank=rank,
            preds=preds,
            clinical_df=clinical_df,
            out_dir=out_dir,
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nSaved diagnostic panels to:")
    print(out_dir)


if __name__ == "__main__":
    main()
