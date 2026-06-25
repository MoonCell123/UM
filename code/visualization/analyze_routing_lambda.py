"""
Analyze DualConsistencyRouter lambda values from model checkpoints.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUTING_HINTS = ("theta", "lambda", "router", "routing", "gamma")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "Routing_Lambda_Analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize lambda_similarity/lambda_attribution from routing checkpoints."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-pattern", default="*.pt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold-regex", default=r"fold_(\d+)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search checkpoint-dir.")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def tensor_to_float(value: object) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() != 1:
        raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def load_checkpoint(path: Path) -> object:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to load checkpoint files. Install torch in this environment."
        ) from exc
    return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: object) -> Mapping[str, object]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict or state_dict, got {type(checkpoint)}")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(hasattr(value, "shape") or hasattr(value, "item") for value in checkpoint.values()):
        return checkpoint

    raise KeyError("Could not find state_dict/model_state_dict or raw state_dict in checkpoint.")


def normalize_key(key: str) -> str:
    prefixes = ("module.", "model.", "net.")
    changed = True
    normalized = key
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
    return normalized


def candidate_theta_keys(state_dict: Mapping[str, object]) -> List[str]:
    exact_suffixes = (
        "theta",
        "router.theta",
        "routing.theta",
        "dual_router.theta",
        "dual_consistency_router.theta",
    )
    keys = []
    for key in state_dict:
        normalized = normalize_key(key)
        if normalized in exact_suffixes or normalized.endswith(".router.theta") or normalized.endswith(".routing.theta"):
            keys.append(key)

    if keys:
        return keys

    fallback = []
    for key in state_dict:
        normalized = normalize_key(key)
        if normalized.endswith(".theta") or normalized == "theta":
            fallback.append(key)
    return fallback


def find_gamma_key(state_dict: Mapping[str, object], theta_key: str) -> str | None:
    candidates = []
    normalized_theta = normalize_key(theta_key)
    if normalized_theta.endswith("theta"):
        base = theta_key[: -len("theta")]
        candidates.extend([f"{base}gamma_logit", f"{base}gamma"])

    candidates.extend([
        "gamma_logit",
        "gamma",
        "router.gamma_logit",
        "router.gamma",
        "routing.gamma_logit",
        "routing.gamma",
        "module.router.gamma_logit",
        "model.router.gamma_logit",
    ])

    for key in candidates:
        if key in state_dict:
            return key
    normalized_lookup = {normalize_key(key): key for key in state_dict}
    for key in candidates:
        normalized = normalize_key(key)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    return None


def suspicious_keys(state_dict: Mapping[str, object], limit: int = 60) -> List[str]:
    keys = [
        key for key in state_dict
        if any(hint in key.lower() for hint in ROUTING_HINTS)
    ]
    return keys[:limit]


def parse_fold(path: Path, fold_regex: str) -> int | None:
    pattern = re.compile(fold_regex)
    for text in (path.name, str(path.parent), str(path)):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def discover_checkpoints(checkpoint_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    files = checkpoint_dir.rglob(pattern) if recursive else checkpoint_dir.glob(pattern)
    checkpoints = sorted(path for path in files if path.is_file())
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found: {checkpoint_dir} / {pattern}")
    return checkpoints


def analyze_checkpoint(path: Path, fold_regex: str) -> Dict[str, object]:
    checkpoint = load_checkpoint(path)
    state_dict = extract_state_dict(checkpoint)
    theta_keys = candidate_theta_keys(state_dict)
    if not theta_keys:
        hints = suspicious_keys(state_dict)
        raise KeyError(
            f"No router theta found in {path}. Suspicious keys: {hints}"
        )

    theta_key = theta_keys[0]
    theta = tensor_to_float(state_dict[theta_key])
    lambda_similarity = sigmoid(theta)
    lambda_attribution = 1.0 - lambda_similarity
    if not np.isclose(lambda_similarity + lambda_attribution, 1.0, atol=1e-6):
        raise AssertionError("lambda_similarity + lambda_attribution must equal 1.")

    gamma_key = find_gamma_key(state_dict, theta_key)
    gamma = np.nan
    gamma_logit = np.nan
    if gamma_key is not None:
        raw_gamma = tensor_to_float(state_dict[gamma_key])
        if gamma_key.endswith("gamma_logit"):
            gamma_logit = raw_gamma
            gamma = sigmoid(raw_gamma)
        else:
            gamma = raw_gamma

    return {
        "checkpoint_path": str(path),
        "checkpoint_name": path.name,
        "fold": parse_fold(path, fold_regex),
        "theta": theta,
        "lambda_similarity": lambda_similarity,
        "lambda_attribution": lambda_attribution,
        "gamma": gamma,
        "gamma_logit": gamma_logit,
        "source_key": theta_key,
        "gamma_source_key": gamma_key or "",
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["theta", "lambda_similarity", "lambda_attribution", "gamma"]
    summary_rows = []
    for metric in metric_cols:
        values = pd.to_numeric(rows[metric], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append({
            "metric": metric,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        })
    return pd.DataFrame(summary_rows)


def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_bar(rows: pd.DataFrame, output_path: Path) -> None:
    df = rows.copy().sort_values(["fold", "checkpoint_name"], na_position="last").reset_index(drop=True)
    labels = [
        f"fold {int(row.fold)}" if pd.notna(row.fold) else row.checkpoint_name
        for row in df.itertuples()
    ]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(df)), 4.6))
    ax.bar(x, df["lambda_attribution"], label="lambda_attribution", color="#4C78A8")
    ax.bar(
        x,
        df["lambda_similarity"],
        bottom=df["lambda_attribution"],
        label="lambda_similarity",
        color="#F58518",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Routing weight")
    ax.set_title("Routing lambda composition by checkpoint")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pie(rows: pd.DataFrame, output_path: Path) -> None:
    means = rows[["lambda_similarity", "lambda_attribution"]].mean()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.pie(
        [means["lambda_similarity"], means["lambda_attribution"]],
        labels=["lambda_similarity", "lambda_attribution"],
        autopct="%.1f%%",
        startangle=90,
        colors=["#F58518", "#4C78A8"],
        textprops={"fontsize": 10},
    )
    ax.set_title("Mean routing lambda composition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fold_curve(rows: pd.DataFrame, output_path: Path) -> None:
    df = rows.dropna(subset=["fold"]).copy()
    if df.empty:
        df = rows.copy()
        df["fold"] = np.arange(1, len(df) + 1)
    df["fold"] = df["fold"].astype(int)
    df = df.sort_values("fold")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["fold"], df["lambda_similarity"], marker="o", label="lambda_similarity", color="#F58518")
    ax.plot(df["fold"], df["lambda_attribution"], marker="o", label="lambda_attribution", color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold" if rows["fold"].notna().any() else "Checkpoint index")
    ax.set_ylabel("Routing weight")
    ax.set_title("Routing lambda across folds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(args.checkpoint_dir, args.checkpoint_pattern, args.recursive)
    rows = []
    errors: List[Tuple[Path, Exception]] = []
    for checkpoint in checkpoints:
        try:
            rows.append(analyze_checkpoint(checkpoint, args.fold_regex))
        except Exception as exc:
            errors.append((checkpoint, exc))

    if not rows:
        message_parts = ["No checkpoint could be analyzed."]
        for checkpoint, exc in errors:
            message_parts.append(f"{checkpoint}: {exc}")
        raise RuntimeError("\n".join(message_parts))

    result_df = pd.DataFrame(rows)
    summary_df = summarize(result_df)

    by_checkpoint_path = args.output_dir / "routing_lambda_by_checkpoint.csv"
    summary_path = args.output_dir / "routing_lambda_summary.csv"
    result_df.to_csv(by_checkpoint_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    plot_bar(result_df, args.output_dir / "routing_lambda_barplot.png")
    plot_pie(result_df, args.output_dir / "routing_lambda_pie_mean.png")
    plot_fold_curve(result_df, args.output_dir / "routing_lambda_fold_curve.png")

    if errors:
        error_path = args.output_dir / "routing_lambda_errors.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            for checkpoint, exc in errors:
                f.write(f"{checkpoint}: {exc}\n")
        print(f"[Warning] Some checkpoints failed. See: {error_path}")

    print(f"Analyzed checkpoints: {len(result_df)}")
    print(f"Saved: {by_checkpoint_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
