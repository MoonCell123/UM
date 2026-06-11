"""
Run top-k late-fusion stacking with rank transform and Elastic-Net Logistic Regression.

The base models are ranked by their single-model OOF AUC on the full OOF matrix.
For k=1..N, the top-k model columns are used as meta-features, then evaluated
with repeated stratified CV.
"""

import argparse
import json
import os
import warnings
from typing import Dict, List

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output", ".matplotlib_cache")),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_stacking_meta import (
    DEFAULT_LABELS_PATH,
    DEFAULT_MATRIX_PATH,
    LABEL_COL,
    PROJECT_ROOT,
    SLIDE_ID_COL,
    cross_validated_predict,
    get_effective_n_splits,
    load_oof_data,
    make_pipeline,
)


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "TopK_Rank_ElasticNet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate top-k base models with rank + Elastic-Net stacking."
    )
    parser.add_argument("--matrix", default=DEFAULT_MATRIX_PATH, help="Path to OOF_matrix.csv.")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Path to OOF_labels.csv.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--cv-repeats", type=int, default=20, help="Number of CV repeats.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--c", type=float, default=0.3, help="Elastic-Net inverse regularization strength.")
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.9, help="Elastic-Net L1 ratio.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max solver iterations.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    args = parser.parse_args()
    if args.cv_repeats < 1:
        raise ValueError("--cv-repeats must be at least 1.")
    if not 0.0 <= args.elasticnet_l1_ratio <= 1.0:
        raise ValueError("--elasticnet-l1-ratio must be between 0 and 1.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    return args


def rank_base_models(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for feature in x.columns:
        prob = x[feature].astype(float).values
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "model": feature,
            "auc": float(roc_auc_score(y, prob)),
            "auprc": float(average_precision_score(y, prob)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
        })
    ranking = pd.DataFrame(rows).sort_values(
        ["auc", "auprc", "balanced_accuracy"],
        ascending=[False, False, False],
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def build_rank_elasticnet(args: argparse.Namespace):
    estimator = LogisticRegression(
        penalty="elasticnet",
        C=args.c,
        solver="saga",
        l1_ratio=args.elasticnet_l1_ratio,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    return make_pipeline(estimator, transform="rank")


def plot_topk(summary_df: pd.DataFrame, ranking_df: pd.DataFrame, output_path: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("auc", "AUC"),
        ("auprc", "AUPRC"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1", "F1"),
    ]
    x = summary_df["k"].values
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        mean = summary_df[metric].values
        best_idx = int(np.argmax(mean))
        ax.plot(x, mean, marker="o", linewidth=1.8, markersize=4, color="#1f77b4")
        ax.scatter(
            [x[best_idx]],
            [mean[best_idx]],
            color="#d62728",
            s=45,
            zorder=4,
            label=f"best k={int(x[best_idx])}",
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Top-k selected base models")
    fig.suptitle("Effect of top-k model selection on Elastic-Net stacking performance", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.makedirs(args.output_dir, exist_ok=True)

    x, y, slide_ids, feature_cols = load_oof_data(args.matrix, args.labels)
    n_splits = get_effective_n_splits(y, args.cv_folds)
    ranking_df = rank_base_models(x, y)
    ranked_features = ranking_df["model"].tolist()

    print("=" * 80)
    print("Top-k rank + Elastic-Net stacking")
    print("=" * 80)
    print(f"Samples: {len(y)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print(f"Meta CV folds x repeats: {n_splits} x {args.cv_repeats}")
    print(f"Elastic-Net: C={args.c}, l1_ratio={args.elasticnet_l1_ratio}")
    print(f"Output dir: {args.output_dir}")
    print()

    ranking_path = os.path.join(args.output_dir, "base_model_auc_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print("Base model ranking by AUC:")
    print(ranking_df[["rank", "model", "auc", "auprc", "balanced_accuracy", "f1"]].to_string(index=False))
    print()

    summaries = []
    repeat_metric_parts = []
    prediction_parts = []
    for k in range(1, len(ranked_features) + 1):
        selected_features = ranked_features[:k]
        estimator = build_rank_elasticnet(args)
        predictions, metrics, repeat_metrics_df = cross_validated_predict(
            estimator=estimator,
            x=x[selected_features],
            y=y,
            slide_ids=slide_ids,
            n_splits=n_splits,
            n_repeats=args.cv_repeats,
            seed=args.seed,
            threshold=args.threshold,
        )
        metrics["k"] = k
        metrics["selected_models"] = ";".join(selected_features)
        metrics["meta_learner"] = "elastic_net"
        metrics["transform"] = "rank"
        metrics["n_samples"] = int(len(y))
        metrics["n_features"] = int(k)
        metrics["cv_folds"] = int(n_splits)
        metrics["cv_repeats"] = int(args.cv_repeats)
        summaries.append(metrics)

        repeat_metrics_df.insert(0, "k", k)
        repeat_metrics_df["selected_models"] = ";".join(selected_features)
        repeat_metric_parts.append(repeat_metrics_df)

        predictions.insert(0, "k", k)
        prediction_parts.append(predictions)

        print(
            f"k={k:02d}: "
            f"AUC={metrics['auc']:.4f}±{metrics['auc_std']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}, "
            f"BalAcc={metrics['balanced_accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("k")
    repeat_metrics_all = pd.concat(repeat_metric_parts, ignore_index=True)
    predictions_all = pd.concat(prediction_parts, ignore_index=True)

    summary_path = os.path.join(args.output_dir, "topk_rank_elasticnet_summary.csv")
    repeat_path = os.path.join(args.output_dir, "topk_rank_elasticnet_repeat_metrics.csv")
    predictions_path = os.path.join(args.output_dir, "topk_rank_elasticnet_predictions.csv")
    plot_path = os.path.join(args.output_dir, "topk_rank_elasticnet_metrics.png")
    config_path = os.path.join(args.output_dir, "topk_rank_elasticnet_config.json")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    repeat_metrics_all.to_csv(repeat_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    predictions_all.to_csv(predictions_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    plot_topk(summary_df, ranking_df, plot_path)

    config = {
        "matrix": os.path.abspath(args.matrix),
        "labels": os.path.abspath(args.labels) if args.labels else None,
        "ranking_rule": "single-model OOF AUC descending on full OOF_matrix",
        "ranked_features": ranked_features,
        "cv_folds": n_splits,
        "cv_repeats": args.cv_repeats,
        "seed": args.seed,
        "c": args.c,
        "elasticnet_l1_ratio": args.elasticnet_l1_ratio,
        "threshold": args.threshold,
        "transform": "rank",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_auc = summary_df.loc[summary_df["auc"].idxmax()]
    best_f1 = summary_df.loc[summary_df["f1"].idxmax()]
    print()
    print("Best by repeated-CV mean:")
    print(
        f"  AUC: k={int(best_auc['k'])}, "
        f"AUC={best_auc['auc']:.4f}, "
        f"AUPRC={best_auc['auprc']:.4f}, "
        f"BalAcc={best_auc['balanced_accuracy']:.4f}, "
        f"F1={best_auc['f1']:.4f}"
    )
    print(
        f"  F1:  k={int(best_f1['k'])}, "
        f"AUC={best_f1['auc']:.4f}, "
        f"AUPRC={best_f1['auprc']:.4f}, "
        f"BalAcc={best_f1['balanced_accuracy']:.4f}, "
        f"F1={best_f1['f1']:.4f}"
    )
    print()
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved repeat metrics: {repeat_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
