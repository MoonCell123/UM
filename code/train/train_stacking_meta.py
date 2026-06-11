"""
Train late-fusion stacking meta-learners on the OOF prediction matrix.

Default input:
    output/OOF_Result/OOF_matrix.csv

The script implements two binary meta-learners:
    1. Logistic Regression with L2 regularization.
    2. Elastic-Net Logistic Regression.

For evaluation, it runs repeated StratifiedKFold on the OOF rows and saves
meta-level OOF predictions. It also fits each meta-learner on the full OOF
matrix and saves the final model for later test-time stacking.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

import warnings


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_MATRIX_PATH = os.path.join(PROJECT_ROOT, "output", "OOF_Result", "OOF_matrix.csv")
DEFAULT_LABELS_PATH = os.path.join(PROJECT_ROOT, "output", "OOF_Result", "OOF_labels.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "Stacking_Result")

SLIDE_ID_COL = "slide_id"
LABEL_COL = "label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression and Elastic-Net meta-learners for OOF stacking."
    )
    parser.add_argument("--matrix", default=DEFAULT_MATRIX_PATH, help="Path to OOF_matrix.csv.")
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS_PATH,
        help="Optional path to OOF_labels.csv. Used for consistency checks when present.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument(
        "--meta-learner",
        choices=["both", "logistic", "elasticnet"],
        default="both",
        help="Which meta-learner to run.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of meta-level CV folds.")
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=20,
        help="Number of repeats for repeated meta-level CV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument(
        "--elasticnet-l1-ratio",
        type=float,
        default=0.5,
        help="Elastic-Net mixing parameter. 0=L2, 1=L1.",
    )
    parser.add_argument("--max-iter", type=int, default=10000, help="Max solver iterations.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    parser.add_argument(
        "--transform",
        choices=["standard", "rank", "none"],
        default="standard",
        help=(
            "Feature transform before the meta-learner. "
            "rank is often better for saturated probability columns."
        ),
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Backward-compatible alias for --transform none.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.elasticnet_l1_ratio <= 1.0:
        raise ValueError("--elasticnet-l1-ratio must be between 0 and 1.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.no_scale:
        args.transform = "none"
    if args.cv_repeats < 1:
        raise ValueError("--cv-repeats must be at least 1.")
    return args


def load_oof_data(
    matrix_path: str,
    labels_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"OOF matrix not found: {matrix_path}")

    matrix_df = pd.read_csv(matrix_path)
    required_cols = {SLIDE_ID_COL, LABEL_COL}
    missing = required_cols - set(matrix_df.columns)
    if missing:
        raise ValueError(f"{matrix_path} is missing required columns: {sorted(missing)}")

    slide_ids = matrix_df[SLIDE_ID_COL].astype(str)
    y = matrix_df[LABEL_COL].astype(int)

    feature_cols = [
        col
        for col in matrix_df.columns
        if col not in {SLIDE_ID_COL, LABEL_COL}
    ]
    if not feature_cols:
        raise ValueError("No model prediction columns found in the OOF matrix.")

    x = matrix_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        bad_cols = x.columns[x.isna().any()].tolist()
        raise ValueError(f"OOF matrix contains missing or non-numeric values in: {bad_cols}")

    if labels_path and os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        if SLIDE_ID_COL not in labels_df.columns or LABEL_COL not in labels_df.columns:
            raise ValueError(f"{labels_path} must contain {SLIDE_ID_COL} and {LABEL_COL}.")

        label_check = pd.DataFrame({
            SLIDE_ID_COL: slide_ids,
            "matrix_label": y.values,
        }).merge(
            labels_df[[SLIDE_ID_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels_label"}),
            on=SLIDE_ID_COL,
            how="left",
        )
        if label_check["labels_label"].isna().any():
            missing_ids = label_check.loc[label_check["labels_label"].isna(), SLIDE_ID_COL].tolist()
            raise ValueError(f"Labels file is missing slide ids: {missing_ids[:10]}")
        mismatch = label_check["matrix_label"].values != label_check["labels_label"].astype(int).values
        if mismatch.any():
            bad_ids = label_check.loc[mismatch, SLIDE_ID_COL].tolist()
            raise ValueError(f"Label mismatch between matrix and labels file: {bad_ids[:10]}")

    return x, y, slide_ids, feature_cols


def make_pipeline(estimator: LogisticRegression, transform: str) -> Pipeline:
    steps = []
    if transform == "standard":
        steps.append(("scaler", StandardScaler()))
    elif transform == "rank":
        steps.append((
            "rank_transform",
            QuantileTransformer(
                n_quantiles=30,
                output_distribution="normal",
                random_state=42,
            ),
        ))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_meta_learners(args: argparse.Namespace) -> Dict[str, Pipeline]:
    learners: Dict[str, Pipeline] = {}

    if args.meta_learner in {"both", "logistic"}:
        estimator = LogisticRegression(
            penalty="l2",
            C=args.c,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        learners["logistic_regression"] = make_pipeline(estimator, args.transform)

    if args.meta_learner in {"both", "elasticnet"}:
        estimator = LogisticRegression(
            penalty="elasticnet",
            C=args.c,
            solver="saga",
            l1_ratio=args.elasticnet_l1_ratio,
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        learners["elastic_net"] = make_pipeline(estimator, args.transform)

    return learners


def get_effective_n_splits(y: pd.Series, requested: int) -> int:
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(requested, min_class_count)
    if n_splits < 2:
        raise ValueError(f"Need at least 2 samples in each class for CV. Class counts: {class_counts.to_dict()}")
    return n_splits


def cross_validated_predict(
    estimator: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    slide_ids: pd.Series,
    n_splits: int,
    n_repeats: int,
    seed: int,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    prediction_parts = []
    repeat_metrics = []

    for split_idx, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
        repeat = split_idx // n_splits + 1
        fold = split_idx % n_splits + 1
        fold_model = clone(estimator)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        prob = fold_model.predict_proba(x.iloc[valid_idx])[:, 1]
        pred = (prob >= threshold).astype(int)
        prediction_parts.append(pd.DataFrame({
            SLIDE_ID_COL: slide_ids.iloc[valid_idx].values,
            "label": y.iloc[valid_idx].values,
            "repeat": repeat,
            "fold": fold,
            "prob_class1": prob,
            "pred": pred,
        }))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    for repeat, repeat_df in predictions.groupby("repeat", sort=True):
        metrics = compute_metrics(
            repeat_df["label"].values,
            repeat_df["prob_class1"].values,
            repeat_df["pred"].values,
            threshold,
        )
        metrics["repeat"] = int(repeat)
        repeat_metrics.append(metrics)

    repeat_metrics_df = pd.DataFrame(repeat_metrics)
    metrics = summarize_repeat_metrics(repeat_metrics_df)
    return predictions, metrics, repeat_metrics_df


def summarize_repeat_metrics(repeat_metrics_df: pd.DataFrame) -> Dict[str, float]:
    metric_cols = [
        "auc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    summary: Dict[str, float] = {}
    n_repeats = len(repeat_metrics_df)
    for col in metric_cols:
        values = repeat_metrics_df[col].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n_repeats > 1 else 0.0
        ci = 1.96 * std / np.sqrt(n_repeats) if n_repeats > 1 else 0.0
        summary[col] = mean
        summary[f"{col}_std"] = std
        summary[f"{col}_ci95_low"] = mean - ci
        summary[f"{col}_ci95_high"] = mean + ci
    summary["threshold"] = float(repeat_metrics_df["threshold"].iloc[0])
    summary["cv_repeats"] = int(n_repeats)
    return summary


def compute_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "auprc": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def extract_coefficients(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    lr = model.named_steps["model"]
    coef = lr.coef_.reshape(-1)
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False)
    coef_df.loc[:, "intercept"] = float(lr.intercept_[0])
    return coef_df


def save_final_model(model: Pipeline, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    x, y, slide_ids, feature_cols = load_oof_data(args.matrix, args.labels)
    n_splits = get_effective_n_splits(y, args.cv_folds)
    learners = build_meta_learners(args)

    print("=" * 80)
    print("OOF stacking meta-learner training")
    print("=" * 80)
    print(f"Matrix: {args.matrix}")
    print(f"Samples: {len(y)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Class counts: {y.value_counts().sort_index().to_dict()}")
    print(f"Meta CV folds x repeats: {n_splits} x {args.cv_repeats}")
    print(f"Output dir: {args.output_dir}")
    print()

    all_metrics = []
    config = {
        "matrix": os.path.abspath(args.matrix),
        "labels": os.path.abspath(args.labels) if args.labels else None,
        "feature_cols": feature_cols,
        "cv_folds": n_splits,
        "cv_repeats": args.cv_repeats,
        "seed": args.seed,
        "c": args.c,
        "elasticnet_l1_ratio": args.elasticnet_l1_ratio,
        "threshold": args.threshold,
        "transform": args.transform,
    }

    for name, estimator in learners.items():
        print(f"Training {name} ...")
        predictions, metrics, repeat_metrics_df = cross_validated_predict(
            estimator=estimator,
            x=x,
            y=y,
            slide_ids=slide_ids,
            n_splits=n_splits,
            n_repeats=args.cv_repeats,
            seed=args.seed,
            threshold=args.threshold,
        )
        metrics["meta_learner"] = name
        metrics["n_samples"] = int(len(y))
        metrics["n_features"] = int(len(feature_cols))
        metrics["cv_folds"] = int(n_splits)
        metrics["cv_repeats"] = int(args.cv_repeats)
        all_metrics.append(metrics)

        final_model = clone(estimator)
        final_model.fit(x, y)
        coef_df = extract_coefficients(final_model, feature_cols)

        predictions_path = os.path.join(args.output_dir, f"{name}_meta_oof_predictions.csv")
        repeat_metrics_path = os.path.join(args.output_dir, f"{name}_repeat_metrics.csv")
        coef_path = os.path.join(args.output_dir, f"{name}_coefficients.csv")
        model_path = os.path.join(args.output_dir, f"{name}_final_model.pkl")

        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        repeat_metrics_df.to_csv(repeat_metrics_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        save_final_model(final_model, model_path)

        print(
            f"  AUC={metrics['auc']:.4f}±{metrics['auc_std']:.4f}, "
            f"AUPRC={metrics['auprc']:.4f}±{metrics['auprc_std']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f}, "
            f"F1={metrics['f1']:.4f}±{metrics['f1_std']:.4f}"
        )
        print(f"  Saved predictions: {predictions_path}")
        print(f"  Saved repeat metrics: {repeat_metrics_path}")
        print(f"  Saved coefficients: {coef_path}")
        print(f"  Saved final model: {model_path}")
        print()

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.output_dir, "meta_learner_metrics.csv")
    config_path = os.path.join(args.output_dir, "meta_learner_config.json")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Summary")
    print(metrics_df[[
        "meta_learner",
        "auc",
        "auc_std",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_std",
    ]].to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
