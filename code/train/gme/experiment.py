"""End-to-end GME/CGME middle-fusion training for cohort-selected classification.

This script is intentionally self-contained for server runs:

1. Read a middle-fusion manifest of multi-encoder h5 embeddings.
2. For each CV fold, train ProjectionHead, router, and the configured downstream head together.
3. Use intervention attribution only when online routing or teacher
   distillation is enabled; otherwise keep it as a separate post-hoc analysis.
4. Report GME performance, attribution-independent routing, and efficiency.
5. Report AUC, AUPRC, efficiency metrics, and save checkpoints/artifacts.

The intervention score is computed as the prediction drop after replacing one
encoder's projected embeddings, using mean-fusion prediction as the attribution
estimator. This avoids the circular dependency where routing needs attribution
while attribution needs a routed model prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_DIR, CODE_DIR / "architecture", CODE_DIR / "modules"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture.gme_model import GMEModel
from architecture.projection_head import MultiEncoderProjectionHead
from data_utils.cohort import load_experiment_data
from data_utils.gme_dataset import MultiEncoderSlideDataset, infer_input_dims
from modules.attribution import EncoderBaselineAccumulator, replace_encoder_embedding
from modules.routing import DualConsistencyRouter
from train.gme.profiling import move_features, profile_gme_efficiency
from train.gme.config import parse_args


DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Manifests" / "Manifests_seed35" / "fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]


def project_path(value: str | Path) -> Path:
    """Resolve repository-relative paths independently of the caller's cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
PATH_ARGS = ("manifest", "manifest_dir", "output_dir", "run_dir")
DEFAULT_DECISION_THRESHOLD = 0.5


@dataclass
class EvalResult:
    auc: float
    auprc: float
    threshold: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    tn: int
    fp: int
    fn: int
    tp: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split_indices(
    dataset: "MultiEncoderSlideDataset", val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Create a deterministic label-stratified split inside outer-train."""
    labels = np.asarray(
        [int(dataset.clinical.loc[sid, dataset.label_col]) for sid in dataset.slide_ids], dtype=int
    )
    rng = np.random.default_rng(int(seed))
    train_indices: List[int] = []
    val_indices: List[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label).astype(int).tolist()
        rng.shuffle(indices)
        if len(indices) < 2:
            train_indices.extend(indices)
            continue
        n_val = min(max(int(round(len(indices) * float(val_fraction))), 1), len(indices) - 1)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not train_indices or not val_indices:
        raise ValueError(f"Could not create inner split: train={len(train_indices)}, val={len(val_indices)}")
    return train_indices, val_indices


def resolve_device(args: argparse.Namespace) -> torch.device:
    device_name = str(args.device).lower()
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
            args.device = "cpu"
            return torch.device("cpu")
        if args.gpu_id is not None and device_name == "cuda":
            args.device = f"cuda:{int(args.gpu_id)}"
        device = torch.device(args.device)
        if device.index is not None:
            torch.cuda.set_device(device)
        return device
    if device_name != "cpu":
        raise ValueError(f"Unsupported device: {args.device}. Use 'cpu', 'cuda', or 'cuda:<index>'.")
    return torch.device("cpu")


def ensure_manifest(args: argparse.Namespace) -> Path:
    """Build the middle-fusion manifest when requested, missing, or stale."""
    manifest_path = project_path(args.manifest)
    if manifest_path.exists() and not args.build_manifest:
        requested = {str(item) for item in args.feature_dirs}
        existing = set(pd.read_csv(manifest_path, usecols=["feature_dir"])["feature_dir"].astype(str).unique())
        metadata_path = project_path(args.manifest_dir) / "middle_fusion_manifest_config.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        manifest_is_stratified = (
            metadata.get("splitter") == "StratifiedGroupKFold"
            and metadata.get("label_col") == str(args.label_col)
            and metadata.get("cohort") == str(args.cohort)
            and int(metadata.get("seed", -1)) == int(args.seed)
        )
        if existing == requested and manifest_is_stratified:
            return manifest_path
        print(
            "\nExisting manifest is missing the requested feature/split metadata; rebuilding manifest.\n"
            f"Existing: {sorted(existing)}\n"
            f"Requested: {sorted(requested)}\n"
            f"Metadata: {metadata}"
        )

    output_dir = project_path(args.manifest_dir)
    feat_base = project_path(args.feat_base)
    clinical_path = project_path(args.clinical_path)
    command = [
        sys.executable,
        str(CODE_DIR / "utils" / "build_embedding_manifest.py"),
        "--feat-base",
        str(feat_base),
        "--output-dir",
        str(output_dir),
        "--feature-dirs",
        *[str(item) for item in args.feature_dirs],
        "--cv-folds",
        str(args.cv_folds),
        "--seed",
        str(args.seed),
        "--clinical-path",
        str(clinical_path),
        "--experiment-name",
        str(args.experiment_name),
    ]
    print("\nBuilding middle-fusion manifest:")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    built_manifest = output_dir / "fusion_manifest.csv"
    if not built_manifest.exists():
        raise FileNotFoundError(f"Manifest builder finished but did not create {built_manifest}")
    return built_manifest


def set_projection_trainable(model: GMEModel, trainable: bool) -> None:
    for param in model.projection_heads.parameters():
        param.requires_grad_(trainable)
    model.projection_heads.train(trainable)


def build_stage2_optimizer(model: GMEModel, args: argparse.Namespace) -> torch.optim.AdamW:
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not backbone_parameters:
        raise RuntimeError("No trainable parameters left for Stage2.")
    return torch.optim.AdamW(
        [{"params": backbone_parameters, "weight_decay": args.weight_decay}],
        lr=args.lr_stage2,
    )


def compute_metrics(labels: Sequence[int], probs: Sequence[float], decision_threshold: float) -> EvalResult:
    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    threshold = float(decision_threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"decision_threshold must be finite, got {threshold}")
    preds_np = (probs_np >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(labels_np, probs_np)
    except ValueError:
        auprc = np.nan

    return EvalResult(
        auc=float(auc),
        auprc=float(auprc),
        threshold=float(threshold),
        accuracy=float(accuracy_score(labels_np, preds_np)),
        f1=float(f1_score(labels_np, preds_np, zero_division=0)),
        precision=float(precision_score(labels_np, preds_np, zero_division=0)),
        recall=float(recall_score(labels_np, preds_np, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def save_checkpoint(path: Path, model: GMEModel, epoch: int, metrics: EvalResult, extra: Mapping[str, object] | None = None) -> None:
    payload = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "projection_heads": model.projection_heads.state_dict(),
        "router_stats": model.router.get_routing_stats(),
        "downstream_head": model.downstream_head,
        "metrics": asdict(metrics),
    }
    if extra:
        payload.update(dict(extra))
    torch.save(payload, path)


@torch.no_grad()
def build_replacement_baselines(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
) -> Tuple[Dict[str, Mapping[str, torch.Tensor]], pd.DataFrame]:
    """Build train-only encoder replacement baselines."""
    was_training = model.training
    model.eval()
    baseline_acc = EncoderBaselineAccumulator()
    counts: Dict[str, int] = {name: 0 for name in model.encoder_names}

    try:
        for idx in range(len(dataset)):
            raw_features, _, _ = dataset[idx]
            raw_features = move_features(raw_features, device)
            projected = model.project(raw_features)
            for name in model.encoder_names:
                baseline_acc.update(name, projected[name])
                counts[name] += int(projected[name].shape[0])
    finally:
        model.train(was_training)

    baselines = baseline_acc.compute()
    baselines = {
        name: {
            stat_name: stat.to(device) if torch.is_tensor(stat) else stat
            for stat_name, stat in stats.items()
        }
        for name, stats in baselines.items()
    }
    summary = pd.DataFrame([
        {"encoder_name": name, "count": counts[name]}
        for name in model.encoder_names
    ])
    return baselines, summary


def _min_max_from_score_list(scores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if not scores:
        raise RuntimeError("Cannot compute routing score stats from an empty score list.")
    values = torch.cat([score.detach().reshape(-1).float().cpu() for score in scores], dim=0)
    return values.min(), values.max(), int(values.numel())


@torch.no_grad()
def update_train_offline_stats(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Estimate train-only routing statistics.

    Pairwise interactions are intentionally not part of training statistics;
    they are computed by the separate post-hoc analysis script.
    """
    was_training = model.training
    model.eval()

    single_values = []
    for idx in range(len(dataset)):
        raw_features, _, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = model.project(raw_features)
        routing_scores, _single, _interactions = model.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=False,
            interaction_target_class=1,
        )
        single_values.append(routing_scores)

    score_min, score_max, score_count = _min_max_from_score_list(single_values)
    model.router.set_score_stats(
        attribution_min=score_min,
        attribution_max=score_max,
        count=score_count,
    )
    interaction_stats: Dict[str, object] = {
        "count": 0,
        "pair_names": [
            f"{model.encoder_names[i]}__{model.encoder_names[j]}"
            for i, j in model.interaction_pairs
        ],
        "analysis_only": True,
    }
    model.train(was_training)
    return model.router.get_score_stats(), interaction_stats


def inactive_attribution_stats(model: GMEModel) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Return placeholder metadata when attribution routing is not used."""
    return model.router.get_score_stats(), {
        "count": 0,
        "pair_names": [
            f"{model.encoder_names[i]}__{model.encoder_names[j]}"
            for i, j in model.interaction_pairs
        ],
        "analysis_only": True,
    }


def train_mean_teacher_epoch(
    model: GMEModel,
    dataset,
    optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Train the attribution teacher with uniform mean fusion only."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for idx in np.random.permutation(len(dataset)):
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        projected = model.project(raw_features)
        logits, _ = model.forward_mean_fusion(projected)
        loss = criterion(logits, label_t)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(dataset), 1)


@torch.no_grad()
def evaluate_mean_teacher(
    model: GMEModel,
    dataset,
    device: torch.device,
    decision_threshold: float,
) -> EvalResult:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    for idx in range(len(dataset)):
        raw_features, label, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        logits, _ = model.forward_mean_fusion(model.project(raw_features))
        labels.append(int(label))
        probabilities.append(float(torch.softmax(logits, dim=1)[0, 1].cpu()))
    return compute_metrics(labels, probabilities, decision_threshold)


@torch.no_grad()
def build_teacher_attribution_targets(
    teacher: GMEModel,
    dataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    temperature: float,
    clip: float,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """Build train-only soft router targets from true-label teacher LOO margins."""
    teacher.eval()
    targets: Dict[str, torch.Tensor] = {}
    rows: List[Dict[str, object]] = []
    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = teacher.project(raw_features)
        scores, _, _ = teacher.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=False,
            target_class=int(label),
        )
        centered = scores - scores.mean()
        normalized = centered / centered.std(unbiased=False).clamp_min(1e-8)
        normalized = normalized.clamp(min=-float(clip), max=float(clip))
        weights = torch.softmax(normalized / float(temperature), dim=-1)
        targets[str(slide_id)] = weights.detach().cpu()
        for name, raw_score, normalized_score, weight in zip(
            teacher.encoder_names, scores, normalized, weights
        ):
            rows.append({
                "slide_id": str(slide_id),
                "label": int(label),
                "encoder": name,
                "loo_true_margin": float(raw_score.cpu()),
                "normalized_score": float(normalized_score.cpu()),
                "teacher_weight": float(weight.cpu()),
            })
    return targets, pd.DataFrame(rows)


def train_teacher_student_epoch(
    model: GMEModel,
    dataset,
    optimizer,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    teacher_targets: Mapping[str, torch.Tensor] | None,
    distill_weight: float,
    max_distill_loss_ratio: float,
    grad_clip: float,
) -> Tuple[float, float, float, float]:
    """Train the student router with classification and optional distillation."""
    projection_is_trainable = any(param.requires_grad for param in model.projection_heads.parameters())
    use_distillation = float(distill_weight) > 0.0
    if use_distillation and teacher_targets is None:
        raise RuntimeError("Teacher targets are required when distillation is enabled.")
    model.train()
    if not projection_is_trainable:
        model.projection_heads.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_distill_loss = 0.0
    total_effective_distill_weight = 0.0

    for idx in np.random.permutation(len(dataset)):
        raw_features, label, slide_id = dataset[int(idx)]
        slide_id = str(slide_id)
        if use_distillation and slide_id not in teacher_targets:
            raise KeyError(f"Missing teacher attribution target for slide {slide_id}")
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        target_weights = (
            teacher_targets[slide_id].to(device=device)
            if use_distillation
            else None
        )

        optimizer.zero_grad(set_to_none=True)
        (
            logits, _attn, routed, _projected, _scores, _single, _interactions,
            _base_logits, _pair_residual, _scaled,
        ) = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            compute_interactions=False,
            use_student_router=True,
        )
        cls_loss = criterion(logits, label_t)
        if use_distillation:
            distill_loss = F.kl_div(
                routed.weights.clamp_min(1e-8).log(),
                target_weights,
                reduction="sum",
            )
            effective_distill_weight = float(distill_weight)
            if max_distill_loss_ratio > 0.0:
                effective_distill_weight = min(
                    effective_distill_weight,
                    float(max_distill_loss_ratio)
                    * float(cls_loss.detach())
                    / (float(distill_loss.detach()) + 1e-8),
                )
        else:
            distill_loss = cls_loss.new_zeros(())
            effective_distill_weight = 0.0
        loss = cls_loss + effective_distill_weight * distill_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_cls_loss += float(cls_loss.detach().cpu())
        total_distill_loss += float(distill_loss.detach().cpu())
        total_effective_distill_weight += effective_distill_weight

    denom = max(len(dataset), 1)
    return (
        total_loss / denom,
        total_cls_loss / denom,
        total_distill_loss / denom,
        total_effective_distill_weight / denom,
    )


def train_stage2_epoch(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    optimizer,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    grad_clip: float,
) -> Tuple[float, float]:
    projection_is_trainable = any(param.requires_grad for param in model.projection_heads.parameters())
    model.train()
    if not projection_is_trainable:
        model.projection_heads.eval()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    total_cls_loss = 0.0

    for idx in indices:
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        (
            logits,
            _attn,
            _routed,
            _projected,
            _attr,
            _single_attr,
            _interactions,
            _base_logits,
            _pair_residual,
            _scaled_interactions,
        ) = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=False,
        )
        cls_loss = criterion(logits, label_t)
        loss = cls_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_cls_loss += float(cls_loss.detach().cpu())

    denom = max(len(dataset), 1)
    return total_loss / denom, total_cls_loss / denom


@torch.no_grad()
def evaluate_stage2(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    decision_threshold: float,
    fused_output_dir: Path | None = None,
    use_student_router: bool = False,
) -> Tuple[EvalResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()
    labels, probs, rows, weight_rows, interaction_rows = [], [], [], [], []
    if fused_output_dir is not None:
        fused_output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        (
            logits,
            _attn,
            routed,
            _projected,
            attr,
            single_attr,
            _interactions,
            _base_logits,
            _pair_residual,
            _scaled_interactions,
        ) = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            return_attribution_logits=True,
            use_student_router=use_student_router,
        )
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        labels.append(int(label))
        probs.append(prob)
        rows.append({
            "slide_id": slide_id,
            "label": int(label),
            "prob_class1": prob,
        })

        weights = routed.weights.detach().cpu().reshape(-1).numpy()
        attr_np = attr.detach().cpu().reshape(-1).numpy()
        single_np = single_attr.detach().cpu().reshape(-1).numpy()
        c_range = float(routed.attribution_range.detach().cpu().reshape(-1).item())
        tau = float(routed.tau.detach().cpu().reshape(-1).item())
        for encoder, weight, attr_value, single_value in zip(
            routed.encoder_names, weights, attr_np, single_np
        ):
            weight_row = {
                "slide_id": slide_id,
                "encoder": encoder,
                "weight": float(weight),
                "routing_score": float(attr_value),
                "c_range": c_range,
                "tau": tau,
            }
            if not use_student_router:
                weight_row["attribution"] = float(single_value)
            weight_rows.append(weight_row)
        if fused_output_dir is not None:
            fused = routed.fused.detach().cpu().numpy().astype(np.float32)
            with h5py.File(fused_output_dir / f"{slide_id}.h5", "w") as f:
                f.create_dataset("features", data=fused, compression="gzip")

    metrics = compute_metrics(labels, probs, decision_threshold=decision_threshold)
    pred_df = pd.DataFrame(rows)
    pred_df["threshold"] = metrics.threshold
    pred_df["pred"] = (pred_df["prob_class1"] >= metrics.threshold).astype(int)
    return metrics, pred_df, pd.DataFrame(weight_rows), pd.DataFrame(interaction_rows)


def save_fold_outputs(
    fold_dir: Path,
    stage_name: str,
    metrics: EvalResult,
    predictions: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    interactions: pd.DataFrame | None = None,
    efficiency: Mapping[str, float] | None = None,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {**{"stage": stage_name}, **asdict(metrics)}
    if {"label", "base_prob_class1"}.issubset(predictions.columns):
        attribution_metrics = compute_metrics(
            predictions["label"],
            predictions["base_prob_class1"],
            decision_threshold=metrics.threshold,
        )
        metric_row.update({
            "attribution_only_auc": attribution_metrics.auc,
            "attribution_only_auprc": attribution_metrics.auprc,
            "attribution_only_accuracy": attribution_metrics.accuracy,
            "attribution_only_f1": attribution_metrics.f1,
            "pair_delta_auc": metrics.auc - attribution_metrics.auc,
            "pair_delta_auprc": metrics.auprc - attribution_metrics.auprc,
            "pair_delta_f1": metrics.f1 - attribution_metrics.f1,
        })
    if efficiency:
        metric_row.update(dict(efficiency))
    pd.DataFrame([metric_row]).to_csv(
        fold_dir / f"{stage_name}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    predictions.to_csv(
        fold_dir / f"{stage_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    pair_stats = {}
    for column in ("prob_delta", "pair_residual_rms", "pair_relative_rms"):
        if column not in predictions.columns:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").dropna()
        if values.empty:
            continue
        pair_stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
            "mean_abs": float(values.abs().mean()),
        }
    if pair_stats:
        with open(fold_dir / f"{stage_name}_pair_branch_stats.json", "w", encoding="utf-8") as f:
            json.dump(pair_stats, f, indent=2)
    cm = pd.DataFrame(
        [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm.to_csv(fold_dir / f"{stage_name}_confusion_matrix.csv", encoding="utf-8-sig")
    if weights is not None:
        weights.to_csv(
            fold_dir / f"{stage_name}_routing_weights.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
    if interactions is not None and not interactions.empty:
        interactions.to_csv(
            fold_dir / f"{stage_name}_pairwise_interactions.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        encoder_names = sorted(set(interactions["encoder_i"]) | set(interactions["encoder_j"]))
        mean_pairs = interactions.groupby(["encoder_i", "encoder_j"])["interaction"].mean()
        matrix = pd.DataFrame(np.nan, index=encoder_names, columns=encoder_names)
        for encoder in encoder_names:
            matrix.loc[encoder, encoder] = 0.0
        for (left, right), value in mean_pairs.items():
            matrix.loc[left, right] = float(value)
        matrix.index.name = "encoder"
        matrix.to_csv(
            fold_dir / f"{stage_name}_interaction_matrix.csv",
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        if stage_name == "final":
            matrix.to_csv(
                fold_dir / "interaction_matrix.csv",
                encoding="utf-8-sig",
                float_format="%.6f",
            )


def summarize_metrics(fold_rows: List[Mapping[str, object]], output_dir: Path) -> None:
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_rows = []
    for metric in ["auc", "auprc", "accuracy", "f1", "precision", "recall", "parameters", "flops", "inference_time"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce")
        if values.notna().sum() == 0:
            summary_rows.append({
                "metric": metric,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
            })
            continue
        summary_rows.append({
            "metric": metric,
            "mean": float(values.mean(skipna=True)),
            "std": float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0,
            "min": float(values.min(skipna=True)),
            "median": float(values.median(skipna=True)),
            "max": float(values.max(skipna=True)),
        })
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )


def run_fold(
    args: argparse.Namespace,
    fold: int,
    manifest: pd.DataFrame,
    clinical_df: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
) -> Mapping[str, object]:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    use_teacher_student = args.routing_mode == "teacher_student"
    use_online_attribution = not use_teacher_student
    use_teacher_distillation = use_teacher_student and float(args.teacher_distill_weight) > 0.0
    teacher_stage_required = use_teacher_student and (
        bool(args.mean_fusion_warm_start) or use_teacher_distillation
    )

    train_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.max_patches,
        training=True,
    )
    val_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="val",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.eval_max_patches,
        training=False,
    )
    train_eval_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.eval_max_patches,
        training=False,
    )
    input_dims = infer_input_dims(manifest, fold=fold)
    input_dims = {name: input_dims[name] for name in sorted(train_ds.encoder_names)}

    seed_everything(args.seed + fold)
    model = GMEModel(
        input_dims=input_dims,
        target_dim=args.target_dim,
        projection_dropout=args.projection_dropout,
        d_inner=args.d_inner,
        d_attn=args.d_attn,
        n_classes=args.n_classes,
        droprate=args.droprate,
        downstream_head=args.downstream_head,
        mlp_hidden_dim=args.mlp_hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        routing_temperature=args.routing_temperature,
        routing_logit_scale=args.routing_logit_scale,
        attribution_target=args.attribution_target,
        student_router_hidden_dim=(
            args.student_router_hidden_dim
            if args.routing_mode == "teacher_student"
            else 0
        ),
        student_router_temperature=args.student_router_temperature,
        student_router_use_consensus=args.student_router_use_consensus,
    ).to(device)

    print(f"\nFold {fold}: train={len(train_ds)}, val={len(val_ds)}, encoders={train_ds.encoder_names}")
    print(f"Input dims: {input_dims}")

    baseline_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.baseline_max_patches,
        training=False,
    )
    score_stats_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=0,
        training=False,
    )
    inner_train_indices, inner_val_indices = stratified_split_indices(
        train_ds, val_fraction=args.inner_val_fraction, seed=args.seed + fold
    )
    inner_train_ds = Subset(train_ds, inner_train_indices)
    inner_val_ds = Subset(train_eval_ds, inner_val_indices)
    inner_baseline_ds = Subset(baseline_ds, inner_train_indices)
    inner_score_stats_ds = Subset(score_stats_ds, inner_train_indices)
    print(
        f"Fold {fold}: outer_train={len(train_ds)}, outer_test={len(val_ds)}, "
        f"inner_train={len(inner_train_ds)}, inner_val={len(inner_val_ds)}"
    )
    teacher_targets: Dict[str, torch.Tensor] | None = None
    if teacher_stage_required:
        teacher = copy.deepcopy(model).to(device)
        set_projection_trainable(teacher, True)
        teacher_optimizer = torch.optim.AdamW(
            [parameter for parameter in teacher.parameters() if parameter.requires_grad],
            lr=args.teacher_lr,
            weight_decay=args.weight_decay,
        )
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer,
            T_max=max(args.teacher_epochs, 1),
            eta_min=args.teacher_lr * 0.01,
        )
        teacher_best_path = fold_dir / "best_attribution_teacher.pt"
        teacher_best_auc = -math.inf
        teacher_no_improve = 0
        teacher_history: List[Dict[str, object]] = []
        print(
            f"Fold {fold} | Frozen-attribution teacher pretrain | "
            f"mean fusion | epochs={args.teacher_epochs} | "
            f"warm_start={bool(args.mean_fusion_warm_start)}"
        )
        for teacher_epoch in range(1, args.teacher_epochs + 1):
            teacher_loss = train_mean_teacher_epoch(
                teacher,
                inner_train_ds,
                teacher_optimizer,
                device,
                args.grad_clip,
            )
            teacher_metrics = evaluate_mean_teacher(
                teacher,
                inner_val_ds,
                device,
                DEFAULT_DECISION_THRESHOLD,
            )
            teacher_scheduler.step()
            teacher_history.append({
                "epoch": int(teacher_epoch),
                "train_loss": float(teacher_loss),
                "inner_auc": float(teacher_metrics.auc),
                "inner_auprc": float(teacher_metrics.auprc),
                "inner_f1": float(teacher_metrics.f1),
            })
            pd.DataFrame(teacher_history).to_csv(
                fold_dir / "teacher_training_history.csv",
                index=False,
                encoding="utf-8-sig",
                float_format="%.6f",
            )
            print(
                f"Fold {fold} | Teacher | Epoch {teacher_epoch:03d}/{args.teacher_epochs} | "
                f"loss={teacher_loss:.4f} | inner_AUC={teacher_metrics.auc:.4f} | "
                f"inner_AUPRC={teacher_metrics.auprc:.4f}"
            )
            if not np.isnan(teacher_metrics.auc) and teacher_metrics.auc > teacher_best_auc:
                teacher_best_auc = teacher_metrics.auc
                teacher_no_improve = 0
                torch.save({
                    "epoch": int(teacher_epoch),
                    "state_dict": teacher.state_dict(),
                    "metrics": asdict(teacher_metrics),
                    "stage": "frozen_mean_fusion_attribution_teacher",
                }, teacher_best_path)
            else:
                teacher_no_improve += 1
                if teacher_no_improve >= args.teacher_patience:
                    break
        if not teacher_best_path.exists():
            raise RuntimeError(f"Fold {fold}: no attribution teacher checkpoint was selected.")
        teacher_payload = torch.load(teacher_best_path, map_location=device)
        teacher.load_state_dict(teacher_payload["state_dict"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        if use_teacher_distillation:
            teacher_baselines, teacher_baseline_summary = build_replacement_baselines(
                teacher,
                inner_baseline_ds,
                device,
            )
            teacher_targets, teacher_target_rows = build_teacher_attribution_targets(
                teacher=teacher,
                dataset=inner_score_stats_ds,
                device=device,
                baselines=teacher_baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
                temperature=args.teacher_target_temperature,
                clip=args.teacher_target_clip,
            )
            teacher_target_rows.to_csv(
                fold_dir / "teacher_attribution_targets.csv",
                index=False,
                encoding="utf-8-sig",
                float_format="%.6f",
            )
            teacher_baseline_summary.to_csv(
                fold_dir / "teacher_replacement_baseline_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            torch.save({
                "targets": teacher_targets,
                "encoder_names": teacher.encoder_names,
                "teacher_checkpoint": str(teacher_best_path),
                "target_temperature": float(args.teacher_target_temperature),
                "target_clip": float(args.teacher_target_clip),
                "target": "true_label_logit_margin_loo",
            }, fold_dir / "teacher_attribution_targets.pt")
        else:
            print(
                f"Fold {fold}: teacher_distill_weight={args.teacher_distill_weight:g}; "
                "skipping teacher LOO targets and replacement baselines."
            )

        # Optionally initialize the student representation and classifier from
        # the frozen teacher. The embedding router remains label-free at inference.
        if args.mean_fusion_warm_start:
            model.load_state_dict(teacher.state_dict())
            set_projection_trainable(model, not bool(args.teacher_freeze_projection))
        else:
            print(
                f"Fold {fold}: mean-fusion warm start disabled; "
                "the main model keeps its independent initialization."
            )
            # Without teacher initialization, the student projection must remain
            # trainable even when teacher_freeze_projection is enabled.
            set_projection_trainable(model, True)
        del teacher
    elif use_teacher_student:
        print(
            f"Fold {fold}: teacher stage skipped because mean-fusion warm start is disabled "
            "and teacher_distill_weight=0."
        )
        set_projection_trainable(model, True)
    else:
        set_projection_trainable(model, True)

    initial_stage2_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    optimizer2 = build_stage2_optimizer(model, args)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1), eta_min=args.lr_stage2 * 0.01)
    best_stage2_auc = -math.inf
    best_stage2_path = fold_dir / "best_gme_model.pt"
    best_stage2_metrics: EvalResult | None = None
    no_improve = 0
    stage2_history: List[Dict[str, object]] = []
    if use_online_attribution:
        baselines, baseline_summary = build_replacement_baselines(
            model,
            inner_baseline_ds,
            device,
        )
        score_stats, interaction_stats = update_train_offline_stats(
            model=model,
            dataset=inner_score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
    else:
        baselines = {}
        baseline_summary = pd.DataFrame()
        score_stats, interaction_stats = inactive_attribution_stats(model)

    for epoch in range(1, args.stage2_epochs + 1):
        if use_teacher_student:
            if use_teacher_distillation and teacher_targets is None:
                raise RuntimeError("Teacher targets were not initialized.")
            (
                train_loss,
                train_cls_loss,
                train_distill_loss,
                effective_distill_weight,
            ) = train_teacher_student_epoch(
                model=model,
                dataset=inner_train_ds,
                optimizer=optimizer2,
                device=device,
                baselines=baselines,
                teacher_targets=teacher_targets,
                distill_weight=args.teacher_distill_weight,
                max_distill_loss_ratio=args.teacher_kl_loss_ratio,
                grad_clip=args.grad_clip,
            )
        else:
            train_loss, train_cls_loss = train_stage2_epoch(
                model=model,
                dataset=inner_train_ds,
                optimizer=optimizer2,
                device=device,
                baselines=baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
                grad_clip=args.grad_clip,
            )
            train_distill_loss = float("nan")
            effective_distill_weight = float("nan")
        if use_online_attribution:
            baselines, baseline_summary = build_replacement_baselines(
                model,
                inner_baseline_ds,
                device,
            )
        if use_online_attribution:
            score_stats, interaction_stats = update_train_offline_stats(
                model=model,
                dataset=inner_score_stats_ds,
                device=device,
                baselines=baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
            )
        scheduler2.step()
        evaluation_weight_source = "raw"
        eval_baselines = baselines
        eval_baseline_summary = baseline_summary
        eval_score_stats = score_stats
        eval_interaction_stats = interaction_stats

        val_metrics, val_pred, val_weights, val_interactions = evaluate_stage2(
            model=model,
            dataset=inner_val_ds,
            device=device,
            baselines=eval_baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            decision_threshold=DEFAULT_DECISION_THRESHOLD,
            use_student_router=use_teacher_student,
        )
        stage2_history.append({
            "epoch": int(epoch),
            "evaluation_weight_source": evaluation_weight_source,
            "train_loss": float(train_loss),
            "train_cls_loss": float(train_cls_loss),
            "train_distill_loss": float(train_distill_loss),
            "effective_distill_weight": float(effective_distill_weight),
            "inner_auc": float(val_metrics.auc),
            "inner_auprc": float(val_metrics.auprc),
            "inner_f1": float(val_metrics.f1),
        })
        pd.DataFrame(stage2_history).to_csv(
            fold_dir / "stage2_training_history.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        print(
            f"Fold {fold} | Epoch {epoch:03d}/{args.stage2_epochs} | "
            f"loss={train_loss:.4f} | cls={train_cls_loss:.4f} | "
            f"inner_AUC={val_metrics.auc:.4f} | inner_AUPRC={val_metrics.auprc:.4f} | "
            f"F1@{DEFAULT_DECISION_THRESHOLD:g}={val_metrics.f1:.4f} | "
        )

        improved = not np.isnan(val_metrics.auc) and val_metrics.auc > best_stage2_auc
        if improved:
            best_stage2_auc = val_metrics.auc
            best_stage2_metrics = val_metrics
            no_improve = 0
            save_checkpoint(
                best_stage2_path,
                model,
                epoch,
                val_metrics,
                {
                    "input_dims": input_dims,
                    "stage": "stage2_gme",
                    "baseline_path": str(fold_dir / "replacement_baselines.pt"),
                    "routing_score_stats": eval_score_stats,
                    "evaluation_weight_source": evaluation_weight_source,
                    "baseline_policy": "Train-only replacement baselines rebuilt from current projection after each epoch.",
                },
            )
            if use_online_attribution:
                baseline_payload = {
                    "fold": int(fold),
                    "baselines": {
                        name: {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in stats.items()
                        }
                        for name, stats in eval_baselines.items()
                    },
                    "input_dims": input_dims,
                    "encoder_names": model.encoder_names,
                    "weight_source": evaluation_weight_source,
                    "policy": "Train-only replacement baselines rebuilt from the current end-to-end projection space.",
                    "routing_score_stats": eval_score_stats,
                }
                torch.save(baseline_payload, fold_dir / "replacement_baselines.pt")
                eval_baseline_summary.to_csv(
                    fold_dir / "replacement_baseline_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            save_fold_outputs(
                fold_dir, "inner_selection", val_metrics, val_pred, val_weights,
                interactions=val_interactions,
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Fold {fold}: early stopping at epoch {epoch}. Best inner Stage2 AUC={best_stage2_auc:.4f}")
                break

    if not best_stage2_path.exists():
        raise RuntimeError(f"Fold {fold}: no Stage2 checkpoint was selected on inner validation.")

    selected_payload = torch.load(best_stage2_path, map_location=device)
    selected_epoch = int(selected_payload.get("epoch", 1))
    if args.training_protocol == "nested_refit":
        model.load_state_dict(initial_stage2_state)
        seed_everything(args.seed + fold)
        retrain_optimizer = build_stage2_optimizer(model, args)
        retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            retrain_optimizer,
            T_max=max(args.stage2_epochs, 1),
            eta_min=args.lr_stage2 * 0.01,
        )
        if use_online_attribution:
            baselines, baseline_summary = build_replacement_baselines(model, baseline_ds, device)
            _, final_interaction_stats = update_train_offline_stats(
                model=model,
                dataset=score_stats_ds,
                device=device,
                baselines=baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
            )
        else:
            baselines = {}
            final_interaction_stats = inactive_attribution_stats(model)[1]
        for retrain_epoch in range(1, selected_epoch + 1):
            _, retrain_cls_loss = train_stage2_epoch(
                model=model,
                dataset=train_ds,
                optimizer=retrain_optimizer,
                device=device,
                baselines=baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
                grad_clip=args.grad_clip,
            )
            retrain_scheduler.step()
            if use_online_attribution:
                baselines, baseline_summary = build_replacement_baselines(model, baseline_ds, device)
            if use_online_attribution:
                _, final_interaction_stats = update_train_offline_stats(
                    model=model,
                    dataset=score_stats_ds,
                    device=device,
                    baselines=baselines,
                    replacement_strategy=args.replacement_strategy,
                    gaussian_std_scale=args.gaussian_std_scale,
                )
        final_weight_source = "raw"
        final_train_count = len(train_ds)
        checkpoint_stage = "stage2_retrained_outer_train"
        selection_policy = "inner_validation_AUC_then_retrain_on_outer_train"
        print(
            f"Fold {fold}: selected inner epoch={selected_epoch}; retrained on all outer-train slides; "
            f"final weights={final_weight_source.upper()}."
        )
    else:
        # The selected checkpoint is already the model evaluated on inner
        # validation. Test it directly; never expose outer-test to training or
        # checkpoint selection, and do not refit on inner-validation slides.
        model.load_state_dict(selected_payload["state_dict"])
        final_weight_source = "raw"
        baselines, baseline_summary = build_replacement_baselines(model, inner_baseline_ds, device)
        if use_online_attribution:
            _, final_interaction_stats = update_train_offline_stats(
                model=model,
                dataset=inner_score_stats_ds,
                device=device,
                baselines=baselines,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
            )
        else:
            final_interaction_stats = inactive_attribution_stats(model)[1]
        baseline_summary.to_csv(
            fold_dir / "replacement_baseline_summary.csv", index=False, encoding="utf-8-sig"
        )
        final_train_count = len(inner_train_ds)
        checkpoint_stage = "stage2_selected_inner_checkpoint_no_refit"
        selection_policy = "fixed_train_validation_test_best_checkpoint_no_refit"
        split_rows = [
            {"slide_id": train_ds.slide_ids[int(index)], "split": "train"}
            for index in inner_train_indices
        ]
        split_rows.extend(
            {"slide_id": train_ds.slide_ids[int(index)], "split": "validation"}
            for index in inner_val_indices
        )
        split_rows.extend(
            {"slide_id": slide_id, "split": "test"}
            for slide_id in val_ds.slide_ids
        )
        pd.DataFrame(split_rows).to_csv(
            fold_dir / "fixed_split_assignments.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"Fold {fold}: selected inner epoch={selected_epoch}; using selected checkpoint directly; "
            f"train={len(inner_train_ds)}, validation={len(inner_val_ds)}, test={len(val_ds)}; "
            f"final weights={final_weight_source.upper()}; no outer retrain."
        )

    # Pairwise-interaction analysis is a separate post-hoc workflow. Keep one
    # baseline artifact matched to the final checkpoint, without running LOO
    # statistics during student-router training.
    if not baselines:
        final_baseline_ds = (
            baseline_ds
            if args.training_protocol == "nested_refit"
            else inner_baseline_ds
        )
        baselines, baseline_summary = build_replacement_baselines(
            model,
            final_baseline_ds,
            device,
        )
    final_baseline_payload = {
        "fold": int(fold),
        "baselines": {
            name: {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in stats.items()
            }
            for name, stats in baselines.items()
        },
        "input_dims": input_dims,
        "encoder_names": model.encoder_names,
        "weight_source": final_weight_source,
        "policy": (
            "Train-only replacement baselines retained for post-hoc interaction analysis; "
            "not used by student routing."
            if use_teacher_student
            else "Train-only replacement baselines matched to the final online-attribution model."
        ),
        "routing_score_stats": model.router.get_score_stats(),
    }
    torch.save(final_baseline_payload, fold_dir / "replacement_baselines.pt")
    baseline_summary.to_csv(
        fold_dir / "replacement_baseline_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decision_threshold = DEFAULT_DECISION_THRESHOLD
    print(f"Fold {fold}: fixed decision threshold={decision_threshold:.6f}")
    with open(fold_dir / "decision_threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(decision_threshold),
            "source": "internal_default",
            },
            f,
            indent=2,
        )
    fused_dir = fold_dir / "fused_val_h5" if args.save_fused_h5 else None
    final_metrics, final_pred, final_weights, final_interactions = evaluate_stage2(
        model=model,
        dataset=val_ds,
        device=device,
        baselines=baselines,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
        decision_threshold=decision_threshold,
        fused_output_dir=fused_dir,
        use_student_router=use_teacher_student,
    )
    efficiency = profile_gme_efficiency(
        model=model,
        dataset=val_ds,
        device=device,
        baselines=baselines,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
        profile_samples=args.profile_samples,
        profile_warmup=args.profile_warmup,
        profile_repeat=args.profile_repeat,
        use_student_router=use_teacher_student,
    )
    save_fold_outputs(
        fold_dir, "final", final_metrics, final_pred, final_weights,
        interactions=final_interactions, efficiency=efficiency,
    )
    save_checkpoint(
        best_stage2_path,
        model,
        selected_epoch,
        final_metrics,
        {
            "input_dims": input_dims,
            "stage": checkpoint_stage,
            "training_protocol": args.training_protocol,
            "selection_policy": selection_policy,
            "selected_inner_epoch": selected_epoch,
            "evaluation_weight_source": final_weight_source,
        },
    )

    with open(fold_dir / "routing_stats.json", "w", encoding="utf-8") as f:
        json.dump(model.router.get_routing_stats(), f, indent=2)
    with open(fold_dir / "interaction_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "analysis_only": True,
                "script": "code/evaluation/compute_pairwise_interactions.py",
                "note": "Pairwise interactions are not part of training or inference.",
            },
            f,
            indent=2,
        )

    row = {
        "fold": int(fold),
        "training_protocol": args.training_protocol,
        "n_train": int(final_train_count),
        "n_selection_val": len(inner_val_ds),
        "n_test": len(val_ds),
        "n_val": len(val_ds),
        "weight_source": final_weight_source,
        **asdict(final_metrics),
        **efficiency,
    }
    return row


def main() -> None:
    args = parse_args()
    if args.workflow_mode != "fusion_only":
        raise ValueError(
            "train_gme.py is the fusion-only entry point. "
            "Use code/train/run_gme_workflow.py for fusion_and_analysis or analysis_only."
        )
    device = resolve_device(args)
    seed_everything(args.seed)

    manifest_path = ensure_manifest(args)
    args.manifest = manifest_path
    manifest = pd.read_csv(manifest_path)
    clinical_df, _, _, cohort = load_experiment_data(args.experiment_name, args.clinical_path)
    if cohort.label_col != args.label_col:
        raise RuntimeError("Resolved cohort label does not match the training configuration.")
    clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)

    all_folds = sorted(manifest["fold"].dropna().astype(int).unique().tolist())
    folds = args.folds if args.folds else all_folds
    missing_folds = sorted(set(folds) - set(all_folds))
    if missing_folds:
        raise ValueError(f"Requested folds not found in manifest: {missing_folds}. Available: {all_folds}")

    if args.run_dir is not None:
        output_dir = args.run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir / args.experiment_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    print("=" * 80)
    print("End-to-end GME middle-fusion training")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Clinical: {args.clinical_path}")
    print(f"Folds: {folds}")
    print(f"Training protocol: {args.training_protocol}")
    if args.training_protocol == "fixed_split_no_refit" and len(folds) > 1:
        print(
            "[Note] fixed_split_no_refit runs once per selected fold. "
            "Set folds: [1] for a single fixed train/validation/test experiment."
        )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.current_device()} | {torch.cuda.get_device_name(device)}")
    print(f"Output: {output_dir}")
    print()

    fold_rows: List[Mapping[str, object]] = []
    for fold in folds:
        fold_row = run_fold(args, int(fold), manifest, clinical_df, device, output_dir)
        fold_rows.append(fold_row)
        summarize_metrics(fold_rows, output_dir)

    summarize_metrics(fold_rows, output_dir)

    print("\nSummary:")
    print(pd.read_csv(output_dir / "summary_metrics.csv").to_string(index=False))
    print(f"\nSaved output: {output_dir}")


if __name__ == "__main__":
    main()
