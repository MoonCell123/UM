"""End-to-end GME/CGME middle-fusion training for UVM D3/M3 classification.

This script is intentionally self-contained for server runs:

1. Read a middle-fusion manifest of multi-encoder h5 embeddings.
2. For each CV fold, train ProjectionHead, router, and ABMIL together.
3. Rebuild train-only replacement baselines from the current projection space.
4. Build a train-only static Beacon as a global semantic prior.
5. Train GME with intervention attribution plus a Beacon constraint loss.
6. Report AUC, AUPRC, efficiency metrics, and save checkpoints/artifacts.

The intervention score is computed as the prediction drop after replacing one
encoder's projected embeddings, using mean-fusion prediction as the attribution
estimator. This avoids the circular dependency where routing needs attribution
while attribution needs a routed model prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
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

from architecture.abmil_cls import ABMIL_Cls
from architecture.gme_model import GMEModel
from architecture.projection_head import MultiEncoderProjectionHead
from data_utils.cls_dataset import load_uvm_data
from data_utils.gme_dataset import MultiEncoderSlideDataset, infer_input_dims
from modules.attribution import EncoderBaselineAccumulator, replace_encoder_embedding
from modules.beacon import BeaconAccumulator
from modules.routing import DualConsistencyRouter
from train.gme.profiling import move_features, profile_gme_efficiency
from train.gme.config import parse_args


DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
PATH_ARGS = ("manifest", "manifest_dir", "output_dir", "run_dir")


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


class ExponentialMovingAverage:
    """Maintain an optimizer-step EMA of model parameters.

    Offline routing statistics are deliberately not averaged. They are rebuilt
    from train-only data after applying the averaged parameters.
    """

    def __init__(self, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.num_updates = 0

    @property
    def initialized(self) -> bool:
        return bool(self.shadow)

    @torch.no_grad()
    def initialize(self, model: nn.Module) -> None:
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not self.initialized:
            raise RuntimeError("EMA must be initialized before update().")
        current_names = {name for name, _ in model.named_parameters()}
        if current_names != set(self.shadow):
            raise RuntimeError("Model parameters changed after EMA initialization.")
        one_minus_decay = 1.0 - self.decay
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=one_minus_decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        if not self.initialized:
            raise RuntimeError("EMA has not been initialized.")
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                raise RuntimeError(f"EMA is missing parameter: {name}")
            parameter.copy_(self.shadow[name])

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily apply EMA parameters and restore parameters/buffers after use."""
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(backup)


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
    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.build_manifest:
        requested = {str(item) for item in args.feature_dirs}
        existing = set(pd.read_csv(manifest_path, usecols=["feature_dir"])["feature_dir"].astype(str).unique())
        metadata_path = Path(args.manifest_dir) / "middle_fusion_manifest_config.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        manifest_is_stratified = (
            metadata.get("splitter") == "StratifiedGroupKFold"
            and metadata.get("label_col") == str(args.label_col)
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

    output_dir = Path(args.manifest_dir)
    command = [
        sys.executable,
        str(CODE_DIR / "utils" / "build_embedding_manifest.py"),
        "--feat-base",
        str(args.feat_base),
        "--output-dir",
        str(output_dir),
        "--feature-dirs",
        *[str(item) for item in args.feature_dirs],
        "--cv-folds",
        str(args.cv_folds),
        "--seed",
        str(args.seed),
        "--clinical-path",
        str(args.clinical_path),
        "--label-col",
        str(args.label_col),
    ]
    print("\nBuilding middle-fusion manifest:")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    built_manifest = output_dir / "middle_fusion_manifest.csv"
    if not built_manifest.exists():
        raise FileNotFoundError(f"Manifest builder finished but did not create {built_manifest}")
    return built_manifest


def set_projection_trainable(model: GMEModel, trainable: bool) -> None:
    for param in model.projection_heads.parameters():
        param.requires_grad_(trainable)
    model.projection_heads.train(trainable)


def build_stage2_optimizer(model: GMEModel, args: argparse.Namespace) -> torch.optim.AdamW:
    pair_parameters = [model.interaction_gate] if model.interaction_gate.requires_grad else []
    pair_parameter_ids = {id(parameter) for parameter in pair_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in pair_parameter_ids
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "weight_decay": args.weight_decay})
    if pair_parameters:
        parameter_groups.append({
            "params": pair_parameters,
            "lr": args.interaction_pair_lr,
            "weight_decay": args.interaction_pair_weight_decay,
        })
    if not parameter_groups:
        raise RuntimeError("No trainable parameters left for Stage2.")
    return torch.optim.AdamW(parameter_groups, lr=args.lr_stage2)


def reset_stage2_classifier(model: GMEModel, args: argparse.Namespace, device: torch.device) -> None:
    model.classifier = ABMIL_Cls(
        D_feat=args.target_dim,
        D_inner=args.d_inner,
        D_attn=args.d_attn,
        n_classes=args.n_classes,
        droprate=args.droprate,
    ).to(device)


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
        "metrics": asdict(metrics),
    }
    if extra:
        payload.update(dict(extra))
    torch.save(payload, path)


def load_model_state(path: Path, model: GMEModel, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)


@torch.no_grad()
def build_beacon_and_baselines(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    target_dim: int,
) -> Tuple[torch.Tensor, pd.DataFrame, Dict[str, Mapping[str, torch.Tensor]], pd.DataFrame]:
    """Build train-only Beacon and replacement baselines in one projection pass."""
    was_training = model.training
    model.eval()
    beacon_acc = BeaconAccumulator(target_dim=target_dim, device=device)
    baseline_acc = EncoderBaselineAccumulator()
    counts: Dict[str, int] = {name: 0 for name in model.encoder_names}

    try:
        for idx in range(len(dataset)):
            raw_features, _, _ = dataset[idx]
            raw_features = move_features(raw_features, device)
            projected = model.project(raw_features)
            for name in model.encoder_names:
                beacon_acc.update(name, projected[name])
                baseline_acc.update(name, projected[name])
                counts[name] += int(projected[name].shape[0])
    finally:
        model.train(was_training)

    beacon = beacon_acc.compute(normalize_beacon=True).to(device)
    beacon_summary = beacon_acc.summary()
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
    return beacon, beacon_summary, baselines, summary


@torch.no_grad()
def build_replacement_baselines(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    target_dim: int,
) -> Tuple[Dict[str, Mapping[str, torch.Tensor]], pd.DataFrame]:
    """Backward-compatible helper for replacement-only baseline construction."""
    _, _, baselines, summary = build_beacon_and_baselines(model, dataset, device, target_dim)
    return baselines, summary


def _min_max_from_score_list(scores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if not scores:
        raise RuntimeError("Cannot compute routing score stats from an empty score list.")
    values = torch.cat([score.detach().reshape(-1).float().cpu() for score in scores], dim=0)
    return values.min(), values.max(), int(values.numel())


def train_stage1_epoch(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    optimizer,
    device: torch.device,
    beacon: torch.Tensor | None,
    beacon_constraint_weight: float,
    consistency_weight: float,
    grad_clip: float,
) -> Tuple[float, float, float]:
    """Pretrain ProjectionHead with geometry-only objectives."""
    model.train()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    total_beacon_loss = 0.0
    total_consistency_loss = 0.0

    for idx in indices:
        raw_features, _, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)

        optimizer.zero_grad(set_to_none=True)
        projected = model.project(raw_features)
        if beacon is not None and beacon_constraint_weight > 0:
            beacon_loss, _ = model.beacon_constraint_loss(projected=projected, beacon=beacon)
        else:
            beacon_loss = next(iter(projected.values())).new_tensor(0.0)
        if consistency_weight > 0:
            consistency_loss = model.encoder_consistency_loss(projected)
        else:
            consistency_loss = beacon_loss.new_tensor(0.0)
        loss = float(beacon_constraint_weight) * beacon_loss + float(consistency_weight) * consistency_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=grad_clip,
            )
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_beacon_loss += float(beacon_loss.detach().cpu())
        total_consistency_loss += float(consistency_loss.detach().cpu())

    denom = max(len(dataset), 1)
    return total_loss / denom, total_beacon_loss / denom, total_consistency_loss / denom


@torch.no_grad()
def evaluate_stage1_geometry(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    beacon: torch.Tensor | None,
    beacon_constraint_weight: float,
    consistency_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_beacon_loss = 0.0
    total_consistency_loss = 0.0
    for idx in range(len(dataset)):
        raw_features, _, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = model.project(raw_features)
        if beacon is not None and beacon_constraint_weight > 0:
            beacon_loss, _ = model.beacon_constraint_loss(projected=projected, beacon=beacon)
        else:
            beacon_loss = next(iter(projected.values())).new_tensor(0.0)
        if consistency_weight > 0:
            consistency_loss = model.encoder_consistency_loss(projected)
        else:
            consistency_loss = beacon_loss.new_tensor(0.0)
        loss = float(beacon_constraint_weight) * beacon_loss + float(consistency_weight) * consistency_loss
        total_loss += float(loss.detach().cpu())
        total_beacon_loss += float(beacon_loss.detach().cpu())
        total_consistency_loss += float(consistency_loss.detach().cpu())

    denom = max(len(dataset), 1)
    return total_loss / denom, total_beacon_loss / denom, total_consistency_loss / denom


@torch.no_grad()
def update_train_offline_stats(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Estimate routing and per-pair interaction statistics from train only."""
    was_training = model.training
    model.eval()

    single_values = []
    interaction_values = []
    for idx in range(len(dataset)):
        raw_features, _, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = model.project(raw_features)
        compute_interactions = model.interaction_pair_beta > 0.0
        routing_scores, _single, interactions = model.intervention_attribution(
                projected=projected,
                baselines=baselines,
                replacement_strategy=replacement_strategy,
                gaussian_std_scale=gaussian_std_scale,
                compute_interactions=compute_interactions,
                interaction_target_class=1,
            )
        single_values.append(routing_scores)
        if compute_interactions:
            interaction_values.append(model.flatten_interactions(interactions))

    score_min, score_max, score_count = _min_max_from_score_list(single_values)
    model.router.set_score_stats(
        attribution_min=score_min,
        attribution_max=score_max,
        count=score_count,
    )
    if interaction_values:
        interaction_tensor = torch.stack(interaction_values, dim=0).float()
        interaction_mean = interaction_tensor.mean(dim=0)
        interaction_std = interaction_tensor.std(dim=0, unbiased=False).clamp_min(1e-8)
        interaction_rms = interaction_tensor.square().mean(dim=0).sqrt().clamp_min(1e-8)
        model.set_interaction_stats(
            mean=interaction_mean,
            std=interaction_std,
            rms=interaction_rms,
            count=interaction_tensor.shape[0],
        )
    interaction_stats: Dict[str, object] = {
        "count": int(model.interaction_stats_count.detach().cpu().item()),
        "pair_names": [
            f"{model.encoder_names[i]}__{model.encoder_names[j]}"
            for i, j in model.interaction_pairs
        ],
        "mean": model.interaction_mean.detach().cpu().tolist(),
        "std": model.interaction_std.detach().cpu().tolist(),
        "rms": model.interaction_rms.detach().cpu().tolist(),
        "scaling": "signed_rms_without_centering",
    }
    model.train(was_training)
    return model.router.get_score_stats(), interaction_stats


def next_stage2_beacon_weight(
    current_weight: float,
    initial_weight: float,
    train_cls_loss: float,
    train_beacon_loss: float,
    adaptive: bool,
    max_loss_ratio: float,
    eps: float = 1e-12,
) -> float:
    """Return the next epoch's monotonic train-only Beacon weight.

    The cap uses detached epoch-average losses:
        lambda_next <= max_loss_ratio * L_cls / (L_beacon + eps)
    so the next weighted Beacon term cannot dominate if loss scales remain
    comparable. The weight never increases after it has decayed.
    """
    current = max(float(current_weight), 0.0)
    initial = max(float(initial_weight), 0.0)
    if not adaptive or current == 0.0 or initial == 0.0:
        return min(current, initial)
    cls_loss = float(train_cls_loss)
    beacon_loss = float(train_beacon_loss)
    if not math.isfinite(cls_loss) or not math.isfinite(beacon_loss):
        return min(current, initial)
    if beacon_loss <= eps:
        return min(current, initial)
    ratio_cap = float(max_loss_ratio) * max(cls_loss, 0.0) / (max(beacon_loss, 0.0) + eps)
    return max(0.0, min(current, initial, ratio_cap))


def train_stage2_epoch(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    optimizer,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    beacon: torch.Tensor,
    beacon_constraint_weight: float,
    replacement_strategy: str,
    gaussian_std_scale: float,
    grad_clip: float,
    weight_averager: ExponentialMovingAverage | None = None,
) -> Tuple[float, float, float]:
    projection_is_trainable = any(param.requires_grad for param in model.projection_heads.parameters())
    model.train()
    if not projection_is_trainable:
        model.projection_heads.eval()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    total_cls_loss = 0.0
    total_beacon_loss = 0.0

    for idx in indices:
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        (
            logits,
            _attn,
            _routed,
            projected,
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
            compute_interactions=True,
        )
        cls_loss = criterion(logits, label_t)
        if beacon_constraint_weight > 0:
            beacon_loss, _ = model.beacon_constraint_loss(projected=projected, beacon=beacon)
        else:
            beacon_loss = cls_loss.new_tensor(0.0)
        loss = cls_loss + float(beacon_constraint_weight) * beacon_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if weight_averager is not None:
            weight_averager.update(model)
        total_loss += float(loss.detach().cpu())
        total_cls_loss += float(cls_loss.detach().cpu())
        total_beacon_loss += float(beacon_loss.detach().cpu())

    denom = max(len(dataset), 1)
    return total_loss / denom, total_cls_loss / denom, total_beacon_loss / denom


@torch.no_grad()
def evaluate_stage2(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    beacon: torch.Tensor | None,
    replacement_strategy: str,
    gaussian_std_scale: float,
    decision_threshold: float,
    fused_output_dir: Path | None = None,
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
            projected,
            attr,
            single_attr,
            interactions,
            base_logits,
            pair_residual,
            scaled_interactions,
        ) = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            return_attribution_logits=True,
        )
        beacon_sims: Dict[str, torch.Tensor] = {}
        if beacon is not None:
            _, beacon_sims = model.beacon_constraint_loss(projected=projected, beacon=beacon)
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        base_prob = float(torch.softmax(base_logits, dim=1)[0, 1].detach().cpu())
        pair_rms = float(pair_residual.detach().float().square().mean().sqrt().cpu())
        attribution_rms = float(routed.fused.detach().float().square().mean().sqrt().cpu())
        pair_relative_rms = pair_rms / max(attribution_rms, 1e-8)
        labels.append(int(label))
        probs.append(prob)
        rows.append({
            "slide_id": slide_id,
            "label": int(label),
            "prob_class1": prob,
            "base_prob_class1": base_prob,
            "prob_delta": prob - base_prob,
            "pair_residual_rms": pair_rms,
            "attribution_fused_rms": attribution_rms,
            "pair_relative_rms": pair_relative_rms,
        })

        weights = routed.weights.detach().cpu().reshape(-1).numpy()
        attr_np = attr.detach().cpu().reshape(-1).numpy()
        single_np = single_attr.detach().cpu().reshape(-1).numpy()
        interaction_np = interactions.detach().cpu().numpy()
        scaled_interaction_np = scaled_interactions.detach().cpu().reshape(-1).numpy()
        c_range = float(routed.attribution_range.detach().cpu().reshape(-1).item())
        tau = float(routed.tau.detach().cpu().reshape(-1).item())
        for encoder, weight, attr_value, single_value in zip(
            routed.encoder_names, weights, attr_np, single_np
        ):
            weight_rows.append({
                "slide_id": slide_id,
                "encoder": encoder,
                "weight": float(weight),
                "attribution": float(single_value),
                "routing_score": float(attr_value),
                "c_range": c_range,
                "tau": tau,
                "beacon_similarity": (
                    float(beacon_sims[encoder].detach().cpu())
                    if encoder in beacon_sims
                    else np.nan
                ),
            })
        for pair_index, (i, j) in enumerate(model.interaction_pairs):
            left = routed.encoder_names[i]
            right = routed.encoder_names[j]
            interaction_rows.append({
                "slide_id": slide_id,
                "encoder_i": left,
                "encoder_j": right,
                "interaction": float(interaction_np[i, j]),
                "scaled_interaction": float(scaled_interaction_np[pair_index]),
            })

        if fused_output_dir is not None:
            fused = (routed.fused + pair_residual).detach().cpu().numpy().astype(np.float32)
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
    stage1_train_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.stage1_max_patches,
        training=True,
    )
    stage1_val_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.stage1_eval_max_patches,
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
        routing_temperature=args.routing_temperature,
        routing_logit_scale=args.routing_logit_scale,
        attribution_target=args.attribution_target,
        interaction_pair_beta=args.interaction_pair_beta,
        interaction_rms_clip=args.interaction_rms_clip,
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
    inner_stage1_train_ds = Subset(stage1_train_ds, inner_train_indices)
    inner_stage1_val_ds = Subset(stage1_val_ds, inner_val_indices)
    print(
        f"Fold {fold}: outer_train={len(train_ds)}, outer_test={len(val_ds)}, "
        f"inner_train={len(inner_train_ds)}, inner_val={len(inner_val_ds)}"
    )
    if args.stage1_epochs <= 0 and args.freeze_projection_stage2:
        raise ValueError("--freeze-projection-stage2 requires --stage1-epochs > 0 so the ProjectionHead is not frozen randomly.")
    if args.stage1_epochs > 0 and args.stage1_beacon_mode == "none" and args.stage1_consistency_weight <= 0:
        raise ValueError("Stage1 would have no loss. Set stage1_consistency_weight > 0 or stage1_beacon_mode: epoch.")

    stage1_path = fold_dir / "best_stage1_projection.pt"
    if args.stage1_epochs > 0:
        set_projection_trainable(model, True)
        optimizer1 = torch.optim.AdamW(
            model.projection_heads.parameters(),
            lr=args.lr_stage1,
            weight_decay=args.weight_decay,
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1,
            T_max=max(args.stage1_epochs, 1),
            eta_min=args.lr_stage1 * 0.01,
        )
        best_stage1_loss = math.inf
        stage1_no_improve = 0

        print(
            f"Fold {fold} | Stage1 geometry-only ProjectionHead pretrain | "
            f"epochs={args.stage1_epochs} | beacon_mode={args.stage1_beacon_mode} | "
            f"freeze_stage2={args.freeze_projection_stage2}"
        )
        for epoch in range(1, args.stage1_epochs + 1):
            stage1_beacon = None
            if args.stage1_beacon_mode == "epoch" and args.beacon_constraint_weight > 0:
                stage1_beacon, stage1_beacon_summary, _, _ = build_beacon_and_baselines(
                    model,
                    baseline_ds,
                    device,
                    args.target_dim,
                )
            train_loss, train_beacon_loss, train_consistency_loss = train_stage1_epoch(
                model=model,
                dataset=inner_stage1_train_ds,
                optimizer=optimizer1,
                device=device,
                beacon=stage1_beacon,
                beacon_constraint_weight=args.beacon_constraint_weight,
                consistency_weight=args.stage1_consistency_weight,
                grad_clip=args.grad_clip,
            )
            val_loss, val_beacon_loss, val_consistency_loss = evaluate_stage1_geometry(
                model=model,
                dataset=inner_stage1_val_ds,
                device=device,
                beacon=stage1_beacon,
                beacon_constraint_weight=args.beacon_constraint_weight,
                consistency_weight=args.stage1_consistency_weight,
            )
            scheduler1.step()

            stage1_log = (
                f"Fold {fold} | Stage1 | Epoch {epoch:03d}/{args.stage1_epochs} | "
                f"train_loss={train_loss:.4f} | consistency={train_consistency_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_consistency={val_consistency_loss:.4f}"
            )
            if args.stage1_beacon_mode == "epoch":
                stage1_log += f" | beacon={train_beacon_loss:.4f} | val_beacon={val_beacon_loss:.4f}"
            print(stage1_log)

            improved = np.isfinite(val_loss) and val_loss < best_stage1_loss
            if improved:
                best_stage1_loss = val_loss
                stage1_no_improve = 0
                metrics_payload = {
                    "val_loss": float(val_loss),
                    "val_consistency_loss": float(val_consistency_loss),
                }
                if args.stage1_beacon_mode == "epoch":
                    metrics_payload["val_beacon_loss"] = float(val_beacon_loss)
                stage1_payload = {
                    "epoch": int(epoch),
                    "state_dict": model.state_dict(),
                    "projection_heads": model.projection_heads.state_dict(),
                    "input_dims": input_dims,
                    "stage": "stage1_projection_pretrain",
                    "metrics": metrics_payload,
                    "beacon_constraint_weight": float(args.beacon_constraint_weight),
                    "stage1_beacon_mode": args.stage1_beacon_mode,
                    "stage1_consistency_weight": float(args.stage1_consistency_weight),
                    "policy": (
                        "ProjectionHead pretrained with cross-encoder consistency; "
                        "epoch-level Beacon geometry enabled."
                        if args.stage1_beacon_mode == "epoch"
                        else "ProjectionHead pretrained with cross-encoder consistency only; Beacon built once after freeze."
                    ),
                }
                torch.save(stage1_payload, stage1_path)
                if args.stage1_beacon_mode == "epoch" and args.beacon_constraint_weight > 0:
                    stage1_beacon_summary.to_csv(fold_dir / "stage1_beacon_summary.csv", index=False, encoding="utf-8-sig")
                stage1_metric_row = {
                    "stage": "stage1",
                    "epoch": int(epoch),
                    "val_loss": float(val_loss),
                    "val_consistency_loss": float(val_consistency_loss),
                }
                if args.stage1_beacon_mode == "epoch":
                    stage1_metric_row["val_beacon_loss"] = float(val_beacon_loss)
                pd.DataFrame([stage1_metric_row]).to_csv(
                    fold_dir / "stage1_geometry_metrics.csv",
                    index=False,
                    encoding="utf-8-sig",
                    float_format="%.6f",
                )
            else:
                stage1_no_improve += 1
                if stage1_no_improve >= args.stage1_patience:
                    print(f"Fold {fold}: Stage1 early stopping at epoch {epoch}. Best Stage1 val loss={best_stage1_loss:.4f}")
                    break

        if stage1_path.exists():
            load_model_state(stage1_path, model, device)

    if args.freeze_projection_stage2:
        set_projection_trainable(model, False)
        if args.stage2_warm_start_classifier:
            print("[Warning] Stage1 is projection-only, so there is no pretrained classifier to warm-start.")
        reset_stage2_classifier(model, args, device)
        stage2_beacon_weight = 0.0
        print("Stage2: ProjectionHead frozen; Beacon is static analysis/geometry prior, not a trainable loss term.")
    else:
        set_projection_trainable(model, True)
        stage2_beacon_weight = float(args.beacon_constraint_weight)
        print("Stage2: ProjectionHead remains trainable; Beacon constraint stays active in Stage2.")

    initial_stage2_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    optimizer2 = build_stage2_optimizer(model, args)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1), eta_min=args.lr_stage2 * 0.01)
    best_stage2_auc = -math.inf
    best_stage2_path = fold_dir / "best_gme_model.pt"
    best_beacon_path = fold_dir / "static_beacon_and_baselines.pt"
    best_stage2_metrics: EvalResult | None = None
    no_improve = 0
    stage2_history: List[Dict[str, object]] = []
    ema = (
        ExponentialMovingAverage(args.ema_decay)
        if args.weight_averaging == "ema"
        else None
    )
    print(
        f"Stage2 evaluation weights: {args.weight_averaging}"
        + (
            f" | decay={args.ema_decay:g} | start_epoch={args.ema_start_epoch}"
            if ema is not None
            else ""
        )
    )
    current_stage2_beacon_weight = float(stage2_beacon_weight)
    print(
        f"Stage2 Beacon weighting: initial={current_stage2_beacon_weight:g} | "
        f"adaptive={bool(args.adaptive_beacon_weight)} | "
        f"max_weighted_loss_ratio={args.beacon_loss_ratio:g}"
    )

    # Initialize train-only offline statistics once. After every epoch they are
    # refreshed for the updated model, used for validation, and carried into
    # the next epoch instead of being rebuilt again at the same checkpoint.
    beacon, beacon_summary, baselines, baseline_summary = build_beacon_and_baselines(
        model,
        inner_baseline_ds,
        device,
        args.target_dim,
    )
    score_stats, interaction_stats = update_train_offline_stats(
        model=model,
        dataset=inner_score_stats_ds,
        device=device,
        baselines=baselines,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
    )

    for epoch in range(1, args.stage2_epochs + 1):
        if ema is not None and not ema.initialized and epoch >= args.ema_start_epoch:
            ema.initialize(model)
        epoch_beacon_weight = float(current_stage2_beacon_weight)
        train_loss, train_cls_loss, train_beacon_loss = train_stage2_epoch(
            model=model,
            dataset=inner_train_ds,
            optimizer=optimizer2,
            device=device,
            baselines=baselines,
            beacon=beacon,
            beacon_constraint_weight=epoch_beacon_weight,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            grad_clip=args.grad_clip,
            weight_averager=ema if ema is not None and ema.initialized else None,
        )
        next_beacon_weight = next_stage2_beacon_weight(
            current_weight=epoch_beacon_weight,
            initial_weight=stage2_beacon_weight,
            train_cls_loss=train_cls_loss,
            train_beacon_loss=train_beacon_loss,
            adaptive=bool(args.adaptive_beacon_weight),
            max_loss_ratio=float(args.beacon_loss_ratio),
        )
        current_stage2_beacon_weight = next_beacon_weight
        if not args.freeze_projection_stage2:
            beacon, beacon_summary, baselines, baseline_summary = build_beacon_and_baselines(
                model,
                inner_baseline_ds,
                device,
                args.target_dim,
            )
        score_stats, interaction_stats = update_train_offline_stats(
            model=model,
            dataset=inner_score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
        scheduler2.step()
        use_ema_for_eval = ema is not None and ema.initialized
        evaluation_context = ema.average_parameters(model) if use_ema_for_eval else nullcontext()
        with evaluation_context:
            evaluation_weight_source = "ema" if use_ema_for_eval else "raw"
            if use_ema_for_eval:
                eval_beacon, eval_beacon_summary, eval_baselines, eval_baseline_summary = build_beacon_and_baselines(
                    model,
                    inner_baseline_ds,
                    device,
                    args.target_dim,
                )
                eval_score_stats, eval_interaction_stats = update_train_offline_stats(
                    model=model,
                    dataset=inner_score_stats_ds,
                    device=device,
                    baselines=eval_baselines,
                    replacement_strategy=args.replacement_strategy,
                    gaussian_std_scale=args.gaussian_std_scale,
                )
            else:
                eval_beacon = beacon
                eval_beacon_summary = beacon_summary
                eval_baselines = baselines
                eval_baseline_summary = baseline_summary
                eval_score_stats = score_stats
                eval_interaction_stats = interaction_stats

            val_metrics, val_pred, val_weights, val_interactions = evaluate_stage2(
                model=model,
                dataset=inner_val_ds,
                device=device,
                baselines=eval_baselines,
                beacon=eval_beacon,
                replacement_strategy=args.replacement_strategy,
                gaussian_std_scale=args.gaussian_std_scale,
                decision_threshold=args.threshold,
            )
            base_val_metrics = compute_metrics(
                val_pred["label"],
                val_pred["base_prob_class1"],
                decision_threshold=args.threshold,
            )
            mean_pair_relative_rms = float(val_pred["pair_relative_rms"].mean())
            mean_abs_prob_delta = float(val_pred["prob_delta"].abs().mean())
            stage2_history.append({
                "epoch": int(epoch),
                "evaluation_weight_source": evaluation_weight_source,
                "ema_updates": int(ema.num_updates) if ema is not None else 0,
                "train_loss": float(train_loss),
                "train_cls_loss": float(train_cls_loss),
                "train_beacon_loss": float(train_beacon_loss),
                "beacon_weight": float(epoch_beacon_weight),
                "weighted_beacon_loss": float(epoch_beacon_weight * train_beacon_loss),
                "next_beacon_weight": float(next_beacon_weight),
                "inner_auc": float(val_metrics.auc),
                "inner_auprc": float(val_metrics.auprc),
                "inner_f1": float(val_metrics.f1),
                "attribution_only_auc": float(base_val_metrics.auc),
                "attribution_only_auprc": float(base_val_metrics.auprc),
                "attribution_only_f1": float(base_val_metrics.f1),
                "mean_pair_relative_rms": mean_pair_relative_rms,
                "mean_abs_prob_delta": mean_abs_prob_delta,
                "max_abs_prob_delta": float(val_pred["prob_delta"].abs().max()),
            })
            pd.DataFrame(stage2_history).to_csv(
                fold_dir / "stage2_training_history.csv",
                index=False,
                encoding="utf-8-sig",
                float_format="%.6f",
            )
            print(
                f"Fold {fold} | Epoch {epoch:03d}/{args.stage2_epochs} | "
                f"eval={evaluation_weight_source.upper()} | "
                f"loss={train_loss:.4f} | cls={train_cls_loss:.4f} | beacon={train_beacon_loss:.4f} | "
                f"inner_AUC={val_metrics.auc:.4f} | inner_AUPRC={val_metrics.auprc:.4f} | "
                f"attr_AUC={base_val_metrics.auc:.4f} | "
                f"ACC@{args.threshold:g}={val_metrics.accuracy:.4f} | F1@{args.threshold:g}={val_metrics.f1:.4f} | "
                f"pair_rel_RMS={mean_pair_relative_rms:.4f} | |delta_p|={mean_abs_prob_delta:.6f} | "
                f"I_min={eval_score_stats['attribution_min']:.4f} | I_max={eval_score_stats['attribution_max']:.4f}"
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
                        "beacon_path": str(best_beacon_path),
                        "beacon_constraint_weight": float(args.beacon_constraint_weight),
                        "stage2_beacon_constraint_weight": float(epoch_beacon_weight),
                        "initial_stage2_beacon_constraint_weight": float(stage2_beacon_weight),
                        "next_stage2_beacon_constraint_weight": float(next_beacon_weight),
                        "adaptive_beacon_weight": bool(args.adaptive_beacon_weight),
                        "beacon_loss_ratio": float(args.beacon_loss_ratio),
                        "freeze_projection_stage2": bool(args.freeze_projection_stage2),
                        "stage1_path": str(stage1_path) if stage1_path.exists() else None,
                        "routing_score_stats": eval_score_stats,
                        "interaction_stats": eval_interaction_stats,
                        "interaction_pair_beta": float(args.interaction_pair_beta),
                        "interaction_scaling": "signed_train_only_rms_without_centering",
                        "weight_averaging": args.weight_averaging,
                        "evaluation_weight_source": evaluation_weight_source,
                        "ema_decay": float(args.ema_decay),
                        "ema_start_epoch": int(args.ema_start_epoch),
                        "ema_num_updates": int(ema.num_updates) if ema is not None else 0,
                        "baseline_policy": (
                            "Train-only replacement baselines built once from frozen Stage1 projection."
                            if args.freeze_projection_stage2
                            else "Train-only replacement baselines rebuilt from current projection after each epoch."
                        ),
                        "beacon_policy": (
                            "Train-only static Beacon anchors Stage1 projection geometry; Stage2 projection is frozen."
                            if args.freeze_projection_stage2
                            else "Train-only static Beacon used only as a detached global semantic prior constraint."
                        ),
                    },
                )
                baseline_payload = {
                    "fold": int(fold),
                    "baselines": {
                        name: {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in stats.items()}
                        for name, stats in eval_baselines.items()
                    },
                    "input_dims": input_dims,
                    "encoder_names": model.encoder_names,
                    "weight_source": evaluation_weight_source,
                    "policy": (
                        "Train-only replacement baselines built once from frozen Stage1 projection."
                        if args.freeze_projection_stage2
                        else "Train-only replacement baselines rebuilt from the current end-to-end projection space."
                    ),
                    "routing_score_stats": eval_score_stats,
                }
                torch.save(baseline_payload, fold_dir / "replacement_baselines.pt")
                torch.save(
                    {
                        **baseline_payload,
                        "beacon": eval_beacon.detach().cpu(),
                        "beacon_constraint_weight": float(args.beacon_constraint_weight),
                        "stage2_beacon_constraint_weight": float(epoch_beacon_weight),
                        "initial_stage2_beacon_constraint_weight": float(stage2_beacon_weight),
                        "next_stage2_beacon_constraint_weight": float(next_beacon_weight),
                        "adaptive_beacon_weight": bool(args.adaptive_beacon_weight),
                        "beacon_loss_ratio": float(args.beacon_loss_ratio),
                        "freeze_projection_stage2": bool(args.freeze_projection_stage2),
                        "stage1_path": str(stage1_path) if stage1_path.exists() else None,
                        "beacon_policy": (
                            "Train-only static Beacon anchors Stage1 projection geometry; Stage2 projection is frozen."
                            if args.freeze_projection_stage2
                            else "Train-only static Beacon used only as a detached global semantic prior constraint."
                        ),
                    },
                    best_beacon_path,
                )
                eval_baseline_summary.to_csv(fold_dir / "replacement_baseline_summary.csv", index=False, encoding="utf-8-sig")
                eval_beacon_summary.to_csv(fold_dir / "static_beacon_summary.csv", index=False, encoding="utf-8-sig")
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
    model.load_state_dict(initial_stage2_state)
    seed_everything(args.seed + fold)
    retrain_optimizer = build_stage2_optimizer(model, args)
    retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        retrain_optimizer,
        T_max=max(args.stage2_epochs, 1),
        eta_min=args.lr_stage2 * 0.01,
    )
    retrain_ema = (
        ExponentialMovingAverage(args.ema_decay)
        if args.weight_averaging == "ema"
        else None
    )
    retrain_beacon_weight = float(stage2_beacon_weight)
    last_retrain_beacon_weight = float(stage2_beacon_weight)
    beacon, _, baselines, _ = build_beacon_and_baselines(
        model, baseline_ds, device, args.target_dim
    )
    _, final_interaction_stats = update_train_offline_stats(
        model=model,
        dataset=score_stats_ds,
        device=device,
        baselines=baselines,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
    )
    for retrain_epoch in range(1, selected_epoch + 1):
        if (
            retrain_ema is not None
            and not retrain_ema.initialized
            and retrain_epoch >= args.ema_start_epoch
        ):
            retrain_ema.initialize(model)
        last_retrain_beacon_weight = float(retrain_beacon_weight)
        _, retrain_cls_loss, retrain_beacon_loss = train_stage2_epoch(
            model=model,
            dataset=train_ds,
            optimizer=retrain_optimizer,
            device=device,
            baselines=baselines,
            beacon=beacon,
            beacon_constraint_weight=last_retrain_beacon_weight,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            grad_clip=args.grad_clip,
            weight_averager=(
                retrain_ema
                if retrain_ema is not None and retrain_ema.initialized
                else None
            ),
        )
        retrain_beacon_weight = next_stage2_beacon_weight(
            current_weight=last_retrain_beacon_weight,
            initial_weight=stage2_beacon_weight,
            train_cls_loss=retrain_cls_loss,
            train_beacon_loss=retrain_beacon_loss,
            adaptive=bool(args.adaptive_beacon_weight),
            max_loss_ratio=float(args.beacon_loss_ratio),
        )
        retrain_scheduler.step()
        if not args.freeze_projection_stage2:
            beacon, _, baselines, _ = build_beacon_and_baselines(
                model, baseline_ds, device, args.target_dim
            )
        _, final_interaction_stats = update_train_offline_stats(
            model=model,
            dataset=score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
    final_weight_source = "raw"
    if retrain_ema is not None and retrain_ema.initialized:
        retrain_ema.copy_to(model)
        final_weight_source = "ema"
        # Beacon, replacement baselines, and routing statistics must match the
        # averaged ProjectionHead/classifier used for final evaluation.
        beacon, _, baselines, _ = build_beacon_and_baselines(
            model, baseline_ds, device, args.target_dim
        )
        _, final_interaction_stats = update_train_offline_stats(
            model=model,
            dataset=score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
    print(
        f"Fold {fold}: selected inner epoch={selected_epoch}; retrained on all outer-train slides; "
        f"final weights={final_weight_source.upper()}; "
        f"last lambda_B={last_retrain_beacon_weight:.6f}."
    )

    decision_threshold = float(args.threshold)
    print(f"Fold {fold}: fixed decision threshold={decision_threshold:.6f}")
    with open(fold_dir / "decision_threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(decision_threshold),
                "source": "fixed_config",
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
        beacon=beacon,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
        decision_threshold=decision_threshold,
        fused_output_dir=fused_dir,
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
            "stage": "stage2_retrained_outer_train",
            "selection_policy": "inner_validation_AUC_then_retrain_on_outer_train",
            "selected_inner_epoch": selected_epoch,
            "weight_averaging": args.weight_averaging,
            "evaluation_weight_source": final_weight_source,
            "ema_decay": float(args.ema_decay),
            "ema_start_epoch": int(args.ema_start_epoch),
            "ema_num_updates": int(retrain_ema.num_updates) if retrain_ema is not None else 0,
            "adaptive_beacon_weight": bool(args.adaptive_beacon_weight),
            "beacon_loss_ratio": float(args.beacon_loss_ratio),
            "initial_stage2_beacon_constraint_weight": float(stage2_beacon_weight),
            "last_stage2_beacon_constraint_weight": float(last_retrain_beacon_weight),
            "next_stage2_beacon_constraint_weight": float(retrain_beacon_weight),
            "interaction_stats": final_interaction_stats,
            "interaction_pair_beta": float(args.interaction_pair_beta),
            "interaction_scaling": "signed_train_only_rms_without_centering",
        },
    )

    with open(fold_dir / "routing_stats.json", "w", encoding="utf-8") as f:
        json.dump(model.router.get_routing_stats(), f, indent=2)
    with open(fold_dir / "weight_averaging.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "configured_mode": args.weight_averaging,
                "final_weight_source": final_weight_source,
                "ema_decay": float(args.ema_decay),
                "ema_start_epoch": int(args.ema_start_epoch),
                "ema_num_updates": int(retrain_ema.num_updates) if retrain_ema is not None else 0,
                "selected_inner_epoch": int(selected_epoch),
                "offline_statistics_rebuilt_after_ema": final_weight_source == "ema",
            },
            f,
            indent=2,
        )
    with open(fold_dir / "adaptive_beacon.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "enabled": bool(args.adaptive_beacon_weight),
                "initial_weight": float(stage2_beacon_weight),
                "max_weighted_loss_ratio": float(args.beacon_loss_ratio),
                "selected_inner_epoch": int(selected_epoch),
                "outer_retrain_last_epoch_weight": float(last_retrain_beacon_weight),
                "outer_retrain_next_epoch_weight": float(retrain_beacon_weight),
                "schedule": "monotonic_train_only_epoch_loss_ratio_cap",
                "formula": "lambda_next=min(lambda_current,lambda_initial,rho*L_cls/(L_beacon+eps))",
            },
            f,
            indent=2,
        )
    with open(fold_dir / "interaction_pair_branch.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **final_interaction_stats,
                "beta": float(args.interaction_pair_beta),
                "rms_clip": float(args.interaction_rms_clip),
                "normalization": "divide_by_train_only_pair_rms_without_centering",
                "all_pairs_used": True,
                "branch": "parameter_free_hadamard_pairs_with_feature_gate",
                "gate_zero_initialized": True,
                "gate_parameter_count": int(model.interaction_gate.numel()),
                "gate_l2": float(model.interaction_gate.detach().float().norm().cpu()),
                "gate_tanh_abs_mean": float(
                    torch.tanh(model.interaction_gate.detach()).abs().mean().cpu()
                ),
                "gate_tanh_abs_max": float(
                    torch.tanh(model.interaction_gate.detach()).abs().max().cpu()
                ),
            },
            f,
            indent=2,
        )

    row = {
        "fold": int(fold),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "weight_source": final_weight_source,
        "ema_num_updates": int(retrain_ema.num_updates) if retrain_ema is not None else 0,
        "adaptive_beacon_weight": bool(args.adaptive_beacon_weight),
        "final_beacon_weight": float(last_retrain_beacon_weight),
        **asdict(final_metrics),
        **efficiency,
    }
    return row


def main() -> None:
    args = parse_args()
    device = resolve_device(args)
    seed_everything(args.seed)

    manifest_path = ensure_manifest(args)
    args.manifest = manifest_path
    manifest = pd.read_csv(manifest_path)
    clinical_df, _, _ = load_uvm_data(args.clinical_path, args.label_col)
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
