"""End-to-end GME/CGME middle-fusion training for UVM D3/M3 classification.

This script is intentionally self-contained for server runs:

1. Read a middle-fusion manifest of multi-encoder h5 embeddings.
2. For each CV fold, train ProjectionHead with a temporary mean-fusion ABMIL head.
3. Freeze the trained ProjectionHead and build train-only static Beacon/baselines.
4. Reinitialize the main router + ABMIL, then train GME with Beacon similarity
   and intervention attribution.
5. Report AUC, AUPRC, sensitivity, specificity, and save checkpoints/artifacts.

The intervention score is computed as the prediction drop after replacing one
encoder's projected embeddings, using similarity-only routing as the attribution
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_DIR, CODE_DIR / "architecture", CODE_DIR / "modules"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture.abmil_cls import ABMIL_Cls
from architecture.projection_head import MultiEncoderProjectionHead
from data_utils.cls_dataset import load_uvm_data
from modules.attribution import EncoderBaselineAccumulator, replace_encoder_embedding
from modules.beacon import BeaconAccumulator, beacon_similarity, infer_input_dims, l2_normalize
from modules.routing import DualConsistencyRouter


DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
FEATURE_KEYS = ("feats", "features")
PATH_ARGS = ("manifest", "manifest_dir", "output_dir")


@dataclass
class EvalResult:
    auc: float
    auprc: float
    sensitivity: float
    specificity: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    tn: int
    fp: int
    fn: int
    tp: int


def load_config_file(config_path: Path | None) -> Dict[str, object]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                f"Reading YAML config requires PyYAML. Install it or use JSON config: {config_path}"
            ) from exc
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping/dict, got {type(data)} from {config_path}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file. CLI args override config.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command end-to-end GME middle-fusion training on multi-encoder h5 embeddings."
    )
    add_config_argument(parser)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-manifest", action="store_true", help="Build the manifest before training.")
    parser.add_argument("--feat-base", default=r"L:\20x_256px_0px_overlap")
    parser.add_argument("--feature-dirs", nargs="+", default=["features_hoptimus1", "features_virchow", "features_hoptimus0"])
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--clinical-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default="gme")
    parser.add_argument("--label-col", default="d3m3")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Fold ids to run. Default: all folds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])

    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-attn", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.25)
    parser.add_argument("--n-classes", type=int, default=2)

    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument(
        "--stage2-warm-start-classifier",
        action="store_true",
        help="Reuse the temporary Stage-1 ABMIL/router weights in Stage 2. Default: reuse only ProjectionHead.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr-stage1", type=float, default=1e-4)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--theta-init", type=float, default=0.0)
    parser.add_argument("--gamma-init", type=float, default=0.0)
    parser.add_argument("--beacon-temperature", type=float, default=1.0)
    parser.add_argument("--use-cosine-similarity", action="store_true")
    parser.add_argument("--replacement-strategy", choices=["mean", "zero", "gaussian"], default="mean")
    parser.add_argument("--gaussian-std-scale", type=float, default=1.0)

    parser.add_argument(
        "--max-patches",
        type=int,
        default=0,
        help="Optional patch subsampling per WSI. 0 means use all patches.",
    )
    parser.add_argument(
        "--eval-max-patches",
        type=int,
        default=0,
        help="Optional deterministic patch cap during validation. 0 means use all patches.",
    )
    parser.add_argument(
        "--beacon-max-patches",
        type=int,
        default=0,
        help="Optional patch cap when building static Beacon/baselines. 0 means use all train patches.",
    )
    parser.add_argument("--save-fused-h5", action="store_true", help="Export validation fused h5 embeddings.")
    parser.add_argument(
        "--skip-routing-lambda-analysis",
        action="store_true",
        help="Skip automatic routing lambda analysis after training.",
    )
    parser.add_argument(
        "--strict-routing-lambda-analysis",
        action="store_true",
        help="Raise an error if post-training routing lambda analysis fails.",
    )
    config_args, remaining = parser.parse_known_args()
    config = load_config_file(config_args.config)
    if config:
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid_dests)
        if unknown:
            raise ValueError(f"Unknown config keys in {config_args.config}: {unknown}")
        parser.set_defaults(**config)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    for name in PATH_ARGS:
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    if args.clinical_path is None:
        raise ValueError("--clinical-path is required, either in CLI or config.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_manifest(args: argparse.Namespace) -> Path:
    """Build the middle-fusion manifest when requested or missing."""
    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.build_manifest:
        return manifest_path

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
    ]
    print("\nBuilding middle-fusion manifest:")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    built_manifest = output_dir / "middle_fusion_manifest.csv"
    if not built_manifest.exists():
        raise FileNotFoundError(f"Manifest builder finished but did not create {built_manifest}")
    return built_manifest


def get_dataset_key(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as f:
        for key in FEATURE_KEYS:
            if key in f:
                return key
        candidates = [key for key in f.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(f.keys())}")
        return candidates[0]


def read_h5_features(h5_path: Path, dataset_key: str | None = None) -> np.ndarray:
    key = dataset_key or get_dataset_key(h5_path)
    with h5py.File(h5_path, "r") as f:
        features = np.asarray(f[key][:], dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected [N, D], got {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def subset_patch_indices(n_patches: int, max_patches: int, training: bool) -> np.ndarray | None:
    if max_patches <= 0 or n_patches <= max_patches:
        return None
    if training:
        return np.sort(np.random.choice(n_patches, size=max_patches, replace=False))
    return np.arange(max_patches)


class MultiEncoderSlideDataset:
    """Slide-level dataset that returns a dict of raw features per encoder."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        fold: int,
        split: str,
        clinical_df: pd.DataFrame,
        label_col: str = "d3m3",
        max_patches: int = 0,
        training: bool = True,
    ):
        rows = manifest[(manifest["fold"].astype(int) == int(fold)) & (manifest["split"].astype(str) == split)].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows found for fold={fold}, split={split}.")

        self.label_col = label_col
        self.max_patches = int(max_patches)
        self.training = bool(training)
        self.clinical = clinical_df.set_index("slide_id")
        self.encoder_names = sorted(rows["feature_dir"].astype(str).unique().tolist())
        self.slide_rows: Dict[str, pd.DataFrame] = {}

        for slide_id, group in rows.groupby("slide_id"):
            sid = str(slide_id)
            if sid not in self.clinical.index:
                continue
            if pd.isna(self.clinical.loc[sid, label_col]):
                continue
            present = set(group["feature_dir"].astype(str))
            if set(self.encoder_names).issubset(present):
                self.slide_rows[sid] = group.copy()

        self.slide_ids = sorted(self.slide_rows.keys())
        if not self.slide_ids:
            raise RuntimeError(f"Fold {fold}, split={split}: no usable slides after clinical/feature filtering.")

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int, str]:
        slide_id = self.slide_ids[int(idx)]
        rows = self.slide_rows[slide_id]
        features: Dict[str, np.ndarray] = {}

        for _, row in rows.iterrows():
            encoder = str(row["feature_dir"])
            h5_path = Path(row["h5_path"])
            dataset_key = str(row["dataset_key"]) if pd.notna(row.get("dataset_key", None)) else None
            features[encoder] = read_h5_features(h5_path, dataset_key=dataset_key)

        min_patches = min(arr.shape[0] for arr in features.values())
        patch_idx = subset_patch_indices(min_patches, self.max_patches, training=self.training)
        tensor_features: Dict[str, torch.Tensor] = {}
        for encoder in self.encoder_names:
            arr = features[encoder][:min_patches]
            if patch_idx is not None:
                arr = arr[patch_idx]
            tensor_features[encoder] = torch.from_numpy(arr.astype(np.float32))

        label = int(self.clinical.loc[slide_id, self.label_col])
        return tensor_features, label, slide_id


class GMEModel(nn.Module):
    """ProjectionHead + DualConsistencyRouter + ABMIL classifier."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        target_dim: int = 512,
        projection_dropout: float = 0.0,
        d_inner: int = 256,
        d_attn: int = 128,
        n_classes: int = 2,
        droprate: float = 0.25,
        score_temperature: float = 1.0,
        theta_init: float = 0.0,
        gamma_init: float = 0.0,
        beacon_temperature: float = 1.0,
        use_cosine_similarity: bool = False,
    ):
        super().__init__()
        self.encoder_names = sorted(input_dims.keys())
        ordered_dims = {name: int(input_dims[name]) for name in self.encoder_names}
        self.projection_heads = MultiEncoderProjectionHead(
            input_dims=ordered_dims,
            target_dim=target_dim,
            dropout=projection_dropout,
        )
        self.router = DualConsistencyRouter(
            score_temperature=score_temperature,
            theta_init=theta_init,
            gamma_init=gamma_init,
        )
        self.classifier = ABMIL_Cls(
            D_feat=target_dim,
            D_inner=d_inner,
            D_attn=d_attn,
            n_classes=n_classes,
            droprate=droprate,
        )
        self.target_dim = int(target_dim)
        self.beacon_temperature = float(beacon_temperature)
        self.use_cosine_similarity = bool(use_cosine_similarity)

    def project(self, raw_features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.projection_heads(raw_features)

    def mean_fuse(self, projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([projected[name] for name in self.encoder_names], dim=0)
        return stacked.mean(dim=0)

    def similarity_scores(self, projected: Mapping[str, torch.Tensor], beacon: torch.Tensor) -> torch.Tensor:
        scores = []
        for name in self.encoder_names:
            patch_scores = beacon_similarity(
                projected[name],
                beacon=beacon,
                temperature=self.beacon_temperature,
                use_cosine=self.use_cosine_similarity,
            )
            scores.append(patch_scores.mean())
        return torch.stack(scores, dim=0)

    def route_with_scores(
        self,
        projected: Mapping[str, torch.Tensor],
        attribution_scores: torch.Tensor,
        similarity_scores: torch.Tensor,
    ):
        return self.router(
            features_by_encoder=projected,
            attribution_scores=attribution_scores,
            similarity_scores=similarity_scores,
            encoder_names=self.encoder_names,
        )

    def forward_stage1(self, raw_features: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        projected = self.project(raw_features)
        fused = self.mean_fuse(projected)
        logits, attn = self.classifier(fused)
        return logits, attn, projected

    def forward_similarity_only(
        self,
        projected: Mapping[str, torch.Tensor],
        beacon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = self.similarity_scores(projected, beacon)
        attr = torch.zeros_like(sim)
        routed = self.route_with_scores(projected, attribution_scores=attr, similarity_scores=sim)
        logits, _ = self.classifier(routed.fused)
        return logits, sim

    def intervention_attribution(
        self,
        projected: Mapping[str, torch.Tensor],
        beacon: torch.Tensor,
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        class_index: int = 1,
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
    ) -> torch.Tensor:
        classifier_was_training = self.classifier.training
        router_was_training = self.router.training
        self.classifier.eval()
        self.router.eval()
        try:
            with torch.no_grad():
                full_logits, _ = self.forward_similarity_only(projected, beacon)
                full_score = torch.softmax(full_logits, dim=1)[0, class_index]
                scores = []
                for name in self.encoder_names:
                    replaced = replace_encoder_embedding(
                        projected,
                        encoder_name=name,
                        replacement_strategy=replacement_strategy,  # type: ignore[arg-type]
                        baselines=baselines,
                        gaussian_std_scale=gaussian_std_scale,
                    )
                    masked_logits, _ = self.forward_similarity_only(replaced, beacon)
                    masked_score = torch.softmax(masked_logits, dim=1)[0, class_index]
                    scores.append(full_score - masked_score)
        finally:
            self.classifier.train(classifier_was_training)
            self.router.train(router_was_training)
        return torch.stack(scores, dim=0)

    def forward_stage2(
        self,
        raw_features: Mapping[str, torch.Tensor],
        beacon: torch.Tensor,
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
    ):
        projected = self.project(raw_features)
        sim = self.similarity_scores(projected, beacon)
        attr = self.intervention_attribution(
            projected=projected,
            beacon=beacon,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
        )
        routed = self.route_with_scores(projected, attribution_scores=attr, similarity_scores=sim)
        logits, attn = self.classifier(routed.fused)
        return logits, attn, routed, projected, attr, sim


def move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.float().to(device) for name, tensor in features.items()}


def compute_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float) -> EvalResult:
    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    preds_np = (probs_np >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
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
        sensitivity=float(sensitivity),
        specificity=float(specificity),
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
) -> Tuple[torch.Tensor, Dict[str, Mapping[str, torch.Tensor]], pd.DataFrame]:
    model.eval()
    beacon_acc = BeaconAccumulator(target_dim=target_dim, device=device)
    baseline_acc = EncoderBaselineAccumulator()

    for idx in range(len(dataset)):
        raw_features, _, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = model.project(raw_features)
        for name in model.encoder_names:
            beacon_acc.update(name, projected[name])
            baseline_acc.update(name, projected[name])

    beacon = beacon_acc.compute(normalize_beacon=True).to(device)
    baselines = baseline_acc.compute()
    baselines = {
        name: {
            stat_name: stat.to(device) if torch.is_tensor(stat) else stat
            for stat_name, stat in stats.items()
        }
        for name, stats in baselines.items()
    }
    return beacon, baselines, beacon_acc.summary()


def train_stage1_epoch(model: GMEModel, dataset: MultiEncoderSlideDataset, optimizer, device: torch.device, grad_clip: float) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0

    for idx in indices:
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model.forward_stage1(raw_features)
        loss = criterion(logits, label_t)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(dataset), 1)


def train_stage2_epoch(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    optimizer,
    device: torch.device,
    beacon: torch.Tensor,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    grad_clip: float,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0

    for idx in indices:
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _, _, _, _, _ = model.forward_stage2(
            raw_features=raw_features,
            beacon=beacon,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
        )
        loss = criterion(logits, label_t)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(dataset), 1)


@torch.no_grad()
def evaluate_stage1(model: GMEModel, dataset: MultiEncoderSlideDataset, device: torch.device, threshold: float) -> Tuple[EvalResult, pd.DataFrame]:
    model.eval()
    labels, probs, rows = [], [], []
    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        logits, _, _ = model.forward_stage1(raw_features)
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        labels.append(int(label))
        probs.append(prob)
        rows.append({"slide_id": slide_id, "label": int(label), "prob_class1": prob})

    metrics = compute_metrics(labels, probs, threshold)
    pred_df = pd.DataFrame(rows)
    pred_df["pred"] = (pred_df["prob_class1"] >= threshold).astype(int)
    return metrics, pred_df


@torch.no_grad()
def evaluate_stage2(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    beacon: torch.Tensor,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    threshold: float,
    fused_output_dir: Path | None = None,
) -> Tuple[EvalResult, pd.DataFrame, pd.DataFrame]:
    model.eval()
    labels, probs, rows, weight_rows = [], [], [], []
    if fused_output_dir is not None:
        fused_output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        logits, _, routed, _, attr, sim = model.forward_stage2(
            raw_features=raw_features,
            beacon=beacon,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
        )
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        labels.append(int(label))
        probs.append(prob)
        rows.append({"slide_id": slide_id, "label": int(label), "prob_class1": prob})

        weights = routed.weights.detach().cpu().reshape(-1).numpy()
        attr_np = attr.detach().cpu().reshape(-1).numpy()
        sim_np = sim.detach().cpu().reshape(-1).numpy()
        for encoder, weight, attr_value, sim_value in zip(routed.encoder_names, weights, attr_np, sim_np):
            weight_rows.append({
                "slide_id": slide_id,
                "encoder": encoder,
                "weight": float(weight),
                "attribution": float(attr_value),
                "similarity": float(sim_value),
            })

        if fused_output_dir is not None:
            fused = routed.fused.detach().cpu().numpy().astype(np.float32)
            with h5py.File(fused_output_dir / f"{slide_id}.h5", "w") as f:
                f.create_dataset("features", data=fused, compression="gzip")

    metrics = compute_metrics(labels, probs, threshold)
    pred_df = pd.DataFrame(rows)
    pred_df["pred"] = (pred_df["prob_class1"] >= threshold).astype(int)
    return metrics, pred_df, pd.DataFrame(weight_rows)


def freeze_projection(model: GMEModel) -> None:
    model.projection_heads.eval()
    for param in model.projection_heads.parameters():
        param.requires_grad_(False)


def unfreeze_projection(model: GMEModel) -> None:
    for param in model.projection_heads.parameters():
        param.requires_grad_(True)


def reset_stage2_modules(model: GMEModel, args: argparse.Namespace, device: torch.device) -> None:
    """Start Stage 2 from trained projection only, with fresh router/classifier."""
    model.router = DualConsistencyRouter(
        score_temperature=args.score_temperature,
        theta_init=args.theta_init,
        gamma_init=args.gamma_init,
    ).to(device)
    model.classifier = ABMIL_Cls(
        D_feat=args.target_dim,
        D_inner=args.d_inner,
        D_attn=args.d_attn,
        n_classes=args.n_classes,
        droprate=args.droprate,
    ).to(device)


def save_fold_outputs(
    fold_dir: Path,
    stage_name: str,
    metrics: EvalResult,
    predictions: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{"stage": stage_name}, **asdict(metrics)}]).to_csv(
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


def summarize_metrics(fold_rows: List[Mapping[str, object]], output_dir: Path) -> None:
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    summary_rows = []
    for metric in ["auc", "auprc", "sensitivity", "specificity", "accuracy", "f1", "precision", "recall"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce")
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


def run_routing_lambda_analysis(output_dir: Path, strict: bool = False) -> None:
    """Run post-training visualization for DualConsistencyRouter lambda values."""
    script_path = CODE_DIR / "visualization" / "analyze_routing_lambda.py"
    analysis_dir = output_dir / "routing_lambda_analysis"
    command = [
        sys.executable,
        str(script_path),
        "--checkpoint-dir",
        str(output_dir),
        "--checkpoint-pattern",
        "best_gme_model.pt",
        "--output-dir",
        str(analysis_dir),
        "--fold-regex",
        r"fold_(\d+)",
        "--recursive",
    ]
    print("\nRouting lambda analysis:")
    print("$ " + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        message = (
            f"Routing lambda analysis failed with exit code {completed.returncode}. "
            f"Checkpoint output is still saved in {output_dir}."
        )
        if strict:
            raise RuntimeError(message)
        print(f"[Warning] {message}")
        return
    print(f"Routing lambda analysis saved to: {analysis_dir}")


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
        score_temperature=args.score_temperature,
        theta_init=args.theta_init,
        gamma_init=args.gamma_init,
        beacon_temperature=args.beacon_temperature,
        use_cosine_similarity=args.use_cosine_similarity,
    ).to(device)

    print(f"\nFold {fold}: train={len(train_ds)}, val={len(val_ds)}, encoders={train_ds.encoder_names}")
    print(f"Input dims: {input_dims}")

    # Stage 1: train ProjectionHead by downstream classification loss. The
    # mean-fusion ABMIL is a temporary supervision head, not the final method.
    unfreeze_projection(model)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.stage1_epochs, 1), eta_min=args.lr_stage1 * 0.01)
    best_stage1_auc = -math.inf
    best_stage1_path = fold_dir / "best_stage1_projection_meanfusion.pt"
    best_stage1_metrics: EvalResult | None = None

    for epoch in range(1, args.stage1_epochs + 1):
        train_loss = train_stage1_epoch(model, train_ds, optimizer1, device, args.grad_clip)
        val_metrics, val_pred = evaluate_stage1(model, val_ds, device, args.threshold)
        scheduler1.step()
        print(
            f"Fold {fold} | Stage1 | Epoch {epoch:03d}/{args.stage1_epochs} | "
            f"loss={train_loss:.4f} | AUC={val_metrics.auc:.4f} | AUPRC={val_metrics.auprc:.4f}"
        )
        if not np.isnan(val_metrics.auc) and val_metrics.auc > best_stage1_auc:
            best_stage1_auc = val_metrics.auc
            best_stage1_metrics = val_metrics
            save_checkpoint(best_stage1_path, model, epoch, val_metrics, {"input_dims": input_dims, "stage": "stage1"})
            save_fold_outputs(fold_dir, "stage1", val_metrics, val_pred)

    if best_stage1_path.exists():
        load_model_state(best_stage1_path, model, device)
    elif best_stage1_metrics is None:
        best_stage1_metrics, val_pred = evaluate_stage1(model, val_ds, device, args.threshold)
        save_fold_outputs(fold_dir, "stage1", best_stage1_metrics, val_pred)

    if not args.stage2_warm_start_classifier:
        reset_stage2_modules(model, args, device)

    # Stage 2 setup: freeze trained projection, build train-only static Beacon and mean baselines.
    freeze_projection(model)
    beacon_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        max_patches=args.beacon_max_patches,
        training=False,
    )
    beacon, baselines, beacon_summary = build_beacon_and_baselines(model, beacon_ds, device, args.target_dim)
    torch.save(
        {
            "fold": int(fold),
            "beacon": beacon.detach().cpu(),
            "baselines": {
                name: {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in stats.items()}
                for name, stats in baselines.items()
            },
            "input_dims": input_dims,
            "encoder_names": model.encoder_names,
            "policy": "Projection frozen; Beacon and replacement baselines built from train split only.",
        },
        fold_dir / "static_beacon_and_baselines.pt",
    )
    beacon_summary.to_csv(fold_dir / "static_beacon_summary.csv", index=False, encoding="utf-8-sig")

    # Stage 2: train router + ABMIL classifier with frozen projection.
    optimizer2 = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1), eta_min=args.lr_stage2 * 0.01)
    best_stage2_auc = -math.inf
    best_stage2_path = fold_dir / "best_gme_model.pt"
    best_stage2_metrics: EvalResult | None = None
    no_improve = 0

    for epoch in range(1, args.stage2_epochs + 1):
        train_loss = train_stage2_epoch(
            model=model,
            dataset=train_ds,
            optimizer=optimizer2,
            device=device,
            beacon=beacon,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            grad_clip=args.grad_clip,
        )
        val_metrics, val_pred, val_weights = evaluate_stage2(
            model=model,
            dataset=val_ds,
            device=device,
            beacon=beacon,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            threshold=args.threshold,
        )
        scheduler2.step()
        stats = model.router.get_routing_stats()
        print(
            f"Fold {fold} | Stage2 | Epoch {epoch:03d}/{args.stage2_epochs} | "
            f"loss={train_loss:.4f} | AUC={val_metrics.auc:.4f} | AUPRC={val_metrics.auprc:.4f} | "
            f"Sens={val_metrics.sensitivity:.4f} | Spec={val_metrics.specificity:.4f} | "
            f"lambda_sim={stats['lambda_similarity']:.3f} | gamma={stats['gamma']:.3f}"
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
                    "beacon_path": str(fold_dir / "static_beacon_and_baselines.pt"),
                },
            )
            save_fold_outputs(fold_dir, "stage2", val_metrics, val_pred, val_weights)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Fold {fold}: early stopping at epoch {epoch}. Best Stage2 AUC={best_stage2_auc:.4f}")
                break

    if best_stage2_path.exists():
        load_model_state(best_stage2_path, model, device)

    fused_dir = fold_dir / "fused_val_h5" if args.save_fused_h5 else None
    final_metrics, final_pred, final_weights = evaluate_stage2(
        model=model,
        dataset=val_ds,
        device=device,
        beacon=beacon,
        baselines=baselines,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
        threshold=args.threshold,
        fused_output_dir=fused_dir,
    )
    save_fold_outputs(fold_dir, "final", final_metrics, final_pred, final_weights)

    with open(fold_dir / "routing_stats.json", "w", encoding="utf-8") as f:
        json.dump(model.router.get_routing_stats(), f, indent=2)

    row = {
        "fold": int(fold),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        **asdict(final_metrics),
        "stage1_auc": best_stage1_metrics.auc if best_stage1_metrics else np.nan,
        "lambda_similarity": model.router.get_routing_stats()["lambda_similarity"],
        "lambda_attribution": model.router.get_routing_stats()["lambda_attribution"],
        "gamma": model.router.get_routing_stats()["gamma"],
    }
    return row


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
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
    print(f"Output: {output_dir}")
    print()

    fold_rows: List[Mapping[str, object]] = []
    for fold in folds:
        fold_row = run_fold(args, int(fold), manifest, clinical_df, device, output_dir)
        fold_rows.append(fold_row)
        summarize_metrics(fold_rows, output_dir)

    summarize_metrics(fold_rows, output_dir)
    if not args.skip_routing_lambda_analysis:
        run_routing_lambda_analysis(output_dir, strict=args.strict_routing_lambda_analysis)

    print("\nSummary:")
    print(pd.read_csv(output_dir / "summary_metrics.csv").to_string(index=False))
    print(f"\nSaved output: {output_dir}")


if __name__ == "__main__":
    main()
