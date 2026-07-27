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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_DIR, CODE_DIR / "architecture", CODE_DIR / "modules"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture.abmil_cls import ABMIL_Cls
from architecture.projection_head import MultiEncoderProjectionHead
from data_utils.cls_dataset import load_uvm_data
from modules.attribution import EncoderBaselineAccumulator, replace_encoder_embedding
from modules.beacon import BeaconAccumulator
from modules.routing import DualConsistencyRouter


DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
FEATURE_KEYS = ("feats", "features")
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
    config_parser = argparse.ArgumentParser(add_help=False)
    add_config_argument(config_parser)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="One-command end-to-end GME middle-fusion training on multi-encoder h5 embeddings."
    )
    add_config_argument(parser)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-manifest", action="store_true", help="Build the manifest before training.")
    parser.add_argument("--feat-base", default=r"L:\20x_256px_0px_overlap")
    parser.add_argument("--feature-dirs", nargs="+", default=DEFAULT_FEATURE_DIRS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--clinical-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default="gme")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact output directory for this run. Overrides --output-dir/--experiment-name timestamp layout.",
    )
    parser.add_argument("--label-col", default="d3m3")
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Fold ids to run. Default: all folds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index used when --device is 'cuda'.")

    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-attn", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.25)
    parser.add_argument("--n-classes", type=int, default=2)

    parser.add_argument("--stage1-epochs", type=int, default=0, help="Optional projection-only warmup epochs. 0 = end-to-end only.")
    parser.add_argument("--lr-stage1", type=float, default=0.0)
    parser.add_argument("--stage1-patience", type=int, default=5)
    parser.add_argument(
        "--stage1-beacon-mode",
        choices=["none", "epoch"],
        default="none",
        help="Stage-1 Beacon loss policy. 'none' is fastest; 'epoch' rebuilds train Beacon each epoch.",
    )
    parser.add_argument(
        "--stage1-consistency-weight",
        type=float,
        default=0.5,
        help="Weight for Stage-1 cross-encoder projection consistency loss.",
    )
    parser.add_argument(
        "--stage1-max-patches",
        type=int,
        default=1024,
        help="Patch cap per WSI for Stage-1 projection pretraining. 0 means use all patches.",
    )
    parser.add_argument(
        "--stage1-eval-max-patches",
        type=int,
        default=1024,
        help="Patch cap per WSI for Stage-1 geometry validation. 0 means use all patches.",
    )
    parser.add_argument(
        "--freeze-projection-stage2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze pretrained ProjectionHead after optional Stage 1. Default is end-to-end training with no freeze.",
    )
    parser.add_argument(
        "--stage2-warm-start-classifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated for projection-only Stage 1; Stage 2 reinitializes the classifier.",
    )

    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed decision threshold used for every fold.",
    )

    parser.add_argument("--replacement-strategy", choices=["mean", "zero", "gaussian"], default="mean")
    parser.add_argument("--gaussian-std-scale", type=float, default=1.0)
    parser.add_argument(
        "--attribution-target",
        choices=["predicted_class", "class_1"],
        default="predicted_class",
        help=(
            "Logit-margin direction used by intervention attribution. "
            "'predicted_class' explains the current mean-fusion decision; "
            "'class_1' uses z1-z0 for every sample."
        ),
    )
    parser.add_argument(
        "--interaction-pair-beta",
        type=float,
        default=0.1,
        help="Scale of the signed pairwise residual added to the attribution-fused representation.",
    )
    parser.add_argument(
        "--interaction-pair-weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay applied only to the interaction feature gate.",
    )
    parser.add_argument(
        "--interaction-pair-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the zero-initialized interaction feature gate.",
    )
    parser.add_argument(
        "--interaction-rms-clip",
        type=float,
        default=3.0,
        help="Absolute clipping bound after train-only per-pair RMS scaling without centering.",
    )
    parser.add_argument(
        "--beacon-constraint-weight",
        type=float,
        default=0.05,
        help="Weight for the static Beacon global semantic prior constraint. 0 disables it.",
    )
    parser.add_argument(
        "--routing-temperature",
        type=float,
        default=0.5,
        help="Temperature for attribution gate logits. Smaller values make routing more selective.",
    )
    parser.add_argument(
        "--routing-logit-scale",
        type=float,
        default=1.0,
        help="Scale applied to normalized attribution before softmax routing.",
    )

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
        "--baseline-max-patches",
        type=int,
        default=0,
        help="Optional patch cap when building train replacement baselines. 0 means use all train patches.",
    )
    parser.add_argument("--profile-samples", type=int, default=3, help="Validation WSIs used for FLOPs/time profiling. 0 disables FLOPs/time profiling.")
    parser.add_argument("--profile-warmup", type=int, default=1, help="Warmup forward passes before timing.")
    parser.add_argument("--profile-repeat", type=int, default=3, help="Timed repeats per profiled WSI.")
    parser.add_argument("--save-fused-h5", action="store_true", help="Export validation fused h5 embeddings.")
    # Backward-compatible no-op arguments for older config files.
    parser.add_argument("--skip-routing-lambda-analysis", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--strict-routing-lambda-analysis", action="store_true", help=argparse.SUPPRESS)
    config = load_config_file(config_args.config)
    if config:
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid_dests)
        if unknown:
            raise ValueError(f"Unknown config keys in {config_args.config}: {unknown}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    args.config = config_args.config
    if not 0.0 < float(args.inner_val_fraction) < 1.0:
        raise ValueError("--inner-val-fraction must be between 0 and 1.")
    if float(args.interaction_pair_beta) < 0.0:
        raise ValueError("--interaction-pair-beta must be non-negative.")
    if float(args.interaction_pair_weight_decay) < 0.0:
        raise ValueError("--interaction-pair-weight-decay must be non-negative.")
    if float(args.interaction_pair_lr) <= 0.0:
        raise ValueError("--interaction-pair-lr must be positive.")
    if float(args.interaction_rms_clip) <= 0.0:
        raise ValueError("--interaction-rms-clip must be positive.")
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


def infer_input_dims(manifest: pd.DataFrame, fold: int | None = None) -> Dict[str, int]:
    rows = manifest if fold is None else manifest[manifest["fold"] == fold]
    if rows.empty:
        raise ValueError(f"No manifest rows found for fold={fold}")

    input_dims: Dict[str, int] = {}
    for feature_dir, group in rows.groupby("feature_dir"):
        dims = sorted(group["feature_dim"].astype(int).unique().tolist())
        if len(dims) != 1:
            raise ValueError(f"{feature_dir}: expected one feature_dim, got {dims}")
        input_dims[str(feature_dir)] = int(dims[0])
    return input_dims


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


def class_logit_margin(logits: torch.Tensor, class_index: int) -> torch.Tensor:
    """Return one-vs-rest logit margin for the selected class."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"Expected logits shaped [batch, classes>=2], got {tuple(logits.shape)}")
    if not 0 <= class_index < logits.shape[1]:
        raise IndexError(f"class_index={class_index} is out of range for {logits.shape[1]} classes")
    other_logits = torch.cat(
        [logits[:, :class_index], logits[:, class_index + 1:]],
        dim=1,
    )
    return logits[:, class_index] - torch.logsumexp(other_logits, dim=1)


class GMEModel(nn.Module):
    """ProjectionHead + attribution router + ABMIL classifier."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        target_dim: int = 512,
        projection_dropout: float = 0.0,
        d_inner: int = 256,
        d_attn: int = 128,
        n_classes: int = 2,
        droprate: float = 0.25,
        routing_temperature: float = 0.5,
        routing_logit_scale: float = 1.0,
        attribution_target: str = "predicted_class",
        interaction_pair_beta: float = 0.1,
        interaction_rms_clip: float = 3.0,
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
            routing_temperature=routing_temperature,
            routing_logit_scale=routing_logit_scale,
        )
        self.classifier = ABMIL_Cls(
            D_feat=target_dim,
            D_inner=d_inner,
            D_attn=d_attn,
            n_classes=n_classes,
            droprate=droprate,
        )
        self.target_dim = int(target_dim)
        if attribution_target not in {"predicted_class", "class_1"}:
            raise ValueError(
                "attribution_target must be 'predicted_class' or 'class_1', "
                f"got {attribution_target!r}"
            )
        if attribution_target == "class_1" and n_classes != 2:
            raise ValueError("attribution_target='class_1' requires binary classification.")
        self.attribution_target = attribution_target
        if n_classes != 2:
            raise ValueError("Interaction pairwise fusion currently requires binary classification.")
        self.interaction_pair_beta = float(interaction_pair_beta)
        self.interaction_rms_clip = float(interaction_rms_clip)
        self.interaction_pairs = [
            (i, j)
            for i in range(len(self.encoder_names))
            for j in range(i + 1, len(self.encoder_names))
        ]
        # One learnable value per projected feature dimension. Zero
        # initialization starts exactly at the attribution-only baseline.
        self.interaction_gate = nn.Parameter(torch.zeros(self.target_dim))
        self.register_buffer("interaction_mean", torch.zeros(len(self.interaction_pairs)))
        self.register_buffer("interaction_std", torch.ones(len(self.interaction_pairs)))
        self.register_buffer("interaction_rms", torch.ones(len(self.interaction_pairs)))
        self.register_buffer("interaction_stats_count", torch.tensor(0, dtype=torch.long))

    def project(self, raw_features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.projection_heads(raw_features)

    def mean_fuse(self, projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([projected[name] for name in self.encoder_names], dim=0)
        return stacked.mean(dim=0)

    def route_with_scores(
        self,
        projected: Mapping[str, torch.Tensor],
        attribution_scores: torch.Tensor,
    ):
        return self.router(
            features_by_encoder=projected,
            attribution_scores=attribution_scores,
            encoder_names=self.encoder_names,
        )

    def beacon_constraint_loss(
        self,
        projected: Mapping[str, torch.Tensor],
        beacon: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Constrain encoder-level slide means toward a detached global Beacon.

        Beacon is used only as a structural/global semantic prior here. It does
        not enter routing weights or classifier features.
        """
        first = next(iter(projected.values()))
        beacon = beacon.to(device=first.device, dtype=first.dtype).reshape(-1)
        if beacon.shape[0] != self.target_dim:
            raise ValueError(f"Expected beacon dim={self.target_dim}, got {beacon.shape[0]}")
        beacon = beacon / beacon.norm(p=2).clamp_min(eps)

        losses = []
        similarities: Dict[str, torch.Tensor] = {}
        for name in self.encoder_names:
            h = projected[name].reshape(-1, projected[name].shape[-1])
            h = h / h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            slide_mean = h.mean(dim=0)
            slide_mean = slide_mean / slide_mean.norm(p=2).clamp_min(eps)
            sim = torch.sum(slide_mean * beacon)
            losses.append(1.0 - sim)
            similarities[name] = sim.detach()

        return torch.stack(losses, dim=0).mean(), similarities

    def encoder_consistency_loss(
        self,
        projected: Mapping[str, torch.Tensor],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Align paired patch embeddings across encoders in the projected space."""
        stacked = torch.stack([projected[name] for name in self.encoder_names], dim=0)
        normalized = stacked / stacked.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        consensus = normalized.mean(dim=0)
        consensus = consensus / consensus.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        similarity = torch.sum(normalized * consensus.unsqueeze(0), dim=-1)
        return (1.0 - similarity).mean()

    def forward_mean_fusion(self, projected: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, attn = self.classifier(self.mean_fuse(projected))
        return logits, attn

    def intervention_attribution(
        self,
        projected: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = True,
        interaction_target_class: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return attribution routing scores and optional signed interactions.

        Every masked coalition is evaluated once and cached. Coalition scores
        either explain the class predicted by the complete encoder set or use
        a fixed class-1 margin, according to ``self.attribution_target``.
        Pairwise interactions never modify the routing score or encoder
        weights; the fixed class-1 version may modulate the separate joint
        representation branch.
        """
        classifier_was_training = self.classifier.training
        router_was_training = self.router.training
        self.classifier.eval()
        self.router.eval()
        try:
            with torch.no_grad():
                full_logits, _ = self.forward_mean_fusion(projected)
                attribution_target_class = (
                    1
                    if self.attribution_target == "class_1"
                    else int(full_logits.argmax(dim=1)[0].item())
                )
                coalition_cache: Dict[frozenset[str], torch.Tensor] = {
                    frozenset(): full_logits
                }

                def coalition_logits(masked_names: frozenset[str]) -> torch.Tensor:
                    cached = coalition_cache.get(masked_names)
                    if cached is not None:
                        return cached
                    replaced = dict(projected)
                    for name in sorted(masked_names):
                        replaced = replace_encoder_embedding(
                            replaced,
                            encoder_name=name,
                            replacement_strategy=replacement_strategy,  # type: ignore[arg-type]
                            baselines=baselines,
                            gaussian_std_scale=gaussian_std_scale,
                        )
                    masked_logits, _ = self.forward_mean_fusion(replaced)
                    coalition_cache[masked_names] = masked_logits
                    return masked_logits

                def coalition_score(masked_names: frozenset[str], class_index: int) -> torch.Tensor:
                    return class_logit_margin(coalition_logits(masked_names), class_index)[0]

                full_score = coalition_score(frozenset(), attribution_target_class)
                single_scores = {
                    name: coalition_score(frozenset({name}), attribution_target_class)
                    for name in self.encoder_names
                }
                single_contributions = torch.stack(
                    [full_score - single_scores[name] for name in self.encoder_names]
                )
                interactions = full_score.new_zeros((len(self.encoder_names), len(self.encoder_names)))
                if compute_interactions:
                    pair_target_class = (
                        attribution_target_class
                        if interaction_target_class is None
                        else int(interaction_target_class)
                    )
                    pair_full_score = coalition_score(frozenset(), pair_target_class)
                    pair_single_scores = {
                        name: coalition_score(frozenset({name}), pair_target_class)
                        for name in self.encoder_names
                    }
                    for i, left in enumerate(self.encoder_names):
                        for j in range(i + 1, len(self.encoder_names)):
                            right = self.encoder_names[j]
                            double_score = coalition_score(
                                frozenset({left, right}), pair_target_class
                            )
                            value = (
                                pair_full_score
                                - pair_single_scores[left]
                                - pair_single_scores[right]
                                + double_score
                            )
                            interactions[i, j] = value
                            interactions[j, i] = value
        finally:
            self.classifier.train(classifier_was_training)
            self.router.train(router_was_training)
        return single_contributions, single_contributions, interactions

    def flatten_interactions(self, interactions: torch.Tensor) -> torch.Tensor:
        """Flatten the signed upper triangle in deterministic encoder order."""
        return torch.stack([interactions[i, j] for i, j in self.interaction_pairs], dim=-1)

    def set_interaction_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        rms: torch.Tensor,
        count: int,
    ) -> None:
        expected = len(self.interaction_pairs)
        if mean.numel() != expected or std.numel() != expected or rms.numel() != expected:
            raise ValueError("Interaction statistics do not match the number of encoder pairs.")
        self.interaction_mean.copy_(mean.to(self.interaction_mean).reshape_as(self.interaction_mean))
        self.interaction_std.copy_(std.to(self.interaction_std).reshape_as(self.interaction_std).clamp_min(1e-8))
        self.interaction_rms.copy_(rms.to(self.interaction_rms).reshape_as(self.interaction_rms).clamp_min(1e-8))
        self.interaction_stats_count.copy_(
            torch.as_tensor(int(count), device=self.interaction_stats_count.device)
        )

    def scale_interaction_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Scale without centering so the original interaction sign is preserved."""
        scaled = vector / self.interaction_rms.to(vector).clamp_min(1e-8)
        return scaled.clamp(
            min=-self.interaction_rms_clip,
            max=self.interaction_rms_clip,
        )

    def interaction_pair_residual_from_vector(
        self,
        projected: Mapping[str, torch.Tensor],
        routing_weights: torch.Tensor,
        interaction_vector: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a signed, attribution-gated joint representation residual.

        All encoder tensors must be patch aligned. Pairwise Hadamard products
        stay in the common projection space; a zero-initialized per-dimension
        gate selects stable joint features. Division by Z_q prevents growth
        with the number of encoder pairs.
        """
        tensors = [projected[name] for name in self.encoder_names]
        first_shape = tensors[0].shape
        for name, tensor in zip(self.encoder_names, tensors):
            if tensor.shape != first_shape:
                raise ValueError(
                    "Pairwise interaction fusion requires aligned projected tensors; "
                    f"expected {tuple(first_shape)}, got {tuple(tensor.shape)} for {name}."
                )
        weights = routing_weights.reshape(-1)
        if weights.numel() != len(self.encoder_names):
            raise ValueError(
                f"Expected {len(self.encoder_names)} routing weights, got {weights.numel()}."
            )
        vector = interaction_vector.reshape(-1)
        if vector.numel() != len(self.interaction_pairs):
            raise ValueError(
                f"Expected {len(self.interaction_pairs)} interactions, got {vector.numel()}."
            )

        scaled = self.scale_interaction_vector(vector)
        normalized = {
            name: F.layer_norm(projected[name], (self.target_dim,))
            for name in self.encoder_names
        }
        numerator = torch.zeros_like(tensors[0])
        z_q = weights.new_zeros(())
        for pair_index, (i, j) in enumerate(self.interaction_pairs):
            pair_gate = torch.sqrt((weights[i] * weights[j]).clamp_min(0.0))
            pair_feature = (
                normalized[self.encoder_names[i]]
                * normalized[self.encoder_names[j]]
            )
            numerator = numerator + pair_gate * scaled[pair_index] * pair_feature
            z_q = z_q + pair_gate
        pair_representation = numerator / z_q.clamp_min(eps)
        pair_residual = (
            self.interaction_pair_beta
            * torch.tanh(self.interaction_gate)
            * torch.tanh(pair_representation)
        )
        return pair_residual, scaled

    def interaction_pair_residual(
        self,
        projected: Mapping[str, torch.Tensor],
        routing_weights: torch.Tensor,
        interactions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.interaction_pair_residual_from_vector(
            projected,
            routing_weights,
            self.flatten_interactions(interactions),
        )

    def forward_stage2(
        self,
        raw_features: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = True,
        return_attribution_logits: bool = False,
    ):
        projected = self.project(raw_features)
        attr, single_attr, interactions = self.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=compute_interactions and self.interaction_pair_beta > 0.0,
            interaction_target_class=1,
        )
        routed = self.route_with_scores(projected, attribution_scores=attr)
        if compute_interactions and self.interaction_pair_beta > 0.0:
            pair_residual, scaled_interactions = self.interaction_pair_residual(
                projected,
                routed.weights,
                interactions,
            )
            final_fused = routed.fused + pair_residual
        else:
            pair_residual = torch.zeros_like(routed.fused)
            scaled_interactions = routed.fused.new_zeros(len(self.interaction_pairs))
            final_fused = routed.fused
            interactions = routed.fused.new_zeros(
                (len(self.encoder_names), len(self.encoder_names))
            )
        logits, attn = self.classifier(final_fused)
        if return_attribution_logits and self.interaction_pair_beta > 0.0:
            attribution_logits, _ = self.classifier(routed.fused)
        else:
            attribution_logits = logits.detach()
        return (
            logits,
            attn,
            routed,
            projected,
            attr,
            single_attr,
            interactions,
            attribution_logits,
            pair_residual,
            scaled_interactions,
        )


def move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.float().to(device) for name, tensor in features.items()}


class GMEProfileWrapper(nn.Module):
    """Traceable online inference with attribution/interaction supplied offline."""

    def __init__(
        self,
        model: GMEModel,
        encoder_names: Sequence[str],
    ):
        super().__init__()
        self.model = model
        self.encoder_names = list(encoder_names)

    def forward(self, *feature_tensors: torch.Tensor) -> torch.Tensor:
        attribution_scores = feature_tensors[-2]
        interaction_vector = feature_tensors[-1]
        raw_features = {
            name: tensor
            for name, tensor in zip(self.encoder_names, feature_tensors[:-2])
        }
        projected = self.model.project(raw_features)
        routed = self.model.route_with_scores(projected, attribution_scores)
        pair_residual, _ = self.model.interaction_pair_residual_from_vector(
            projected,
            routed.weights,
            interaction_vector,
        )
        logits, _ = self.model.classifier(routed.fused + pair_residual)
        return logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_profile_inputs(
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    max_samples: int,
) -> List[Dict[str, torch.Tensor]]:
    samples = []
    for idx in range(min(max(int(max_samples), 0), len(dataset))):
        raw_features, _, _ = dataset[idx]
        samples.append(move_features(raw_features, device))
    return samples


def profile_sample_tuples(
    samples: Sequence[Mapping[str, torch.Tensor]],
    encoder_names: Sequence[str],
    attribution_scores: Sequence[torch.Tensor] | None = None,
    interaction_vectors: Sequence[torch.Tensor] | None = None,
) -> List[Tuple[torch.Tensor, ...]]:
    feature_tuples = [
        tuple(raw_features[name] for name in encoder_names)
        for raw_features in samples
    ]
    if attribution_scores is None and interaction_vectors is None:
        return feature_tuples
    if attribution_scores is None or interaction_vectors is None:
        raise ValueError("Attribution scores and interaction vectors must be supplied together.")
    if len(feature_tuples) != len(attribution_scores) or len(feature_tuples) != len(interaction_vectors):
        raise ValueError("Profile samples and offline routing inputs must have the same length.")
    return [
        features + (scores, interaction)
        for features, scores, interaction in zip(feature_tuples, attribution_scores, interaction_vectors)
    ]


def estimate_fvcore_flops(wrapper: nn.Module, samples: Sequence[Tuple[torch.Tensor, ...]]) -> float:
    if not samples:
        return float("nan")
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[Warning] fvcore is not installed; FLOPs will be written as NaN.")
        return float("nan")

    values = []
    was_training = wrapper.training
    wrapper.eval()
    first_error = None
    try:
        for feature_tensors in samples:
            try:
                analysis = FlopCountAnalysis(wrapper, feature_tensors)
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                analysis.tracer_warnings("none")
                values.append(float(analysis.total()))
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                values.append(float("nan"))
    finally:
        wrapper.train(was_training)

    finite = [value for value in values if np.isfinite(value)]
    if not finite and first_error is not None:
        print(f"[Warning] fvcore FLOPs tracing failed; FLOPs will be written as NaN. First error: {first_error}")
    return float(np.mean(finite)) if finite else float("nan")


@torch.no_grad()
def measure_forward_time_seconds(
    wrapper: nn.Module,
    samples: Sequence[Tuple[torch.Tensor, ...]],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    if not samples or repeat <= 0:
        return float("nan")
    was_training = wrapper.training
    wrapper.eval()
    try:
        for _ in range(max(int(warmup), 0)):
            for feature_tensors in samples:
                wrapper(*feature_tensors)
        cuda_synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(int(repeat)):
            for feature_tensors in samples:
                wrapper(*feature_tensors)
        cuda_synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    finally:
        wrapper.train(was_training)
    return float(elapsed / (int(repeat) * len(samples)))


def profile_gme_efficiency(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    profile_samples: int,
    profile_warmup: int,
    profile_repeat: int,
) -> Dict[str, float]:
    parameters = count_parameters(model)
    raw_samples = collect_profile_inputs(dataset, device, profile_samples)
    # Attribution is an offline preprocessing product and is deliberately
    # computed before FLOPs/timing.  The reported inference_time measures the
    # same online boundary as fusion baselines: projection + fusion/router +
    # classifier.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        offline_scores = []
        offline_interactions = []
        for raw_features in raw_samples:
            projected = model.project(raw_features)
            routing_scores, _, _ = model.intervention_attribution(
                projected=projected,
                baselines=baselines,
                replacement_strategy=replacement_strategy,
                gaussian_std_scale=gaussian_std_scale,
                compute_interactions=False,
            )
            _, _, interactions = model.intervention_attribution(
                projected=projected,
                baselines=baselines,
                replacement_strategy=replacement_strategy,
                gaussian_std_scale=gaussian_std_scale,
                compute_interactions=True,
                interaction_target_class=1,
            )
            offline_scores.append(routing_scores.detach())
            offline_interactions.append(model.flatten_interactions(interactions).detach())
    model.train(was_training)
    profile_inputs = profile_sample_tuples(
        raw_samples, model.encoder_names, offline_scores, offline_interactions
    )
    wrapper = GMEProfileWrapper(model, model.encoder_names).to(device)
    flops = estimate_fvcore_flops(wrapper, profile_inputs)
    inference_time = measure_forward_time_seconds(wrapper, profile_inputs, device, profile_warmup, profile_repeat)
    return {
        "parameters": round(float(parameters) / 1_000_000.0, 2),
        "flops": round(float(flops) / 1_000_000_000.0, 2),
        "inference_time": round(float(inference_time) * 1000.0, 3),
    }


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
        max_patches=args.max_patches,
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
    stage2_history: List[Dict[str, float | int]] = []

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
        train_loss, train_cls_loss, train_beacon_loss = train_stage2_epoch(
            model=model,
            dataset=inner_train_ds,
            optimizer=optimizer2,
            device=device,
            baselines=baselines,
            beacon=beacon,
            beacon_constraint_weight=stage2_beacon_weight,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            grad_clip=args.grad_clip,
        )
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
        val_metrics, val_pred, val_weights, val_interactions = evaluate_stage2(
            model=model,
            dataset=inner_val_ds,
            device=device,
            baselines=baselines,
            beacon=beacon,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            decision_threshold=args.threshold,
        )
        scheduler2.step()
        stats = model.router.get_routing_stats()
        base_val_metrics = compute_metrics(
            val_pred["label"],
            val_pred["base_prob_class1"],
            decision_threshold=args.threshold,
        )
        mean_pair_relative_rms = float(val_pred["pair_relative_rms"].mean())
        mean_abs_prob_delta = float(val_pred["prob_delta"].abs().mean())
        stage2_history.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_cls_loss": float(train_cls_loss),
            "train_beacon_loss": float(train_beacon_loss),
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
            f"Fold {fold} | Stage2 | Epoch {epoch:03d}/{args.stage2_epochs} | "
            f"loss={train_loss:.4f} | cls={train_cls_loss:.4f} | beacon={train_beacon_loss:.4f} | "
            f"inner_AUC={val_metrics.auc:.4f} | inner_AUPRC={val_metrics.auprc:.4f} | "
            f"attr_AUC={base_val_metrics.auc:.4f} | "
            f"ACC@{args.threshold:g}={val_metrics.accuracy:.4f} | F1@{args.threshold:g}={val_metrics.f1:.4f} | "
            f"pair_rel_RMS={mean_pair_relative_rms:.4f} | |delta_p|={mean_abs_prob_delta:.6f} | "
            f"I_min={score_stats['attribution_min']:.4f} | I_max={score_stats['attribution_max']:.4f}"
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
                    "stage2_beacon_constraint_weight": float(stage2_beacon_weight),
                    "freeze_projection_stage2": bool(args.freeze_projection_stage2),
                    "stage1_path": str(stage1_path) if stage1_path.exists() else None,
                    "routing_score_stats": score_stats,
                    "interaction_stats": interaction_stats,
                    "interaction_pair_beta": float(args.interaction_pair_beta),
                    "interaction_scaling": "signed_train_only_rms_without_centering",
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
                    for name, stats in baselines.items()
                },
                "input_dims": input_dims,
                "encoder_names": model.encoder_names,
                "policy": (
                    "Train-only replacement baselines built once from frozen Stage1 projection."
                    if args.freeze_projection_stage2
                    else "Train-only replacement baselines rebuilt from the current end-to-end projection space."
                ),
                "routing_score_stats": score_stats,
            }
            torch.save(baseline_payload, fold_dir / "replacement_baselines.pt")
            torch.save(
                {
                    **baseline_payload,
                    "beacon": beacon.detach().cpu(),
                    "beacon_constraint_weight": float(args.beacon_constraint_weight),
                    "stage2_beacon_constraint_weight": float(stage2_beacon_weight),
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
            baseline_summary.to_csv(fold_dir / "replacement_baseline_summary.csv", index=False, encoding="utf-8-sig")
            beacon_summary.to_csv(fold_dir / "static_beacon_summary.csv", index=False, encoding="utf-8-sig")
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
        T_max=max(selected_epoch, 1),
        eta_min=args.lr_stage2 * 0.01,
    )
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
    for _ in range(selected_epoch):
        train_stage2_epoch(
            model=model,
            dataset=train_ds,
            optimizer=retrain_optimizer,
            device=device,
            baselines=baselines,
            beacon=beacon,
            beacon_constraint_weight=stage2_beacon_weight,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            grad_clip=args.grad_clip,
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
    print(f"Fold {fold}: selected inner epoch={selected_epoch}; retrained on all outer-train slides.")

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
            "interaction_stats": final_interaction_stats,
            "interaction_pair_beta": float(args.interaction_pair_beta),
            "interaction_scaling": "signed_train_only_rms_without_centering",
        },
    )

    with open(fold_dir / "routing_stats.json", "w", encoding="utf-8") as f:
        json.dump(model.router.get_routing_stats(), f, indent=2)
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
