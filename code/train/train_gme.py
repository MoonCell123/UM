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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--replacement-strategy", choices=["mean", "zero", "gaussian"], default="mean")
    parser.add_argument("--gaussian-std-scale", type=float, default=1.0)
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
        if existing == requested:
            return manifest_path
        print(
            "\nExisting manifest feature_dirs do not match config; rebuilding manifest.\n"
            f"Existing: {sorted(existing)}\n"
            f"Requested: {sorted(requested)}"
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
                full_logits, _ = self.forward_mean_fusion(projected)
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
                    masked_logits, _ = self.forward_mean_fusion(replaced)
                    masked_score = torch.softmax(masked_logits, dim=1)[0, class_index]
                    scores.append(full_score - masked_score)
        finally:
            self.classifier.train(classifier_was_training)
            self.router.train(router_was_training)
        return torch.stack(scores, dim=0)

    def forward_stage2(
        self,
        raw_features: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
    ):
        projected = self.project(raw_features)
        attr = self.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
        )
        routed = self.route_with_scores(projected, attribution_scores=attr)
        logits, attn = self.classifier(routed.fused)
        return logits, attn, routed, projected, attr


def move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.float().to(device) for name, tensor in features.items()}


class GMEProfileWrapper(nn.Module):
    """Traceable inference wrapper for final GME forward profiling."""

    def __init__(
        self,
        model: GMEModel,
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str,
        gaussian_std_scale: float,
        encoder_names: Sequence[str],
    ):
        super().__init__()
        self.model = model
        self.baselines = baselines
        self.replacement_strategy = replacement_strategy
        self.gaussian_std_scale = float(gaussian_std_scale)
        self.encoder_names = list(encoder_names)

    def forward(self, *feature_tensors: torch.Tensor) -> torch.Tensor:
        raw_features = {
            name: tensor
            for name, tensor in zip(self.encoder_names, feature_tensors)
        }
        logits, *_ = self.model.forward_stage2(
            raw_features=raw_features,
            baselines=self.baselines,
            replacement_strategy=self.replacement_strategy,
            gaussian_std_scale=self.gaussian_std_scale,
        )
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
) -> List[Tuple[torch.Tensor, ...]]:
    return [
        tuple(raw_features[name] for name in encoder_names)
        for raw_features in samples
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
    profile_inputs = profile_sample_tuples(raw_samples, model.encoder_names)
    wrapper = GMEProfileWrapper(
        model,
        baselines,
        replacement_strategy,
        gaussian_std_scale,
        model.encoder_names,
    ).to(device)
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


def reset_stage2_classifier(model: GMEModel, args: argparse.Namespace, device: torch.device) -> None:
    model.classifier = ABMIL_Cls(
        D_feat=args.target_dim,
        D_inner=args.d_inner,
        D_attn=args.d_attn,
        n_classes=args.n_classes,
        droprate=args.droprate,
    ).to(device)


def youden_threshold(labels: np.ndarray, probs: np.ndarray, fallback: float = 0.5) -> float:
    if np.unique(labels).size < 2:
        return float(fallback)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return float(fallback)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    youden = tpr - fpr
    return float(thresholds[int(np.argmax(youden))])


def compute_metrics(labels: Sequence[int], probs: Sequence[float], fallback_threshold: float = 0.5) -> EvalResult:
    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    threshold = youden_threshold(labels_np, probs_np, fallback=fallback_threshold)
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
def update_train_routing_score_stats(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
) -> Dict[str, float]:
    """Estimate scalar attribution normalization stats from the whole train split."""
    was_training = model.training
    model.eval()

    attr_scores = []
    for idx in range(len(dataset)):
        raw_features, _, _ = dataset[idx]
        raw_features = move_features(raw_features, device)
        projected = model.project(raw_features)
        attr_scores.append(
            model.intervention_attribution(
                projected=projected,
                baselines=baselines,
                replacement_strategy=replacement_strategy,
                gaussian_std_scale=gaussian_std_scale,
            )
        )

    attr_min, attr_max, attr_count = _min_max_from_score_list(attr_scores)
    model.router.set_score_stats(
        attribution_min=attr_min,
        attribution_max=attr_max,
        count=attr_count,
    )
    model.train(was_training)
    return model.router.get_score_stats()


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
        logits, _attn, _routed, projected, _attr = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
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
    fallback_threshold: float,
    fused_output_dir: Path | None = None,
) -> Tuple[EvalResult, pd.DataFrame, pd.DataFrame]:
    model.eval()
    labels, probs, rows, weight_rows = [], [], [], []
    if fused_output_dir is not None:
        fused_output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        logits, _attn, routed, projected, attr = model.forward_stage2(
            raw_features=raw_features,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
        )
        beacon_sims: Dict[str, torch.Tensor] = {}
        if beacon is not None:
            _, beacon_sims = model.beacon_constraint_loss(projected=projected, beacon=beacon)
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        labels.append(int(label))
        probs.append(prob)
        rows.append({"slide_id": slide_id, "label": int(label), "prob_class1": prob})

        weights = routed.weights.detach().cpu().reshape(-1).numpy()
        attr_np = attr.detach().cpu().reshape(-1).numpy()
        for encoder, weight, attr_value in zip(routed.encoder_names, weights, attr_np):
            weight_rows.append({
                "slide_id": slide_id,
                "encoder": encoder,
                "weight": float(weight),
                "attribution": float(attr_value),
                "beacon_similarity": (
                    float(beacon_sims[encoder].detach().cpu())
                    if encoder in beacon_sims
                    else np.nan
                ),
            })

        if fused_output_dir is not None:
            fused = routed.fused.detach().cpu().numpy().astype(np.float32)
            with h5py.File(fused_output_dir / f"{slide_id}.h5", "w") as f:
                f.create_dataset("features", data=fused, compression="gzip")

    metrics = compute_metrics(labels, probs, fallback_threshold=fallback_threshold)
    pred_df = pd.DataFrame(rows)
    pred_df["threshold"] = metrics.threshold
    pred_df["pred"] = (pred_df["prob_class1"] >= metrics.threshold).astype(int)
    return metrics, pred_df, pd.DataFrame(weight_rows)


def save_fold_outputs(
    fold_dir: Path,
    stage_name: str,
    metrics: EvalResult,
    predictions: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    efficiency: Mapping[str, float] | None = None,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {**{"stage": stage_name}, **asdict(metrics)}
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
        split="val",
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
                dataset=stage1_train_ds,
                optimizer=optimizer1,
                device=device,
                beacon=stage1_beacon,
                beacon_constraint_weight=args.beacon_constraint_weight,
                consistency_weight=args.stage1_consistency_weight,
                grad_clip=args.grad_clip,
            )
            val_loss, val_beacon_loss, val_consistency_loss = evaluate_stage1_geometry(
                model=model,
                dataset=stage1_val_ds,
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

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters left for Stage2.")

    optimizer2 = torch.optim.AdamW(
        trainable_params,
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1), eta_min=args.lr_stage2 * 0.01)
    best_stage2_auc = -math.inf
    best_stage2_path = fold_dir / "best_gme_model.pt"
    best_beacon_path = fold_dir / "static_beacon_and_baselines.pt"
    best_stage2_metrics: EvalResult | None = None
    no_improve = 0

    static_beacon = static_beacon_summary = static_baselines = static_baseline_summary = None
    if args.freeze_projection_stage2:
        static_beacon, static_beacon_summary, static_baselines, static_baseline_summary = build_beacon_and_baselines(
            model,
            baseline_ds,
            device,
            args.target_dim,
        )

    for epoch in range(1, args.stage2_epochs + 1):
        if args.freeze_projection_stage2:
            beacon = static_beacon
            beacon_summary = static_beacon_summary
            baselines = static_baselines
            baseline_summary = static_baseline_summary
        else:
            beacon, beacon_summary, baselines, baseline_summary = build_beacon_and_baselines(
                model,
                baseline_ds,
                device,
                args.target_dim,
            )
        score_stats = update_train_routing_score_stats(
            model=model,
            dataset=score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
        train_loss, train_cls_loss, train_beacon_loss = train_stage2_epoch(
            model=model,
            dataset=train_ds,
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
                baseline_ds,
                device,
                args.target_dim,
            )
        score_stats = update_train_routing_score_stats(
            model=model,
            dataset=score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )
        val_metrics, val_pred, val_weights = evaluate_stage2(
            model=model,
            dataset=val_ds,
            device=device,
            baselines=baselines,
            beacon=beacon,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
            fallback_threshold=args.threshold,
        )
        scheduler2.step()
        stats = model.router.get_routing_stats()
        print(
            f"Fold {fold} | Stage2 | Epoch {epoch:03d}/{args.stage2_epochs} | "
            f"loss={train_loss:.4f} | cls={train_cls_loss:.4f} | beacon={train_beacon_loss:.4f} | "
            f"AUC={val_metrics.auc:.4f} | AUPRC={val_metrics.auprc:.4f} | "
            f"ACC={val_metrics.accuracy:.4f} | F1={val_metrics.f1:.4f} | "
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
            save_fold_outputs(fold_dir, "stage2", val_metrics, val_pred, val_weights)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Fold {fold}: early stopping at epoch {epoch}. Best Stage2 AUC={best_stage2_auc:.4f}")
                break

    if best_stage2_path.exists():
        load_model_state(best_stage2_path, model, device)
        payload_path = best_beacon_path if best_beacon_path.exists() else fold_dir / "replacement_baselines.pt"
        baseline_payload = torch.load(payload_path, map_location=device)
        baselines = {
            name: {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in stats.items()
            }
            for name, stats in baseline_payload["baselines"].items()
        }
        if "beacon" in baseline_payload:
            beacon = baseline_payload["beacon"].to(device).float()
        else:
            beacon, _, _, _ = build_beacon_and_baselines(model, baseline_ds, device, args.target_dim)
        if "routing_score_stats" in baseline_payload:
            score_stats = baseline_payload["routing_score_stats"]
            model.router.set_score_stats(
                attribution_min=score_stats["attribution_min"],
                attribution_max=score_stats["attribution_max"],
                count=score_stats["count"],
            )
    else:
        beacon, _, baselines, _ = build_beacon_and_baselines(model, baseline_ds, device, args.target_dim)
        update_train_routing_score_stats(
            model=model,
            dataset=score_stats_ds,
            device=device,
            baselines=baselines,
            replacement_strategy=args.replacement_strategy,
            gaussian_std_scale=args.gaussian_std_scale,
        )

    fused_dir = fold_dir / "fused_val_h5" if args.save_fused_h5 else None
    final_metrics, final_pred, final_weights = evaluate_stage2(
        model=model,
        dataset=val_ds,
        device=device,
        baselines=baselines,
        beacon=beacon,
        replacement_strategy=args.replacement_strategy,
        gaussian_std_scale=args.gaussian_std_scale,
        fallback_threshold=args.threshold,
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
    save_fold_outputs(fold_dir, "final", final_metrics, final_pred, final_weights, efficiency=efficiency)

    with open(fold_dir / "routing_stats.json", "w", encoding="utf-8") as f:
        json.dump(model.router.get_routing_stats(), f, indent=2)

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
