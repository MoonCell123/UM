"""Offline fusion baselines for multi-encoder WSI embeddings.

Baselines implemented here:

1. no_fusion       : train ABMIL on one encoder at a time.
2. mean            : per-encoder ProjectionHead -> mean over encoders -> ABMIL.
3. concat          : per-encoder ProjectionHead -> concat -> linear reduction -> ABMIL.
4. cross_attention : per-encoder ProjectionHead -> encoder-token cross attention -> ABMIL.
5. self_attention  : per-encoder ProjectionHead -> encoder-token self attention -> ABMIL.

These are conventional baselines for comparing against train_gme.py. They do
not use Beacon, intervention attribution, or GME routing.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
for path in (PROJECT_ROOT, CODE_DIR, CODE_DIR / "architecture"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture.abmil_cls import ABMIL_Cls
from architecture.projection_head import MultiEncoderProjectionHead, initialize_projection_weights
from data_utils.cls_dataset import load_uvm_data
from modules.beacon import infer_input_dims


DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Offline_Fusion_Baselines"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
FEATURE_KEYS = ("feats", "features")
METHODS = ("mean", "concat", "cross_attention", "self_attention")
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

    parser = argparse.ArgumentParser(description="Train offline fusion baselines on multi-encoder h5 embeddings.")
    add_config_argument(parser)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--feat-base", default=r"L:\20x_256px_0px_overlap")
    parser.add_argument("--feature-dirs", nargs="+", default=DEFAULT_FEATURE_DIRS)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--cv-folds", type=int, default=5)

    parser.add_argument("--clinical-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default="offline_fusion_baselines")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact output directory for this run. Overrides --output-dir/--experiment-name timestamp layout.",
    )
    parser.add_argument("--label-col", default="d3m3")
    parser.add_argument("--methods", nargs="+", default=["all"], choices=["all", *METHODS])
    parser.add_argument(
        "--single-encoders",
        nargs="*",
        default=None,
        help="Optional encoder names for no_fusion. Default: all encoders in each fold.",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device index used when --device is 'cuda'.")

    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--cross-attn-heads", type=int, default=4)
    parser.add_argument("--cross-attn-layers", type=int, default=1)
    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-attn", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.25)
    parser.add_argument("--n-classes", type=int, default=2)

    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed decision threshold used for every fold.",
    )

    parser.add_argument("--max-patches", type=int, default=0, help="Random train patch cap per WSI. 0 = all patches.")
    parser.add_argument("--eval-max-patches", type=int, default=0, help="Deterministic val patch cap per WSI. 0 = all patches.")
    parser.add_argument("--profile-samples", type=int, default=3, help="Validation WSIs used for FLOPs/time profiling. 0 disables FLOPs/time profiling.")
    parser.add_argument("--profile-warmup", type=int, default=1, help="Warmup forward passes before timing.")
    parser.add_argument("--profile-repeat", type=int, default=3, help="Timed repeats per profiled WSI.")
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


def normalize_methods(methods: Sequence[str]) -> List[str]:
    if "all" in methods:
        return list(METHODS)
    seen = []
    for method in methods:
        if method not in seen:
            seen.append(method)
    return seen


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


def stable_name_offset(name: str) -> int:
    """Deterministic small integer offset for per-baseline seeding."""
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name)) % 10000


def ensure_manifest(args: argparse.Namespace) -> Path:
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


def subset_patch_indices(n_patches: int, max_patches: int, training: bool) -> np.ndarray | None:
    if max_patches <= 0 or n_patches <= max_patches:
        return None
    if training:
        return np.sort(np.random.choice(n_patches, size=max_patches, replace=False))
    return np.arange(max_patches)


class MultiEncoderSlideDataset:
    """Slide-level dataset returning raw features for each encoder."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        fold: int,
        split: str,
        clinical_df: pd.DataFrame,
        label_col: str = "d3m3",
        encoder_names: Sequence[str] | None = None,
        max_patches: int = 0,
        training: bool = True,
    ):
        rows = manifest[(manifest["fold"].astype(int) == int(fold)) & (manifest["split"].astype(str) == split)].copy()
        if rows.empty:
            raise ValueError(f"No manifest rows found for fold={fold}, split={split}.")

        available = sorted(rows["feature_dir"].astype(str).unique().tolist())
        self.encoder_names = sorted(encoder_names) if encoder_names is not None else available
        missing = sorted(set(self.encoder_names) - set(available))
        if missing:
            raise ValueError(f"Fold {fold}, split={split}: requested encoders not in manifest: {missing}")

        rows = rows[rows["feature_dir"].astype(str).isin(self.encoder_names)].copy()
        self.label_col = label_col
        self.max_patches = int(max_patches)
        self.training = bool(training)
        self.clinical = clinical_df.set_index("slide_id")
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
            raise RuntimeError(f"Fold {fold}, split={split}: no usable slides after filtering.")

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
            features[encoder] = read_h5_features(h5_path, dataset_key)

        min_patches = min(arr.shape[0] for arr in features.values())
        patch_idx = subset_patch_indices(min_patches, self.max_patches, self.training)
        tensor_features = {}
        for encoder in self.encoder_names:
            arr = features[encoder][:min_patches]
            if patch_idx is not None:
                arr = arr[patch_idx]
            tensor_features[encoder] = torch.from_numpy(arr.astype(np.float32))

        label = int(self.clinical.loc[slide_id, self.label_col])
        return tensor_features, label, slide_id


class SingleEncoderABMIL(nn.Module):
    """No-fusion baseline: one raw encoder embedding -> ABMIL."""

    def __init__(self, encoder_name: str, input_dim: int, args: argparse.Namespace):
        super().__init__()
        self.encoder_name = encoder_name
        self.classifier = ABMIL_Cls(
            D_feat=input_dim,
            D_inner=args.d_inner,
            D_attn=args.d_attn,
            n_classes=args.n_classes,
            droprate=args.droprate,
        )

    def forward(self, raw_features: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(raw_features[self.encoder_name])


class CrossAttentionFusion(nn.Module):
    """Per-patch cross-attention over encoder tokens."""

    def __init__(self, dim: int, heads: int = 4, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"target_dim={dim} must be divisible by cross_attn_heads={heads}.")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                ),
            })
            for _ in range(layers)
        ])
        self.apply(initialize_projection_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_patches, M_encoders, D]
        for layer in self.layers:
            h = layer["norm1"](x)
            attn_out, _ = layer["attn"](h, h, h, need_weights=False)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm2"](x))
        return x.mean(dim=1)


class SelfAttentionFusion(nn.Module):
    """Per-patch self-attention over encoder tokens followed by learned reduction."""

    def __init__(self, dim: int, num_encoders: int, heads: int = 4, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"target_dim={dim} must be divisible by cross_attn_heads={heads}.")
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.reduce = nn.Sequential(
            nn.Linear(dim * num_encoders, dim),
            nn.LayerNorm(dim),
        )
        self.apply(initialize_projection_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_patches, M_encoders, D]
        for layer in self.layers:
            x = layer(x)
        return self.reduce(x.flatten(start_dim=1))


class ProjectedFusionABMIL(nn.Module):
    """Projected multi-encoder fusion baseline followed by ABMIL."""

    def __init__(self, method: str, input_dims: Mapping[str, int], args: argparse.Namespace):
        super().__init__()
        if method not in {"mean", "concat", "cross_attention", "self_attention"}:
            raise ValueError(f"Unsupported projected fusion method: {method}")
        self.method = method
        self.encoder_names = sorted(input_dims.keys())
        self.projection_heads = MultiEncoderProjectionHead(
            input_dims={name: int(input_dims[name]) for name in self.encoder_names},
            target_dim=args.target_dim,
            dropout=args.projection_dropout,
        )
        if method == "concat":
            self.concat_reduce = nn.Sequential(
                nn.Linear(args.target_dim * len(self.encoder_names), args.target_dim),
                nn.LayerNorm(args.target_dim),
            )
            self.concat_reduce.apply(initialize_projection_weights)
        else:
            self.concat_reduce = None

        if method == "cross_attention":
            self.cross_attention = CrossAttentionFusion(
                dim=args.target_dim,
                heads=args.cross_attn_heads,
                layers=args.cross_attn_layers,
                dropout=args.projection_dropout,
            )
        else:
            self.cross_attention = None

        if method == "self_attention":
            self.self_attention = SelfAttentionFusion(
                dim=args.target_dim,
                num_encoders=len(self.encoder_names),
                heads=args.cross_attn_heads,
                layers=args.cross_attn_layers,
                dropout=args.projection_dropout,
            )
        else:
            self.self_attention = None

        self.classifier = ABMIL_Cls(
            D_feat=args.target_dim,
            D_inner=args.d_inner,
            D_attn=args.d_attn,
            n_classes=args.n_classes,
            droprate=args.droprate,
        )

    def fuse(self, raw_features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        projected = self.projection_heads(raw_features)
        tensors = [projected[name] for name in self.encoder_names]
        if self.method == "mean":
            return torch.stack(tensors, dim=0).mean(dim=0)
        if self.method == "concat":
            return self.concat_reduce(torch.cat(tensors, dim=-1))
        if self.method == "cross_attention":
            return self.cross_attention(torch.stack(tensors, dim=1))
        if self.method == "self_attention":
            return self.self_attention(torch.stack(tensors, dim=1))
        raise RuntimeError(f"Unknown method: {self.method}")

    def forward(self, raw_features: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(self.fuse(raw_features))


def move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.float().to(device) for name, value in features.items()}


class OfflineProfileWrapper(nn.Module):
    """Traceable inference wrapper for offline baseline profiling."""

    def __init__(self, model: nn.Module, encoder_names: Sequence[str]):
        super().__init__()
        self.model = model
        self.encoder_names = list(encoder_names)

    def forward(self, *feature_tensors: torch.Tensor) -> torch.Tensor:
        raw_features = {
            name: tensor
            for name, tensor in zip(self.encoder_names, feature_tensors)
        }
        logits, _ = self.model(raw_features)
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


def profile_offline_efficiency(
    model: nn.Module,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    profile_samples: int,
    profile_warmup: int,
    profile_repeat: int,
) -> Dict[str, float]:
    parameters = count_parameters(model)
    raw_samples = collect_profile_inputs(dataset, device, profile_samples)
    encoder_names = (
        list(model.encoder_names)
        if hasattr(model, "encoder_names")
        else [str(model.encoder_name)]
    )
    profile_inputs = profile_sample_tuples(raw_samples, encoder_names)
    wrapper = OfflineProfileWrapper(model, encoder_names).to(device)
    flops = estimate_fvcore_flops(wrapper, profile_inputs)
    inference_time = measure_forward_time_seconds(wrapper, profile_inputs, device, profile_warmup, profile_repeat)
    return {
        "parameters": round(float(parameters) / 1_000_000.0, 2),
        "flops": round(float(flops) / 1_000_000_000.0, 2),
        "inference_time": round(float(inference_time) * 1000.0, 3),
    }


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


def train_one_epoch(model: nn.Module, dataset: MultiEncoderSlideDataset, optimizer, device: torch.device, grad_clip: float) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    for idx in indices:
        raw_features, label, _ = dataset[int(idx)]
        raw_features = move_features(raw_features, device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(raw_features)
        loss = criterion(logits, label_t)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(dataset), 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: MultiEncoderSlideDataset, device: torch.device, decision_threshold: float) -> Tuple[EvalResult, pd.DataFrame]:
    model.eval()
    labels, probs, rows = [], [], []
    for idx in range(len(dataset)):
        raw_features, label, slide_id = dataset[idx]
        raw_features = move_features(raw_features, device)
        logits, _ = model(raw_features)
        prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
        labels.append(int(label))
        probs.append(prob)
        rows.append({"slide_id": slide_id, "label": int(label), "prob_class1": prob})
    metrics = compute_metrics(labels, probs, decision_threshold=decision_threshold)
    pred_df = pd.DataFrame(rows)
    pred_df["threshold"] = metrics.threshold
    pred_df["pred"] = (pred_df["prob_class1"] >= metrics.threshold).astype(int)
    return metrics, pred_df


def save_fold_outputs(
    fold_dir: Path,
    method_name: str,
    metrics: EvalResult,
    predictions: pd.DataFrame,
    efficiency: Mapping[str, float] | None = None,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    metric_row = {**{"method": method_name}, **asdict(metrics)}
    if efficiency:
        metric_row.update(dict(efficiency))
    pd.DataFrame([metric_row]).to_csv(
        fold_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    predictions.to_csv(
        fold_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    cm = pd.DataFrame(
        [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm.to_csv(fold_dir / "confusion_matrix.csv", encoding="utf-8-sig")


def train_fold(
    args: argparse.Namespace,
    method: str,
    model_name: str,
    fold: int,
    input_dims: Mapping[str, int],
    manifest: pd.DataFrame,
    clinical_df: pd.DataFrame,
    device: torch.device,
    output_dir: Path,
) -> Mapping[str, object]:
    encoder_names = list(input_dims.keys())
    train_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="train",
        clinical_df=clinical_df,
        label_col=args.label_col,
        encoder_names=encoder_names,
        max_patches=args.max_patches,
        training=True,
    )
    val_ds = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=fold,
        split="val",
        clinical_df=clinical_df,
        label_col=args.label_col,
        encoder_names=encoder_names,
        max_patches=args.eval_max_patches,
        training=False,
    )
    seed_everything(args.seed + fold + stable_name_offset(model_name))
    if method == "no_fusion":
        if len(input_dims) != 1:
            raise ValueError("no_fusion expects exactly one encoder.")
        encoder_name = next(iter(input_dims.keys()))
        model = SingleEncoderABMIL(encoder_name, int(input_dims[encoder_name]), args).to(device)
    else:
        model = ProjectedFusionABMIL(method, input_dims, args).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_epochs, 1), eta_min=args.lr * 0.01)
    fold_dir = output_dir / model_name / f"fold_{fold}"
    best_path = fold_dir / "best_model.pt"
    best_auc = -np.inf
    best_metrics: EvalResult | None = None
    no_improve = 0

    print(f"\n{model_name} | fold {fold}: train={len(train_ds)}, val={len(val_ds)}, encoders={encoder_names}")
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_ds, optimizer, device, args.grad_clip)
        metrics, pred_df = evaluate(model, val_ds, device, args.threshold)
        scheduler.step()
        print(
            f"{model_name} | Fold {fold} | Epoch {epoch:03d}/{args.max_epochs} | "
            f"loss={train_loss:.4f} | AUC={metrics.auc:.4f} | AUPRC={metrics.auprc:.4f} | "
            f"ACC@{args.threshold:g}={metrics.accuracy:.4f} | F1@{args.threshold:g}={metrics.f1:.4f}"
        )

        improved = not np.isnan(metrics.auc) and metrics.auc > best_auc
        if improved:
            best_auc = metrics.auc
            best_metrics = metrics
            no_improve = 0
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": method,
                    "model_name": model_name,
                    "fold": int(fold),
                    "input_dims": dict(input_dims),
                    "metrics": asdict(metrics),
                },
                best_path,
            )
            save_fold_outputs(fold_dir, model_name, metrics, pred_df)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"{model_name} | fold {fold}: early stopping at epoch {epoch}. Best AUC={best_auc:.4f}")
                break

    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
    decision_threshold = float(args.threshold)
    print(f"{model_name} | fold {fold}: fixed decision threshold={decision_threshold:.6f}")
    with open(fold_dir / "decision_threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(decision_threshold),
                "source": "fixed_config",
            },
            f,
            indent=2,
        )
    final_metrics, final_pred = evaluate(model, val_ds, device, decision_threshold)
    efficiency = profile_offline_efficiency(
        model=model,
        dataset=val_ds,
        device=device,
        profile_samples=args.profile_samples,
        profile_warmup=args.profile_warmup,
        profile_repeat=args.profile_repeat,
    )
    save_fold_outputs(fold_dir, model_name, final_metrics, final_pred, efficiency=efficiency)
    if best_metrics is None:
        best_metrics = final_metrics

    return {
        "method": method,
        "model_name": model_name,
        "fold": int(fold),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "encoders": ",".join(encoder_names),
        **asdict(final_metrics),
        **efficiency,
    }


def method_input_dims(method: str, fold_dims: Mapping[str, int], single_encoders: Sequence[str] | None = None) -> List[Tuple[str, Dict[str, int]]]:
    if method == "no_fusion":
        encoder_names = list(single_encoders) if single_encoders else sorted(fold_dims.keys())
        missing = sorted(set(encoder_names) - set(fold_dims.keys()))
        if missing:
            raise ValueError(f"Requested no_fusion encoders not found: {missing}")
        return [(f"no_fusion__{name}", {name: int(fold_dims[name])}) for name in encoder_names]
    return [(method, {name: int(fold_dims[name]) for name in sorted(fold_dims.keys())})]


def summarize(fold_rows: List[Mapping[str, object]], output_dir: Path) -> None:
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    if fold_df.empty:
        return

    group_cols = ["method", "model_name", "encoders"]
    metric_cols = ["auc", "auprc", "accuracy", "f1", "precision", "recall", "parameters", "flops", "inference_time"]
    summary_rows = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, key_values))
        summary_key_dict = {"method": key_dict["method"]}
        model_dir = output_dir / str(key_dict["model_name"])
        model_dir.mkdir(parents=True, exist_ok=True)
        group = group.sort_values("fold")
        group.to_csv(model_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

        rows = []
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.notna().sum() == 0:
                rows.append({
                    **summary_key_dict,
                    "metric": metric,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                })
                continue
            rows.append({
                **summary_key_dict,
                "metric": metric,
                "mean": float(values.mean(skipna=True)),
                "std": float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0,
                "min": float(values.min(skipna=True)),
                "median": float(values.median(skipna=True)),
                "max": float(values.max(skipna=True)),
            })
        summary_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            model_dir / "summary_metrics.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.2f",
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args)
    seed_everything(args.seed)

    manifest_path = ensure_manifest(args)
    args.manifest = manifest_path
    manifest = pd.read_csv(manifest_path)
    clinical_df, _, _ = load_uvm_data(args.clinical_path, args.label_col)
    clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)

    methods = normalize_methods(args.methods)
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
    print("Offline fusion baseline training")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Clinical: {args.clinical_path}")
    print(f"Methods: {methods}")
    print(f"Folds: {folds}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.current_device()} | {torch.cuda.get_device_name(device)}")
    print(f"Output: {output_dir}")

    fold_rows: List[Mapping[str, object]] = []
    for method in methods:
        for fold in folds:
            fold_dims = infer_input_dims(manifest, fold=int(fold))
            for model_name, input_dims in method_input_dims(method, fold_dims, args.single_encoders):
                row = train_fold(
                    args=args,
                    method=method,
                    model_name=model_name,
                    fold=int(fold),
                    input_dims=input_dims,
                    manifest=manifest,
                    clinical_df=clinical_df,
                    device=device,
                    output_dir=output_dir,
                )
                fold_rows.append(row)
                summarize(fold_rows, output_dir)

    summarize(fold_rows, output_dir)
    print("\nSummary files:")
    if fold_rows:
        summary_dirs = sorted({str(row["model_name"]) for row in fold_rows})
        for model_name in summary_dirs:
            summary_path = output_dir / model_name / "summary_metrics.csv"
            if summary_path.exists():
                print(f"  {summary_path}")
    print(f"\nSaved output: {output_dir}")


if __name__ == "__main__":
    main()
