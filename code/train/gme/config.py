"""Command-line and configuration handling for GME training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
PATH_ARGS = ("manifest", "manifest_dir", "output_dir", "run_dir")


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
    parser.add_argument(
        "--training-protocol",
        choices=["nested_refit", "fixed_split_no_refit"],
        default="nested_refit",
        help=(
            "'nested_refit' selects an epoch on inner validation and retrains on all outer-train data. "
            "'fixed_split_no_refit' evaluates the selected inner checkpoint directly on outer-test."
        ),
    )
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
        "--weight-averaging",
        choices=["none", "ema"],
        default="none",
        help="Weight averaging used for inner selection and final outer-test evaluation.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="Optimizer-step EMA decay. Used only when --weight-averaging=ema.",
    )
    parser.add_argument(
        "--ema-start-epoch",
        type=int,
        default=2,
        help="First Stage2 epoch included in EMA. Earlier epochs use raw weights.",
    )
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
        "--adaptive-beacon-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Adapt the Stage2 Beacon weight from train-only epoch losses so its weighted "
            "contribution does not dominate classification."
        ),
    )
    parser.add_argument(
        "--beacon-loss-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum target ratio (weighted Beacon loss / classification loss) used by "
            "adaptive Stage2 Beacon weighting."
        ),
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
    if float(args.beacon_constraint_weight) < 0.0:
        raise ValueError("--beacon-constraint-weight must be non-negative.")
    if float(args.beacon_loss_ratio) <= 0.0:
        raise ValueError("--beacon-loss-ratio must be positive.")
    if not 0.0 < float(args.ema_decay) < 1.0:
        raise ValueError("--ema-decay must be between 0 and 1.")
    if int(args.ema_start_epoch) < 1:
        raise ValueError("--ema-start-epoch must be at least 1.")
    for name in PATH_ARGS:
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    if args.clinical_path is None:
        raise ValueError("--clinical-path is required, either in CLI or config.")
    return args


