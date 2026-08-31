"""Command-line and configuration handling for GME training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from data_utils.cohort import resolve_cohort_spec


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Manifests" / "Manifests_seed35" / "fusion_manifest.csv"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "GME"
DEFAULT_FEATURE_DIRS = [
    "features_hoptimus1",
    "features_virchow",
    "features_hoptimus0",
]
PATH_ARGS = ("manifest", "manifest_dir", "output_dir", "run_dir")
WORKFLOW_MODES = ("fusion_only", "analysis_only", "fusion_and_analysis")


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
    parser.add_argument(
        "--workflow-mode",
        choices=WORKFLOW_MODES,
        default="fusion_only",
        help="Workflow launcher mode. train_gme.py itself only accepts fusion_only.",
    )
    parser.add_argument(
        "--analysis-source-run-dir",
        type=Path,
        default=None,
        help="Existing GME run directory used by the analysis_only workflow.",
    )
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
    parser.add_argument(
        "--downstream-head",
        type=str.upper,
        choices=["ABMIL", "TRANSMIL", "GNN", "MLP"],
        default="ABMIL",
        help="Downstream head applied to the routed patch bag.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=256,
        help="Hidden width for the MLP downstream head.",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=256,
        help="Hidden width for the GNN downstream head.",
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
        help="Number of local graph-convolution layers in the GNN head.",
    )

    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
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
        "--routing-mode",
        choices=["online_attribution", "teacher_student"],
        default="online_attribution",
        help="Use existing online attribution routing or frozen-teacher router distillation.",
    )
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=20,
        help="Maximum mean-fusion teacher epochs when --routing-mode=teacher_student.",
    )
    parser.add_argument("--teacher-patience", type=int, default=5)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument(
        "--mean-fusion-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize the Stage-2 student from the trained mean-fusion teacher. "
            "Disable for the warm-start ablation."
        ),
    )
    parser.add_argument(
        "--teacher-distill-weight",
        type=float,
        default=0.5,
        help=(
            "KL loss weight matching student router weights to frozen teacher attribution. "
            "Set to 0 to skip teacher-target/LOO construction and KL lookup."
        ),
    )
    parser.add_argument(
        "--teacher-kl-loss-ratio",
        type=float,
        default=0.1,
        help=(
            "Maximum weighted KL / classification-loss ratio per sample. "
            "Set to 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--teacher-target-temperature",
        type=float,
        default=1.0,
        help="Temperature converting normalized true-label LOO scores into teacher router targets.",
    )
    parser.add_argument(
        "--teacher-target-clip",
        type=float,
        default=3.0,
        help="Symmetric clip applied after per-slide teacher attribution z-scoring.",
    )
    parser.add_argument(
        "--student-router-hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the embedding-only student router.",
    )
    parser.add_argument(
        "--student-router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used by the embedding-only student router.",
    )
    parser.add_argument(
        "--student-router-use-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Condition student routing on each encoder's slide descriptor and the "
            "cross-encoder consensus descriptor. Disable for the no-consensus ablation."
        ),
    )
    parser.add_argument(
        "--teacher-freeze-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep student projections fixed to the teacher feature space during distillation.",
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
    cohort = resolve_cohort_spec(args.experiment_name)
    args.cohort = cohort.name
    args.label_col = cohort.label_col
    if not 0.0 < float(args.inner_val_fraction) < 1.0:
        raise ValueError("--inner-val-fraction must be between 0 and 1.")
    if int(args.teacher_epochs) < 1:
        raise ValueError("--teacher-epochs must be at least 1.")
    if int(args.teacher_patience) < 1:
        raise ValueError("--teacher-patience must be at least 1.")
    if float(args.teacher_lr) <= 0.0:
        raise ValueError("--teacher-lr must be positive.")
    if float(args.teacher_distill_weight) < 0.0:
        raise ValueError("--teacher-distill-weight must be non-negative.")
    if float(args.teacher_kl_loss_ratio) < 0.0:
        raise ValueError("--teacher-kl-loss-ratio must be non-negative.")
    if float(args.teacher_target_temperature) <= 0.0:
        raise ValueError("--teacher-target-temperature must be positive.")
    if float(args.teacher_target_clip) <= 0.0:
        raise ValueError("--teacher-target-clip must be positive.")
    if int(args.student_router_hidden_dim) < 1:
        raise ValueError("--student-router-hidden-dim must be at least 1.")
    if float(args.student_router_temperature) <= 0.0:
        raise ValueError("--student-router-temperature must be positive.")
    if int(args.mlp_hidden_dim) < 1:
        raise ValueError("--mlp-hidden-dim must be at least 1.")
    if int(args.gnn_hidden_dim) < 1:
        raise ValueError("--gnn-hidden-dim must be at least 1.")
    if int(args.gnn_layers) < 1:
        raise ValueError("--gnn-layers must be at least 1.")
    if args.routing_mode == "teacher_student" and args.training_protocol != "fixed_split_no_refit":
        raise ValueError(
            "--routing-mode=teacher_student currently requires "
            "--training-protocol=fixed_split_no_refit so teacher targets remain train-only."
        )
    for name in PATH_ARGS:
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    if args.clinical_path is None:
        raise ValueError("--clinical-path is required, either in CLI or config.")
    return args


