"""Run the GME fusion branch and/or its post-hoc analysis branch.

The training entry point remains ``train_gme.py``.  This launcher owns the
workflow-level sequencing so CKA, Spearman, pairwise interactions, and
post-hoc routing analyses never become part of model training or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.output_guard import allocate_run_dir, resolve_project_path as resolve_guard_path


DEFAULT_CONFIG = PROJECT_ROOT / "code" / "config" / "gme.yml"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "output" / "Analysis_Result"
WORKFLOW_MODES = ("fusion_only", "analysis_only", "fusion_and_analysis")


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Reading YAML workflow config requires PyYAML.") from exc
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Workflow config must be a mapping, got {type(data)}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GME fusion and post-hoc analysis branches as one reproducible workflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--workflow-mode",
        choices=WORKFLOW_MODES,
        default=None,
        help="Override workflow_mode from the config file.",
    )
    parser.add_argument(
        "--analysis-source-run-dir",
        type=Path,
        default=None,
        help="Override analysis_source_run_dir for analysis_only.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used by analysis scripts. Default: device from the run config.",
    )
    return parser.parse_args()


def resolve_project_path(value: object, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("A required path is missing from the workflow configuration.")
        return default
    return resolve_guard_path(str(value))


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}] $ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_spearman_pairs(spearman_root: Path, feature_dirs: list[str]) -> Path:
    candidates = sorted(
        spearman_root.rglob("encoder_pair_spearman_summary.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Spearman completed without producing encoder_pair_spearman_summary.csv.")
    summary_path = candidates[0]
    with open(summary_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {
        tuple(sorted((str(row["encoder_a"]), str(row["encoder_b"]))))
        for row in rows
    }
    expected = {
        tuple(sorted((feature_dirs[left], feature_dirs[right])))
        for left in range(len(feature_dirs))
        for right in range(left + 1, len(feature_dirs))
    }
    if pairs != expected:
        raise RuntimeError(
            "Spearman pair coverage is incomplete or inconsistent. "
            f"Expected {len(expected)} pairs from {feature_dirs}, found {len(pairs)} in {summary_path}."
        )
    return summary_path


def run_fusion(config_path: Path, run_dir: Path) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "train" / "train_gme.py"),
        "--config",
        str(config_path),
        "--workflow-mode",
        "fusion_only",
        "--run-dir",
        str(run_dir),
    ]
    run_command(command, "fusion")


def source_run_config(run_dir: Path) -> dict[str, object]:
    path = run_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"GME run config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def analysis_device(run_config: Mapping[str, object], override: str | None) -> str:
    device = str(override or run_config.get("device", "cuda"))
    if device == "cuda" and run_config.get("gpu_id") is not None:
        return f"cuda:{int(run_config['gpu_id'])}"
    return device


def run_analysis(source_run_dir: Path, analysis_dir: Path, device_override: str | None) -> None:
    run_config = source_run_config(source_run_dir)
    manifest = resolve_project_path(run_config.get("manifest"))
    clinical_path = resolve_project_path(run_config.get("clinical_path"))
    feature_dirs = [str(item) for item in run_config.get("feature_dirs", [])]
    if len(feature_dirs) < 2:
        raise ValueError("At least two feature_dirs are required for CKA and Spearman analysis.")
    device = analysis_device(run_config, device_override)
    label_col = str(run_config.get("label_col", "d3m3"))
    replacement_strategy = str(run_config.get("replacement_strategy", "mean"))
    gaussian_std_scale = str(run_config.get("gaussian_std_scale", 1.0))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    cka_command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "evaluation" / "compute_linear_cka.py"),
        "--manifest",
        str(manifest),
        "--feature-dirs",
        *feature_dirs,
        "--output-dir",
        str(analysis_dir / "cka"),
        "--device",
        device,
    ]
    run_command(cka_command, "analysis: CKA")
    cka_files = sorted(
        (analysis_dir / "cka").rglob("pairwise_cka_by_wsi.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not cka_files:
        raise RuntimeError("CKA completed without producing pairwise_cka_by_wsi.csv.")

    spearman_command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "evaluation" / "Spearman.py"),
        "--manifest",
        str(manifest),
        "--feature-dirs",
        *feature_dirs,
        "--output-dir",
        str(analysis_dir / "spearman"),
        "--device",
        device,
        "--cka-file",
        str(cka_files[0]),
    ]
    run_command(spearman_command, "analysis: Spearman")
    validate_spearman_pairs(analysis_dir / "spearman", feature_dirs)

    fold_dirs = sorted(
        (path for path in source_run_dir.glob("fold_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {source_run_dir}")
    for fold_dir in fold_dirs:
        fold = int(fold_dir.name.rsplit("_", 1)[-1])
        interaction_command = [
            sys.executable,
            str(PROJECT_ROOT / "code" / "evaluation" / "compute_pairwise_interactions.py"),
            "--fold-dir",
            str(fold_dir),
            "--manifest",
            str(manifest),
            "--clinical-path",
            str(clinical_path),
            "--label-col",
            label_col,
            "--fold",
            str(fold),
            "--split",
            "val",
            "--output-dir",
            str(analysis_dir / "interactions" / f"fold_{fold}"),
            "--device",
            device,
            "--replacement-strategy",
            replacement_strategy,
            "--gaussian-std-scale",
            gaussian_std_scale,
        ]
        run_command(interaction_command, f"analysis: interaction fold {fold}")

    quadrant_command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "evaluation" / "plot_cka_interaction_quadrants.py"),
        "--cka-root",
        str(analysis_dir / "cka"),
        "--interaction-root",
        str(analysis_dir / "interactions"),
        "--output-dir",
        str(analysis_dir / "cka_interaction_quadrants"),
    ]
    run_command(quadrant_command, "analysis: CKA-interaction quadrants")

    routing_weight_command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "evaluation" / "plot_gme_routing_weight_distribution.py"),
        "--run-dir",
        str(source_run_dir),
        "--output-dir",
        str(analysis_dir / "routing_weight_distribution"),
    ]
    run_command(routing_weight_command, "analysis: OOF routing-weight distribution")

    with open(analysis_dir / "workflow_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_run_dir": str(source_run_dir),
                "manifest": str(manifest),
                "clinical_path": str(clinical_path),
                "feature_dirs": feature_dirs,
                "device": device,
                "analysis": [
                    "linear_cka",
                    "spearman",
                    "pairwise_interactions",
                    "cka_interaction_quadrants",
                    "oof_routing_weight_distribution",
                ],
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = str(args.workflow_mode or config.get("workflow_mode", "fusion_and_analysis"))
    if mode not in WORKFLOW_MODES:
        raise ValueError(f"Unsupported workflow_mode={mode!r}; choose one of {WORKFLOW_MODES}.")

    source_run_dir = args.analysis_source_run_dir
    if source_run_dir is None and config.get("analysis_source_run_dir"):
        source_run_dir = resolve_project_path(config["analysis_source_run_dir"])

    run_dir: Path | None = None
    if mode in {"fusion_only", "fusion_and_analysis"}:
        output_dir = resolve_project_path(config.get("output_dir"), PROJECT_ROOT / "output" / "GME")
        experiment_name = str(config.get("experiment_name", "gme"))
        run_dir = allocate_run_dir(output_dir, experiment_name)
        run_fusion(args.config, run_dir)
        source_run_dir = run_dir

    if mode == "analysis_only":
        if source_run_dir is None:
            raise ValueError("analysis_only requires analysis_source_run_dir in gme.yml or --analysis-source-run-dir.")
        if not source_run_dir.exists():
            raise FileNotFoundError(f"Analysis source run directory not found: {source_run_dir}")

    if mode in {"analysis_only", "fusion_and_analysis"}:
        assert source_run_dir is not None
        analysis_dir = DEFAULT_ANALYSIS_ROOT / source_run_dir.name
        run_analysis(source_run_dir, analysis_dir, args.device)
        print(f"\nAnalysis output: {analysis_dir}")
    elif run_dir is not None:
        print(f"\nFusion output: {run_dir}")


if __name__ == "__main__":
    main()
