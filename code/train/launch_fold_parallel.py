"""Launch fold-level parallel training across multiple GPUs.

This is the pragmatic multi-GPU mode for small WSI MIL datasets: each GPU runs
an independent Python process on a different subset of CV folds.

Examples:
    python code/train/launch_fold_parallel.py --config code/config/fold_parallel.yml

    python code/train/launch_fold_parallel.py --config code/config/fold_parallel_offline.yml
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
DEFAULT_CONFIGS = {
    "gme": CODE_DIR / "config" / "gme.yml",
    "offline": CODE_DIR / "config" / "offline_fusion_baselines.yml",
}
DEFAULT_LAUNCH_CONFIG = CODE_DIR / "config" / "fold_parallel.yml"
TARGET_SCRIPTS = {
    "gme": CODE_DIR / "train" / "train_gme.py",
    "offline": CODE_DIR / "train" / "train_offline_fusion_baselines.py",
}
DEFAULT_OUTPUT_DIRS = {
    "gme": PROJECT_ROOT / "output" / "GME",
    "offline": PROJECT_ROOT / "output" / "Offline_Fusion_Baselines",
}
METRICS = ("auc", "auprc", "sensitivity", "specificity", "accuracy", "f1", "precision", "recall")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_LAUNCH_CONFIG)
    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Run CV folds in parallel across GPUs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LAUNCH_CONFIG, help="YAML/JSON launcher config. CLI args override config.")
    parser.add_argument("--target", choices=["gme", "offline"], default="gme")
    parser.add_argument("--train-config", type=Path, default=None, help="Config passed to the target training script.")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"], help="GPU ids, e.g. 0 1.")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-mode", choices=["contiguous", "round_robin"], default="contiguous")
    parser.add_argument("--experiment-name", default=None, help="Base experiment name. GPU suffixes are added.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override train script output dir.")
    parser.add_argument("--launch-output-dir", type=Path, default=PROJECT_ROOT / "output" / "Fold_Parallel")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional args passed to the training script after '--'.",
    )
    config = load_config(config_args.config)
    if config:
        config = {str(key).replace("-", "_"): value for key, value in config.items()}
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid_dests)
        if unknown:
            raise ValueError(f"Unknown launcher config keys in {config_args.config}: {unknown}")
        parser.set_defaults(**config)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    for name in ("train_config", "output_dir", "launch_output_dir"):
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    return args


def load_config(config_path: Path | None) -> Dict[str, object]:
    if config_path is None or not config_path.exists():
        return {}
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except ImportError:
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def split_folds(folds: Sequence[int], n_groups: int, mode: str) -> List[List[int]]:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive.")
    groups: List[List[int]] = [[] for _ in range(n_groups)]
    folds = list(folds)
    if mode == "round_robin":
        for idx, fold in enumerate(folds):
            groups[idx % n_groups].append(int(fold))
        return [group for group in groups if group]

    chunk_size = (len(folds) + n_groups - 1) // n_groups
    return [
        [int(fold) for fold in folds[start:start + chunk_size]]
        for start in range(0, len(folds), chunk_size)
    ]


def fold_label(folds: Sequence[int]) -> str:
    if not folds:
        return "none"
    if len(folds) == 1:
        return str(folds[0])
    return f"{folds[0]}-{folds[-1]}"


def normalize_extra_args(extra_args: Sequence[str]) -> List[str]:
    if isinstance(extra_args, str):
        extra_args = shlex.split(extra_args)
    args = list(extra_args)
    if args and args[0] == "--":
        return args[1:]
    return args


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in ("method", "model_name", "encoders") if col in fold_df.columns]
    rows = []
    if group_cols:
        groups = fold_df.groupby(group_cols, dropna=False)
    else:
        groups = [((), fold_df)]

    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys)) if group_cols else {}
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            rows.append({
                **key_values,
                "metric": metric,
                "mean": float(values.mean(skipna=True)),
                "std": float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0,
                "min": float(values.min(skipna=True)),
                "median": float(values.median(skipna=True)),
                "max": float(values.max(skipna=True)),
            })
    return pd.DataFrame(rows)


def aggregate_subruns(subrun_dirs: Sequence[Path], launch_dir: Path) -> None:
    rows = []
    subrun_rows = []
    for run_dir in subrun_dirs:
        metrics_path = run_dir / "fold_metrics.csv"
        subrun_rows.append({"subrun_dir": str(run_dir), "fold_metrics": str(metrics_path), "exists": metrics_path.exists()})
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            df["subrun_dir"] = str(run_dir)
            rows.append(df)

    pd.DataFrame(subrun_rows).to_csv(launch_dir / "subruns.csv", index=False, encoding="utf-8-sig")
    if not rows:
        print("[Warning] No fold_metrics.csv found to aggregate.")
        return

    fold_df = pd.concat(rows, ignore_index=True)
    fold_df.to_csv(launch_dir / "parallel_fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    summary_df = summarize_metrics(fold_df)
    summary_df.to_csv(launch_dir / "parallel_summary_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")


def main() -> None:
    args = parse_args()
    train_config_path = args.train_config or DEFAULT_CONFIGS[args.target]
    train_config = load_config(train_config_path)
    output_dir = args.output_dir or Path(str(train_config.get("output_dir", DEFAULT_OUTPUT_DIRS[args.target])))
    base_experiment = args.experiment_name or str(train_config.get("experiment_name", f"{args.target}_parallel"))
    python_executable = args.python or sys.executable
    extra_args = normalize_extra_args(args.extra_args)
    fold_groups = split_folds(args.folds, len(args.gpus), args.split_mode)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_dir = args.launch_output_dir / args.target / f"{base_experiment}_{timestamp}"
    train_run_dir = output_dir / base_experiment / timestamp
    launch_dir.mkdir(parents=True, exist_ok=True)
    train_run_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    expected_subrun_dirs = []
    script = TARGET_SCRIPTS[args.target]
    for idx, (gpu_id, folds) in enumerate(zip(args.gpus, fold_groups)):
        exp_name = f"{base_experiment}_gpu{gpu_id}_folds{fold_label(folds)}"
        subrun_dir = train_run_dir / exp_name
        command = [
            python_executable,
            str(script),
            "--config",
            str(train_config_path),
            "--folds",
            *[str(fold) for fold in folds],
            "--experiment-name",
            exp_name,
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(subrun_dir),
            *extra_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = launch_dir / f"gpu{gpu_id}_folds{fold_label(folds)}.log"
        expected_subrun_dirs.append(subrun_dir)

        print(f"\nGPU {gpu_id} folds {folds}")
        print("$ " + " ".join(command))
        print(f"output: {subrun_dir}")
        print(f"log: {log_path}")
        if args.dry_run:
            continue

        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu_id, folds, process, log_file))

    with open(launch_dir / "launch_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "launcher_config": str(args.config) if args.config else "",
                "train_config": str(train_config_path),
                "gpus": args.gpus,
                "folds": args.folds,
                "split_mode": args.split_mode,
                "output_dir": str(output_dir),
                "train_run_dir": str(train_run_dir),
                "base_experiment": base_experiment,
                "extra_args": extra_args,
            },
            f,
            indent=2,
        )

    if args.dry_run:
        print(f"\nDry run only. Launch metadata saved to: {launch_dir}")
        return

    failures = []
    for gpu_id, folds, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failures.append((gpu_id, folds, return_code))
            print(f"[Failed] GPU {gpu_id}, folds {folds}, exit={return_code}")
        else:
            print(f"[Done] GPU {gpu_id}, folds {folds}")

    aggregate_subruns(expected_subrun_dirs, launch_dir)

    if failures:
        raise RuntimeError(f"Some fold-parallel jobs failed: {failures}. Logs are in {launch_dir}")

    print(f"\nFold-parallel run finished.")
    print(f"Launch output: {launch_dir}")
    print(f"Training output: {train_run_dir}")
    print(f"Aggregated metrics: {launch_dir / 'parallel_summary_metrics.csv'}")


if __name__ == "__main__":
    main()
