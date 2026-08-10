"""Post-hoc pairwise interaction analysis for a trained GME checkpoint.

This script deliberately lives outside the training path.  It evaluates the
second-order intervention quantity

    I(i, j) = f(all) - f(all\\i) - f(all\\j) + f(all\\{i,j})

on held-out WSI bags using the frozen checkpoint.  The interaction values do
not alter routing, fused representations, predictions, parameter counts, or
efficiency measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch

# Keep direct execution from the repository root compatible with the training
# entry points, whose package root is ``code/``.
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from architecture.gme_model import GMEModel
from data_utils.cls_dataset import load_uvm_data
from data_utils.gme_dataset import MultiEncoderSlideDataset
from train.gme.experiment import build_replacement_baselines, move_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "Middle_Fusion_Manifests" / "middle_fusion_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Pairwise_Interactions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute post-hoc GME pairwise interaction matrices.")
    parser.add_argument("--fold-dir", type=Path, required=True, help="Completed GME fold directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: <fold-dir>/best_gme_model.pt.")
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Train-only replacement baselines. Default: <fold-dir>/replacement_baselines.pt.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clinical-path", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--slide-ids", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=0)
    parser.add_argument("--replacement-strategy", choices=["mean", "zero", "gaussian"], default=None)
    parser.add_argument("--gaussian-std-scale", type=float, default=None)
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Class whose logit margin defines interaction sign; class 1 is the binary positive class.",
    )
    parser.add_argument("--strict", action="store_true", help="Stop at the first invalid WSI.")
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if args.gpu_id is not None and str(args.device) == "cuda":
            args.device = f"cuda:{int(args.gpu_id)}"
        torch.cuda.set_device(torch.device(args.device))
    return torch.device(args.device)


def load_run_config(fold_dir: Path) -> Dict[str, object]:
    config_path = fold_dir.parent / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_baselines(path: Path, device: torch.device) -> Mapping[str, Mapping[str, torch.Tensor]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing train-only baseline file: {path}. "
            "Run GME training first or pass --baseline-file explicitly."
        )
    payload = torch.load(path, map_location=device)
    raw = payload.get("baselines") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid baseline payload in {path}; expected a 'baselines' mapping.")
    return {
        str(name): {
            str(key): value.to(device) if torch.is_tensor(value) else value
            for key, value in stats.items()
        }
        for name, stats in raw.items()
    }


def build_model(run_config: Mapping[str, object], checkpoint_payload: Mapping[str, object], device: torch.device) -> GMEModel:
    input_dims = checkpoint_payload.get("input_dims")
    if not isinstance(input_dims, dict):
        raise ValueError("Checkpoint metadata is missing input_dims.")
    routing_mode = str(run_config.get("routing_mode", "online_attribution"))
    model = GMEModel(
        input_dims={str(k): int(v) for k, v in input_dims.items()},
        target_dim=int(run_config.get("target_dim", 512)),
        projection_dropout=float(run_config.get("projection_dropout", 0.0)),
        d_inner=int(run_config.get("d_inner", 256)),
        d_attn=int(run_config.get("d_attn", 128)),
        n_classes=int(run_config.get("n_classes", 2)),
        droprate=float(run_config.get("droprate", 0.25)),
        routing_temperature=float(run_config.get("routing_temperature", 0.5)),
        routing_logit_scale=float(run_config.get("routing_logit_scale", 1.0)),
        attribution_target=str(run_config.get("attribution_target", "predicted_class")),
        student_router_hidden_dim=(
            int(run_config.get("student_router_hidden_dim", 0)) if routing_mode == "teacher_student" else 0
        ),
        student_router_temperature=float(run_config.get("student_router_temperature", 1.0)),
    ).to(device)
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a state_dict.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Checkpoints produced before interaction was moved to the analysis branch
    # contain the retired gate/statistics tensors. They are intentionally
    # ignored; projection, router, and classifier weights remain compatible.
    unexpected = [name for name in unexpected if not name.startswith("interaction_")]
    if unexpected:
        raise ValueError(f"Checkpoint has incompatible unexpected parameters: {unexpected}")
    missing = [name for name in missing if not name.startswith("interaction_")]
    if missing:
        raise ValueError(f"Checkpoint is missing model parameters: {missing}")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.max_patches < 0:
        raise ValueError("--max-patches must be non-negative.")
    device = resolve_device(args)
    fold_dir = args.fold_dir
    checkpoint_path = args.checkpoint or fold_dir / "best_gme_model.pt"
    baseline_path = args.baseline_file or fold_dir / "replacement_baselines.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    run_config = load_run_config(fold_dir)
    model = build_model(run_config, checkpoint, device)

    fold = args.fold
    if fold is None:
        fold = int(checkpoint.get("fold", 0)) or int(fold_dir.name.rsplit("_", 1)[-1])
    clinical_path = args.clinical_path or run_config.get("clinical_path")
    label_col = args.label_col or run_config.get("label_col", "d3m3")
    if not clinical_path:
        raise ValueError("Clinical path is required; pass --clinical-path or use a run config containing it.")
    # Match the training path: the clinical CSV stores ``SCNA Cluster No.``
    # and load_uvm_data derives the binary ``d3m3`` label used by GME.
    clinical_df, _, _ = load_uvm_data(str(clinical_path), str(label_col))
    clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)
    manifest = pd.read_csv(args.manifest)
    if fold not in set(manifest["fold"].dropna().astype(int)):
        raise ValueError(f"Fold {fold} was not found in {args.manifest}.")
    feature_dirs = list(model.encoder_names)
    dataset = MultiEncoderSlideDataset(
        manifest=manifest,
        fold=int(fold),
        split=args.split,
        clinical_df=clinical_df,
        label_col=str(label_col),
        max_patches=args.max_patches,
        training=False,
    )
    requested = set(map(str, args.slide_ids)) if args.slide_ids else None
    if requested is not None:
        missing = sorted(requested - set(dataset.slide_ids))
        if missing:
            raise ValueError(f"Requested slides are not in fold={fold}, split={args.split}: {missing}")
    baselines = load_baselines(baseline_path, device)
    replacement_strategy = args.replacement_strategy or str(run_config.get("replacement_strategy", "mean"))
    gaussian_std_scale = (
        float(args.gaussian_std_scale)
        if args.gaussian_std_scale is not None
        else float(run_config.get("gaussian_std_scale", 1.0))
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    matrix_dir = output_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold_dir": str(fold_dir),
                "checkpoint": str(checkpoint_path),
                "baseline_file": str(baseline_path),
                "manifest": str(args.manifest),
                "fold": int(fold),
                "split": args.split,
                "feature_dirs": feature_dirs,
                "replacement_strategy": replacement_strategy,
                "gaussian_std_scale": gaussian_std_scale,
                "target_class": int(args.target_class),
                "analysis_only": True,
            },
            handle,
            indent=2,
        )

    pair_rows = []
    summary_rows = []
    errors = []
    with torch.no_grad():
        for index in range(len(dataset)):
            raw_features, label, slide_id = dataset[index]
            slide_id = str(slide_id)
            if requested is not None and slide_id not in requested:
                continue
            print(f"[{index + 1}/{len(dataset)}] {slide_id}")
            try:
                raw_features = move_features(raw_features, device)
                projected = model.project(raw_features)
                _, _, interaction_matrix = model.intervention_attribution(
                    projected=projected,
                    baselines=baselines,
                    replacement_strategy=replacement_strategy,
                    gaussian_std_scale=gaussian_std_scale,
                    compute_interactions=True,
                    interaction_target_class=int(args.target_class),
                )
                matrix = interaction_matrix.detach().cpu().numpy().astype(np.float64)
                pd.DataFrame(matrix, index=feature_dirs, columns=feature_dirs).to_csv(
                    matrix_dir / f"{slide_id}_interaction_matrix.csv", encoding="utf-8-sig", float_format="%.6f"
                )
                values = []
                for i, left in enumerate(feature_dirs):
                    for j in range(i + 1, len(feature_dirs)):
                        right = feature_dirs[j]
                        value = float(matrix[i, j])
                        values.append(value)
                        pair_rows.append(
                            {
                                "slide_id": slide_id,
                                "label": int(label),
                                "encoder_i": left,
                                "encoder_j": right,
                                "interaction": value,
                                "abs_interaction": abs(value),
                                "target_class": int(args.target_class),
                            }
                        )
                summary_rows.append(
                    {
                        "slide_id": slide_id,
                        "label": int(label),
                        "mean_interaction": float(np.mean(values)),
                        "mean_abs_interaction": float(np.mean(np.abs(values))),
                        "max_abs_interaction": float(np.max(np.abs(values))),
                        "n_pairs": len(values),
                    }
                )
            except Exception as exc:
                if args.strict:
                    raise
                errors.append({"slide_id": slide_id, "error": str(exc)})
                print(f"[Warning] {slide_id}: {exc}")

    if not summary_rows:
        raise RuntimeError("Interaction analysis failed for every selected WSI.")
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(output_dir / "pairwise_interactions.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "wsi_interaction_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    matrix = pd.DataFrame(np.nan, index=feature_dirs, columns=feature_dirs)
    for name in feature_dirs:
        matrix.loc[name, name] = 0.0
    for (left, right), group in pairs_df.groupby(["encoder_i", "encoder_j"]):
        value = float(group["interaction"].mean())
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    matrix.index.name = "encoder"
    matrix.to_csv(output_dir / "interaction_matrix.csv", encoding="utf-8-sig", float_format="%.6f")
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    print(f"Completed WSIs: {len(summary_rows)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
