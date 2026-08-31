"""
Downstream evaluation for fused embeddings on UVM D3/M3 classification.

Input is a directory of fused WSI embeddings:

    fused_feat_dir/
        TCGA-xxx.h5
        TCGA-yyy.h5

Each h5 must contain a patch/instance feature matrix under "features", "feats",
or the first non-"coords" dataset. The script trains the selected downstream
head with 5-fold CV and reports AUC, AUPRC, sensitivity, and specificity.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture.gme_heads import build_downstream_head
from data_utils.cls_dataset import ClsDataset, load_uvm_data


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "Downstream_ABMIL"
FEATURE_KEYS = ("feats", "features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configurable downstream head on fused embeddings for UVM D3/M3 classification."
    )
    parser.add_argument("--fused-feat-dir", type=Path, required=True, help="Directory containing fused .h5 embeddings.")
    parser.add_argument("--clinical-path", required=True, help="Clinical CSV/XLSX with slide_id and SCNA Cluster No.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional fold manifest to reuse train/val splits.")
    parser.add_argument("--label-col", default="d3m3")
    parser.add_argument("--dataset-name", default="fused_embedding")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])

    parser.add_argument("--d-inner", type=int, default=256)
    parser.add_argument("--d-attn", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.25)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument(
        "--downstream-head",
        type=str.upper,
        choices=["ABMIL", "TRANSMIL", "GNN", "MLP"],
        default="ABMIL",
        help="Downstream head applied to each fused patch bag.",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_key(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as f:
        for key in FEATURE_KEYS:
            if key in f:
                return key
        candidates = [key for key in f.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(f.keys())}")
        return candidates[0]


def detect_feature_dim(feat_dir: Path) -> int:
    h5_files = sorted(feat_dir.glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {feat_dir}")
    key = get_dataset_key(h5_files[0])
    with h5py.File(h5_files[0], "r") as f:
        shape = f[key].shape
    if len(shape) != 2:
        raise RuntimeError(f"{h5_files[0]}: expected [N, D], got {shape}")
    return int(shape[1])


def available_slide_ids(feat_dir: Path, slide_ids: Sequence[str], labels: Sequence[int]) -> Tuple[List[str], List[int]]:
    valid_ids, valid_labels = [], []
    for sid, label in zip(slide_ids, labels):
        sid = str(sid)
        if (feat_dir / f"{sid}.h5").exists():
            valid_ids.append(sid)
            valid_labels.append(int(label))
    return valid_ids, valid_labels


def splits_from_manifest(manifest_path: Path, fused_feat_dir: Path, clinical_df: pd.DataFrame, label_col: str) -> List[Tuple[int, List[str], List[str]]]:
    manifest = pd.read_csv(manifest_path)
    required = {"fold", "split", "slide_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

    clinical_index = clinical_df.set_index("slide_id")
    splits = []
    for fold in sorted(manifest["fold"].dropna().astype(int).unique()):
        fold_rows = manifest[manifest["fold"].astype(int).eq(fold)]
        train_ids = sorted(set(fold_rows.loc[fold_rows["split"].eq("train"), "slide_id"].astype(str)))
        val_ids = sorted(set(fold_rows.loc[fold_rows["split"].eq("val"), "slide_id"].astype(str)))
        train_ids = [
            sid for sid in train_ids
            if (fused_feat_dir / f"{sid}.h5").exists() and sid in clinical_index.index and pd.notna(clinical_index.loc[sid, label_col])
        ]
        val_ids = [
            sid for sid in val_ids
            if (fused_feat_dir / f"{sid}.h5").exists() and sid in clinical_index.index and pd.notna(clinical_index.loc[sid, label_col])
        ]
        if train_ids and val_ids:
            splits.append((fold, train_ids, val_ids))
    if not splits:
        raise ValueError(f"No usable train/val splits found in {manifest_path}")
    return splits


def stratified_splits(valid_ids: Sequence[str], valid_labels: Sequence[int], n_folds: int, seed: int) -> List[Tuple[int, List[str], List[str]]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    ids = np.asarray(valid_ids)
    labels = np.asarray(valid_labels)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, labels), start=1):
        splits.append((fold_idx, ids[train_idx].tolist(), ids[val_idx].tolist()))
    return splits


def train_one_epoch(model: nn.Module, dataset: ClsDataset, optimizer, device: torch.device, accum_steps: int = 1) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, idx in enumerate(indices):
        feats, label, _ = dataset[int(idx)]
        feats = feats.to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)
        logits, _ = model(feats)
        loss = criterion(logits, label_t) / accum_steps
        loss.backward()
        total_loss += float(loss.detach().cpu()) * accum_steps

        if (step + 1) % accum_steps == 0 or step == len(indices) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / max(len(dataset), 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: ClsDataset, device: torch.device, threshold: float = 0.5) -> Dict[str, object]:
    model.eval()
    labels, probs, preds, slide_ids = [], [], [], []
    for idx in range(len(dataset)):
        feats, label, sid = dataset[idx]
        feats = feats.to(device)
        logits, _ = model(feats)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs.append(prob)
        labels.append(int(label))
        preds.append(int(prob[1] >= threshold))
        slide_ids.append(sid)

    labels_np = np.asarray(labels, dtype=int)
    probs_np = np.asarray(probs, dtype=float)
    preds_np = np.asarray(preds, dtype=int)

    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    try:
        auc = roc_auc_score(labels_np, probs_np[:, 1])
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(labels_np, probs_np[:, 1])
    except ValueError:
        auprc = np.nan

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "labels": labels_np,
        "probs": probs_np,
        "preds": preds_np,
        "slide_ids": slide_ids,
    }


def train_fold(args: argparse.Namespace, fold: int, train_ids: List[str], val_ids: List[str], clinical_df: pd.DataFrame, d_feat: int, device: torch.device, output_dir: Path) -> Dict[str, object]:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_ds = ClsDataset(train_ids, str(args.fused_feat_dir), clinical_df, args.label_col)
    val_ds = ClsDataset(val_ids, str(args.fused_feat_dir), clinical_df, args.label_col)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Fold {fold}: empty train or validation dataset.")

    seed_everything(args.seed + fold)
    model = build_downstream_head(
        name=args.downstream_head,
        d_feat=d_feat,
        d_inner=args.d_inner,
        d_attn=args.d_attn,
        n_classes=args.n_classes,
        droprate=args.droprate,
        mlp_hidden_dim=args.mlp_hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01)

    best_auc = -np.inf
    best_metrics = None
    epochs_no_improve = 0
    best_path = fold_dir / f"best_{args.downstream_head.lower()}_model.pth"

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_ds, optimizer, device, args.accum_steps)
        metrics = evaluate(model, val_ds, device, threshold=args.threshold)
        scheduler.step()
        val_auc = metrics["auc"]
        print(
            f"Fold {fold} | Epoch {epoch:03d}/{args.max_epochs} | "
            f"loss={train_loss:.4f} | AUC={val_auc:.4f} | AUPRC={metrics['auprc']:.4f} | "
            f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
        )

        improved = not np.isnan(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_metrics = metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Fold {fold}: early stopping after {epoch} epochs. Best AUC={best_auc:.4f}")
                break

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        best_metrics = evaluate(model, val_ds, device, threshold=args.threshold)
    if best_metrics is None:
        best_metrics = evaluate(model, val_ds, device, threshold=args.threshold)

    pred_df = pd.DataFrame({
        "slide_id": best_metrics["slide_ids"],
        "label": best_metrics["labels"],
        "prob_class0": best_metrics["probs"][:, 0],
        "prob_class1": best_metrics["probs"][:, 1],
        "pred": best_metrics["preds"],
    })
    pred_df.to_csv(fold_dir / "predictions.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    cm = pd.DataFrame(
        [[best_metrics["tn"], best_metrics["fp"]], [best_metrics["fn"], best_metrics["tp"]]],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm.to_csv(fold_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    row = {
        "fold": fold,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "auc": best_metrics["auc"],
        "auprc": best_metrics["auprc"],
        "sensitivity": best_metrics["sensitivity"],
        "specificity": best_metrics["specificity"],
        "accuracy": best_metrics["accuracy"],
        "f1": best_metrics["f1"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "tn": best_metrics["tn"],
        "fp": best_metrics["fp"],
        "fn": best_metrics["fn"],
        "tp": best_metrics["tp"],
    }
    pd.DataFrame([row]).to_csv(fold_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    return row


def save_summary(fold_df: pd.DataFrame, output_dir: Path) -> None:
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    rows = []
    for metric in ["auc", "auprc", "sensitivity", "specificity", "accuracy", "f1", "precision", "recall"]:
        values = pd.to_numeric(fold_df[metric], errors="coerce")
        rows.append({
            "metric": metric,
            "mean": float(values.mean(skipna=True)),
            "std": float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0,
            "min": float(values.min(skipna=True)),
            "median": float(values.median(skipna=True)),
            "max": float(values.max(skipna=True)),
        })
    pd.DataFrame(rows).to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_df, slide_ids, labels = load_uvm_data(args.clinical_path, args.label_col)
    valid_ids, valid_labels = available_slide_ids(args.fused_feat_dir, slide_ids, labels)
    if not valid_ids:
        raise RuntimeError(f"No clinical slides found in fused feature dir: {args.fused_feat_dir}")
    d_feat = detect_feature_dim(args.fused_feat_dir)

    if args.manifest is not None:
        splits = splits_from_manifest(args.manifest, args.fused_feat_dir, clinical_df, args.label_col)
    else:
        splits = stratified_splits(valid_ids, valid_labels, args.n_folds, args.seed)

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "output_dir": str(output_dir),
        "D_feat": d_feat,
        "n_valid": len(valid_ids),
    })
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, f, indent=2)

    print("=" * 80)
    print(f"Downstream {args.downstream_head} evaluation on fused embeddings")
    print("=" * 80)
    print(f"Fused feature dir: {args.fused_feat_dir}")
    print(f"D_feat: {d_feat}")
    print(f"Valid slides: {len(valid_ids)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    fold_rows = []
    for fold, train_ids, val_ids in splits:
        print(f"\nFold {fold}: train={len(train_ids)}, val={len(val_ids)}")
        fold_rows.append(train_fold(args, fold, train_ids, val_ids, clinical_df, d_feat, device, output_dir))
        save_summary(pd.DataFrame(fold_rows), output_dir)

    fold_df = pd.DataFrame(fold_rows)
    save_summary(fold_df, output_dir)
    print("\nSummary:")
    print(pd.read_csv(output_dir / "summary_metrics.csv").to_string(index=False))
    print(f"\nSaved output: {output_dir}")


if __name__ == "__main__":
    main()
