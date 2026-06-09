"""
run_benchmark.py
UVM D3/M3 二分类任务 —— 病理基础大模型横评主脚本。

科学问题:
    通用病理基础模型的特征表示，能否识别 UVM 染色体 3 状态（D3 vs M3）？
    在 TCGA-UVM 这样的小样本场景（80 例）中，
    哪类预训练范式（自监督 / 对比学习 / 视觉-语言对齐）更鲁棒？
    D3 (Disomy 3) = SCNA Cluster 1/2 → label 0
    M3 (Monosomy 3) = SCNA Cluster 3/4 → label 1

评测方案:
    对 21 个大模型统一使用 ACMIL 聚合头 + 5 折分层交叉验证。
    指标: AUC / Accuracy / Weighted F1 / Cohen's Kappa

使用方法:
    python run_benchmark.py                         # 全部模型
    python run_benchmark.py --config config/benchmark.yml
    python run_benchmark.py --model CONCH           # 仅指定模型
    python run_benchmark.py --folds 5               # 覆盖折数
    python run_benchmark.py --seed 123              # 覆盖随机种子
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

# 将 _3_predictmodel 及其父目录均加入 sys.path
# 前者用于相对 import（cls_train、datasets 等）
# 后者用于 transformer.py 中的绝对 import（from _3_predictmodel.xxx import ...）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from cls_train import evaluate_cls, train_cls_model
from architecture.transformer import ACMIL_GA
from datasets.cls_dataset import ClsDataset, load_uvm_data


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """设置所有随机源，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_D_feat(feat_dir: str) -> int:
    """
    从特征目录中读取第一个 .h5 文件，自动检测特征维度 D_feat。

    支持 key: feats / features / 第一个非 coords 数据集。

    Returns:
        D_feat : int，特征向量维度
    Raises:
        RuntimeError : 目录为空或文件格式不符
    """
    import h5py

    h5_files = [f for f in os.listdir(feat_dir) if f.endswith(".h5")]
    if not h5_files:
        raise RuntimeError(f"特征目录中未找到 .h5 文件: {feat_dir}")

    sample_path = os.path.join(feat_dir, h5_files[0])
    with h5py.File(sample_path, "r") as f:
        if "feats" in f:
            shape = f["feats"].shape
        elif "features" in f:
            shape = f["features"].shape
        else:
            keys = [k for k in f.keys() if k != "coords"]
            if not keys:
                raise RuntimeError(f"{sample_path}: 未找到特征 key，可用: {list(f.keys())}")
            shape = f[keys[0]].shape

    if len(shape) != 2:
        raise RuntimeError(f"{sample_path}: 特征 shape 应为 [N, D_feat]，实际为 {shape}")

    return int(shape[1])


# ──────────────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────────────

class Config:
    """从 YAML 加载横评配置，支持默认值和 CLI 覆盖。"""

    DEFAULTS = {
        "clinical_path": "",
        "feat_base": "",
        "label_col": "d3m3",
        "models": [],
        "n_classes": 2,
        # MIL 架构（D_inner 由运行时 auto-detect: D_inner = D_feat // 2）
        "D_attn": 128,
        "droprate": 0.25,
        # ACMIL 专有参数
        "n_token": 1,
        "n_masked_patch": 10,
        "mask_drop": 0.6,
        # 训练
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "max_epochs": 100,
        "patience": 20,
        "accum_steps": 1,
        # 交叉验证
        "n_folds": 5,
        "random_state": 42,
        # 输出
        "output_dir": "benchmark_output",
    }

    def __init__(self, yaml_path=None):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f)
            if yml:
                for k, v in yml.items():
                    setattr(self, k, v)

    def override(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in self.DEFAULTS}


# ──────────────────────────────────────────────────────────────────────
# 单模型 5 折 CV
# ──────────────────────────────────────────────────────────────────────

def run_single_model(model_cfg, conf, clinical_df, slide_ids, all_labels, device, out_dir):
    """
    对单个基础模型运行 n_folds 折分层交叉验证。

    Args:
        model_cfg  : 模型配置字典 {name, feat_subdir, D_feat}
        conf       : Config 对象
        clinical_df: 完整临床表 DataFrame
        slide_ids  : 全部 slide_id 列表
        all_labels : 全部标签列表（与 slide_ids 一一对应）
        device     : torch.device
        out_dir    : 本次横评的输出根目录

    Returns:
        summary : dict（含各指标均值/标准差），若跳过则返回 None
    """
    model_name = model_cfg["name"]
    feat_subdir = model_cfg.get("feat_subdir", "") or model_name
    feat_dir = os.path.join(conf.feat_base, feat_subdir) if conf.feat_base else feat_subdir

    print(f"\n{'=' * 60}")
    print(f"模型: {model_name}")
    print(f"特征目录: {feat_dir}")
    print(f"{'=' * 60}")

    # 检查特征目录
    if not os.path.isdir(feat_dir):
        print(f"  [跳过] 特征目录不存在: {feat_dir}")
        return None

    # 自动检测特征维度，D_inner = D_feat // 2
    try:
        D_feat = detect_D_feat(feat_dir)
    except RuntimeError as e:
        print(f"  [跳过] 无法检测特征维度: {e}")
        return None
    D_inner = D_feat // 2
    print(f"  自动检测: D_feat={D_feat}  →  D_inner={D_inner}")

    # 过滤出在该目录中有对应 .h5 文件的 slide
    valid_ids, valid_labels = [], []
    for sid, lbl in zip(slide_ids, all_labels):
        if os.path.exists(os.path.join(feat_dir, f"{sid}.h5")):
            valid_ids.append(sid)
            valid_labels.append(lbl)

    if len(valid_ids) == 0:
        print(f"  [跳过] 未找到任何 .h5 文件")
        return None

    print(f"  有效样本: {len(valid_ids)} / {len(slide_ids)}")
    class_dist = np.bincount(valid_labels, minlength=conf.n_classes)
    print(f"  类别分布: {dict(enumerate(class_dist))}")

    # 5 折分层 CV
    skf = StratifiedKFold(
        n_splits=conf.n_folds, shuffle=True, random_state=conf.random_state
    )
    fold_metrics_list = []
    model_out = os.path.join(out_dir, model_name)
    os.makedirs(model_out, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(valid_ids, valid_labels)
    ):
        print(f"\n  ── Fold {fold_idx + 1}/{conf.n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ──")

        train_ids = [valid_ids[i] for i in train_idx]
        val_ids = [valid_ids[i] for i in val_idx]

        train_ds = ClsDataset(train_ids, feat_dir, clinical_df, conf.label_col)
        val_ds = ClsDataset(val_ids, feat_dir, clinical_df, conf.label_col)

        # 每折使用不同种子（确保权重初始化多样性）
        seed_everything(conf.random_state + fold_idx)

        # ACMIL_GA 用 conf 对象传基础维度，注意字段名是 n_class（非 n_classes）
        conf_model = SimpleNamespace(
            D_feat=D_feat,
            D_inner=D_inner,
            n_class=conf.n_classes,
        )
        model = ACMIL_GA(
            conf_model,
            D=conf.D_attn,
            droprate=conf.droprate,
            n_token=conf.n_token,
            n_masked_patch=conf.n_masked_patch,
            mask_drop=conf.mask_drop,
        ).to(device)

        fold_dir = os.path.join(model_out, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        save_path = os.path.join(fold_dir, "best_model.pth")

        best_auc, best_metrics = train_cls_model(
            model, train_ds, val_ds, device,
            lr=conf.lr,
            weight_decay=conf.weight_decay,
            max_epochs=conf.max_epochs,
            patience=conf.patience,
            n_classes=conf.n_classes,
            accum_steps=conf.accum_steps,
            save_path=save_path,
        )

        # 用保存的最优权重重新评估
        if os.path.exists(save_path):
            model.load_state_dict(
                torch.load(save_path, map_location=device, weights_only=True)
            )
            final_metrics = evaluate_cls(model, val_ds, device, conf.n_classes)
        else:
            final_metrics = best_metrics or {}

        fold_result = {
            "fold": fold_idx + 1,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "acc": final_metrics.get("acc", float("nan")),
            "auc_macro": final_metrics.get("auc_macro", float("nan")),
            "f1_weighted": final_metrics.get("f1_weighted", float("nan")),
            "kappa": final_metrics.get("kappa", float("nan")),
        }
        fold_metrics_list.append(fold_result)

        # 保存混淆矩阵
        if "confusion_matrix" in final_metrics:
            cm_df = pd.DataFrame(
                final_metrics["confusion_matrix"],
                index=[f"true_{i}" for i in range(conf.n_classes)],
                columns=[f"pred_{i}" for i in range(conf.n_classes)],
            )
            cm_df.to_csv(os.path.join(fold_dir, "confusion_matrix.csv"))

        # 保存 pred 概率
        if "probs" in final_metrics:
            prob_df = pd.DataFrame(
                final_metrics["probs"],
                columns=[f"prob_class{i}" for i in range(conf.n_classes)],
            )
            prob_df["label"] = final_metrics["labels"]
            prob_df["pred"] = final_metrics["preds"]
            prob_df["slide_id"] = val_ids[: len(prob_df)]
            prob_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        print(
            f"  Fold {fold_idx + 1} | "
            f"AUC: {fold_result['auc_macro']:.4f} | "
            f"Acc: {fold_result['acc']:.4f} | "
            f"F1: {fold_result['f1_weighted']:.4f} | "
            f"Kappa: {fold_result['kappa']:.4f}"
        )

    # 汇总各折结果
    fold_df = pd.DataFrame(fold_metrics_list)
    fold_df.to_csv(os.path.join(model_out, "fold_metrics.csv"), index=False)

    summary = {
        "model": model_name,
        "tcga_pretrained": model_cfg.get("tcga_pretrained", None),
        "D_feat": D_feat,
        "D_inner": D_inner,
        "n_valid": len(valid_ids),
    }
    for metric in ["acc", "auc_macro", "f1_weighted", "kappa"]:
        vals = fold_df[metric].values
        summary[f"{metric}_mean"] = float(np.nanmean(vals))
        summary[f"{metric}_std"] = float(np.nanstd(vals))

    print(
        f"\n  [{model_name}] 5 折汇总 | "
        f"AUC: {summary['auc_macro_mean']:.4f}±{summary['auc_macro_std']:.4f} | "
        f"Acc: {summary['acc_mean']:.4f}±{summary['acc_std']:.4f} | "
        f"F1: {summary['f1_weighted_mean']:.4f}±{summary['f1_weighted_std']:.4f} | "
        f"Kappa: {summary['kappa_mean']:.4f}±{summary['kappa_std']:.4f}"
    )

    return summary


# ──────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UVM D3/M3 二分类大模型横评"
    )
    parser.add_argument(
        "--config", default="config/benchmark.yml",
        help="配置文件路径（默认 config/benchmark.yml）"
    )
    parser.add_argument(
        "--model", default=None,
        help="仅运行指定模型（按 name 匹配），不填则跑全部"
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="覆盖配置中的 n_folds"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="覆盖配置中的 random_state"
    )
    args = parser.parse_args()

    # 加载配置
    conf = Config(os.path.join(SCRIPT_DIR, args.config))
    conf.override(n_folds=args.folds, random_state=args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, conf.output_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # 保存本次配置快照
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(conf.to_dict(), f, ensure_ascii=False, indent=2)

    # 加载临床表
    if not conf.clinical_path:
        raise ValueError(
            "请在 config/benchmark.yml 中填写 clinical_path，"
            "或通过 --config 指定配置文件。"
        )

    clinical_df, slide_ids, all_labels = load_uvm_data(
        conf.clinical_path, conf.label_col
    )
    print(f"\n共 {len(slide_ids)} 例，{conf.n_folds} 折分层 CV，{len(conf.models)} 个模型")

    # 筛选要跑的模型
    models_to_run = conf.models
    if args.model:
        models_to_run = [m for m in conf.models if m["name"] == args.model]
        if not models_to_run:
            raise ValueError(
                f"配置中未找到模型 '{args.model}'。"
                f"可用: {[m['name'] for m in conf.models]}"
            )

    # 逐模型横评
    all_summaries = []
    for model_cfg in models_to_run:
        summary = run_single_model(
            model_cfg, conf, clinical_df, slide_ids, all_labels, device, out_dir
        )
        if summary is not None:
            all_summaries.append(summary)
            # 及时保存中间汇总（便于中途查看排名）
            _save_summary(all_summaries, out_dir)

    # 最终汇总
    if all_summaries:
        _save_summary(all_summaries, out_dir, final=True)
    else:
        print("\n[警告] 所有模型均被跳过，请检查特征目录和临床表路径。")

    print(f"\n输出目录: {out_dir}")


def _save_summary(summaries, out_dir, final=False):
    """将汇总 DataFrame 按 AUC 降序保存，并打印排名表。"""
    df = pd.DataFrame(summaries).sort_values("auc_macro_mean", ascending=False)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.4f")

    if final:
        cols = [
            "model", "D_feat",
            "auc_macro_mean", "auc_macro_std",
            "acc_mean", "acc_std",
            "f1_weighted_mean", "f1_weighted_std",
            "kappa_mean", "kappa_std",
        ]
        print(f"\n{'=' * 60}")
        print("最终排名（按 AUC 降序）:")
        print(df[cols].to_string(index=False))
        print(f"\n后续分析: python analyze_results.py --results {out_dir}")


if __name__ == "__main__":
    main()
