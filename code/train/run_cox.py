"""
run_cox.py
Cox回归生存预测主脚本。

使用方法:
    python run_cox.py                       # 默认读取 config/cox.yml
    python run_cox.py --config xxx.yml      # 指定配置文件

完整流程:
1. 加载数据（3个队列）
2. HMU 7:3分层划分
3. 训练ACMIL-Cox模型
4. 训练集最优阈值搜索
5. 验证集/外部测试集评估
6. 生成KM曲线和结果报告
"""

import os
import sys
import json
import random
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加 _3_predictmodel 和项目根目录到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.survival_dataset import (
    SurvivalDataset, load_cohort, split_hmu_train_val
)
from architecture.acmil_cox import ACMIL_Cox
from cox_train import train_cox_model, evaluate
from cox_evaluate import (
    search_optimal_threshold, evaluate_cohort, save_results_report,
    plot_time_dependent_roc, plot_dca_curves, YEAR_LABELS_ZH,
)


# ──────────────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────────────

class Config:
    """从 YAML 文件加载配置，支持默认值。"""

    # 默认值（与 config/cox.yml 一致）
    DEFAULTS = {
        # 数据路径
        "clinical_dir": r"D:\Datas of lab\crc\临床表\CRC_stage2\匹配特征后",
        "eligible_dir": r"D:\Datas of lab\crc\临床表\CRC_stage2",
        "hmu_clinical": "HMU_stage2_化疗_匹配特征_withDFS.xlsx",
        "wenfu_clinical": "温附一_cohort1+2_stage2_化疗_匹配特征_withDFS.xlsx",
        "fujian_clinical": "福建协和_stage2_化疗_匹配特征_withDFS.xlsx",
        "feat_base": r"D:\Datas of lab\crc\特征\CONCH\重命名",
        "hmu_feat": "哈医大_CONCH特征_CLAM_5X_pth_stage2_renamed",
        "wenfu_feat": "温附一_CONCH特征_CLAM_5X_pth_stage2_renamed",
        "fujian_feat": "福建协和_CONCH特征_CLAM_5X_pth_stage2_renamed",
        "output_dir": "cox_output",
        # 模型架构
        "D_feat": 512,
        "D_inner": 128,
        "D_attn": 64,
        "n_token": 1,
        "n_masked_patch": 0,
        "mask_drop": 0.6,
        "droprate": 0.5,
        # 训练参数
        "lr": 5e-5,
        "weight_decay": 1e-3,
        "max_epochs": 200,
        "patience": 30,
        "diff_weight": 0.0,
        "sub_weight": 0.0,
        # 数据划分
        "val_ratio": 0.3,
        "random_state": 42,
        # 运行模式
        "mode": "train",
        "checkpoint_path": "",
    }

    def __init__(self, yaml_path=None):
        # 先加载默认值
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)

        # 再用 YAML 覆盖
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f)
            if yml:
                for k, v in yml.items():
                    if v is not None and v != "":
                        setattr(self, k, v)
            print(f"已加载配置: {yaml_path}")
        elif yaml_path:
            print(f"警告: 配置文件不存在 {yaml_path}，使用默认值")

        # 拼接完整路径
        self.hmu_clinical = os.path.join(self.clinical_dir, self.hmu_clinical) \
            if not os.path.isabs(self.hmu_clinical) else self.hmu_clinical
        self.wenfu_clinical = os.path.join(self.clinical_dir, self.wenfu_clinical) \
            if not os.path.isabs(self.wenfu_clinical) else self.wenfu_clinical
        self.fujian_clinical = os.path.join(self.clinical_dir, self.fujian_clinical) \
            if not os.path.isabs(self.fujian_clinical) else self.fujian_clinical

        self.hmu_feat = os.path.join(self.feat_base, self.hmu_feat) \
            if not os.path.isabs(self.hmu_feat) else self.hmu_feat
        self.wenfu_feat = os.path.join(self.feat_base, self.wenfu_feat) \
            if not os.path.isabs(self.wenfu_feat) else self.wenfu_feat
        self.fujian_feat = os.path.join(self.feat_base, self.fujian_feat) \
            if not os.path.isabs(self.fujian_feat) else self.fujian_feat

        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(SCRIPT_DIR, self.output_dir)

        # 在 output_dir 下创建带时间戳的子目录，避免覆盖历史结果
        # evaluate 模式不追加时间戳，以便从已有目录加载模型
        # 如果 output_dir 末尾已经是 "exp_XXXX" 格式（由 roll_cox.py 预分配），也跳过
        dir_tail = os.path.basename(self.output_dir)
        is_roll_exp = dir_tail.startswith("exp_")
        is_evaluate = getattr(self, "mode", "train") == "evaluate"
        if not is_roll_exp and not is_evaluate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.output_dir, timestamp)

    def print_config(self):
        print("─" * 50)
        print("当前配置:")
        print("─" * 50)
        for k in sorted(vars(self)):
            if not k.startswith("_"):
                print(f"  {k}: {getattr(self, k)}")
        print("─" * 50)


# ──────────────────────────────────────────────────────────────────────
# 可复现性
# ──────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    """设置所有随机源的种子，确保同参数两次运行结果一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_dir):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss曲线
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cox Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # C-index曲线
    axes[1].plot(epochs, history["train_ci"], "b-", label="Train C-index")
    axes[1].plot(epochs, history["val_ci"], "r-", label="Val C-index")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("C-index")
    axes[1].set_title("Training & Validation C-index")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def save_risk_scores(results, threshold, save_path, clinical_df=None):
    """保存单个队列的risk scores到CSV。"""
    df = pd.DataFrame({
        "slide_id": results["slide_ids"],
        "risk_score": results["risk_scores"],
        "DFS_month": results["times"],
        "DFS_event": results["events"],
        "risk_group": (results["risk_scores"] >= threshold).astype(int),
    })
    if clinical_df is not None and {"slide_id", "OS_month", "OS_event"}.issubset(clinical_df.columns):
        os_df = clinical_df[["slide_id", "OS_month", "OS_event"]].copy()
        os_df["slide_id"] = os_df["slide_id"].astype(str)
        df = df.merge(os_df, on="slide_id", how="left")
    df.to_csv(save_path, index=False)


def build_endpoint_results(base_results, clinical_df, endpoint_label, cohort_name=""):
    """基于已有 risk score 回填指定终点（DFS / OS）的 time/event。"""
    month_col = f"{endpoint_label}_month"
    event_col = f"{endpoint_label}_event"

    if month_col not in clinical_df.columns or event_col not in clinical_df.columns:
        print(f"[{cohort_name}] 缺少 {month_col}/{event_col}，跳过 {endpoint_label} 评估")
        return None

    endpoint_df = clinical_df[["slide_id", month_col, event_col]].copy()
    endpoint_df["slide_id"] = endpoint_df["slide_id"].astype(str)

    merged = pd.DataFrame({
        "slide_id": [str(sid) for sid in base_results["slide_ids"]],
        "risk_score": base_results["risk_scores"],
    }).merge(endpoint_df, on="slide_id", how="left")

    valid_mask = merged[month_col].notna() & merged[event_col].notna()
    valid_n = int(valid_mask.sum())
    if valid_n == 0:
        print(f"[{cohort_name}] 没有可用的 {endpoint_label} 终点，跳过")
        return None

    if valid_n < len(merged):
        print(f"[{cohort_name}] {endpoint_label} 有效样本 {valid_n}/{len(merged)}，已过滤缺失值")

    return {
        "slide_ids": merged.loc[valid_mask, "slide_id"].tolist(),
        "risk_scores": merged.loc[valid_mask, "risk_score"].to_numpy(dtype=float),
        "times": pd.to_numeric(merged.loc[valid_mask, month_col], errors="coerce").to_numpy(dtype=float),
        "events": pd.to_numeric(merged.loc[valid_mask, event_col], errors="coerce").to_numpy(dtype=int),
    }


def evaluate_no_chemo_subgroup(cohort_specs, thresholds, cfg):
    """
    未化疗亚组 (chemo=0) KM 评估。

    从 eligible 表中筛选 chemo=0 的患者，用已有的 risk scores 和全集阈值
    进行 KM 评估，验证模型在未化疗患者中的分层能力。

    Args:
        cohort_specs: list of (cohort_name, results_dict, clinical_df)
        thresholds: dict {策略名: 阈值}（全集上搜索到的）
        cfg: Config 对象（含 output_dir, eligible_dir）
    """
    eligible_dir = cfg.eligible_dir
    if not eligible_dir or not os.path.isdir(eligible_dir):
        print(f"  ⚠️ eligible_dir 不存在或未配置: {eligible_dir}，跳过未化疗亚组评估")
        return

    # 队列名 → eligible 表文件名映射
    eligible_map = {
        "HMU_Train":   "HMU_stage2_eligible.xlsx",
        "HMU_Val":     "HMU_stage2_eligible.xlsx",
        "Wenfu_Test":  "温附一_stage2_eligible.xlsx",
        "Fujian_Test": "福建协和_stage2_eligible.xlsx",
    }

    km_dir = os.path.join(cfg.output_dir, "km_curves")
    os.makedirs(km_dir, exist_ok=True)

    all_metrics = []
    no_chemo_cache = {}  # eligible_path → set[str] | None（避免重复读取同一文件）

    for cohort_name, results, _ in cohort_specs:
        eligible_file = eligible_map.get(cohort_name)
        if not eligible_file:
            print(f"  [{cohort_name}] 无对应 eligible 表，跳过")
            continue

        eligible_path = os.path.join(eligible_dir, eligible_file)

        # 带缓存的读取与预处理（HMU_Train / HMU_Val 共用同一文件）
        if eligible_path not in no_chemo_cache:
            try:
                eligible_df = pd.read_excel(
                    eligible_path, usecols=["slide_id", "chemo"]
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  [{cohort_name}] 读取 eligible 表失败: {e}，跳过")
                no_chemo_cache[eligible_path] = None
                continue
            no_chemo_cache[eligible_path] = set(
                eligible_df.loc[eligible_df["chemo"] == 0, "slide_id"]
                           .astype(str)
            )

        no_chemo_ids = no_chemo_cache[eligible_path]
        if no_chemo_ids is None:
            continue

        # 向量化匹配：从 results 中提取 chemo=0 的患者
        mask = pd.Series(results["slide_ids"]).astype(str).isin(no_chemo_ids).values

        n_matched = int(mask.sum())
        if n_matched < 5:
            print(f"  [{cohort_name}] 未化疗患者匹配数不足 ({n_matched})，跳过")
            continue

        sub_risk_scores = results["risk_scores"][mask]
        sub_times = results["times"][mask]
        sub_events = results["events"][mask]

        n_events = int(sub_events.sum())
        print(f"\n  [{cohort_name}] 未化疗亚组: {n_matched}人, 事件数: {n_events}")

        # 用全集阈值评估
        metrics = evaluate_cohort(
            sub_risk_scores,
            sub_times,
            sub_events,
            thresholds,
            cohort_name=f"NoChemo_{cohort_name}",
            save_dir=km_dir,
            endpoint_label="DFS",
            file_prefix="NoChemo_",
        )
        all_metrics.extend(metrics)

    if all_metrics:
        save_results_report(all_metrics, thresholds, cfg.output_dir,
                            file_suffix="no_chemo")


def evaluate_secondary_endpoint(endpoint_label, cohort_specs, output_dir):
    """使用与 DFS 相同的阈值搜索和 KM 出图流程评估额外终点。"""
    if not cohort_specs:
        return {}, {}

    print("\n" + "=" * 60)
    print(f"{endpoint_label} KM 评估")
    print("=" * 60)

    train_cohort_name, train_base_results, train_clinical_df = cohort_specs[0]
    train_endpoint_results = build_endpoint_results(
        train_base_results, train_clinical_df, endpoint_label, train_cohort_name
    )
    if train_endpoint_results is None:
        return {}, {}

    print(f"\n[{endpoint_label}] 训练集阈值搜索")
    endpoint_thresholds, endpoint_pvalues = search_optimal_threshold(
        train_endpoint_results["risk_scores"],
        train_endpoint_results["times"],
        train_endpoint_results["events"],
    )

    km_dir = os.path.join(output_dir, "km_curves")
    os.makedirs(km_dir, exist_ok=True)

    endpoint_metrics = []
    for cohort_name, base_results, clinical_df in cohort_specs:
        endpoint_results = build_endpoint_results(
            base_results, clinical_df, endpoint_label, cohort_name
        )
        if endpoint_results is None:
            continue

        endpoint_metrics.extend(
            evaluate_cohort(
                endpoint_results["risk_scores"],
                endpoint_results["times"],
                endpoint_results["events"],
                endpoint_thresholds,
                cohort_name=cohort_name,
                save_dir=km_dir,
                endpoint_label=endpoint_label,
                y_label=f"{endpoint_label} Survival Probability",
                file_prefix=f"{endpoint_label}_",
            )
        )

    if endpoint_metrics:
        save_results_report(
            endpoint_metrics,
            endpoint_thresholds,
            output_dir,
            file_suffix=endpoint_label.lower(),
        )

    return endpoint_thresholds, endpoint_pvalues


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(description="ACMIL-Cox 生存预测")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(SCRIPT_DIR, "config", "cox.yml"),
        help="配置文件路径 (默认: config/cox.yml)",
    )
    args = parser.parse_args()

    cfg = Config(args.config)
    cfg.print_config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 设置全局随机种子，确保可复现
    seed_everything(cfg.random_state)
    print(f"随机种子: {cfg.random_state}")

    # ──────────────────────────────────────────────────────────────
    # Step 1: 检查DFS表是否存在
    # ──────────────────────────────────────────────────────────────
    if not os.path.exists(cfg.hmu_clinical):
        print("未找到DFS表，先运行 prepare_dfs.py ...")
        from _1_prework import prepare_dfs
        prepare_dfs.main()
        print()

    # ──────────────────────────────────────────────────────────────
    # Step 2: 加载数据
    # ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    hmu_df, _, hmu_ids = load_cohort(cfg.hmu_clinical, cfg.hmu_feat, "HMU")
    wenfu_df, _, wenfu_ids = load_cohort(cfg.wenfu_clinical, cfg.wenfu_feat, "温附一")
    fujian_df, _, fujian_ids = load_cohort(cfg.fujian_clinical, cfg.fujian_feat, "福建协和")

    # ──────────────────────────────────────────────────────────────
    # Step 3: HMU 7:3分层划分
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HMU 7:3 分层划分")
    print("=" * 60)

    train_ids, val_ids = split_hmu_train_val(
        hmu_ids, hmu_df,
        val_ratio=cfg.val_ratio,
        random_state=cfg.random_state,
    )

    # 构建数据集
    print("\n构建数据集...")
    train_dataset = SurvivalDataset(train_ids, cfg.hmu_feat, hmu_df)
    val_dataset = SurvivalDataset(val_ids, cfg.hmu_feat, hmu_df)
    wenfu_dataset = SurvivalDataset(wenfu_ids, cfg.wenfu_feat, wenfu_df)
    fujian_dataset = SurvivalDataset(fujian_ids, cfg.fujian_feat, fujian_df)

    print(f"  训练集: {len(train_dataset)}人")
    print(f"  验证集: {len(val_dataset)}人")
    print(f"  温附一测试集: {len(wenfu_dataset)}人")
    print(f"  福建协和测试集: {len(fujian_dataset)}人")

    # ──────────────────────────────────────────────────────────────
    # Step 4: 构建模型
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("构建 ACMIL-Cox 模型")
    print("=" * 60)

    model = ACMIL_Cox(
        D_feat=cfg.D_feat,
        D_inner=cfg.D_inner,
        D_attn=cfg.D_attn,
        n_token=cfg.n_token,
        n_masked_patch=cfg.n_masked_patch,
        mask_drop=cfg.mask_drop,
        droprate=cfg.droprate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,} (可训练: {n_trainable:,})")

    model_save_path = os.path.join(cfg.output_dir, "best_model.pth")

    # ──────────────────────────────────────────────────────────────
    # Step 5: 训练 或 加载模型
    # ──────────────────────────────────────────────────────────────
    if cfg.mode == "train":
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        best_val_ci, history = train_cox_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            diff_weight=cfg.diff_weight,
            sub_weight=cfg.sub_weight,
            save_path=model_save_path,
        )

        # 绘制训练曲线
        plot_training_curves(history, cfg.output_dir)

    elif cfg.mode == "evaluate":
        ckpt = cfg.checkpoint_path if cfg.checkpoint_path else model_save_path
        if not os.path.exists(ckpt):
            print(f"错误: 模型文件不存在 {ckpt}")
            return
        print(f"\n跳过训练，直接加载模型: {ckpt}")
        best_val_ci = -1.0

    # ──────────────────────────────────────────────────────────────
    # Step 6: 加载最优模型，进行评估
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("加载最优模型，进行评估")
    print("=" * 60)

    ckpt = cfg.checkpoint_path if (cfg.mode == "evaluate" and cfg.checkpoint_path) else model_save_path
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    _, train_ci, train_results = evaluate(model, train_dataset, device)
    _, val_ci, val_results = evaluate(model, val_dataset, device)

    print(f"\n最优模型 - 训练集 C-index: {train_ci:.4f}, 验证集 C-index: {val_ci:.4f}")

    # ──────────────────────────────────────────────────────────────
    # Step 7: 训练集阈值搜索（多策略）
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("训练集阈值搜索（多策略）")
    print("=" * 60)

    thresholds, threshold_pvalues = search_optimal_threshold(
        train_results["risk_scores"],
        train_results["times"],
        train_results["events"],
    )

    # ──────────────────────────────────────────────────────────────
    # Step 8: 全部队列评估（每种阈值策略各画一套KM）
    # ──────────────────────────────────────────────────────────────
    km_dir = os.path.join(cfg.output_dir, "km_curves")
    os.makedirs(km_dir, exist_ok=True)

    all_metrics = []

    # 训练集
    m = evaluate_cohort(
        train_results["risk_scores"], train_results["times"],
        train_results["events"], thresholds,
        cohort_name="HMU_Train", save_dir=km_dir,
    )
    all_metrics.extend(m)

    # 验证集
    m = evaluate_cohort(
        val_results["risk_scores"], val_results["times"],
        val_results["events"], thresholds,
        cohort_name="HMU_Val", save_dir=km_dir,
    )
    all_metrics.extend(m)

    # 外部测试集 - 温附一
    wenfu_results = None
    if len(wenfu_dataset) > 0:
        _, wenfu_ci, wenfu_results = evaluate(model, wenfu_dataset, device)
        m = evaluate_cohort(
            wenfu_results["risk_scores"], wenfu_results["times"],
            wenfu_results["events"], thresholds,
            cohort_name="Wenfu_Test", save_dir=km_dir,
        )
        all_metrics.extend(m)

    # 外部测试集 - 福建协和
    fujian_results = None
    if len(fujian_dataset) > 0:
        _, fujian_ci, fujian_results = evaluate(model, fujian_dataset, device)
        m = evaluate_cohort(
            fujian_results["risk_scores"], fujian_results["times"],
            fujian_results["events"], thresholds,
            cohort_name="Fujian_Test", save_dir=km_dir,
        )
        all_metrics.extend(m)

    cohort_specs = [
        ("HMU_Train", train_results, hmu_df),
        ("HMU_Val", val_results, hmu_df),
    ]
    if wenfu_results is not None:
        cohort_specs.append(("Wenfu_Test", wenfu_results, wenfu_df))
    if fujian_results is not None:
        cohort_specs.append(("Fujian_Test", fujian_results, fujian_df))

    os_thresholds, os_threshold_pvalues = evaluate_secondary_endpoint(
        "OS", cohort_specs, cfg.output_dir
    )

    # ──────────────────────────────────────────────────────────────
    # Step 8.1: 未化疗亚组 KM 评估
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("未化疗亚组 (chemo=0) KM 评估")
    print("=" * 60)

    evaluate_no_chemo_subgroup(cohort_specs, thresholds, cfg)

    # ──────────────────────────────────────────────────────────────
    # Step 8.5: Time-dependent ROC 曲线（1/3/5年）
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Time-dependent ROC 曲线 (1/3/5年)")
    print("=" * 60)

    roc_dir = os.path.join(cfg.output_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)

    td_auc_records = []

    # 训练集的时间/事件作为 IPCW 权重的基准
    train_times_arr = train_results["times"]
    train_events_arr = train_results["events"]

    cohort_data = [(cohort_name, results) for cohort_name, results, _ in cohort_specs]

    for cohort_name, results in cohort_data:
        print(f"\n  [{cohort_name}]")
        roc_path = os.path.join(roc_dir, f"ROC_{cohort_name}.png")
        auc_dict, plot_auc_dict, mean_auc = plot_time_dependent_roc(
            train_times_arr, train_events_arr,
            results["risk_scores"], results["times"], results["events"],
            time_points_month=(12, 36, 60),
            cohort_name=cohort_name,
            save_path=roc_path,
        )
        record = {"cohort": cohort_name, "mean_auc": mean_auc}
        for t, auc in auc_dict.items():
            record[f"AUC_{t}m"] = auc
        # 同时记录普通二分类 AUC（与 ROC 图一致）
        for t, auc in plot_auc_dict.items():
            record[f"plotAUC_{t}m"] = auc
        td_auc_records.append(record)

        # 打印（IPCW td-AUC 和图上 AUC 都显示，方便对比）
        for t in auc_dict:
            td_auc = auc_dict[t]
            p_auc = plot_auc_dict.get(t, float("nan"))
            td_str = f"{td_auc:.4f}" if not np.isnan(td_auc) else "N/A"
            p_str = f"{p_auc:.4f}" if not np.isnan(p_auc) else "N/A"
            year_label = YEAR_LABELS_ZH.get(t, f"{t}月")
            print(f"    {year_label} td-AUC(IPCW): {td_str}  ROC-AUC: {p_str}")
        mean_str = f"{mean_auc:.4f}" if not np.isnan(mean_auc) else "N/A"
        print(f"    Mean td-AUC: {mean_str}")

    # 保存 td-AUC 汇总表
    td_auc_df = pd.DataFrame(td_auc_records)
    td_auc_path = os.path.join(cfg.output_dir, "time_dependent_auc.csv")
    td_auc_df.to_csv(td_auc_path, index=False)
    print(f"\n  Time-dependent AUC 汇总已保存: {td_auc_path}")

    # ──────────────────────────────────────────────────────────────
    # Step 8.6: Decision Curve Analysis (DCA) — 1/3/5年
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Decision Curve Analysis (1/3/5年)")
    print("=" * 60)

    dca_dir = os.path.join(cfg.output_dir, "dca_curves")
    os.makedirs(dca_dir, exist_ok=True)

    td_dca_records = []
    train_risk_arr = train_results["risk_scores"]

    for cohort_name, results in cohort_data:
        print(f"\n  [{cohort_name}]")
        dca_path = os.path.join(dca_dir, f"DCA_{cohort_name}.png")
        dca_summary, mean_dca = plot_dca_curves(
            train_times_arr, train_events_arr, train_risk_arr,
            results["risk_scores"], results["times"], results["events"],
            time_points_month=(12, 36, 60),
            cohort_name=cohort_name,
            save_path=dca_path,
        )

        dca_record = {"cohort": cohort_name, "mean_inb": mean_dca}
        for t, inb in dca_summary.items():
            dca_record[f"INB_{t}m"] = inb
        td_dca_records.append(dca_record)

        # 打印
        for t, inb in dca_summary.items():
            inb_str = f"{inb:.4f}" if not np.isnan(inb) else "N/A"
            year_label = YEAR_LABELS_ZH.get(t, f"{t}月")
            print(f"    {year_label} INB: {inb_str}")
        mean_str = f"{mean_dca:.4f}" if not np.isnan(mean_dca) else "N/A"
        print(f"    Mean INB: {mean_str}")

    # 保存独立的 DCA 汇总表
    td_dca_df = pd.DataFrame(td_dca_records)
    td_dca_path = os.path.join(cfg.output_dir, "time_dependent_dca.csv")
    td_dca_df.to_csv(td_dca_path, index=False)
    print(f"\n  DCA 汇总已保存: {td_dca_path}")

    # 将 INB 列合并到 td_auc_df，覆写 time_dependent_auc.csv（兼容 roll_cox 读取）
    td_auc_df = td_auc_df.merge(td_dca_df, on="cohort", how="left")
    td_auc_df.to_csv(td_auc_path, index=False)
    print(f"  AUC+INB 合并表已覆写: {td_auc_path}")

    # ──────────────────────────────────────────────────────────────
    # Step 9: 保存结果
    # ──────────────────────────────────────────────────────────────
    save_results_report(all_metrics, thresholds, cfg.output_dir)

    # 保存配置快照
    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    # 序列化阈值（xtile 是元组，需转为 list）
    config_dict["thresholds"] = {
        k: list(v) if isinstance(v, (tuple, list)) else float(v)
        for k, v in thresholds.items()
    }
    config_dict["threshold_pvalues"] = {k: float(v) for k, v in threshold_pvalues.items()}
    if os_thresholds:
        config_dict["os_thresholds"] = {
            k: list(v) if isinstance(v, (tuple, list)) else float(v)
            for k, v in os_thresholds.items()
        }
        config_dict["os_threshold_pvalues"] = {
            k: float(v) for k, v in os_threshold_pvalues.items()
        }
    config_dict["best_val_cindex"] = float(best_val_ci if best_val_ci > 0 else val_ci)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 保存各队列 risk scores（用 logrank 最优阈值标注分组）
    logrank_thr = thresholds["logrank"]
    save_risk_scores(train_results, logrank_thr,
                     os.path.join(cfg.output_dir, "risk_scores_train.csv"),
                     clinical_df=hmu_df)
    save_risk_scores(val_results, logrank_thr,
                     os.path.join(cfg.output_dir, "risk_scores_val.csv"),
                     clinical_df=hmu_df)
    if wenfu_results is not None:
        save_risk_scores(wenfu_results, logrank_thr,
                         os.path.join(cfg.output_dir, "risk_scores_wenfu.csv"),
                         clinical_df=wenfu_df)
    if fujian_results is not None:
        save_risk_scores(fujian_results, logrank_thr,
                         os.path.join(cfg.output_dir, "risk_scores_fujian.csv"),
                         clinical_df=fujian_df)

    print("\n" + "=" * 60)
    print("完成！所有结果已保存到:", cfg.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
