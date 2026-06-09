"""
analyze_results.py
benchmark 跑完后的独立分析脚本。

读取 run_benchmark.py 生成的 summary.csv，结合 benchmark.yml 中的
tcga_pretrained 分组标记，回答两个科学问题：

  Q1: 预训练病理知识的迁移价值
      病理基础模型（20个） vs ImageNet 预训练 ResNet50 基线

  Q2: 数据污染对评估公平性的影响
      TCGA 预训练组 vs 私有数据预训练组

统计方法：
  - Q1 主要检验: 符号检验（二项检验），不依赖模型间独立性假设
  - Q2 主要检验: 置换检验（permutation test），对观测到的分组均值差进行非参检验
  - 注意: 模型不是独立统计样本，不能用模型级 AUC 做 MWU/Wilcoxon 推断

分组随时可在 benchmark.yml 中修改 tcga_pretrained 标记，
无需重跑 benchmark。

使用方法:
    python analyze_results.py --results benchmark_output/20260331_153928
    python analyze_results.py --results benchmark_output/20260331_153928 --config config/benchmark.yml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_grouping(config_path):
    """从 benchmark.yml 读取每个模型的分组标记。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    grouping = {}
    confirmed = {}
    for m in cfg.get("models", []):
        grouping[m["name"]] = m.get("tcga_pretrained", None)
        confirmed[m["name"]] = m.get("group_confirmed", True)
    return grouping, confirmed



def permutation_test(group_a, group_b, n_perm=10000, seed=42):
    """
    置换检验：检验两组观测值的均值差是否显著。
    不要求样本来自同一总体的独立采样，仅检验给定分组标签下
    观测到的组间差异是否可能来自随机分配。
    """
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count_extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_perm


def sig_label(p):
    if p < 0.001:
        return "*** p<0.001"
    elif p < 0.01:
        return f"** p={p:.4f}"
    elif p < 0.05:
        return f"* p={p:.4f}"
    else:
        return f"n.s. p={p:.4f}"


def analyze(results_dir, config_path):
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到 summary.csv: {summary_path}")

    df = pd.read_csv(summary_path)
    grouping, confirmed = load_grouping(config_path)

    # 将分组标记合并进 df
    df["tcga_pretrained"] = df["model"].map(grouping)
    df["group_confirmed"] = df["model"].map(confirmed).fillna(True)

    unlabeled = df[df["tcga_pretrained"].isna()]["model"].tolist()
    if unlabeled:
        print(f"[提示] 以下模型未在 benchmark.yml 中设置 tcga_pretrained，将排除在分组分析之外:")
        for m in unlabeled:
            print(f"  - {m}")

    # 警告分组未确认的模型
    unconfirmed = df[df["group_confirmed"] == False]["model"].tolist()
    if unconfirmed:
        print(f"\n[警告] 以下模型的 tcga_pretrained 分组尚未确认（group_confirmed=false）:")
        for m in unconfirmed:
            grp = grouping.get(m, "?")
            print(f"  - {m}  (当前标记: tcga_pretrained={grp})")
        print("  请核实预训练数据来源后更新 benchmark.yml，当前分析结果可能需要修正。\n")

    # ── 分组 ────────────────────────────────────────────────────────
    baseline_df   = df[df["model"] == "resnet50"]
    foundation_df = df[df["model"] != "resnet50"]
    tcga_df       = df[df["tcga_pretrained"] == True]
    private_df    = df[(df["tcga_pretrained"] == False) & (df["model"] != "resnet50")]

    baseline_aucs = baseline_df["auc_macro_mean"].values
    foundation_aucs = foundation_df["auc_macro_mean"].values
    tcga_aucs    = tcga_df["auc_macro_mean"].values
    private_aucs = private_df["auc_macro_mean"].values

    records = []

    # ── 全排名 ──────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("全模型排名（按 AUC 降序）")
    print(f"{'=' * 65}")
    ranked = df.sort_values("auc_macro_mean", ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        flag = "[TCGA]" if row["tcga_pretrained"] == True else \
               "[priv]" if row["tcga_pretrained"] == False else "[----]"
        conf = "" if row["group_confirmed"] else " (?)"
        print(f"  {i:2d}. {flag}{conf} {row['model']:15s}  "
              f"AUC={row['auc_macro_mean']:.4f}±{row['auc_macro_std']:.4f}  "
              f"Acc={row['acc_mean']:.4f}  F1={row['f1_weighted_mean']:.4f}")

    # ── Q1: 迁移价值 ────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Q1：预训练病理知识的迁移价值")
    print(f"{'=' * 65}")
    if len(baseline_aucs):
        print(f"  ResNet50 基线      ({len(baseline_aucs):2d}个): "
              f"AUC = {baseline_aucs[0]:.4f}")
    else:
        print("  ResNet50 未运行")

    if len(foundation_aucs):
        print(f"  病理基础模型       ({len(foundation_aucs):2d}个): "
              f"AUC = {np.nanmean(foundation_aucs):.4f} ± {np.nanstd(foundation_aucs):.4f}"
              f"  [min={np.nanmin(foundation_aucs):.4f}, max={np.nanmax(foundation_aucs):.4f}]")

    if len(baseline_aucs) and len(foundation_aucs):
        delta_q1 = float(np.nanmean(foundation_aucs) - baseline_aucs[0])
        resnet_auc = float(baseline_aucs[0])
        n_above = int(np.sum(foundation_aucs > resnet_auc))
        n_fm = len(foundation_aucs)

        # [主要] 符号检验（二项检验）：n_above/n_fm vs 随机猜测
        # 不依赖模型间独立性假设，仅检验超越基线的比例
        p_q1_sign = stats.binomtest(n_above, n=n_fm, p=0.5, alternative="greater").pvalue

        print(f"  ΔAUC (基础 - 基线) = {delta_q1:+.4f}")
        print(f"  基础模型超越基线: {n_above}/{n_fm}")
        print(f"  符号检验 (二项, 单侧): {sig_label(p_q1_sign)}")

        records.append({
            "question": "Q1_transfer_value",
            "group_A": "foundation_models",    "n_A": n_fm,
            "auc_A_mean": float(np.nanmean(foundation_aucs)),
            "auc_A_std":  float(np.nanstd(foundation_aucs)),
            "group_B": "resnet50_baseline",    "n_B": len(baseline_aucs),
            "auc_B_mean": resnet_auc,
            "auc_B_std":  0.0,
            "delta_AUC":  delta_q1,
            "n_above_baseline": n_above,
            "p_value_sign":     float(p_q1_sign),
        })

    # ── Q2: 数据污染影响 ─────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Q2：数据污染对评估公平性的影响")
    print(f"{'=' * 65}")

    if unconfirmed:
        print(f"  [注意] 以下模型分组未确认，当前结果可能变化: {unconfirmed}")

    if len(tcga_aucs):
        print(f"  TCGA 预训练组      ({len(tcga_aucs):2d}个): "
              f"AUC = {np.nanmean(tcga_aucs):.4f} ± {np.nanstd(tcga_aucs):.4f}")
        for _, row in tcga_df.sort_values("auc_macro_mean", ascending=False).iterrows():
            conf = "" if row["group_confirmed"] else " (?)"
            print(f"    {row['model']:15s}{conf}  AUC={row['auc_macro_mean']:.4f}±{row['auc_macro_std']:.4f}")
    if len(private_aucs):
        print(f"  私有数据预训练组   ({len(private_aucs):2d}个): "
              f"AUC = {np.nanmean(private_aucs):.4f} ± {np.nanstd(private_aucs):.4f}")
        for _, row in private_df.sort_values("auc_macro_mean", ascending=False).iterrows():
            conf = "" if row["group_confirmed"] else " (?)"
            print(f"    {row['model']:15s}{conf}  AUC={row['auc_macro_mean']:.4f}±{row['auc_macro_std']:.4f}")

    if len(tcga_aucs) >= 2 and len(private_aucs) >= 2:
        delta_q2 = float(np.nanmean(tcga_aucs) - np.nanmean(private_aucs))
        direction = "TCGA组更高（可能存在泄露效应）" if delta_q2 > 0 else "私有组更高（TCGA预训练无优势）"

        # 置换检验：不假设模型是独立样本
        p_q2_perm = permutation_test(tcga_aucs, private_aucs)

        print(f"  ΔAUC (TCGA - 私有) = {delta_q2:+.4f}  →  {direction}")
        print(f"  置换检验 (双侧, n_perm=10000): {sig_label(p_q2_perm)}")

        records.append({
            "question": "Q2_contamination_effect",
            "group_A": "tcga_pretrained",      "n_A": len(tcga_aucs),
            "auc_A_mean": float(np.nanmean(tcga_aucs)),
            "auc_A_std":  float(np.nanstd(tcga_aucs)),
            "group_B": "private_pretrained",   "n_B": len(private_aucs),
            "auc_B_mean": float(np.nanmean(private_aucs)),
            "auc_B_std":  float(np.nanstd(private_aucs)),
            "delta_AUC":  delta_q2,
            "n_above_baseline": None,
            "p_value_permutation": float(p_q2_perm),
        })

    # ── 保存 ────────────────────────────────────────────────────────
    if records:
        out_path = os.path.join(results_dir, "group_analysis.csv")
        pd.DataFrame(records).to_csv(out_path, index=False, float_format="%.4f")
        print(f"\n已保存: {out_path}")

    # 保存带分组标记的完整排名
    ranked_out = os.path.join(results_dir, "summary_with_groups.csv")
    ranked.to_csv(ranked_out, index=False, float_format="%.4f")
    print(f"已保存: {ranked_out}")


def main():
    parser = argparse.ArgumentParser(description="benchmark 后续分组分析")
    parser.add_argument(
        "--results", required=True,
        help="benchmark 输出目录（含 summary.csv）"
    )
    parser.add_argument(
        "--config", default="config/benchmark.yml",
        help="benchmark 配置文件（含 tcga_pretrained 标记）"
    )
    args = parser.parse_args()

    config_path = os.path.join(SCRIPT_DIR, args.config)
    results_dir = args.results if os.path.isabs(args.results) else \
                  os.path.join(SCRIPT_DIR, args.results)

    analyze(results_dir, config_path)


if __name__ == "__main__":
    main()
