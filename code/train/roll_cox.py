"""
roll_cox.py
Cox 回归超参数网格遍历脚本（支持并行）。

使用方法:
    python roll_cox.py                              # 默认读取 config/roll_cox.yml
    python roll_cox.py --config config/roll_cox.yml
    python roll_cox.py --workers 8                  # 8 进程并行（默认 1 = 串行）

工作流程:
1. 读取 roll_cox.yml 中的 base_config + search_space
2. 对 search_space 做笛卡尔积，生成所有参数组合
3. 为每组参数预分配输出目录（exp_0001/, exp_0002/, ...）
4. 串行或并行调用 run_cox.py，每个实验是独立 subprocess
5. 汇总所有实验到 summary.csv，按联合评分降序排列

并行安全说明:
- 每个实验是独立 subprocess（独立 Python 进程、独立 CUDA context）
- 种子、cudnn 设置、随机状态完全隔离，不会互相干扰
- 输出目录由 roll_cox.py 预分配（exp_XXXX/），无冲突
- 特征文件只读，无写入竞争
- 并行数受 GPU 显存限制（每进程约 400-600 MB），建议:
    24 GB GPU: --workers 8~12
    12 GB GPU: --workers 4~6
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_roll_config(yaml_path):
    """加载遍历配置文件。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("base_config", {}), cfg.get("search_space", {}), cfg.get("workers", 1)


def build_param_grid(search_space):
    """将 search_space 展开为笛卡尔积。"""
    if not search_space:
        return [{}]
    grid = {k: v if isinstance(v, list) else [v] for k, v in search_space.items()}
    return list(ParameterGrid(grid))


def make_experiment_name(params):
    """根据参数生成简短的实验名。"""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "__".join(parts) if parts else "default"


# ──────────────────────────────────────────────────────────────────────
# 结果收集
# ──────────────────────────────────────────────────────────────────────

COHORTS = ["HMU_Train", "HMU_Val", "Wenfu_Test", "Fujian_Test"]
EVAL_COHORTS = ["HMU_Val", "Wenfu_Test", "Fujian_Test"]


def collect_results(exp_output_dir, override_params, exp_index):
    """
    从单次实验的输出目录中收集所有指标。

    读取:
      - config.json          → val_cindex
      - evaluation_results.csv → logrank 策略各队列 C-index / p / HR
      - time_dependent_auc.csv → 各队列 3年/5年 AUC
    """
    config_json = os.path.join(exp_output_dir, "config.json")
    if not os.path.exists(config_json):
        return None

    with open(config_json, "r", encoding="utf-8") as f:
        saved_config = json.load(f)

    result = dict(override_params)
    result["exp_index"] = exp_index
    result["exp_name"] = make_experiment_name(override_params)
    result["exp_dir"] = exp_output_dir
    result["val_cindex"] = saved_config.get("best_val_cindex", float("nan"))

    # logrank 策略指标
    eval_csv = os.path.join(exp_output_dir, "evaluation_results.csv")
    if os.path.exists(eval_csv):
        eval_df = pd.read_csv(eval_csv)
        logrank_df = eval_df[eval_df["strategy"] == "logrank"]
        for _, row in logrank_df.iterrows():
            cohort = row.get("cohort", "")
            if cohort in COHORTS:
                result[f"{cohort}_cindex"] = row.get("c_index", float("nan"))
                result[f"{cohort}_p"] = row.get("log_rank_p", float("nan"))
                result[f"{cohort}_hr"] = row.get("hr", float("nan"))
                result[f"{cohort}_n_high"] = row.get("n_high_risk", 0)
                result[f"{cohort}_n_low"] = row.get("n_low_risk", 0)

    # 3年/5年 AUC + INB
    auc_csv = os.path.join(exp_output_dir, "time_dependent_auc.csv")
    if os.path.exists(auc_csv):
        auc_df = pd.read_csv(auc_csv)
        for _, row in auc_df.iterrows():
            cohort = row.get("cohort", "")
            if cohort in COHORTS:
                result[f"{cohort}_AUC_36m"] = row.get("AUC_36m", float("nan"))
                result[f"{cohort}_AUC_60m"] = row.get("AUC_60m", float("nan"))
                result[f"{cohort}_mean_auc"] = row.get("mean_auc", float("nan"))
                result[f"{cohort}_INB_36m"] = row.get("INB_36m", float("nan"))
                result[f"{cohort}_INB_60m"] = row.get("INB_60m", float("nan"))
                result[f"{cohort}_mean_inb"] = row.get("mean_inb", float("nan"))

    result["joint_score"] = compute_joint_score(result)
    return result


def compute_joint_score(result):
    """
    联合评分 = val_cindex
               - 0.02 × (外部队列中 p >= 0.05 的个数)
               - 0.01 × (外部队列中 3y/5y AUC < 0.7 的指标数)

    注: DCA (INB) 指标暂不纳入评分公式（先观察值分布），
        后续可考虑加入 INB 的惩罚/奖励项。
    """
    val_ci = result.get("val_cindex", 0.5)
    if np.isnan(val_ci):
        return -1.0

    score = val_ci

    for cohort in EVAL_COHORTS:
        p = result.get(f"{cohort}_p", float("nan"))
        if np.isnan(p) or p >= 0.05:
            score -= 0.02

    for cohort in EVAL_COHORTS:
        for suffix in ["AUC_36m", "AUC_60m"]:
            auc = result.get(f"{cohort}_{suffix}", float("nan"))
            if np.isnan(auc) or auc < 0.7:
                score -= 0.01

    return round(score, 6)


# ──────────────────────────────────────────────────────────────────────
# 单次实验（可在子进程中执行）
# ──────────────────────────────────────────────────────────────────────

def run_single_experiment(task):
    """
    运行单次实验。作为独立函数以便 ProcessPoolExecutor 调用。

    Args:
        task: dict with keys: base_config, override_params, exp_dir, exp_index, total

    Returns:
        result_dict 或 None
    """
    base_config = task["base_config"]
    override_params = task["override_params"]
    exp_dir = task["exp_dir"]
    exp_index = task["exp_index"]
    total = task["total"]

    exp_name = make_experiment_name(override_params)

    # 合并配置，指向预分配的输出目录
    exp_config = dict(base_config)
    exp_config.update(override_params)
    exp_config["output_dir"] = exp_dir

    # 写临时 YAML
    tmp_dir = os.path.join(os.path.dirname(exp_dir), "_tmp_configs")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_yaml = os.path.join(tmp_dir, f"exp_{exp_index:04d}.yml")
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(exp_config, f, allow_unicode=True, default_flow_style=False)

    # 打印进度
    print(f"\n{'█' * 60}")
    print(f"  实验 [{exp_index + 1}/{total}]  {exp_name}")
    print(f"{'█' * 60}", flush=True)

    # 调用 run_cox.py
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "run_cox.py"), "--config", tmp_yaml]
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3600,
        )
        if proc.returncode != 0:
            print(f"  ⚠️ 实验 {exp_index + 1} 失败 (exit code {proc.returncode})")
            if proc.stderr:
                print(proc.stderr[-500:])
            return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ 实验 {exp_index + 1} 超时 (>1小时)")
        return None
    except Exception as e:
        print(f"  ⚠️ 实验 {exp_index + 1} 异常: {e}")
        return None

    # 收集结果
    result = collect_results(exp_dir, override_params, exp_index)
    if result:
        js = result["joint_score"]
        vc = result["val_cindex"]
        print(f"  ✅ 实验 {exp_index + 1} 完成: "
              f"joint={js:.4f}  val_ci={vc:.4f}  [{exp_name}]", flush=True)
    return result


# ──────────────────────────────────────────────────────────────────────
# 汇总 & 报告
# ──────────────────────────────────────────────────────────────────────

def save_summary(results, output_root):
    """保存汇总 CSV，按联合评分降序。"""
    df = pd.DataFrame(results)
    if "joint_score" in df.columns:
        df = df.sort_values("joint_score", ascending=False)
    summary_path = os.path.join(output_root, "summary.csv")
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def print_final_report(results, summary_path):
    """打印最终报告。"""
    df = pd.DataFrame(results)
    if "joint_score" in df.columns:
        df = df.sort_values("joint_score", ascending=False)

    print("\n" + "=" * 70)
    print("超参数遍历完成！")
    print("=" * 70)
    print(f"成功: {len(df)} 组")
    print(f"汇总表: {summary_path}")

    print(f"\n{'─' * 70}")
    print("Top-5 (按联合评分排序):")
    print(f"{'─' * 70}")
    for rank, (_, row) in enumerate(df.head(5).iterrows(), 1):
        js = row.get("joint_score", float("nan"))
        vc = row.get("val_cindex", float("nan"))
        print(f"\n  #{rank}  joint_score={js:.4f}  val_cindex={vc:.4f}")
        print(f"       {row.get('exp_name', '')}")

        for cohort in COHORTS:
            ci = row.get(f"{cohort}_cindex", float("nan"))
            if np.isnan(ci):
                continue
            p = row.get(f"{cohort}_p", float("nan"))
            auc3 = row.get(f"{cohort}_AUC_36m", float("nan"))
            auc5 = row.get(f"{cohort}_AUC_60m", float("nan"))

            ci_s = f"C={ci:.4f}"
            p_s = f"p={p:.2e}" if not np.isnan(p) else "p=N/A"
            auc3_s = f"3y={auc3:.3f}" if not np.isnan(auc3) else "3y=N/A"
            auc5_s = f"5y={auc5:.3f}" if not np.isnan(auc5) else "5y=N/A"

            inb3 = row.get(f"{cohort}_INB_36m", float("nan"))
            inb5 = row.get(f"{cohort}_INB_60m", float("nan"))
            inb3_s = f"3yINB={inb3:.4f}" if not np.isnan(inb3) else "3yINB=N/A"
            inb5_s = f"5yINB={inb5:.4f}" if not np.isnan(inb5) else "5yINB=N/A"

            flags = []
            if not np.isnan(p) and p < 0.05:
                flags.append("✅")
            elif cohort != "HMU_Train":
                flags.append("❌p")
            if not np.isnan(auc3) and auc3 >= 0.7:
                flags.append("✅3y")
            elif cohort != "HMU_Train" and not np.isnan(auc3):
                flags.append("❌3y")

            print(f"       {cohort:>14}: {ci_s}  {p_s}  {auc3_s}  {auc5_s}  {inb3_s}  {inb5_s}  {' '.join(flags)}")

    print(f"\n{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cox 超参数网格遍历（支持并行）")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(SCRIPT_DIR, "config", "roll_cox.yml"),
        help="遍历配置文件路径 (默认: config/roll_cox.yml)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="并行进程数（覆盖配置文件中的 workers 值）",
    )
    args = parser.parse_args()

    base_config, search_space, cfg_workers = load_roll_config(args.config)
    param_grid = build_param_grid(search_space)
    total = len(param_grid)
    workers = min(args.workers if args.workers is not None else cfg_workers, total)

    # 创建输出根目录
    output_root = base_config.get("output_dir", "cox_roll_output")
    if not os.path.isabs(output_root):
        output_root = os.path.join(SCRIPT_DIR, output_root)
    os.makedirs(output_root, exist_ok=True)

    print("=" * 60)
    print("Cox 超参数网格遍历")
    print("=" * 60)
    print(f"搜索空间:")
    for k, v in search_space.items():
        print(f"  {k}: {v}")
    print(f"\n总组合数: {total}")
    print(f"并行进程: {workers}")
    if workers > 1:
        print(f"预计耗时: {total / workers * 5:.0f} ~ {total / workers * 8:.0f} 分钟")
    else:
        print(f"预计耗时: {total * 3:.0f} ~ {total * 8:.0f} 分钟")
    print()
    print("评分规则: joint_score = val_cindex")
    print("           - 0.02 × (外部队列 p≥0.05 的个数)")
    print("           - 0.01 × (外部队列 3y/5y AUC<0.7 的个数)")
    print("=" * 60)

    # 为每个实验预分配输出目录（exp_0001/, exp_0002/, ...）
    tasks = []
    for i, params in enumerate(param_grid):
        exp_dir = os.path.join(output_root, f"exp_{i + 1:04d}")
        tasks.append({
            "base_config": base_config,
            "override_params": params,
            "exp_dir": exp_dir,
            "exp_index": i,
            "total": total,
        })

    # 执行实验
    all_results = []

    if workers <= 1:
        # 串行模式（保持原来的行为）
        for task in tasks:
            result = run_single_experiment(task)
            if result is not None:
                all_results.append(result)
            if all_results:
                save_summary(all_results, output_root)
    else:
        # 并行模式
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_single_experiment, task): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as e:
                    idx = task["exp_index"]
                    print(f"  ⚠️ 实验 {idx + 1} 异常: {e}")

                # 每收到一个结果就保存一次汇总
                if all_results:
                    save_summary(all_results, output_root)

    # 最终汇总
    if all_results:
        summary_path = save_summary(all_results, output_root)
        print_final_report(all_results, summary_path)
    else:
        print("\n⚠️ 所有实验均失败，无结果可汇总")


if __name__ == "__main__":
    main()
