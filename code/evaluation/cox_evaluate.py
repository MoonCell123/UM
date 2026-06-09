"""
cox_evaluate.py
阈值搜索 + 评估 + KM曲线绘制。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.metrics import roc_curve, auc as sklearn_auc
from cox_train import compute_cindex

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 时间点标签常量
YEAR_LABELS_EN = {12: "1-Year", 24: "2-Year", 36: "3-Year", 48: "4-Year", 60: "5-Year"}
YEAR_LABELS_ZH = {12: "1年", 24: "2年", 36: "3年", 48: "4年", 60: "5年"}


def _logrank_p(risk_scores, times, events, threshold):
    """对给定阈值计算 log-rank p 值的辅助函数。"""
    high_mask = risk_scores >= threshold
    low_mask = ~high_mask
    if high_mask.sum() < 2 or low_mask.sum() < 2:
        return float("nan")
    try:
        result = logrank_test(
            times[high_mask], times[low_mask],
            events[high_mask], events[low_mask],
        )
        return result.p_value
    except Exception:
        return float("nan")


def _print_threshold(name, threshold, risk_scores, pvalue, N):
    """打印单个阈值策略的统计信息。"""
    high_n = (risk_scores >= threshold).sum()
    low_n = N - high_n
    p_str = f"{pvalue:.2e}" if not np.isnan(pvalue) else "N/A"
    print(f"[{name}] 阈值: {threshold:.4f}, Log-rank p = {p_str}")
    print(f"  分组: 高风险 {high_n}人 ({high_n/N*100:.1f}%), "
          f"低风险 {low_n}人 ({low_n/N*100:.1f}%)")


def search_optimal_threshold(risk_scores, times, events, min_group_ratio=0.2):
    """
    在训练集上搜索阈值策略，同时返回以便对比。

    返回的阈值字典包含:
      - "youden":     Youden-like 最优点（time-dependent AUC 思路，不依赖多重检验）
      - "logrank":    log-rank穷举最优（探索性分析，有过拟合风险）

    Args:
        risk_scores: numpy array [N]
        times: numpy array [N]
        events: numpy array [N]
        min_group_ratio: float, 每组最少占总人数的比例（默认0.2=20%）

    Returns:
        thresholds: dict  {策略名: threshold值}
        pvalues:    dict  {策略名: p值}
    """
    N = len(risk_scores)
    min_group_size = max(3, int(N * min_group_ratio))
    median_thr = float(np.median(risk_scores))

    thresholds = {}
    pvalues = {}

    # ── 策略1: Youden-like 最优点 ──
    # 把 event 当作二分类标签，risk_score 当作预测值
    # 对每个候选阈值算 sensitivity + specificity - 1，取最大
    sorted_scores = np.sort(np.unique(risk_scores))
    if len(sorted_scores) >= 2:
        candidates = (sorted_scores[:-1] + sorted_scores[1:]) / 2.0

        best_youden = -1.0
        youden_thr = median_thr

        n_event = events.sum()
        n_nonevent = N - n_event

        if n_event > 0 and n_nonevent > 0:
            for threshold in candidates:
                high_mask = risk_scores >= threshold
                # sensitivity: 复发者中被判为高风险的比例
                sens = events[high_mask].sum() / n_event
                # specificity: 未复发者中被判为低风险的比例
                spec = (1 - events[~high_mask]).sum() / n_nonevent
                youden = sens + spec - 1
                if youden > best_youden:
                    best_youden = youden
                    youden_thr = threshold

        thresholds["youden"] = float(youden_thr)
        pvalues["youden"] = _logrank_p(risk_scores, times, events, youden_thr)
        print(f"\n  Youden index: {best_youden:.3f}")
        _print_threshold("Youden", youden_thr, risk_scores, pvalues["youden"], N)

        # ── 策略2: log-rank 最优 ──
        best_pvalue = 1.0
        best_threshold = median_thr

        for threshold in candidates:
            high_mask = risk_scores >= threshold
            low_mask = risk_scores < threshold

            if high_mask.sum() < min_group_size or low_mask.sum() < min_group_size:
                continue
            if events[high_mask].sum() == 0 and events[low_mask].sum() == 0:
                continue

            try:
                result = logrank_test(
                    times[high_mask], times[low_mask],
                    events[high_mask], events[low_mask],
                )
                if result.p_value < best_pvalue:
                    best_pvalue = result.p_value
                    best_threshold = threshold
            except Exception:
                continue

        thresholds["logrank"] = best_threshold
        pvalues["logrank"] = best_pvalue
        _print_threshold("\nLog-rank最优", best_threshold, risk_scores, pvalues["logrank"], N)

    else:
        thresholds["youden"] = median_thr
        pvalues["youden"] = _logrank_p(risk_scores, times, events, median_thr)
        thresholds["logrank"] = median_thr
        pvalues["logrank"] = pvalues["youden"]

    return thresholds, pvalues


def compute_hazard_ratio(risk_scores, times, events, threshold):
    """
    计算高/低风险组的Hazard Ratio。

    Returns:
        hr: float, hazard ratio
        ci_lower: float, 95% CI下界
        ci_upper: float, 95% CI上界
    """
    groups = (risk_scores >= threshold).astype(int)

    df = pd.DataFrame({
        "T": times,
        "E": events,
        "group": groups,
    })

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="T", event_col="E", formula="group")
        hr = cph.hazard_ratios_["group"]
        ci = cph.confidence_intervals_
        ci_lower = np.exp(ci.iloc[0, 0])
        ci_upper = np.exp(ci.iloc[0, 1])
        return hr, ci_lower, ci_upper
    except Exception as e:
        print(f"  HR计算失败: {e}")
        return float("nan"), float("nan"), float("nan")


def plot_km_curve(risk_scores, times, events, threshold,
                  title="KM Survival Curve", save_path=None,
                  y_label="Disease-Free Survival Probability"):
    """
    绘制Kaplan-Meier生存曲线。

    Args:
        risk_scores: numpy array [N]
        times: numpy array [N] (月)
        events: numpy array [N]
        threshold: float（二分组）或 (float, float) 元组（三分组）
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    is_three_group = isinstance(threshold, (tuple, list))

    if is_three_group:
        thr_low, thr_high = threshold
        low_mask = risk_scores < thr_low
        mid_mask = (risk_scores >= thr_low) & (risk_scores < thr_high)
        high_mask = risk_scores >= thr_high

        n_low, n_mid, n_high = low_mask.sum(), mid_mask.sum(), high_mask.sum()

        colors = ["#2196F3", "#FF9800", "#F44336"]
        labels = [f"Low Risk (n={n_low})", f"Mid Risk (n={n_mid})",
                  f"High Risk (n={n_high})"]
        masks = [low_mask, mid_mask, high_mask]

        for mask, label, color in zip(masks, labels, colors):
            if mask.sum() > 0:
                kmf.fit(times[mask], events[mask], label=label)
                kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True,
                                           color=color, linewidth=2)

        # multivariate log-rank p
        try:
            groups = np.zeros(len(risk_scores), dtype=int)
            groups[mid_mask] = 1
            groups[high_mask] = 2
            from lifelines.statistics import multivariate_logrank_test
            result = multivariate_logrank_test(times, groups, events)
            p_text = (f"p = {result.p_value:.4f}" if result.p_value >= 0.0001
                      else f"p = {result.p_value:.2e}")
        except Exception:
            p_text = "p = N/A"

    else:
        high_mask = risk_scores >= threshold
        low_mask = ~high_mask
        n_high = high_mask.sum()
        n_low = low_mask.sum()

        if n_low > 0:
            kmf.fit(times[low_mask], events[low_mask],
                    label=f"Low Risk (n={n_low})")
            kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True,
                                       color="#2196F3", linewidth=2)
        if n_high > 0:
            kmf.fit(times[high_mask], events[high_mask],
                    label=f"High Risk (n={n_high})")
            kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True,
                                       color="#F44336", linewidth=2)

        if n_high > 0 and n_low > 0:
            try:
                result = logrank_test(
                    times[high_mask], times[low_mask],
                    events[high_mask], events[low_mask],
                )
                p_val = result.p_value
                p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else f"p = {p_val:.2e}"
            except Exception:
                p_text = "p = N/A"
        else:
            p_text = "p = N/A"

    ax.text(0.95, 0.95, p_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower left", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf",
                    bbox_inches="tight")
        print(f"  KM曲线已保存: {save_path}")
    plt.close()


def evaluate_cohort(risk_scores, times, events, thresholds,
                    cohort_name="", save_dir=None,
                    endpoint_label="DFS", y_label=None, file_prefix=""):
    """
    对单个队列使用多种阈值策略进行完整评估。

    Args:
        thresholds: dict {策略名: threshold值 或 (thr_low, thr_high)}

    Returns:
        all_metrics: list of dict, 每种阈值策略一条记录
    """
    # C-index 不依赖阈值，只算一次
    ci = compute_cindex(risk_scores, times, events)

    if y_label is None:
        y_label = f"{endpoint_label} Survival Probability"

    all_metrics = []
    for strategy, threshold in thresholds.items():
        is_three_group = isinstance(threshold, (tuple, list))

        if is_three_group:
            thr_low, thr_high = threshold
            low_mask = risk_scores < thr_low
            mid_mask = (risk_scores >= thr_low) & (risk_scores < thr_high)
            high_mask = risk_scores >= thr_high
            n_low, n_mid, n_high = int(low_mask.sum()), int(mid_mask.sum()), int(high_mask.sum())

            # multivariate log-rank
            try:
                groups = np.zeros(len(risk_scores), dtype=int)
                groups[mid_mask] = 1
                groups[high_mask] = 2
                from lifelines.statistics import multivariate_logrank_test
                result = multivariate_logrank_test(times, groups, events)
                p_val = result.p_value
            except Exception:
                p_val = float("nan")

            # HR: 高风险 vs 低风险（跳过中间组）
            hr, hr_ci_lower, hr_ci_upper = compute_hazard_ratio(
                risk_scores, times, events, thr_high
            )

            metrics = {
                "cohort": cohort_name,
                "strategy": strategy,
                "threshold": f"({thr_low:.4f}, {thr_high:.4f})",
                "n_total": len(risk_scores),
                "n_high_risk": n_high,
                "n_mid_risk": n_mid,
                "n_low_risk": n_low,
                "c_index": ci,
                "log_rank_p": p_val,
                "hr": hr,
                "hr_ci_lower": hr_ci_lower,
                "hr_ci_upper": hr_ci_upper,
            }

            print(f"\n{'=' * 50}")
            print(f"[{cohort_name}] 阈值策略: {strategy} (三分组)")
            print(f"{'=' * 50}")
            print(f"  阈值: ({thr_low:.4f}, {thr_high:.4f})")
            print(f"  样本数: {len(risk_scores)} "
                  f"(高: {n_high}, 中: {n_mid}, 低: {n_low})")
            print(f"  C-index: {ci:.4f}")
            print(f"  Log-rank p: {p_val:.2e}")
            print(f"  HR(高vs低): {hr:.2f} (95%CI: {hr_ci_lower:.2f}-{hr_ci_upper:.2f})")

        else:
            high_mask = risk_scores >= threshold
            low_mask = ~high_mask

            try:
                result = logrank_test(
                    times[high_mask], times[low_mask],
                    events[high_mask], events[low_mask],
                )
                p_val = result.p_value
            except Exception:
                p_val = float("nan")

            hr, hr_ci_lower, hr_ci_upper = compute_hazard_ratio(
                risk_scores, times, events, threshold
            )

            n_high = int(high_mask.sum())
            n_low = int(low_mask.sum())

            metrics = {
                "cohort": cohort_name,
                "strategy": strategy,
                "threshold": f"{threshold:.4f}",
                "n_total": len(risk_scores),
                "n_high_risk": n_high,
                "n_low_risk": n_low,
                "c_index": ci,
                "log_rank_p": p_val,
                "hr": hr,
                "hr_ci_lower": hr_ci_lower,
                "hr_ci_upper": hr_ci_upper,
            }

            print(f"\n{'=' * 50}")
            print(f"[{cohort_name}] 阈值策略: {strategy}")
            print(f"{'=' * 50}")
            print(f"  阈值: {threshold:.4f}")
            print(f"  样本数: {len(risk_scores)} (高风险: {n_high}, 低风险: {n_low})")
            print(f"  C-index: {ci:.4f}")
            print(f"  Log-rank p: {p_val:.2e}")
            print(f"  HR: {hr:.2f} (95%CI: {hr_ci_lower:.2f}-{hr_ci_upper:.2f})")

        all_metrics.append(metrics)

        # KM曲线
        if save_dir:
            km_path = os.path.join(save_dir, f"KM_{file_prefix}{cohort_name}_{strategy}.png")
            plot_km_curve(
                risk_scores, times, events, threshold,
                title=f"{endpoint_label} - {cohort_name} ({strategy})",
                save_path=km_path,
                y_label=y_label,
            )

    return all_metrics


def save_results_report(all_metrics, thresholds, save_dir, file_suffix=""):
    """保存评估结果汇总表（支持多阈值策略）。

    Args:
        all_metrics: list of dict, 每条记录包含 strategy 字段
        thresholds: dict {策略名: threshold值}
        save_dir: 输出目录
    """
    df = pd.DataFrame(all_metrics)
    suffix = f"_{file_suffix}" if file_suffix else ""
    csv_path = os.path.join(save_dir, f"evaluation_results{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n评估结果已保存: {csv_path}")

    # 文本报告
    report_path = os.path.join(save_dir, f"evaluation_report{suffix}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Cox回归生存预测评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("阈值汇总:\n")
        for name, thr in thresholds.items():
            if isinstance(thr, (tuple, list)):
                f.write(f"  [{name}] ({thr[0]:.4f}, {thr[1]:.4f})\n")
            else:
                f.write(f"  [{name}] {thr:.4f}\n")
        f.write("\n")

        for strategy in thresholds.keys():
            thr = thresholds[strategy]
            if isinstance(thr, (tuple, list)):
                thr_str = f"({thr[0]:.4f}, {thr[1]:.4f})"
            else:
                thr_str = f"{thr:.4f}"
            f.write(f"{'─' * 60}\n")
            f.write(f"阈值策略: {strategy} (threshold = {thr_str})\n")
            f.write(f"{'─' * 60}\n\n")

            strategy_metrics = [m for m in all_metrics if m["strategy"] == strategy]
            for m in strategy_metrics:
                f.write(f"  --- {m['cohort']} ---\n")
                if "n_mid_risk" in m:
                    f.write(f"    样本数: {m['n_total']} "
                            f"(高: {m['n_high_risk']}, 中: {m['n_mid_risk']}, "
                            f"低: {m['n_low_risk']})\n")
                else:
                    f.write(f"    样本数: {m['n_total']} "
                            f"(高风险: {m['n_high_risk']}, 低风险: {m['n_low_risk']})\n")
                f.write(f"    C-index: {m['c_index']:.4f}\n")
                f.write(f"    Log-rank p: {m['log_rank_p']:.2e}\n")
                f.write(f"    HR: {m['hr']:.2f} "
                        f"(95%CI: {m['hr_ci_lower']:.2f}-{m['hr_ci_upper']:.2f})\n\n")

    print(f"评估报告已保存: {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Time-dependent AUC & ROC 曲线
# ──────────────────────────────────────────────────────────────────────

def _make_survival_array(times, events):
    """将 times/events 转为 scikit-survival 需要的结构化数组。"""
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


def compute_time_dependent_auc(train_times, train_events,
                                test_risk_scores, test_times, test_events,
                                time_points_month=(12, 36, 60)):
    """
    计算指定时间点的 time-dependent AUC (Uno's estimator)。

    需要训练集的生存数据来估计 IPCW 权重，这是 scikit-survival 的要求。

    Args:
        train_times, train_events: 训练集的时间和事件（用于 IPCW 权重估计）
        test_risk_scores: 待评估队列的连续 risk score [N]
        test_times, test_events: 待评估队列的时间和事件 [N]
        time_points_month: 评估时间点（月），默认 (12, 36, 60) 即 1/3/5 年

    Returns:
        auc_dict: dict {时间点(月): AUC值}，无法计算的时间点为 NaN
        mean_auc: float, 所有有效时间点的平均 AUC
    """
    train_surv = _make_survival_array(train_times, train_events)
    test_surv = _make_survival_array(test_times, test_events)

    # 筛选有效时间点：必须在训练集和测试集的观察范围内
    # scikit-survival 要求 time_points 在 (train_min, train_max) 范围内
    train_min = train_times[train_events == 1].min() if train_events.sum() > 0 else train_times.min()
    train_max = train_times.max()
    test_max = test_times.max()
    upper_bound = min(train_max, test_max)

    valid_times = [t for t in time_points_month if train_min < t < upper_bound]

    auc_dict = {}
    for t in time_points_month:
        if t not in valid_times:
            auc_dict[t] = float("nan")

    if not valid_times:
        print(f"  ⚠️ 所有时间点 {time_points_month} 超出观察范围 "
              f"({train_min:.0f}, {upper_bound:.0f})月，无法计算 td-AUC")
        return auc_dict, float("nan")

    try:
        aucs, mean_auc = cumulative_dynamic_auc(
            train_surv, test_surv,
            test_risk_scores,
            times=valid_times,
        )
        for t, auc in zip(valid_times, aucs):
            auc_dict[t] = float(auc)

        return auc_dict, float(mean_auc)

    except Exception as e:
        print(f"  ⚠️ Time-dependent AUC 计算失败: {e}")
        for t in valid_times:
            auc_dict[t] = float("nan")
        return auc_dict, float("nan")


def plot_time_dependent_roc(train_times, train_events,
                             test_risk_scores, test_times, test_events,
                             time_points_month=(12, 36, 60),
                             cohort_name="", save_path=None):
    """
    绘制多个时间点的 time-dependent ROC 曲线（同一张图）。

    在每个时间点 t 定义:
      - 阳性: 在 t 之前发生事件 (E=1, T≤t)
      - 阴性: 在 t 之后仍存活 (T>t)
    对不同 risk score 阈值扫描计算 TPR/FPR。

    图上标注的 AUC 使用与 ROC 曲线一致的普通二分类 AUC，确保图文统一。
    IPCW td-AUC（Uno's estimator）作为独立指标通过 auc_dict 返回，
    两者在汇总表中同时呈现，方便对比。

    Args:
        train_times, train_events: 训练集数据（IPCW 权重估计）
        test_risk_scores: 待评估队列的 risk score
        test_times, test_events: 待评估队列的生存数据
        time_points_month: 时间点（月）
        cohort_name: 队列名
        save_path: 图片保存路径

    Returns:
        auc_dict: dict {t: IPCW td-AUC} — Uno's estimator（用于汇总表）
        plot_auc_dict: dict {t: 普通二分类 AUC} — 与 ROC 曲线一致
        mean_auc: float — IPCW td-AUC 均值
    """
    # 计算 IPCW td-AUC（Uno's estimator），用于汇总表
    auc_dict, mean_auc = compute_time_dependent_auc(
        train_times, train_events,
        test_risk_scores, test_times, test_events,
        time_points_month,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

    plot_auc_dict = {}

    # 对每个时间点画 ROC 曲线
    for i, t in enumerate(time_points_month):
        label = YEAR_LABELS_EN.get(t, f"{t}m")

        # 在时间点 t：case = 事件发生在 t 之前，control = 在 t 之后仍存活
        case_mask = (test_events == 1) & (test_times <= t)
        control_mask = test_times > t

        n_case = case_mask.sum()
        n_control = control_mask.sum()

        # 需要 case 和 control 都至少有 1 个才能画 ROC
        if n_case < 1 or n_control < 1:
            plot_auc_dict[t] = float("nan")
            ax.plot([], [], color=colors[i % len(colors)],
                    label=f"{label} (N/A)", linewidth=2)
            continue

        case_scores = test_risk_scores[case_mask]
        control_scores = test_risk_scores[control_mask]

        # 用 sklearn 计算 ROC 曲线和对应的 AUC（普通二分类）
        y_true = np.concatenate([np.ones(n_case), np.zeros(n_control)])
        y_score = np.concatenate([case_scores, control_scores])
        fprs, tprs, _ = roc_curve(y_true, y_score)
        plot_auc = sklearn_auc(fprs, tprs)
        plot_auc_dict[t] = float(plot_auc)

        ax.plot(fprs, tprs, color=colors[i % len(colors)],
                label=f"{label} AUC={plot_auc:.3f}",
                linewidth=2)

    # 对角线
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

    # 标注 mean AUC（使用图上一致的普通 AUC）
    valid_plot_aucs = [v for v in plot_auc_dict.values() if not np.isnan(v)]
    mean_plot_auc = float(np.mean(valid_plot_aucs)) if valid_plot_aucs else float("nan")
    mean_str = f"{mean_plot_auc:.3f}" if not np.isnan(mean_plot_auc) else "N/A"
    ax.text(0.55, 0.05, f"Mean AUC = {mean_str}",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"Time-Dependent ROC - {cohort_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf",
                    bbox_inches="tight")
        print(f"  Time-dependent ROC 已保存: {save_path}")
    plt.close()

    return auc_dict, plot_auc_dict, mean_auc


# ──────────────────────────────────────────────────────────────────────
# Decision Curve Analysis (DCA) for survival data
# ──────────────────────────────────────────────────────────────────────

def _fit_coxph_risk_model(train_risk_scores, train_times, train_events):
    """
    用训练集拟合单变量 CoxPH (risk_score → survival)。
    返回拟合好的 CoxPHFitter 对象，可多次调用 predict_survival_function。
    """
    train_df = pd.DataFrame({
        "T": train_times,
        "E": train_events,
        "risk_score": train_risk_scores,
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="T", event_col="E", formula="risk_score")
    return cph


def compute_event_probabilities(cph, test_risk_scores, time_points):
    """
    用已拟合的 CoxPH 模型，一次性预测测试集在多个时间点的事件概率。

    Args:
        cph: 已拟合的 CoxPHFitter
        test_risk_scores: 测试集 risk scores [N_test]
        time_points: 时间点列表（月）

    Returns:
        pred_probs: dict {t: np.ndarray[N_test]}, 每个时间点的事件概率 P(T ≤ t)
    """
    test_df = pd.DataFrame({"risk_score": test_risk_scores})
    surv_func = cph.predict_survival_function(test_df, times=list(time_points))
    # surv_func: DataFrame, index=time_points, columns=个体索引
    return {t: 1.0 - surv_func.loc[t].values for t in time_points}


def _compute_integrated_nb(thresholds, nb_model):
    """
    计算积分净获益 (Integrated Net Benefit) 作为 DCA 的单一汇总值。

    在阈值概率 [0.01, 0.50] 范围内，对 max(NB_model, 0) 做梯形积分，
    再除以区间宽度，得到平均净获益。类似 AUC 的概念：值越大越好。

    Args:
        thresholds: np.ndarray, 阈值概率序列
        nb_model: np.ndarray, 每个阈值处模型的净获益

    Returns:
        inb: float, Integrated Net Benefit
    """
    mask = (thresholds >= 0.01) & (thresholds <= 0.50)
    if mask.sum() < 2:
        return float("nan")
    t_sub = thresholds[mask]
    nb_sub = np.maximum(nb_model[mask], 0.0)
    area = np.trapz(nb_sub, t_sub)
    inb = area / (t_sub[-1] - t_sub[0])
    return float(inb)


def compute_dca(train_times, train_events, train_risk_scores,
                test_risk_scores, test_times, test_events,
                time_points_month=(12, 36, 60),
                threshold_range=(0.01, 0.99), n_thresholds=200):
    """
    核心 DCA 计算：对每个时间点 t，将 risk score 转为预测概率，
    计算 Model / Treat-All 的净获益（经验方法，未做 IPCW 加权）。

    Args:
        train_times, train_events, train_risk_scores: 训练集数据
        test_risk_scores: 待评估队列的 risk score [N]
        test_times, test_events: 待评估队列的生存数据 [N]
        time_points_month: 评估时间点（月）
        threshold_range: 阈值概率扫描范围
        n_thresholds: 阈值个数

    Returns:
        dca_results: dict {t: {"thresholds", "nb_model", "nb_all"}}
        dca_summary: dict {t: INB}
        mean_dca: float
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    # CoxPH 只拟合一次（训练数据不变，只是预测不同时间点的生存概率）
    try:
        cph = _fit_coxph_risk_model(train_risk_scores, train_times, train_events)
        pred_probs_all = compute_event_probabilities(
            cph, test_risk_scores, time_points_month,
        )
    except Exception as e:
        print(f"  ⚠️ DCA: CoxPH 拟合或预测失败: {e}")
        dca_results = {t: None for t in time_points_month}
        dca_summary = {t: float("nan") for t in time_points_month}
        return dca_results, dca_summary, float("nan")

    dca_results = {}
    dca_summary = {}

    for t in time_points_month:
        pred_probs = pred_probs_all[t]

        # 定义在时间点 t 的状态
        # case: 事件发生在 t 之前（T ≤ t 且 E=1）
        # control: 在 t 之后仍存活（T > t）
        # 排除: t 前已删失（E=0 且 T ≤ t），状态未知
        event_by_t = (test_events == 1) & (test_times <= t)
        survived_t = test_times > t
        evaluable = event_by_t | survived_t  # 可判定状态的样本
        N_eff = int(evaluable.sum())

        if N_eff == 0:
            dca_results[t] = None
            dca_summary[t] = float("nan")
            continue

        prevalence = event_by_t.sum() / N_eff

        # 向量化计算所有阈值的 TP/FP（避免 Python for 循环）
        # treat_mask: [n_thresholds, N] — 每行表示一个阈值下的治疗决策
        treat_mask = pred_probs[np.newaxis, :] >= thresholds[:, np.newaxis]
        tp = (treat_mask & event_by_t[np.newaxis, :]).sum(axis=1)    # [n_thresholds]
        fp = (treat_mask & survived_t[np.newaxis, :]).sum(axis=1)    # [n_thresholds]

        # odds ratio: pt / (1 - pt), 避免 pt=1 时除零
        odds = np.where(thresholds < 1.0, thresholds / (1.0 - thresholds), 0.0)

        # Model NB — 分母使用可判定样本数
        nb_model = tp / N_eff - fp / N_eff * odds

        # Treat-All NB
        nb_all = np.where(thresholds < 1.0,
                          prevalence - (1 - prevalence) * odds, 0.0)

        dca_results[t] = {
            "thresholds": thresholds,
            "nb_model": nb_model,
            "nb_all": nb_all,
        }

        inb = _compute_integrated_nb(thresholds, nb_model)
        dca_summary[t] = inb

    # 均值
    valid_inbs = [v for v in dca_summary.values() if not np.isnan(v)]
    mean_dca = float(np.mean(valid_inbs)) if valid_inbs else float("nan")

    return dca_results, dca_summary, mean_dca


def plot_dca_curves(train_times, train_events, train_risk_scores,
                    test_risk_scores, test_times, test_events,
                    time_points_month=(12, 36, 60),
                    cohort_name="", save_path=None):
    """
    绘制 DCA 曲线图：每个队列一张图，N 个子图（按时间点）。

    每个子图画三条线：
      - Model（蓝色实线）
      - Treat All（灰色点划线）
      - Treat None（黑色虚线 y=0）

    Args:
        train_times, train_events: 训练集的生存数据
        train_risk_scores: 训练集的 risk scores（用于拟合 CoxPH 概率模型）
        test_risk_scores: 待评估队列的 risk score
        test_times, test_events: 待评估队列的生存数据
        time_points_month: 时间点（月）
        cohort_name: 队列名
        save_path: 图片保存路径

    Returns:
        dca_summary: dict {t: INB}
        mean_dca: float
    """
    dca_results, dca_summary, mean_dca = compute_dca(
        train_times, train_events, train_risk_scores,
        test_risk_scores, test_times, test_events,
        time_points_month=time_points_month,
    )

    # 绘图
    n_times = len(time_points_month)
    fig, axes = plt.subplots(1, n_times, figsize=(6 * n_times, 5))
    if n_times == 1:
        axes = [axes]

    for idx, t in enumerate(time_points_month):
        ax = axes[idx]
        label = YEAR_LABELS_EN.get(t, f"{t}m")
        inb = dca_summary.get(t, float("nan"))
        inb_str = f"{inb:.4f}" if not np.isnan(inb) else "N/A"

        if dca_results.get(t) is None:
            ax.text(0.5, 0.5, f"N/A\n(时间点 {t}月 无法计算)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            ax.set_title(f"{label} DCA", fontsize=13)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 0.3)
            continue

        data = dca_results[t]
        th = data["thresholds"]
        nb_model = data["nb_model"]
        nb_all = data["nb_all"]

        # 绘制三条线
        ax.plot(th, nb_model, color="#2196F3", linewidth=2,
                label=f"Model (INB={inb_str})")
        ax.plot(th, nb_all, color="gray", linewidth=1.5, linestyle="-.",
                label="Treat All")
        ax.axhline(y=0, color="black", linewidth=1.5, linestyle="--",
                   label="Treat None")

        ax.set_xlabel("Threshold Probability", fontsize=11)
        ax.set_ylabel("Net Benefit", fontsize=11)
        ax.set_title(f"{label} DCA", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0, 1)

        # 动态 y 轴范围
        y_max = max(0.1, np.nanmax(nb_model) * 1.2, np.nanmax(nb_all) * 1.2)
        y_min = min(-0.05, np.nanmin(nb_all) * 1.1)
        ax.set_ylim(y_min, y_max)

        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Decision Curve Analysis - {cohort_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), format="pdf",
                    bbox_inches="tight")
        print(f"  DCA曲线已保存: {save_path}")
    plt.close()

    return dca_summary, mean_dca
