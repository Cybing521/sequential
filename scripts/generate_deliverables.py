#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终交付物：
1. 每一步的性能比较表（带95%置信区间）
2. 最终两步加起来的性能指标比较表（带95%置信区间）
3. Referrals Avoided, Odds Ratio, P值表
4. 桑基图数据
5. 桑基图（类似Chen 2024论文风格）
6. PPV/NPV vs Prevalence曲线图

Author: Auto-generated
Date: 2026-01-21
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrow
from matplotlib.sankey import Sankey
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import chi2_contingency

# Try to import plotly for interactive Sankey
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Sankey diagrams will use matplotlib fallback.")

# ============================================================
# 配置
# ============================================================
@dataclass
class Config:
    """配置参数"""
    # 数据路径
    pred_path: str = "data/all_models_sample_predictions.csv"
    threshold_path: str = "data/thresholds_from_val.csv"
    output_dir: str = "outputs/deliverables"
    
    # 数据集名称
    split_val: str = "Val"
    split_internal_test: str = "Test"
    split_external_test: str = "External"
    
    # 模型标签
    tag_m1: str = "Radiomics_Only"              # M1: DL模型
    tag_m2: str = "Radiomics_plus_A"            # M2: DL + Clinical A
    tag_m3: str = "Radiomics_plus_AllClinical"  # M3: Fusion-Net
    tag_m4: str = "ClinicalA_Only"              # M4: Clinical A
    tag_m5: str = "BaseClinical_Only"           # M5: Clinical-Net
    
    # 模型显示名称
    model_display_names: Dict[str, str] = None
    
    def __post_init__(self):
        if self.model_display_names is None:
            self.model_display_names = {
                self.tag_m1: "M1 (Radiomics-Only)",
                self.tag_m2: "M2 (Radiomics+A)",
                self.tag_m3: "M3 (Fusion-Net)",
                self.tag_m4: "M4 (Clinical-A)",
                self.tag_m5: "M5 (Clinical-Net)",
            }
    
    @property
    def all_model_tags(self) -> List[str]:
        return [self.tag_m1, self.tag_m2, self.tag_m3, self.tag_m4, self.tag_m5]


# ============================================================
# 工具函数
# ============================================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    return a / b if b > 1e-12 else default


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    比简单的正态近似更适合小样本和极端比例。
    
    Returns:
        (lower_bound, upper_bound) of the confidence interval
    """
    if n == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def compute_metrics_with_ci(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    confidence: float = 0.95
) -> Dict:
    """
    计算诊断性能指标及其95%置信区间
    
    指标包括: Sen, Spec, PPV, NPV, Accuracy, TN, TP, FN, FP
    """
    n = len(y_true)
    
    # 混淆矩阵
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    # 计算点估计
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    acc = safe_div(tp + tn, n)
    
    # 计算95%置信区间 (Wilson score)
    sens_ci = wilson_ci(tp, tp + fn, confidence)
    spec_ci = wilson_ci(tn, tn + fp, confidence)
    ppv_ci = wilson_ci(tp, tp + fp, confidence)
    npv_ci = wilson_ci(tn, tn + fn, confidence)
    acc_ci = wilson_ci(tp + tn, n, confidence)
    
    return {
        "N": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": sens,
        "Sensitivity_CI_Lower": sens_ci[0],
        "Sensitivity_CI_Upper": sens_ci[1],
        "Specificity": spec,
        "Specificity_CI_Lower": spec_ci[0],
        "Specificity_CI_Upper": spec_ci[1],
        "PPV": ppv,
        "PPV_CI_Lower": ppv_ci[0],
        "PPV_CI_Upper": ppv_ci[1],
        "NPV": npv,
        "NPV_CI_Lower": npv_ci[0],
        "NPV_CI_Upper": npv_ci[1],
        "Accuracy": acc,
        "Accuracy_CI_Lower": acc_ci[0],
        "Accuracy_CI_Upper": acc_ci[1],
        "Prevalence": safe_div(tp + fn, n),
        "Positive_Rate": safe_div(tp + fp, n),  # 转诊率
    }


def compute_odds_ratio(
    case_exposed: int,    # a: 病例组暴露
    case_unexposed: int,  # b: 病例组未暴露
    ctrl_exposed: int,    # c: 对照组暴露
    ctrl_unexposed: int   # d: 对照组未暴露
) -> Tuple[float, float, float, float]:
    """
    计算优势比及其95%置信区间和P值
    
    OR = (a*d) / (b*c)
    
    Returns:
        (OR, CI_lower, CI_upper, p_value)
    """
    # 添加0.5校正避免除零
    a, b, c, d = case_exposed + 0.5, case_unexposed + 0.5, ctrl_exposed + 0.5, ctrl_unexposed + 0.5
    
    odds_ratio = (a * d) / (b * c)
    
    # 95% CI (Woolf method with continuity correction)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)
    
    # Chi-square test for p-value
    table = np.array([[case_exposed, case_unexposed], [ctrl_exposed, ctrl_unexposed]])
    try:
        _, p_value, _, _ = chi2_contingency(table, correction=True)
    except:
        p_value = np.nan
    
    return (odds_ratio, ci_lower, ci_upper, p_value)


def format_ci(value: float, ci_lower: float, ci_upper: float, precision: int = 3) -> str:
    """格式化带置信区间的值"""
    return f"{value:.{precision}f} ({ci_lower:.{precision}f}-{ci_upper:.{precision}f})"


def format_or_ci(or_val: float, ci_lower: float, ci_upper: float) -> str:
    """格式化优势比及置信区间"""
    return f"{or_val:.2f} ({ci_lower:.2f}, {ci_upper:.2f})"


def format_pvalue(p: float) -> str:
    """格式化P值"""
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


# ============================================================
# 两步筛查算法
# ============================================================

def stepwise_two_stage(
    y_true: np.ndarray,
    p_step1: np.ndarray,
    thr_step1: float,
    p_step2: np.ndarray,
    thr_step2: float,
) -> Dict:
    """
    两步顺序筛查算法
    
    逻辑:
        Step 1: 用临床参数筛查
            - p1 < thr1 → 排除（预测为0）
            - p1 >= thr1 → 进入Step 2
        Step 2: 对通过Step 1的患者
            - p2 >= thr2 → 纳入（预测为1）
            - p2 < thr2 → 排除（预测为0）
    
    Returns:
        Dict with step1/step2/final metrics and zone_counts for Sankey
    """
    n = len(y_true)
    
    # Step 1: 初步筛查
    step1_pred = (p_step1 >= thr_step1).astype(int)
    step1_metrics = compute_metrics_with_ci(y_true, step1_pred)
    
    # 哪些患者进入Step 2
    pass_mask = step1_pred == 1
    n_pass = int(pass_mask.sum())
    n_ruled_out = n - n_pass
    
    # Step 2: 二次筛查（仅对通过Step 1的患者）
    if n_pass > 0:
        step2_pred_subset = (p_step2[pass_mask] >= thr_step2).astype(int)
        step2_metrics = compute_metrics_with_ci(y_true[pass_mask], step2_pred_subset)
    else:
        step2_pred_subset = np.array([])
        step2_metrics = {k: np.nan for k in step1_metrics}
    
    # 最终预测
    final_pred = np.zeros(n, dtype=int)
    if n_pass > 0:
        final_pred[pass_mask] = step2_pred_subset
    
    final_metrics = compute_metrics_with_ci(y_true, final_pred)
    
    # 各区域统计（用于桑基图）
    rule_out_mask = step1_pred == 0
    gray_mask = pass_mask & (final_pred == 0)
    rule_in_mask = pass_mask & (final_pred == 1)
    
    # Step 1 阳性组中的细分
    step1_pos_tp = int(((y_true == 1) & pass_mask).sum())  # Step1+且真阳性
    step1_pos_fp = int(((y_true == 0) & pass_mask).sum())  # Step1+且假阳性
    
    zone_counts = {
        # Step 1 阴性组
        "Step1_Neg_TN": int(((y_true == 0) & rule_out_mask).sum()),
        "Step1_Neg_FN": int(((y_true == 1) & rule_out_mask).sum()),
        # Step 2 阴性组（Gray zone）
        "Step2_Neg_TN": int(((y_true == 0) & gray_mask).sum()),
        "Step2_Neg_FN": int(((y_true == 1) & gray_mask).sum()),
        # Step 2 阳性组（最终预测为阳性）
        "Step2_Pos_FP": int(((y_true == 0) & rule_in_mask).sum()),
        "Step2_Pos_TP": int(((y_true == 1) & rule_in_mask).sum()),
        # 各阶段人数
        "N_Total": n,
        "N_Step1_Pos": n_pass,
        "N_Step1_Neg": n_ruled_out,
        "N_Step2_Pos": int(rule_in_mask.sum()),
        "N_Step2_Neg": int(gray_mask.sum()),
        # Step 1+组中的真/假阳性
        "Step1_Pos_TP": step1_pos_tp,
        "Step1_Pos_FP": step1_pos_fp,
    }
    
    # 计算转诊避免率 (Referrals Avoided)
    # 这里定义为：Step 1被排除的比例
    referrals_avoided = n_ruled_out / n if n > 0 else 0
    
    return {
        "step1_pred": step1_pred,
        "step1_metrics": step1_metrics,
        "step2_pred": step2_pred_subset,
        "step2_metrics": step2_metrics,
        "final_pred": final_pred,
        "final_metrics": final_metrics,
        "zone_counts": zone_counts,
        "referrals_avoided": referrals_avoided,
        "n_pass_step1": n_pass,
    }


# ============================================================
# 可视化函数
# ============================================================

def plot_sankey_chen_style(
    zone_counts: Dict,
    step1_model: str,
    step1_threshold: float,
    step2_model: str,
    step2_threshold: float,
    cohort_name: str,
    outpath: str
):
    """
    绘制Chen 2024论文风格的桑基图
    
    流程:
    [全部患者] → [Step1+/Step1-] → [Step2结果] → [最终TN/FN/FP/TP]
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 提取数据
    n_total = zone_counts["N_Total"]
    n_step1_pos = zone_counts["N_Step1_Pos"]
    n_step1_neg = zone_counts["N_Step1_Neg"]
    
    # Step 1+中的细分
    step1_pos_tp = zone_counts["Step1_Pos_TP"]
    step1_pos_fp = zone_counts["Step1_Pos_FP"]
    
    # Step 1-中的细分
    step1_neg_tn = zone_counts["Step1_Neg_TN"]
    step1_neg_fn = zone_counts["Step1_Neg_FN"]
    
    # Step 2结果
    step2_pos = zone_counts["N_Step2_Pos"]
    step2_neg = zone_counts["N_Step2_Neg"]
    
    # 最终分类
    final_tn = zone_counts["Step1_Neg_TN"] + zone_counts["Step2_Neg_TN"]
    final_fn = zone_counts["Step1_Neg_FN"] + zone_counts["Step2_Neg_FN"]
    final_fp = zone_counts["Step2_Pos_FP"]
    final_tp = zone_counts["Step2_Pos_TP"]
    
    # 颜色定义（类似论文配色）
    colors = {
        "total": "#4A90D9",       # 蓝色 - 全部患者
        "step1_pos": "#F5A623",   # 橙色 - Step1+
        "step1_neg": "#4A90D9",   # 蓝色 - Step1-
        "step2_pos": "#F5A623",   # 橙色 - Step2+ (需要进一步检查)
        "step2_neg": "#7ED321",   # 绿色 - Step2- (排除)
        "TN": "#7ED321",          # 绿色 - 真阴性
        "FN": "#D0021B",          # 红色 - 假阴性
        "FP": "#F8E71C",          # 黄色 - 假阳性（浅色背景）
        "TP": "#9013FE",          # 紫色 - 真阳性
    }
    
    def pct(n): 
        return f"{100*n/n_total:.0f}%" if n_total > 0 else "0%"
    
    # ===== 绘制各节点 =====
    # X坐标位置
    x_total = 0.05
    x_step1 = 0.25
    x_step1_detail = 0.42
    x_step2 = 0.60
    x_final = 0.85
    
    # 节点高度计算
    total_h = 0.7
    y_base = 0.15
    
    # 1. 全部患者（左侧）
    ax.add_patch(plt.Rectangle((x_total, y_base), 0.08, total_h,
                               facecolor=colors["total"], edgecolor='black', lw=2, alpha=0.85))
    ax.text(x_total + 0.04, y_base + total_h/2, 
            f"{cohort_name}\n(N={n_total})",
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 2. Step 1 分流
    # 上半部分：Step1+
    pos_ratio = n_step1_pos / n_total if n_total > 0 else 0.5
    pos_h = total_h * max(pos_ratio, 0.1)
    pos_y = y_base + total_h - pos_h
    
    ax.add_patch(plt.Rectangle((x_step1, pos_y), 0.10, pos_h,
                               facecolor=colors["step1_pos"], edgecolor='black', lw=2, alpha=0.85))
    ax.text(x_step1 + 0.05, pos_y + pos_h/2,
            f"{step1_model}\nThreshold≥{step1_threshold:.3f}\nPositive (n={n_step1_pos})",
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 下半部分：Step1-
    neg_h = total_h - pos_h
    neg_y = y_base
    
    ax.add_patch(plt.Rectangle((x_step1, neg_y), 0.10, neg_h,
                               facecolor=colors["step1_neg"], edgecolor='black', lw=2, alpha=0.85))
    ax.text(x_step1 + 0.05, neg_y + neg_h/2,
            f"{step1_model}\nThreshold<{step1_threshold:.3f}\nNegative (n={n_step1_neg})",
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 3. Step 1+ 细分 (TP和FP)
    if n_step1_pos > 0:
        tp_ratio = step1_pos_tp / n_step1_pos
        tp_h = pos_h * max(tp_ratio, 0.1)
        fp_h = pos_h - tp_h
        
        # TP部分（上）
        ax.add_patch(plt.Rectangle((x_step1_detail, pos_y + fp_h), 0.08, tp_h,
                                   facecolor='#F5A623', edgecolor='black', lw=1.5, alpha=0.8))
        if tp_h > 0.04:
            ax.text(x_step1_detail + 0.04, pos_y + fp_h + tp_h/2,
                    f"TP={step1_pos_tp}, FP={step1_pos_fp}",
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # 4. Step 1- 细分 (TN和FN)
    if n_step1_neg > 0:
        tn_ratio = step1_neg_tn / n_step1_neg if n_step1_neg > 0 else 0.5
        tn_h = neg_h * max(tn_ratio, 0.1)
        fn_h = neg_h - tn_h
        
        ax.add_patch(plt.Rectangle((x_step1_detail, neg_y), 0.08, tn_h,
                                   facecolor=colors["TN"], edgecolor='black', lw=1.5, alpha=0.8))
        if tn_h > 0.03:
            ax.text(x_step1_detail + 0.04, neg_y + tn_h/2,
                    f"TN={step1_neg_tn}, FN={step1_neg_fn}",
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # 5. Step 2 结果
    if n_step1_pos > 0:
        step2_pos_ratio = step2_pos / n_step1_pos
        s2_pos_h = pos_h * max(step2_pos_ratio, 0.1)
        s2_neg_h = pos_h - s2_pos_h
        s2_y = pos_y
        
        # Step2+
        ax.add_patch(plt.Rectangle((x_step2, s2_y + s2_neg_h), 0.10, s2_pos_h,
                                   facecolor=colors["step2_pos"], edgecolor='black', lw=2, alpha=0.85))
        ax.text(x_step2 + 0.05, s2_y + s2_neg_h + s2_pos_h/2,
                f"{step2_model}\nThreshold≥{step2_threshold:.2f}\nPositive (n={step2_pos})",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Step2-
        ax.add_patch(plt.Rectangle((x_step2, s2_y), 0.10, s2_neg_h,
                                   facecolor=colors["step2_neg"], edgecolor='black', lw=2, alpha=0.85))
        ax.text(x_step2 + 0.05, s2_y + s2_neg_h/2,
                f"{step2_model}\nThreshold<{step2_threshold:.2f}\nNegative (n={step2_neg})",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 6. 最终结果（右侧）
    # 按 FN, FP, TP, TN 从上到下排列
    final_counts = [
        ("FN", final_fn, colors["FN"]),
        ("FP", final_fp, "#F8E71C"),  # 黄色用于FP
        ("TP", final_tp, colors["TP"]),
        ("TN", final_tn, colors["TN"]),
    ]
    
    # 计算高度
    final_heights = []
    for label, count, _ in final_counts:
        h = total_h * (count / n_total) if n_total > 0 else total_h / 4
        h = max(h, 0.03)
        final_heights.append(h)
    
    # 归一化
    total_final_h = sum(final_heights)
    if total_final_h > 0:
        scale = total_h / total_final_h
        final_heights = [h * scale for h in final_heights]
    
    y_curr = y_base
    for (label, count, color), h in zip(final_counts, final_heights):
        ax.add_patch(plt.Rectangle((x_final, y_curr), 0.10, h,
                                   facecolor=color, edgecolor='black', lw=2, alpha=0.85))
        text_color = 'black' if color == '#F8E71C' else 'white'
        if h > 0.05:
            ax.text(x_final + 0.05, y_curr + h/2,
                    f"{label} (n={count}, {pct(count)})",
                    ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)
        y_curr += h
    
    # ===== 绘制流向（Sankey风格的多边形） =====
    # 从Total到Step1+
    poly1 = Polygon([
        (x_total + 0.08, y_base + total_h),
        (x_total + 0.08, y_base + total_h * (1 - pos_ratio)),
        (x_step1, pos_y),
        (x_step1, pos_y + pos_h),
    ], facecolor=colors["step1_pos"], alpha=0.3, edgecolor='none')
    ax.add_patch(poly1)
    
    # 从Total到Step1-
    poly2 = Polygon([
        (x_total + 0.08, y_base + total_h * (1 - pos_ratio)),
        (x_total + 0.08, y_base),
        (x_step1, neg_y),
        (x_step1, neg_y + neg_h),
    ], facecolor=colors["step1_neg"], alpha=0.3, edgecolor='none')
    ax.add_patch(poly2)
    
    # 从Step1+到Step2
    if n_step1_pos > 0:
        poly3 = Polygon([
            (x_step1 + 0.10, pos_y + pos_h),
            (x_step1 + 0.10, pos_y),
            (x_step1_detail, pos_y),
            (x_step1_detail, pos_y + pos_h),
        ], facecolor=colors["step1_pos"], alpha=0.2, edgecolor='none')
        ax.add_patch(poly3)
        
        # 从Step1 detail到Step2
        poly4 = Polygon([
            (x_step1_detail + 0.08, pos_y + pos_h),
            (x_step1_detail + 0.08, pos_y),
            (x_step2, pos_y),
            (x_step2, pos_y + pos_h),
        ], facecolor=colors["step1_pos"], alpha=0.2, edgecolor='none')
        ax.add_patch(poly4)
    
    # ===== 标签 =====
    ax.text(x_total + 0.04, 0.92, "Total", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x_step1 + 0.05, 0.92, "Step 1", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x_step2 + 0.05, 0.92, "Step 2", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x_final + 0.05, 0.92, "Final", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 设置画布
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    title = f"Sequential Screening: {step1_model} → {step2_model}\n{cohort_name}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors["TN"], edgecolor='black', label=f'TN (n={final_tn})', alpha=0.85),
        mpatches.Patch(facecolor=colors["FN"], edgecolor='black', label=f'FN (n={final_fn})', alpha=0.85),
        mpatches.Patch(facecolor='#F8E71C', edgecolor='black', label=f'FP (n={final_fp})', alpha=0.85),
        mpatches.Patch(facecolor=colors["TP"], edgecolor='black', label=f'TP (n={final_tp})', alpha=0.85),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=10, ncol=4,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Sankey saved: {outpath}")


def plot_sankey_plotly(
    zone_counts: Dict,
    step1_model: str,
    step1_threshold: float,
    step2_model: str,
    step2_threshold: float,
    cohort_name: str,
    outpath: str
):
    """
    使用Plotly绘制交互式桑基图（类似用户提供的图片风格）
    """
    if not HAS_PLOTLY:
        print(f"    Plotly not available, skipping: {outpath}")
        return
    
    # 提取数据
    n_total = zone_counts["N_Total"]
    n_step1_pos = zone_counts["N_Step1_Pos"]
    n_step1_neg = zone_counts["N_Step1_Neg"]
    
    step1_pos_tp = zone_counts["Step1_Pos_TP"]
    step1_pos_fp = zone_counts["Step1_Pos_FP"]
    step1_neg_tn = zone_counts["Step1_Neg_TN"]
    step1_neg_fn = zone_counts["Step1_Neg_FN"]
    
    step2_neg_tn = zone_counts["Step2_Neg_TN"]
    step2_neg_fn = zone_counts["Step2_Neg_FN"]
    step2_pos_fp = zone_counts["Step2_Pos_FP"]
    step2_pos_tp = zone_counts["Step2_Pos_TP"]
    
    n_step2_pos = zone_counts["N_Step2_Pos"]
    n_step2_neg = zone_counts["N_Step2_Neg"]
    
    # 最终统计
    final_tn = step1_neg_tn + step2_neg_tn
    final_fn = step1_neg_fn + step2_neg_fn
    final_fp = step2_pos_fp
    final_tp = step2_pos_tp
    
    def pct(n):
        return f"{100*n/n_total:.0f}%" if n_total > 0 else "0%"
    
    # 节点定义 (0-indexed)
    # 0: Total
    # 1: Step1+ (需要进一步检查)
    # 2: Step1- (排除)
    # 3: Step1+ TP/FP详情
    # 4: Step2+ (最终阳性)
    # 5: Step2- (排除)
    # 6: FN (最终假阴性)
    # 7: FP (最终假阳性)
    # 8: TP (最终真阳性)
    # 9: TN (最终真阴性)
    
    labels = [
        f"{cohort_name} (N={n_total})",  # 0
        f"{step1_model} Threshold≥{step1_threshold:.3f} Positive (n={n_step1_pos})",  # 1
        f"{step1_model} Threshold<{step1_threshold:.3f} Negative (n={n_step1_neg}) TN={step1_neg_tn}, FN={step1_neg_fn}",  # 2
        f"TP={step1_pos_tp}, FP={step1_pos_fp}",  # 3
        f"{step2_model} Threshold≥{step2_threshold:.2f} Positive (n={n_step2_pos})",  # 4
        f"{step2_model} Threshold<{step2_threshold:.2f} Negative (n={n_step2_neg})",  # 5
        f"FN (n={final_fn}, {pct(final_fn)})",  # 6
        f"FP (n={final_fp}, {pct(final_fp)})",  # 7
        f"TP (n={final_tp}, {pct(final_tp)})",  # 8
        f"TN (n={final_tn}, {pct(final_tn)})",  # 9
    ]
    
    # 节点颜色
    node_colors = [
        "#4A90D9",   # 0: Total - 蓝色
        "#F5A623",   # 1: Step1+ - 橙色
        "#4A90D9",   # 2: Step1- - 蓝色
        "#F5A623",   # 3: Detail - 橙色
        "#F5A623",   # 4: Step2+ - 橙色
        "#7ED321",   # 5: Step2- - 绿色
        "#D0021B",   # 6: FN - 红色
        "#F8E71C",   # 7: FP - 黄色
        "#9013FE",   # 8: TP - 紫色
        "#7ED321",   # 9: TN - 绿色
    ]
    
    # 链接定义
    sources = [0, 0, 1, 3, 3, 4, 4, 5, 5, 2, 2]
    targets = [1, 2, 3, 4, 5, 7, 8, 9, 6, 9, 6]
    values = [
        n_step1_pos,    # Total → Step1+
        n_step1_neg,    # Total → Step1-
        n_step1_pos,    # Step1+ → Detail
        n_step2_pos,    # Detail → Step2+
        n_step2_neg,    # Detail → Step2-
        step2_pos_fp,   # Step2+ → FP
        step2_pos_tp,   # Step2+ → TP
        step2_neg_tn,   # Step2- → TN
        step2_neg_fn,   # Step2- → FN
        step1_neg_tn,   # Step1- → TN
        step1_neg_fn,   # Step1- → FN
    ]
    
    link_colors = [
        "rgba(245, 166, 35, 0.4)",   # Total → Step1+
        "rgba(74, 144, 217, 0.4)",   # Total → Step1-
        "rgba(245, 166, 35, 0.3)",   # Step1+ → Detail
        "rgba(245, 166, 35, 0.4)",   # Detail → Step2+
        "rgba(126, 211, 33, 0.4)",   # Detail → Step2-
        "rgba(248, 231, 28, 0.5)",   # Step2+ → FP
        "rgba(144, 19, 254, 0.4)",   # Step2+ → TP
        "rgba(126, 211, 33, 0.4)",   # Step2- → TN
        "rgba(208, 2, 27, 0.4)",     # Step2- → FN
        "rgba(126, 211, 33, 0.4)",   # Step1- → TN
        "rgba(208, 2, 27, 0.4)",     # Step1- → FN
    ]
    
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=1),
            label=labels,
            color=node_colors,
            x=[0.01, 0.25, 0.25, 0.45, 0.65, 0.65, 0.95, 0.95, 0.95, 0.95],
            y=[0.5, 0.3, 0.8, 0.3, 0.2, 0.5, 0.9, 0.65, 0.35, 0.1],
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Sequential Screening: {step1_model} → {step2_model}<br>{cohort_name}",
            font=dict(size=16),
        ),
        font=dict(size=11),
        height=700,
        width=1400,
    )
    
    # 保存为HTML和PNG
    html_path = outpath.replace('.png', '.html')
    fig.write_html(html_path)
    
    try:
        fig.write_image(outpath, scale=2)
        print(f"    Plotly Sankey saved: {outpath}")
    except Exception as e:
        print(f"    Plotly image export failed ({e}), HTML saved: {html_path}")


def plot_ppv_npv_combined(
    strategies: Dict[str, Dict],
    cohort_name: str,
    outpath: str
):
    """
    绘制PPV/NPV vs Prevalence曲线（合并在一张图上）
    
    PPV用实线，NPV用虚线，不同策略用不同颜色
    """
    prevalence = np.linspace(0.01, 0.99, 200)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 颜色配置（与用户图片类似）
    color_cycle = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿
        '#17becf',  # 青色
    ]
    
    for idx, (name, metrics) in enumerate(strategies.items()):
        sens = metrics.get("Sensitivity", 0.5)
        spec = metrics.get("Specificity", 0.5)
        color = color_cycle[idx % len(color_cycle)]
        
        # 计算PPV和NPV（贝叶斯定理）
        ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence) + 1e-12)
        npv = (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence) + 1e-12)
        
        # PPV用实线
        ax.plot(prevalence, ppv, label=f"{name} (PPV)", color=color, linewidth=2, linestyle='-')
        # NPV用虚线
        ax.plot(prevalence, npv, label=f"{name} (NPV)", color=color, linewidth=2, linestyle='--')
    
    ax.set_xlabel("Prevalence of advanced fibrosis", fontsize=12)
    ax.set_ylabel("Predictive value", fontsize=12)
    ax.set_title(f"PPV and NPV vs Prevalence ({cohort_name})", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    PPV/NPV curve saved: {outpath}")


def plot_ppv_npv_two_panel(
    strategies_dict: Dict[str, Dict[str, Dict]],
    outpath: str
):
    """
    绘制两面板的PPV/NPV图（左右分别显示不同队列）
    """
    prevalence = np.linspace(0.01, 0.99, 200)
    
    cohort_names = list(strategies_dict.keys())
    fig, axes = plt.subplots(1, len(cohort_names), figsize=(7*len(cohort_names), 6))
    
    if len(cohort_names) == 1:
        axes = [axes]
    
    color_cycle = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    
    for panel_idx, (cohort_name, strategies) in enumerate(strategies_dict.items()):
        ax = axes[panel_idx]
        
        for idx, (name, metrics) in enumerate(strategies.items()):
            sens = metrics.get("Sensitivity", 0.5)
            spec = metrics.get("Specificity", 0.5)
            color = color_cycle[idx % len(color_cycle)]
            
            ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence) + 1e-12)
            npv = (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence) + 1e-12)
            
            ax.plot(prevalence, ppv, label=f"{name} (PPV)", color=color, linewidth=2, linestyle='-')
            ax.plot(prevalence, npv, label=f"{name} (NPV)", color=color, linewidth=2, linestyle='--')
        
        ax.set_xlabel("Prevalence of advanced fibrosis", fontsize=11)
        ax.set_ylabel("Predictive value", fontsize=11)
        ax.set_title(f"{chr(65+panel_idx)}    PPV and NPV vs Prevalence ({cohort_name})", 
                     fontsize=12, fontweight='bold', loc='left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Two-panel PPV/NPV saved: {outpath}")


# ============================================================
# 数据加载
# ============================================================

def load_predictions(pred_path: str, cfg: Config) -> pd.DataFrame:
    """加载预测数据并转换为宽格式"""
    df = pd.read_csv(pred_path)
    
    # 转为宽格式
    wide = df.pivot_table(
        index=["Split", "PatientID"],
        columns="Model",
        values=["TrueLabel", "PredProb"],
        aggfunc="first"
    ).reset_index()
    
    # 展平列名
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    
    # 重命名
    rename_dict = {f"TrueLabel_{cfg.tag_m1}": "TrueLabel"}
    for tag in cfg.all_model_tags:
        rename_dict[f"PredProb_{tag}"] = tag
    
    wide = wide.rename(columns=rename_dict)
    
    keep_cols = ["Split", "PatientID", "TrueLabel"] + cfg.all_model_tags
    wide = wide[[c for c in keep_cols if c in wide.columns]].dropna()
    wide["TrueLabel"] = wide["TrueLabel"].astype(int)
    
    return wide


def load_thresholds(threshold_path: str, cfg: Config) -> Dict[str, float]:
    """加载阈值"""
    df = pd.read_csv(threshold_path)
    thresholds = {}
    
    # 匹配模型名称
    model_map = {
        "Radiomics_Only（M1）": cfg.tag_m1,
        "Radiomics_plus_A（M2）": cfg.tag_m2,
        "Radiomics_plus_AllClinical（M3）": cfg.tag_m3,
        "ClinicalA_Only（M4）": cfg.tag_m4,
        "BaseClinical_Only（M5）": cfg.tag_m5,
    }
    
    for _, row in df.iterrows():
        model_name = row["Model"]
        if model_name in model_map:
            thresholds[model_map[model_name]] = row["Threshold_Youden"]
    
    return thresholds


# ============================================================
# 主分析流程
# ============================================================

def run_deliverables_generation(cfg: Config):
    """生成所有交付物"""
    
    # 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/tables", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/sankey", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/curves", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/sankey_data", exist_ok=True)
    
    print("=" * 70)
    print("生成最终交付物")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    wide = load_predictions(cfg.pred_path, cfg)
    thresholds = load_thresholds(cfg.threshold_path, cfg)
    
    print(f"    总样本数: {len(wide)}")
    print(f"    数据集分布: {wide['Split'].value_counts().to_dict()}")
    print(f"    阈值: {thresholds}")
    
    # 2. 定义策略
    # 两步策略 (Step1: Clinical → Step2: DL)
    stepwise_strategies = {
        "stepwise1 (M4->M1)": (cfg.tag_m4, cfg.tag_m1),
        "stepwise2 (M5->M1)": (cfg.tag_m5, cfg.tag_m1),
        "stepwise3 (M5->M2)": (cfg.tag_m5, cfg.tag_m2),
        "stepwise4 (M4->M3)": (cfg.tag_m4, cfg.tag_m3),
        "stepwise5 (M5->M3)": (cfg.tag_m5, cfg.tag_m3),
    }
    
    # 队列定义
    cohorts = {
        "InternalTest": cfg.split_internal_test,
        "ProspectiveTest": cfg.split_external_test,
    }
    
    # 存储所有结果
    all_step1_metrics = []
    all_step2_metrics = []
    all_final_metrics = []
    all_sankey_data = []
    all_or_pvalue = []
    
    strategies_for_ppv_npv = {}
    
    # 3. 对每个队列进行分析
    for cohort_name, split_val in cohorts.items():
        print(f"\n[2] 分析队列: {cohort_name}")
        
        cohort_data = wide[wide["Split"] == split_val]
        y_true = cohort_data["TrueLabel"].values
        n_samples = len(y_true)
        prevalence = y_true.mean()
        
        print(f"    样本数: {n_samples}, 患病率: {prevalence:.2%}")
        
        strategies_for_ppv_npv[cohort_name] = {}
        
        # ========== 单模型基线 ==========
        for tag in cfg.all_model_tags:
            if tag not in thresholds:
                continue
            
            y_prob = cohort_data[tag].values
            y_pred = (y_prob >= thresholds[tag]).astype(int)
            metrics = compute_metrics_with_ci(y_true, y_pred)
            
            display_name = cfg.model_display_names.get(tag, tag)
            
            all_final_metrics.append({
                "Cohort": cohort_name,
                "Strategy": f"Single: {display_name}",
                "Type": "Single",
                **metrics
            })
            
            strategies_for_ppv_npv[cohort_name][display_name] = metrics
        
        # ========== 两步策略 ==========
        for strategy_name, (step1_tag, step2_tag) in stepwise_strategies.items():
            if step1_tag not in thresholds or step2_tag not in thresholds:
                continue
            
            result = stepwise_two_stage(
                y_true=y_true,
                p_step1=cohort_data[step1_tag].values,
                thr_step1=thresholds[step1_tag],
                p_step2=cohort_data[step2_tag].values,
                thr_step2=thresholds[step2_tag]
            )
            
            step1_display = cfg.model_display_names.get(step1_tag, step1_tag)
            step2_display = cfg.model_display_names.get(step2_tag, step2_tag)
            
            # Step 1 指标
            all_step1_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Step": "Step1",
                "Model": step1_display,
                "Threshold": thresholds[step1_tag],
                **result["step1_metrics"]
            })
            
            # Step 2 指标
            all_step2_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Step": "Step2",
                "Model": step2_display,
                "Threshold": thresholds[step2_tag],
                "N_Evaluated": result["n_pass_step1"],
                **result["step2_metrics"]
            })
            
            # 最终指标
            all_final_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Type": "Stepwise",
                "Referrals_Avoided": result["referrals_avoided"],
                **result["final_metrics"]
            })
            
            # 桑基图数据
            sankey_data = {
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Step1_Model": step1_display,
                "Step1_Threshold": thresholds[step1_tag],
                "Step2_Model": step2_display,
                "Step2_Threshold": thresholds[step2_tag],
                **result["zone_counts"]
            }
            all_sankey_data.append(sankey_data)
            
            # 绘制桑基图
            safe_name = strategy_name.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
            
            # Matplotlib版本
            plot_sankey_chen_style(
                zone_counts=result["zone_counts"],
                step1_model=step1_display,
                step1_threshold=thresholds[step1_tag],
                step2_model=step2_display,
                step2_threshold=thresholds[step2_tag],
                cohort_name=f"{cohort_name} (N={n_samples})",
                outpath=f"{cfg.output_dir}/sankey/sankey_{cohort_name}_{safe_name}_mpl.png"
            )
            
            # Plotly版本（交互式）
            plot_sankey_plotly(
                zone_counts=result["zone_counts"],
                step1_model=step1_display,
                step1_threshold=thresholds[step1_tag],
                step2_model=step2_display,
                step2_threshold=thresholds[step2_tag],
                cohort_name=f"{cohort_name} (N={n_samples})",
                outpath=f"{cfg.output_dir}/sankey/sankey_{cohort_name}_{safe_name}.png"
            )
            
            # PPV/NPV用
            strategies_for_ppv_npv[cohort_name][strategy_name] = result["final_metrics"]
            
            # ========== 计算OR和P值（与单模型比较） ==========
            # 比较两步策略与单独使用Step2模型
            y_pred_stepwise = result["final_pred"]
            y_pred_single = (cohort_data[step2_tag].values >= thresholds[step2_tag]).astype(int)
            
            # 检测高级纤维化的能力比较
            # 暴露 = stepwise预测为阳性
            # 结局 = 真实阳性 (有高级纤维化)
            
            # For OR: 比较两种方法的敏感性差异
            # 使用配对数据构建2x2表
            # 但这里简单使用各自的TP/FN/FP/TN
            
            stepwise_tp = result["final_metrics"]["TP"]
            stepwise_fn = result["final_metrics"]["FN"]
            stepwise_fp = result["final_metrics"]["FP"]
            stepwise_tn = result["final_metrics"]["TN"]
            
            single_metrics = compute_metrics_with_ci(y_true, y_pred_single)
            single_tp = single_metrics["TP"]
            single_fn = single_metrics["FN"]
            
            # OR计算（基于正确检测阳性病例的能力）
            or_val, or_ci_l, or_ci_u, p_val = compute_odds_ratio(
                stepwise_tp, stepwise_fn,  # 两步策略
                single_tp, single_fn       # 单模型
            )
            
            all_or_pvalue.append({
                "Cohort": cohort_name,
                "Algorithm": strategy_name,
                "Comparator": f"Single {step2_display}",
                "Referrals_Avoided_%": f"{result['referrals_avoided']*100:.0f}",
                "Detection_OR": or_val,
                "Detection_OR_CI_Lower": or_ci_l,
                "Detection_OR_CI_Upper": or_ci_u,
                "Detection_OR_Formatted": format_or_ci(or_val, or_ci_l, or_ci_u),
                "P_Value": p_val,
                "P_Value_Formatted": format_pvalue(p_val),
            })
        
        # ========== 绘制PPV/NPV曲线（单队列） ==========
        plot_ppv_npv_combined(
            strategies_for_ppv_npv[cohort_name],
            cohort_name,
            f"{cfg.output_dir}/curves/ppv_npv_{cohort_name}.png"
        )
    
    # 4. 绘制两面板PPV/NPV图
    print("\n[3] 绘制两面板PPV/NPV图...")
    plot_ppv_npv_two_panel(
        strategies_for_ppv_npv,
        f"{cfg.output_dir}/curves/ppv_npv_two_panel.png"
    )
    
    # 5. 保存表格
    print("\n[4] 保存结果表格...")
    
    # Step 1 指标表
    step1_df = pd.DataFrame(all_step1_metrics)
    step1_df.to_csv(f"{cfg.output_dir}/tables/step1_performance_with_CI.csv", index=False)
    
    # Step 2 指标表
    step2_df = pd.DataFrame(all_step2_metrics)
    step2_df.to_csv(f"{cfg.output_dir}/tables/step2_performance_with_CI.csv", index=False)
    
    # 最终指标表
    final_df = pd.DataFrame(all_final_metrics)
    final_df.to_csv(f"{cfg.output_dir}/tables/final_performance_with_CI.csv", index=False)
    
    # 桑基图数据
    sankey_df = pd.DataFrame(all_sankey_data)
    sankey_df.to_csv(f"{cfg.output_dir}/sankey_data/sankey_flow_data.csv", index=False)
    
    # OR和P值表
    or_df = pd.DataFrame(all_or_pvalue)
    or_df.to_csv(f"{cfg.output_dir}/tables/odds_ratio_pvalue.csv", index=False)
    
    # 6. 生成格式化汇总表（类似论文Table格式）
    print("\n[5] 生成格式化汇总表...")
    
    # 最终性能对比表（带CI）
    summary_rows = []
    for cohort_name in cohorts.keys():
        cohort_final = final_df[final_df["Cohort"] == cohort_name]
        
        for _, row in cohort_final.iterrows():
            summary_rows.append({
                "Cohort": cohort_name,
                "Strategy": row["Strategy"],
                "Type": row.get("Type", ""),
                "N": row["N"],
                "Sensitivity": format_ci(row["Sensitivity"], row["Sensitivity_CI_Lower"], row["Sensitivity_CI_Upper"]),
                "Specificity": format_ci(row["Specificity"], row["Specificity_CI_Lower"], row["Specificity_CI_Upper"]),
                "PPV": format_ci(row["PPV"], row["PPV_CI_Lower"], row["PPV_CI_Upper"]),
                "NPV": format_ci(row["NPV"], row["NPV_CI_Lower"], row["NPV_CI_Upper"]),
                "Accuracy": format_ci(row["Accuracy"], row["Accuracy_CI_Lower"], row["Accuracy_CI_Upper"]),
                "TP": row["TP"],
                "TN": row["TN"],
                "FP": row["FP"],
                "FN": row["FN"],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{cfg.output_dir}/tables/summary_formatted.csv", index=False)
    
    # 7. 打印摘要
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)
    
    print("\n【最终性能对比】")
    display_cols = ["Cohort", "Strategy", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy"]
    print(summary_df[display_cols].to_string(index=False))
    
    print("\n【OR和P值】")
    or_display = or_df[["Cohort", "Algorithm", "Comparator", "Referrals_Avoided_%", 
                        "Detection_OR_Formatted", "P_Value_Formatted"]]
    print(or_display.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"✔ 所有交付物已保存至: {cfg.output_dir}")
    print("  - tables/: 性能指标表格（带95% CI）")
    print("  - sankey/: 桑基图")
    print("  - sankey_data/: 桑基图原始数据")
    print("  - curves/: PPV/NPV曲线图")
    print("=" * 70)
    
    return final_df, sankey_df, or_df


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    config = Config()
    run_deliverables_generation(config)
