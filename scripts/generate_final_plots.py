#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于已有的多步骤分析结果生成桑基图和PPV/NPV曲线

输入文件:
- table_stepwise_step_metrics.csv: 多步骤分析的详细结果
- all_models_多步骤参照V2.csv: 单模型性能参照

输出:
- 桑基图 (Sankey diagrams)
- PPV/NPV vs 患病率曲线
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 配置 (相对于项目根目录)
# ============================================================
import sys
from pathlib import Path

# 获取项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

STEPWISE_METRICS_PATH = PROJECT_ROOT / "data" / "table_stepwise_step_metrics.csv"
SINGLE_MODEL_PATH = PROJECT_ROOT / "data" / "all_models_多步骤参照V2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sankey", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/curves", exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
def load_data():
    """加载多步骤分析结果和单模型性能"""
    stepwise_df = pd.read_csv(STEPWISE_METRICS_PATH)
    single_df = pd.read_csv(SINGLE_MODEL_PATH)
    return stepwise_df, single_df


def compute_stepwise_final_metrics(stepwise_df, cohort, strategy):
    """
    根据Step1和Step2的结果计算多步骤方法的最终性能
    
    逻辑：
    - Step1阴性 → 最终阴性 (Rule-out)
    - Step1阳性 → 进入Step2
        - Step2阴性 → 最终阴性
        - Step2阳性 → 最终阳性 (Rule-in)
    """
    # 获取该cohort和strategy的Step1和Step2数据
    step1 = stepwise_df[(stepwise_df['Cohort'] == cohort) & 
                        (stepwise_df['Strategy'] == strategy) & 
                        (stepwise_df['Step'] == 'Step1')].iloc[0]
    step2 = stepwise_df[(stepwise_df['Cohort'] == cohort) & 
                        (stepwise_df['Strategy'] == strategy) & 
                        (stepwise_df['Step'] == 'Step2')].iloc[0]
    
    # Step1的混淆矩阵
    step1_tp = int(step1['TP'])
    step1_tn = int(step1['TN'])
    step1_fp = int(step1['FP'])
    step1_fn = int(step1['FN'])
    
    # Step2的混淆矩阵 (只针对Step1阳性的人群)
    step2_tp = int(step2['TP'])
    step2_tn = int(step2['TN'])
    step2_fp = int(step2['FP'])
    step2_fn = int(step2['FN'])
    
    # 计算总人数
    n_total = step1_tp + step1_tn + step1_fp + step1_fn
    
    # 计算各个流向
    n_step1_neg = step1_tn + step1_fn  # Step1阴性 (Rule-out)
    n_step1_pos = step1_tp + step1_fp  # Step1阳性 (进入Step2)
    
    # 最终混淆矩阵
    final_tp = step2_tp  # Step2真阳性
    final_fp = step2_fp  # Step2假阳性
    final_tn = step1_tn + step2_tn  # Step1真阴 + Step2真阴
    final_fn = step1_fn + step2_fn  # Step1假阴 + Step2假阴
    
    # 最终性能指标
    sens = final_tp / (final_tp + final_fn) if (final_tp + final_fn) > 0 else 0
    spec = final_tn / (final_tn + final_fp) if (final_tn + final_fp) > 0 else 0
    ppv = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0
    npv = final_tn / (final_tn + final_fn) if (final_tn + final_fn) > 0 else 0
    acc = (final_tp + final_tn) / n_total
    
    # 流向统计 (用于桑基图)
    flow = {
        'N_Total': n_total,
        'N_Step1_Neg': n_step1_neg,  # Rule-out
        'N_Step1_Pos': n_step1_pos,  # 进入Step2
        'N_Step2_Neg': step2_tn + step2_fn,  # Step2阴性
        'N_Step2_Pos': step2_tp + step2_fp,  # Step2阳性 (最终阳性)
        
        # 详细分类
        'RuleOut_TN': step1_tn,
        'RuleOut_FN': step1_fn,
        'Gray_TN': step2_tn,
        'Gray_FN': step2_fn,
        'RuleIn_FP': step2_fp,
        'RuleIn_TP': step2_tp,
        
        # 最终混淆矩阵
        'Final_TP': final_tp,
        'Final_TN': final_tn,
        'Final_FP': final_fp,
        'Final_FN': final_fn,
    }
    
    metrics = {
        'Sensitivity': sens,
        'Specificity': spec,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': acc,
        'Prevalence': (final_tp + final_fn) / n_total,
        'Referral_Rate': (final_tp + final_fp) / n_total,
    }
    
    return metrics, flow


# ============================================================
# 桑基图绘制
# ============================================================
def plot_sankey_chen_style(flow, title, outpath, step1_name="Step 1", step2_name="Step 2"):
    """
    绘制类似 Chen 2024 论文风格的桑基图
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 提取数据
    n_total = flow['N_Total']
    n_step1_neg = flow['N_Step1_Neg']
    n_step1_pos = flow['N_Step1_Pos']
    
    # 颜色方案 (类似文献)
    colors = {
        'total': '#3498db',       # 蓝色 - 总体
        'step1_pos': '#e67e22',   # 橙色 - Step1阳性
        'step1_neg': '#3498db',   # 蓝色 - Step1阴性
        'TN': '#3498db',          # 蓝色 - TN
        'FN': '#2c3e50',          # 深蓝 - FN
        'FP': '#e74c3c',          # 红色 - FP
        'TP': '#c0392b',          # 深红 - TP
    }
    
    # 计算百分比
    def pct(n): 
        return f"({100*n/n_total:.0f}%)" if n_total > 0 else "(0%)"
    
    # ========== 左侧: 总体 ==========
    total_h = 0.7
    total_box = plt.Rectangle((0.02, 0.15), 0.10, total_h, 
                               facecolor=colors['total'], edgecolor='black', lw=2, alpha=0.9)
    ax.add_patch(total_box)
    ax.text(0.07, 0.5, f"Total\nn={n_total}", ha='center', va='center', 
            fontsize=13, fontweight='bold', color='white')
    
    # ========== 中间: Step1分流 ==========
    # Step1 阳性 (上方, 橙色)
    pos_ratio = n_step1_pos / n_total if n_total > 0 else 0.5
    pos_h = total_h * pos_ratio
    pos_y = 0.15 + total_h - pos_h
    
    pos_box = plt.Rectangle((0.25, pos_y), 0.12, pos_h,
                            facecolor=colors['step1_pos'], edgecolor='black', lw=2, alpha=0.9)
    ax.add_patch(pos_box)
    ax.text(0.31, pos_y + pos_h/2, 
            f"{step1_name}+\n→{step2_name}\nn={n_step1_pos}\n{pct(n_step1_pos)}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Step1 阴性 (下方, 蓝色) - Rule-out
    neg_h = total_h * (1 - pos_ratio)
    neg_y = 0.15
    
    neg_box = plt.Rectangle((0.25, neg_y), 0.12, neg_h,
                            facecolor=colors['step1_neg'], edgecolor='black', lw=2, alpha=0.9)
    ax.add_patch(neg_box)
    ax.text(0.31, neg_y + neg_h/2, 
            f"{step1_name}-\nRule-out\nn={n_step1_neg}\n{pct(n_step1_neg)}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # ========== Step2框 ==========
    step2_h = pos_h * 0.9
    step2_y = pos_y + (pos_h - step2_h) / 2
    
    step2_box = plt.Rectangle((0.48, step2_y), 0.12, step2_h,
                              facecolor='#f39c12', edgecolor='black', lw=2, alpha=0.9)
    ax.add_patch(step2_box)
    
    n_step2_pos = flow['N_Step2_Pos']
    n_step2_neg = flow['N_Step2_Neg']
    ax.text(0.54, step2_y + step2_h/2, 
            f"{step2_name}\n+:{n_step2_pos}\n-:{n_step2_neg}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # ========== 右侧: 最终结果 ==========
    # 计算各类人数
    outcomes = [
        ('TN', flow['RuleOut_TN'] + flow['Gray_TN'], colors['TN']),
        ('FN', flow['RuleOut_FN'] + flow['Gray_FN'], colors['FN']),
        ('FP', flow['RuleIn_FP'], colors['FP']),
        ('TP', flow['RuleIn_TP'], colors['TP']),
    ]
    
    # 计算高度
    outcome_heights = []
    for label, count, _ in outcomes:
        h = total_h * (count / n_total) if n_total > 0 else total_h / 4
        h = max(h, 0.06)  # 最小高度
        outcome_heights.append(h)
    
    # 归一化
    total_oh = sum(outcome_heights)
    scale = total_h / total_oh if total_oh > 0 else 1
    outcome_heights = [h * scale for h in outcome_heights]
    
    # 绘制结果框
    y_current = 0.15
    for (label, count, color), h in zip(outcomes, outcome_heights):
        box = plt.Rectangle((0.78, y_current), 0.12, h,
                            facecolor=color, edgecolor='black', lw=2, alpha=0.9)
        ax.add_patch(box)
        
        text = f"{label}\n{count}\n{pct(count)}" if h > 0.1 else f"{label}:{count}"
        fontsize = 10 if h > 0.1 else 8
        ax.text(0.84, y_current + h/2, text,
                ha='center', va='center', fontsize=fontsize, 
                fontweight='bold', color='white')
        y_current += h
    
    # ========== 绘制流向带 ==========
    # Total → Step1+
    flow_poly1 = Polygon([
        (0.12, 0.15 + total_h),
        (0.12, 0.15 + total_h * (1 - pos_ratio)),
        (0.25, pos_y),
        (0.25, pos_y + pos_h),
    ], facecolor=colors['step1_pos'], alpha=0.3, edgecolor='none')
    ax.add_patch(flow_poly1)
    
    # Total → Step1-
    flow_poly2 = Polygon([
        (0.12, 0.15 + total_h * (1 - pos_ratio)),
        (0.12, 0.15),
        (0.25, neg_y),
        (0.25, neg_y + neg_h),
    ], facecolor=colors['step1_neg'], alpha=0.3, edgecolor='none')
    ax.add_patch(flow_poly2)
    
    # Step1+ → Step2
    flow_poly3 = Polygon([
        (0.37, pos_y + pos_h),
        (0.37, pos_y),
        (0.48, step2_y),
        (0.48, step2_y + step2_h),
    ], facecolor=colors['step1_pos'], alpha=0.25, edgecolor='none')
    ax.add_patch(flow_poly3)
    
    # ========== 标签 ==========
    ax.text(0.07, 0.92, "All\nPatients", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.31, 0.92, step1_name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.54, 0.92, step2_name, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.84, 0.92, "Outcome", ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=colors['TN'], edgecolor='black', 
                      label=f'True Negative (n={flow["RuleOut_TN"] + flow["Gray_TN"]})', alpha=0.9),
        mpatches.Patch(facecolor=colors['FN'], edgecolor='black', 
                      label=f'False Negative (n={flow["RuleOut_FN"] + flow["Gray_FN"]})', alpha=0.9),
        mpatches.Patch(facecolor=colors['FP'], edgecolor='black', 
                      label=f'False Positive (n={flow["RuleIn_FP"]})', alpha=0.9),
        mpatches.Patch(facecolor=colors['TP'], edgecolor='black', 
                      label=f'True Positive (n={flow["RuleIn_TP"]})', alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=2,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✔ 桑基图已保存: {outpath}")


# ============================================================
# PPV/NPV vs 患病率曲线
# ============================================================
def ppv_bayes(prevalence, sens, spec):
    """根据贝叶斯定理计算PPV"""
    return (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence) + 1e-12)

def npv_bayes(prevalence, sens, spec):
    """根据贝叶斯定理计算NPV"""
    return (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence) + 1e-12)


def plot_ppv_npv_curves(strategies_metrics, title, outpath):
    """
    绘制PPV和NPV随患病率变化的曲线
    
    类似 Chen 2024 论文 Figure 6 的风格
    """
    prevalence = np.linspace(0.01, 0.99, 200)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies_metrics)))
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for idx, (name, metrics) in enumerate(strategies_metrics.items()):
        sens = metrics['Sensitivity']
        spec = metrics['Specificity']
        color = colors[idx]
        ls = linestyles[idx % len(linestyles)]
        
        # PPV曲线
        ppv_values = ppv_bayes(prevalence, sens, spec)
        axes[0].plot(prevalence, ppv_values, label=name, color=color, 
                    linewidth=2, linestyle=ls)
        
        # NPV曲线
        npv_values = npv_bayes(prevalence, sens, spec)
        axes[1].plot(prevalence, npv_values, label=name, color=color, 
                    linewidth=2, linestyle=ls)
    
    # PPV子图设置
    axes[0].set_xlabel("Prevalence of Advanced Fibrosis", fontsize=12)
    axes[0].set_ylabel("Positive Predictive Value (PPV)", fontsize=12)
    axes[0].set_title("PPV vs Prevalence", fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(np.arange(0, 1.1, 0.1))
    
    # NPV子图设置
    axes[1].set_xlabel("Prevalence of Advanced Fibrosis", fontsize=12)
    axes[1].set_ylabel("Negative Predictive Value (NPV)", fontsize=12)
    axes[1].set_title("NPV vs Prevalence", fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(np.arange(0, 1.1, 0.1))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✔ PPV/NPV曲线已保存: {outpath}")


def plot_ppv_npv_combined(strategies_metrics, title, outpath):
    """
    绘制PPV和NPV在同一图中 (实线PPV, 虚线NPV)
    
    类似 Chen 2024 论文风格
    """
    prevalence = np.linspace(0.01, 0.99, 200)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 为每个策略选择颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies_metrics)))
    
    for idx, (name, metrics) in enumerate(strategies_metrics.items()):
        sens = metrics['Sensitivity']
        spec = metrics['Specificity']
        color = colors[idx]
        
        # PPV曲线 (实线)
        ppv_values = ppv_bayes(prevalence, sens, spec)
        ax.plot(prevalence, ppv_values, label=f"{name} (PPV)", 
               color=color, linewidth=2, linestyle='-')
        
        # NPV曲线 (虚线)
        npv_values = npv_bayes(prevalence, sens, spec)
        ax.plot(prevalence, npv_values, label=f"{name} (NPV)", 
               color=color, linewidth=2, linestyle='--')
    
    ax.set_xlabel("Prevalence of Advanced Fibrosis", fontsize=12)
    ax.set_ylabel("Predictive Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✔ PPV/NPV组合图已保存: {outpath}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("生成最终结果图表")
    print("=" * 60)
    
    # 加载数据
    stepwise_df, single_df = load_data()
    
    print(f"\n加载数据:")
    print(f"  - 多步骤结果: {len(stepwise_df)} 行")
    print(f"  - 单模型参照: {len(single_df)} 行")
    
    # 定义要分析的队列和策略
    cohorts = {
        'InternalTest': 'Test',
        'ProspectiveTest': 'External',
    }
    
    strategies = [
        'stepwise1 (M4→M3)',
        'stepwise2 (M5→M3)',
    ]
    
    strategy_display = {
        'stepwise1 (M4→M3)': 'Two-step (M4→M3)',
        'stepwise2 (M5→M3)': 'Two-step (M5→M3)',
    }
    
    # 为每个队列生成图表
    for cohort, split in cohorts.items():
        print(f"\n{'='*40}")
        print(f"处理队列: {cohort}")
        print(f"{'='*40}")
        
        # 收集所有策略的性能指标
        all_metrics = {}
        
        # 多步骤策略
        for strategy in strategies:
            metrics, flow = compute_stepwise_final_metrics(stepwise_df, cohort, strategy)
            display_name = strategy_display.get(strategy, strategy)
            all_metrics[display_name] = metrics
            
            print(f"\n{display_name}:")
            print(f"  Sens={metrics['Sensitivity']:.3f}, Spec={metrics['Specificity']:.3f}")
            print(f"  PPV={metrics['PPV']:.3f}, NPV={metrics['NPV']:.3f}")
            print(f"  Accuracy={metrics['Accuracy']:.3f}")
            print(f"  转诊率: {metrics['Referral_Rate']:.1%}")
            
            # 绘制桑基图
            plot_sankey_chen_style(
                flow,
                f"{display_name}\n{cohort}",
                f"{OUTPUT_DIR}/sankey/sankey_{cohort}_{strategy.replace(' ', '_').replace('→', 'to')}.png",
                step1_name="M4" if "M4" in strategy else "M5",
                step2_name="M3"
            )
        
        # 单模型性能 (从参照表获取)
        single_cohort = single_df[single_df['Split'] == split]
        for _, row in single_cohort.iterrows():
            model = row['Model']
            all_metrics[f'Single ({model})'] = {
                'Sensitivity': row['Sensitivity'],
                'Specificity': row['Specificity'],
                'PPV': row['PPV'],
                'NPV': row['NPV'],
                'Accuracy': row['Accuracy_mean'],
            }
        
        # 绘制PPV/NPV曲线 (分开)
        plot_ppv_npv_curves(
            all_metrics,
            f"PPV and NPV vs Prevalence\n{cohort}",
            f"{OUTPUT_DIR}/curves/ppv_npv_separate_{cohort}.png"
        )
        
        # 绘制PPV/NPV曲线 (组合)
        plot_ppv_npv_combined(
            all_metrics,
            f"PPV and NPV vs Prevalence - {cohort}",
            f"{OUTPUT_DIR}/curves/ppv_npv_combined_{cohort}.png"
        )
    
    # ========== 汇总表格 ==========
    print(f"\n{'='*60}")
    print("生成汇总表格")
    print(f"{'='*60}")
    
    summary_rows = []
    for cohort, split in cohorts.items():
        for strategy in strategies:
            metrics, flow = compute_stepwise_final_metrics(stepwise_df, cohort, strategy)
            display_name = strategy_display.get(strategy, strategy)
            
            summary_rows.append({
                'Cohort': cohort,
                'Strategy': display_name,
                'N': flow['N_Total'],
                'Sensitivity': round(metrics['Sensitivity'], 3),
                'Specificity': round(metrics['Specificity'], 3),
                'PPV': round(metrics['PPV'], 3),
                'NPV': round(metrics['NPV'], 3),
                'Accuracy': round(metrics['Accuracy'], 3),
                'Step2_Entry': flow['N_Step1_Pos'],
                'Referral_Rate': round(metrics['Referral_Rate'], 3),
                'TP': flow['Final_TP'],
                'TN': flow['Final_TN'],
                'FP': flow['Final_FP'],
                'FN': flow['Final_FN'],
            })
        
        # 添加单模型
        single_cohort = single_df[single_df['Split'] == split]
        for _, row in single_cohort.iterrows():
            summary_rows.append({
                'Cohort': cohort,
                'Strategy': f"Single ({row['Model']})",
                'Sensitivity': round(row['Sensitivity'], 3),
                'Specificity': round(row['Specificity'], 3),
                'PPV': round(row['PPV'], 3),
                'NPV': round(row['NPV'], 3),
                'Accuracy': round(row['Accuracy_mean'], 3),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/summary_final.csv", index=False)
    
    print(f"\n汇总表:")
    print(summary_df.to_string(index=False))
    print(f"\n✔ 汇总表已保存: {OUTPUT_DIR}/summary_final.csv")
    
    print(f"\n{'='*60}")
    print(f"✔ 所有图表已生成完毕!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
