#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential Screening Algorithm for Advanced Liver Fibrosis
Based on: Chen et al. 2024 - US-based Sequential Algorithm Integrating an AI Model

This script implements:
1. Two-step screening algorithms (Clinical → Deep Learning model)
2. Sankey diagrams showing patient flow (TN/FN/FP/TP)
3. PPV/NPV vs prevalence curves (Bayes theorem)
4. Comprehensive diagnostic performance metrics
5. Threshold selection using Youden's Index on validation set

Author: Auto-generated
Date: 2026-01-21
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve
)

# Try to import plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Sankey diagrams will use matplotlib fallback.")


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class Config:
    """Configuration for the stepwise screening analysis."""
    
    # Input/Output paths (相对于项目根目录)
    pred_path: str = "data/all_models_sample_predictions.csv"
    output_dir: str = "outputs/intermediate"
    
    # Split names in CSV
    split_val: str = "Val"
    split_internal_test: str = "Test"
    split_external_test: str = "External"
    
    # Model tags (must match 'Model' column in CSV)
    tag_m1: str = "Radiomics_Only"              # Echo-Net / DL model
    tag_m2: str = "Radiomics_plus_A"            # DL + Clinical A
    tag_m3: str = "Radiomics_plus_AllClinical"  # DL + All Clinical
    tag_m4: str = "ClinicalA_Only"              # Clinical A parameter (like FIB-4)
    tag_m5: str = "BaseClinical_Only"           # Base clinical features
    
    # Display names for plotting
    model_display_names: Dict[str, str] = None
    
    def __post_init__(self):
        if self.model_display_names is None:
            self.model_display_names = {
                self.tag_m1: "Echo-Net",
                self.tag_m2: "Echo-Net+A",
                self.tag_m3: "Echo-Net+All",
                self.tag_m4: "Clinical-A",
                self.tag_m5: "Clinical-Base",
            }
    
    @property
    def all_model_tags(self) -> List[str]:
        return [self.tag_m1, self.tag_m2, self.tag_m3, self.tag_m4, self.tag_m5]


# Global config instance
CFG = Config()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division to avoid divide by zero."""
    return a / b if b > 1e-12 else default


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute confusion matrix elements.
    
    Returns:
        (TP, TN, FP, FN)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(tn), int(fp), int(fn)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_prob: Optional[np.ndarray] = None) -> Dict:
    """Compute comprehensive diagnostic metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
    
    Returns:
        Dictionary with all metrics
    """
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    n = len(y_true)
    
    metrics = {
        "N": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": safe_div(tp, tp + fn),
        "Specificity": safe_div(tn, tn + fp),
        "PPV": safe_div(tp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "Accuracy": safe_div(tp + tn, n),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "Prevalence": safe_div(tp + fn, n),
        "Positive_Rate": safe_div(tp + fp, n),  # Referral rate
    }
    
    # AUC metrics (require probability scores)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["AUROC"] = roc_auc_score(y_true, y_prob)
            metrics["AUPRC"] = average_precision_score(y_true, y_prob)
        except:
            metrics["AUROC"] = np.nan
            metrics["AUPRC"] = np.nan
    else:
        metrics["AUROC"] = np.nan
        metrics["AUPRC"] = np.nan
    
    return metrics


def find_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find optimal threshold using Youden's J statistic.
    
    J = Sensitivity + Specificity - 1
    
    Returns:
        (optimal_threshold, max_youden_j)
    """
    best_thr, best_j = 0.5, -np.inf
    
    # Search over probability grid
    for thr in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob >= thr).astype(int)
        tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        j = sens + spec - 1
        
        if j > best_j:
            best_j, best_thr = j, thr
    
    return float(best_thr), float(best_j)


# ============================================================
# STEPWISE SCREENING ALGORITHMS
# ============================================================

@dataclass
class StepwiseResult:
    """Results from a two-step screening algorithm."""
    
    # Step 1 results
    step1_pred: np.ndarray
    step1_metrics: Dict
    n_pass_step1: int  # Number passing to step 2
    
    # Step 2 results (on subset)
    step2_pred: np.ndarray  # Only for those who passed step 1
    step2_metrics: Dict
    
    # Final combined results
    final_pred: np.ndarray
    final_metrics: Dict
    
    # Flow statistics
    zone_counts: Dict  # TN, FN, FP, TP counts for Sankey


def stepwise_two_stage(
    y_true: np.ndarray,
    p_step1: np.ndarray,
    thr_step1: float,
    p_step2: np.ndarray,
    thr_step2: float,
    rule_in_mode: bool = True
) -> StepwiseResult:
    """
    Two-step sequential screening algorithm.
    
    Logic (rule-in mode, like FIB-4 → Echo-Net):
        Step 1: screen with clinical parameter
            - p1 < thr1 → rule OUT (predict 0)
            - p1 >= thr1 → proceed to step 2
        Step 2: for those passing step 1
            - p2 >= thr2 → rule IN (predict 1)
            - p2 < thr2 → rule OUT (predict 0)
    
    Args:
        y_true: Ground truth labels
        p_step1: Probabilities from step 1 model
        thr_step1: Threshold for step 1
        p_step2: Probabilities from step 2 model
        thr_step2: Threshold for step 2
        rule_in_mode: If True, step1 >= thr is "needs further testing"
    
    Returns:
        StepwiseResult with all metrics and predictions
    """
    n = len(y_true)
    
    # Step 1: Initial screening
    step1_pred = (p_step1 >= thr_step1).astype(int)
    step1_metrics = compute_metrics(y_true, step1_pred, p_step1)
    
    # Who passes to step 2
    pass_mask = step1_pred == 1
    n_pass = int(pass_mask.sum())
    
    # Step 2: Secondary screening (on subset)
    step2_pred_subset = np.zeros(n_pass, dtype=int) if n_pass > 0 else np.array([])
    
    if n_pass > 0:
        step2_pred_subset = (p_step2[pass_mask] >= thr_step2).astype(int)
        step2_metrics = compute_metrics(
            y_true[pass_mask], 
            step2_pred_subset, 
            p_step2[pass_mask]
        )
    else:
        step2_metrics = {k: np.nan for k in step1_metrics}
    
    # Final prediction: 
    # - Ruled out in step 1 → predict 0
    # - Passed step 1, then ruled out in step 2 → predict 0
    # - Passed both steps → predict 1
    final_pred = np.zeros(n, dtype=int)
    if n_pass > 0:
        final_pred[pass_mask] = step2_pred_subset
    
    final_metrics = compute_metrics(y_true, final_pred, p_step2)
    
    # Zone counts for Sankey diagram
    # Rule-out zone: step1_pred == 0
    rule_out_mask = step1_pred == 0
    # Gray zone: passed step 1, failed step 2
    gray_mask = pass_mask & (final_pred == 0)
    # Rule-in zone: passed both
    rule_in_mask = pass_mask & (final_pred == 1)
    
    zone_counts = {
        # Rule-out zone
        "RuleOut_TN": int(((y_true == 0) & rule_out_mask).sum()),
        "RuleOut_FN": int(((y_true == 1) & rule_out_mask).sum()),
        # Gray zone (failed step 2)
        "Gray_TN": int(((y_true == 0) & gray_mask).sum()),
        "Gray_FN": int(((y_true == 1) & gray_mask).sum()),
        # Rule-in zone
        "RuleIn_FP": int(((y_true == 0) & rule_in_mask).sum()),
        "RuleIn_TP": int(((y_true == 1) & rule_in_mask).sum()),
        # Total
        "N_RuleOut": int(rule_out_mask.sum()),
        "N_Gray": int(gray_mask.sum()),
        "N_RuleIn": int(rule_in_mask.sum()),
        "N_Step2": n_pass,
    }
    
    return StepwiseResult(
        step1_pred=step1_pred,
        step1_metrics=step1_metrics,
        n_pass_step1=n_pass,
        step2_pred=step2_pred_subset,
        step2_metrics=step2_metrics,
        final_pred=final_pred,
        final_metrics=final_metrics,
        zone_counts=zone_counts
    )


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_sankey_matplotlib(
    zone_counts: Dict,
    title: str,
    outpath: str
):
    """
    Create a Sankey-style flow diagram using matplotlib.
    Shows patient flow through two-step screening - similar to Chen 2024 paper Figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract counts
    n_total = zone_counts["N_RuleOut"] + zone_counts["N_Gray"] + zone_counts["N_RuleIn"]
    n_step1_neg = zone_counts["N_RuleOut"]
    n_step1_pos = zone_counts["N_Step2"]
    
    # Colors matching paper style
    colors = {
        "total": "#3498db",       # Blue for total
        "step1_pos": "#e67e22",   # Orange for positive (needs further testing)
        "step1_neg": "#3498db",   # Blue for negative (ruled out)
        "TN": "#3498db",          # Blue - True Negative
        "FN": "#2980b9",          # Darker Blue - False Negative  
        "FP": "#e74c3c",          # Red - False Positive
        "TP": "#c0392b",          # Darker Red - True Positive
        "gray_zone": "#f39c12",   # Yellow/Orange for gray zone
    }
    
    # Calculate percentages
    def pct(n): return f"({100*n/n_total:.0f}%)" if n_total > 0 else "(0%)"
    
    # ========== LEFT SIDE: Total Patients ==========
    # Draw as a tall rectangle
    total_h = 0.7
    total_box = plt.Rectangle((0.02, 0.15), 0.12, total_h, 
                               facecolor=colors["total"], edgecolor='black', lw=2, alpha=0.8)
    ax.add_patch(total_box)
    ax.text(0.08, 0.5, f"Total\nn={n_total}", ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # ========== MIDDLE: Step 1 Split ==========
    # Step 1 splits patients into two groups
    
    # Step 1 Positive (upper) - Orange, goes to Step 2
    pos_ratio = n_step1_pos / n_total if n_total > 0 else 0.5
    pos_h = total_h * pos_ratio
    pos_y = 0.15 + total_h - pos_h
    
    pos_box = plt.Rectangle((0.28, pos_y), 0.14, pos_h,
                            facecolor=colors["step1_pos"], edgecolor='black', lw=2, alpha=0.8)
    ax.add_patch(pos_box)
    ax.text(0.35, pos_y + pos_h/2, f"Step1 +\n→ Step2\nn={n_step1_pos}\n{pct(n_step1_pos)}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Step 1 Negative (lower) - Blue, ruled out
    neg_h = total_h * (1 - pos_ratio)
    neg_y = 0.15
    
    neg_box = plt.Rectangle((0.28, neg_y), 0.14, neg_h,
                            facecolor=colors["step1_neg"], edgecolor='black', lw=2, alpha=0.8)
    ax.add_patch(neg_box)
    ax.text(0.35, neg_y + neg_h/2, f"Step1 -\nRule-out\nn={n_step1_neg}\n{pct(n_step1_neg)}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # ========== RIGHT SIDE: Final Outcomes ==========
    # TN from Step1- 
    tn_step1 = zone_counts["RuleOut_TN"]
    fn_step1 = zone_counts["RuleOut_FN"]
    
    # TN from Gray zone (Step2-)
    tn_step2 = zone_counts["Gray_TN"]
    fn_step2 = zone_counts["Gray_FN"]
    
    # FP/TP from Step2+
    fp = zone_counts["RuleIn_FP"]
    tp = zone_counts["RuleIn_TP"]
    
    # Total outcomes
    total_tn = tn_step1 + tn_step2
    total_fn = fn_step1 + fn_step2
    
    # Draw outcome boxes on the right
    outcomes = [
        ("TN", total_tn, colors["TN"], f"TN\nn={total_tn}\n{pct(total_tn)}"),
        ("FN", total_fn, colors["FN"], f"FN\nn={total_fn}\n{pct(total_fn)}"),
        ("FP", fp, colors["FP"], f"FP\nn={fp}\n{pct(fp)}"),
        ("TP", tp, colors["TP"], f"TP\nn={tp}\n{pct(tp)}"),
    ]
    
    # Calculate heights proportionally
    outcome_heights = []
    for label, count, _, _ in outcomes:
        h = total_h * (count / n_total) if n_total > 0 else total_h / 4
        h = max(h, 0.05)  # Minimum height for visibility
        outcome_heights.append(h)
    
    # Normalize heights to fit
    total_outcome_h = sum(outcome_heights)
    scale_factor = total_h / total_outcome_h if total_outcome_h > 0 else 1
    outcome_heights = [h * scale_factor for h in outcome_heights]
    
    y_current = 0.15
    for i, ((label, count, color, text), h) in enumerate(zip(outcomes, outcome_heights)):
        box = plt.Rectangle((0.82, y_current), 0.12, h,
                            facecolor=color, edgecolor='black', lw=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(0.88, y_current + h/2, text if h > 0.08 else f"{label}\n{count}",
                ha='center', va='center', fontsize=9 if h > 0.08 else 7, 
                fontweight='bold', color='white')
        y_current += h
    
    # ========== DRAW FLOW BANDS (Sankey-style) ==========
    from matplotlib.patches import FancyBboxPatch, Polygon
    
    # Flow from Total to Step1+
    flow_vertices_pos = [
        (0.14, 0.15 + total_h),  # Top-left of Total
        (0.14, 0.15 + total_h * (1-pos_ratio)),  # Bottom of upper portion
        (0.28, pos_y),  # Bottom-left of Step1+
        (0.28, pos_y + pos_h),  # Top-left of Step1+
    ]
    poly_pos = Polygon(flow_vertices_pos, facecolor=colors["step1_pos"], alpha=0.4, edgecolor='none')
    ax.add_patch(poly_pos)
    
    # Flow from Total to Step1-
    flow_vertices_neg = [
        (0.14, 0.15 + total_h * (1-pos_ratio)),  # Top of lower portion
        (0.14, 0.15),  # Bottom-left of Total
        (0.28, neg_y),  # Bottom-left of Step1-
        (0.28, neg_y + neg_h),  # Top-left of Step1-
    ]
    poly_neg = Polygon(flow_vertices_neg, facecolor=colors["step1_neg"], alpha=0.4, edgecolor='none')
    ax.add_patch(poly_neg)
    
    # ========== STEP 2 intermediate (Gray zone processing) ==========
    # Draw Step 2 box in the middle-right
    step2_h = pos_h * 0.8  # Slightly smaller
    step2_y = pos_y + (pos_h - step2_h) / 2
    
    step2_box = plt.Rectangle((0.52, step2_y), 0.14, step2_h,
                              facecolor=colors["gray_zone"], edgecolor='black', lw=2, alpha=0.8)
    ax.add_patch(step2_box)
    
    step2_pos = tp + fp  # Those who pass Step 2
    step2_neg = tn_step2 + fn_step2  # Those who fail Step 2
    ax.text(0.59, step2_y + step2_h/2, f"Step2\n+: {step2_pos}\n-: {step2_neg}",
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Flow from Step1+ to Step2
    flow_to_step2 = [
        (0.42, pos_y + pos_h),
        (0.42, pos_y),
        (0.52, step2_y),
        (0.52, step2_y + step2_h),
    ]
    poly_to_step2 = Polygon(flow_to_step2, facecolor=colors["step1_pos"], alpha=0.3, edgecolor='none')
    ax.add_patch(poly_to_step2)
    
    # ========== LABELS ==========
    ax.text(0.08, 0.92, "Step 1", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.35, 0.92, "Screen", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.59, 0.92, "Step 2", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0.88, 0.92, "Outcome", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Threshold labels (if available in title)
    ax.text(0.35, 0.08, "Clinical\nParameter", ha='center', va='center', fontsize=9, style='italic')
    ax.text(0.59, 0.08, "DL Model", ha='center', va='center', fontsize=9, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors["TN"], edgecolor='black', label=f'True Negative (n={total_tn})', alpha=0.8),
        mpatches.Patch(facecolor=colors["FN"], edgecolor='black', label=f'False Negative (n={total_fn})', alpha=0.8),
        mpatches.Patch(facecolor=colors["FP"], edgecolor='black', label=f'False Positive (n={fp})', alpha=0.8),
        mpatches.Patch(facecolor=colors["TP"], edgecolor='black', label=f'True Positive (n={tp})', alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=9, ncol=2,
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sankey_plotly(
    zone_counts: Dict,
    title: str,
    outpath: str
):
    """Create interactive Sankey diagram using Plotly."""
    if not HAS_PLOTLY:
        plot_sankey_matplotlib(zone_counts, title, outpath.replace('.html', '.png'))
        return
    
    # Node labels
    labels = [
        "All Patients",      # 0
        "Step1 Positive",    # 1
        "Step1 Negative",    # 2
        "True Negative",     # 3
        "False Negative",    # 4
        "False Positive",    # 5
        "True Positive",     # 6
    ]
    
    # Node colors
    colors = [
        "#4A90D9",  # All
        "#F5A623",  # Step1+
        "#7ED321",  # Step1-
        "#50E3C2",  # TN
        "#F8E71C",  # FN
        "#FF6B6B",  # FP
        "#4ECDC4",  # TP
    ]
    
    # Calculate values
    n_total = zone_counts["N_RuleOut"] + zone_counts["N_Step2"]
    
    # Links: source, target, value
    links = {
        "source": [0, 0,  1, 1, 2, 2],
        "target": [1, 2,  5, 6, 3, 4],
        "value": [
            zone_counts["N_Step2"],       # All → Step1+
            zone_counts["N_RuleOut"],     # All → Step1-
            zone_counts["RuleIn_FP"],     # Step1+ → FP
            zone_counts["RuleIn_TP"],     # Step1+ → TP
            zone_counts["RuleOut_TN"],    # Step1- → TN
            zone_counts["RuleOut_FN"],    # Step1- → FN
        ],
        "color": [
            "rgba(245, 166, 35, 0.5)",   # to Step1+
            "rgba(126, 211, 33, 0.5)",   # to Step1-
            "rgba(255, 107, 107, 0.5)",  # to FP
            "rgba(78, 205, 196, 0.5)",   # to TP
            "rgba(80, 227, 194, 0.5)",   # to TN
            "rgba(248, 231, 28, 0.5)",   # to FN
        ]
    }
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=1),
            label=labels,
            color=colors
        ),
        link=dict(
            source=links["source"],
            target=links["target"],
            value=links["value"],
            color=links["color"]
        )
    ))
    
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=600,
        width=900
    )
    
    fig.write_html(outpath)
    # Also save as image
    try:
        fig.write_image(outpath.replace('.html', '.png'), scale=2)
    except:
        pass


def plot_ppv_npv_vs_prevalence(
    strategies: Dict[str, Dict],
    title: str,
    outpath: str
):
    """
    Plot PPV and NPV as a function of disease prevalence using Bayes theorem.
    
    PPV = (Sens × Prev) / (Sens × Prev + (1-Spec) × (1-Prev))
    NPV = (Spec × (1-Prev)) / ((1-Sens) × Prev + Spec × (1-Prev))
    
    Args:
        strategies: Dict mapping strategy name to dict with 'Sensitivity' and 'Specificity'
        title: Plot title
        outpath: Output file path
    """
    prevalence = np.linspace(0.01, 0.99, 200)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, (name, metrics) in enumerate(strategies.items()):
        sens = metrics.get("Sensitivity", 0.5)
        spec = metrics.get("Specificity", 0.5)
        
        # Calculate PPV and NPV
        ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence) + 1e-12)
        npv = (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence) + 1e-12)
        
        # Plot
        axes[0].plot(prevalence, ppv, label=name, color=colors[idx], linewidth=2)
        axes[1].plot(prevalence, npv, label=name, color=colors[idx], linewidth=2)
    
    # PPV subplot
    axes[0].set_xlabel("Prevalence of Advanced Fibrosis", fontsize=12)
    axes[0].set_ylabel("Positive Predictive Value (PPV)", fontsize=12)
    axes[0].set_title("PPV vs Prevalence", fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # NPV subplot
    axes[1].set_xlabel("Prevalence of Advanced Fibrosis", fontsize=12)
    axes[1].set_ylabel("Negative Predictive Value (NPV)", fontsize=12)
    axes[1].set_title("NPV vs Prevalence", fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=9, loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    outpath: str
):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted 0\n(Rule-out)', 'Predicted 1\n(Rule-in)'])
    ax.set_yticklabels(['Actual 0\n(No AF)', 'Actual 1\n(AF)'])
    
    # Annotate
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f"{cm[i, j]}",
                          ha="center", va="center", fontsize=14, fontweight='bold')
    
    # Add labels
    ax.text(0, -0.15, "TN", ha="center", va="center", fontsize=12, color='green', 
            transform=ax.transData)
    ax.text(1, -0.15, "FP", ha="center", va="center", fontsize=12, color='red',
            transform=ax.transData)
    ax.text(0, 1.15, "FN", ha="center", va="center", fontsize=12, color='orange',
            transform=ax.transData)
    ax.text(1, 1.15, "TP", ha="center", va="center", fontsize=12, color='blue',
            transform=ax.transData)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=30)
    
    # Metrics annotation
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    
    metrics_text = f"Sens={sens:.2f}  Spec={spec:.2f}  PPV={ppv:.2f}  NPV={npv:.2f}"
    ax.text(0.5, -0.25, metrics_text, ha='center', va='center', fontsize=10,
            transform=ax.transAxes)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    outpath: str
):
    """Plot ROC curves for multiple models/strategies."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, (name, (y_true, y_prob)) in enumerate(results.items()):
        if len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", 
                   color=colors[idx], linewidth=2)
        except:
            pass
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# DATA LOADING
# ============================================================

def load_predictions(pred_path: str, cfg: Config) -> pd.DataFrame:
    """
    Load predictions CSV and reshape to wide format.
    
    Expected CSV format:
        Model, Split, PatientID, TrueLabel, PredProb, PredLabel
    
    Returns:
        DataFrame with columns: Split, PatientID, TrueLabel, [model_tags...]
    """
    df = pd.read_csv(pred_path)
    
    # Pivot to wide format
    wide = df.pivot_table(
        index=["Split", "PatientID"],
        columns="Model",
        values=["TrueLabel", "PredProb"],
        aggfunc="first"
    ).reset_index()
    
    # Flatten column names
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    
    # Rename columns
    rename_dict = {f"TrueLabel_{cfg.tag_m1}": "TrueLabel"}
    for tag in cfg.all_model_tags:
        rename_dict[f"PredProb_{tag}"] = tag
    
    wide = wide.rename(columns=rename_dict)
    
    # Select needed columns
    keep_cols = ["Split", "PatientID", "TrueLabel"] + cfg.all_model_tags
    wide = wide[[c for c in keep_cols if c in wide.columns]].dropna()
    wide["TrueLabel"] = wide["TrueLabel"].astype(int)
    
    return wide


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis(cfg: Config):
    """Run the complete stepwise screening analysis."""
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/tables", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/thresholds", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/sankey", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/curves", exist_ok=True)
    
    print("=" * 60)
    print("STEPWISE SCREENING ANALYSIS")
    print("Based on Chen et al. 2024 methodology")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading predictions...")
    wide = load_predictions(cfg.pred_path, cfg)
    print(f"    Total samples: {len(wide)}")
    print(f"    Splits: {wide['Split'].value_counts().to_dict()}")
    
    # ========== THRESHOLD SELECTION ON VALIDATION SET ==========
    print("\n[2] Selecting thresholds from validation set (Youden's J)...")
    val_data = wide[wide["Split"] == cfg.split_val]
    y_val = val_data["TrueLabel"].values
    
    thresholds = {}
    for tag in cfg.all_model_tags:
        if tag in val_data.columns:
            thr, youden_j = find_threshold_youden(y_val, val_data[tag].values)
            thresholds[tag] = thr
            print(f"    {cfg.model_display_names.get(tag, tag)}: threshold={thr:.4f}, Youden J={youden_j:.4f}")
    
    # Save thresholds
    thr_df = pd.DataFrame([
        {"Model": k, "DisplayName": cfg.model_display_names.get(k, k), 
         "Threshold_Youden": v}
        for k, v in thresholds.items()
    ])
    thr_df.to_csv(f"{cfg.output_dir}/thresholds/thresholds_youden.csv", index=False)
    
    # ========== DEFINE STEPWISE STRATEGIES ==========
    # Following paper: Clinical parameter → DL model
    strategies = {
        "Two-step (Clinical-A → Echo-Net+All)": (cfg.tag_m4, cfg.tag_m3),
        "Two-step (Clinical-Base → Echo-Net+All)": (cfg.tag_m5, cfg.tag_m3),
        "Two-step (Clinical-A → Echo-Net)": (cfg.tag_m4, cfg.tag_m1),
        "Two-step (Clinical-Base → Echo-Net)": (cfg.tag_m5, cfg.tag_m1),
    }
    
    # ========== EVALUATION ON TEST SETS ==========
    cohorts = {
        "Internal_Test": cfg.split_internal_test,
        "External_Test": cfg.split_external_test,
        "Combined_Test": [cfg.split_internal_test, cfg.split_external_test],
    }
    
    all_step_metrics = []
    all_overall_metrics = []
    all_preds_for_roc = {}
    
    for cohort_name, split_val in cohorts.items():
        print(f"\n[3] Evaluating on {cohort_name}...")
        
        if isinstance(split_val, list):
            cohort_data = wide[wide["Split"].isin(split_val)]
        else:
            cohort_data = wide[wide["Split"] == split_val]
        
        y_true = cohort_data["TrueLabel"].values
        n_samples = len(y_true)
        prevalence = y_true.mean()
        
        print(f"    Samples: {n_samples}, Prevalence: {prevalence:.2%}")
        
        all_preds_for_roc[cohort_name] = {}
        strategies_for_plot = {}
        
        # ---------- STEPWISE STRATEGIES ----------
        for strategy_name, (step1_tag, step2_tag) in strategies.items():
            if step1_tag not in thresholds or step2_tag not in thresholds:
                continue
            
            result = stepwise_two_stage(
                y_true=y_true,
                p_step1=cohort_data[step1_tag].values,
                thr_step1=thresholds[step1_tag],
                p_step2=cohort_data[step2_tag].values,
                thr_step2=thresholds[step2_tag]
            )
            
            # Record step metrics
            all_step_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Step": "Step1",
                "Model": cfg.model_display_names.get(step1_tag, step1_tag),
                "Threshold": thresholds[step1_tag],
                "N": n_samples,
                "N_Pass": result.n_pass_step1,
                "Pass_Rate": result.n_pass_step1 / n_samples,
                **{k: v for k, v in result.step1_metrics.items() if k not in ["N"]}
            })
            
            all_step_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "Step": "Step2",
                "Model": cfg.model_display_names.get(step2_tag, step2_tag),
                "Threshold": thresholds[step2_tag],
                "N": result.n_pass_step1,
                "N_Pass": result.zone_counts["N_RuleIn"],
                "Pass_Rate": result.zone_counts["N_RuleIn"] / result.n_pass_step1 if result.n_pass_step1 > 0 else 0,
                **{k: v for k, v in result.step2_metrics.items() if k not in ["N"]}
            })
            
            # Record overall metrics
            all_overall_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "N": n_samples,
                "Prevalence": prevalence,
                "Step2_Entry_N": result.n_pass_step1,
                "Step2_Entry_Rate": result.n_pass_step1 / n_samples,
                "Referral_Rate": result.final_metrics["Positive_Rate"],
                **{k: v for k, v in result.final_metrics.items() 
                   if k not in ["N", "Prevalence", "Positive_Rate"]}
            })
            
            strategies_for_plot[strategy_name] = result.final_metrics
            
            # Plot Sankey diagram
            safe_name = strategy_name.replace(" ", "_").replace("→", "to").replace("(", "").replace(")", "")
            plot_sankey_matplotlib(
                result.zone_counts,
                f"{strategy_name}\n{cohort_name}",
                f"{cfg.output_dir}/sankey/sankey_{cohort_name}_{safe_name}.png"
            )
            
            # Plot confusion matrix
            plot_confusion_matrix(
                y_true, result.final_pred,
                f"{strategy_name}\n{cohort_name}",
                f"{cfg.output_dir}/confusion_matrices/cm_{cohort_name}_{safe_name}.png"
            )
        
        # ---------- SINGLE MODEL BASELINES ----------
        for tag in cfg.all_model_tags:
            if tag not in thresholds:
                continue
            
            y_prob = cohort_data[tag].values
            y_pred = (y_prob >= thresholds[tag]).astype(int)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            
            display_name = f"Single ({cfg.model_display_names.get(tag, tag)})"
            
            all_overall_metrics.append({
                "Cohort": cohort_name,
                "Strategy": display_name,
                "N": n_samples,
                "Prevalence": prevalence,
                "Step2_Entry_N": np.nan,
                "Step2_Entry_Rate": np.nan,
                "Referral_Rate": metrics["Positive_Rate"],
                **{k: v for k, v in metrics.items() 
                   if k not in ["N", "Prevalence", "Positive_Rate"]}
            })
            
            strategies_for_plot[display_name] = metrics
            all_preds_for_roc[cohort_name][display_name] = (y_true, y_prob)
            
            # Plot confusion matrix for single models
            safe_tag = tag.replace(" ", "_")
            plot_confusion_matrix(
                y_true, y_pred,
                f"Single Model: {cfg.model_display_names.get(tag, tag)}\n{cohort_name}",
                f"{cfg.output_dir}/confusion_matrices/cm_{cohort_name}_single_{safe_tag}.png"
            )
        
        # ---------- PLOTS FOR THIS COHORT ----------
        # PPV/NPV vs Prevalence
        plot_ppv_npv_vs_prevalence(
            strategies_for_plot,
            f"PPV and NPV vs Disease Prevalence\n{cohort_name}",
            f"{cfg.output_dir}/curves/ppv_npv_vs_prevalence_{cohort_name}.png"
        )
        
        # ROC curves (for single models with probabilities)
        if all_preds_for_roc[cohort_name]:
            plot_roc_curves(
                all_preds_for_roc[cohort_name],
                f"ROC Curves\n{cohort_name}",
                f"{cfg.output_dir}/curves/roc_curves_{cohort_name}.png"
            )
    
    # ========== SAVE RESULTS ==========
    print("\n[4] Saving results...")
    
    step_df = pd.DataFrame(all_step_metrics)
    overall_df = pd.DataFrame(all_overall_metrics)
    
    step_df.to_csv(f"{cfg.output_dir}/tables/stepwise_step_metrics.csv", index=False)
    overall_df.to_csv(f"{cfg.output_dir}/tables/overall_metrics_by_cohort.csv", index=False)
    
    # Create summary comparison table
    print("\n" + "=" * 60)
    print("SUMMARY - Key Performance Metrics")
    print("=" * 60)
    
    summary_cols = ["Cohort", "Strategy", "Sensitivity", "Specificity", "PPV", "NPV", 
                    "Accuracy", "AUROC", "Referral_Rate"]
    summary_df = overall_df[[c for c in summary_cols if c in overall_df.columns]].copy()
    
    for col in ["Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "AUROC", "Referral_Rate"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "NA"
            )
    
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{cfg.output_dir}/tables/summary_comparison.csv", index=False)
    
    print("\n" + "=" * 60)
    print(f"✔ Analysis complete! Results saved to: {cfg.output_dir}")
    print("=" * 60)
    
    return overall_df, step_df


# ============================================================
# ADDITIONAL: Support for manual threshold/prediction adjustment
# ============================================================

def run_with_custom_thresholds(
    cfg: Config,
    custom_thresholds: Dict[str, float],
    output_suffix: str = "_custom"
):
    """
    Run analysis with custom (manually specified) thresholds.
    
    Args:
        cfg: Configuration object
        custom_thresholds: Dict mapping model tag to threshold value
        output_suffix: Suffix for output directory
    
    Usage:
        custom_thr = {
            "Radiomics_Only": 0.5,
            "ClinicalA_Only": 0.35,
            ...
        }
        run_with_custom_thresholds(config, custom_thr, "_adjusted")
    """
    # Modify output directory
    original_output = cfg.output_dir
    cfg.output_dir = f"{original_output}{output_suffix}"
    
    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/tables", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/thresholds", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/sankey", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/curves", exist_ok=True)
    
    print("=" * 60)
    print("STEPWISE SCREENING ANALYSIS (Custom Thresholds)")
    print("=" * 60)
    
    # Load data
    wide = load_predictions(cfg.pred_path, cfg)
    
    # Use custom thresholds
    thresholds = custom_thresholds.copy()
    
    # Save custom thresholds
    thr_df = pd.DataFrame([
        {"Model": k, "DisplayName": cfg.model_display_names.get(k, k), 
         "Threshold_Custom": v}
        for k, v in thresholds.items()
    ])
    thr_df.to_csv(f"{cfg.output_dir}/thresholds/thresholds_custom.csv", index=False)
    
    print("\nUsing custom thresholds:")
    for k, v in thresholds.items():
        print(f"    {cfg.model_display_names.get(k, k)}: {v:.4f}")
    
    # Run evaluation (same as main analysis)
    strategies = {
        "Two-step (Clinical-A → Echo-Net+All)": (cfg.tag_m4, cfg.tag_m3),
        "Two-step (Clinical-Base → Echo-Net+All)": (cfg.tag_m5, cfg.tag_m3),
        "Two-step (Clinical-A → Echo-Net)": (cfg.tag_m4, cfg.tag_m1),
        "Two-step (Clinical-Base → Echo-Net)": (cfg.tag_m5, cfg.tag_m1),
    }
    
    cohorts = {
        "Internal_Test": cfg.split_internal_test,
        "External_Test": cfg.split_external_test,
        "Combined_Test": [cfg.split_internal_test, cfg.split_external_test],
    }
    
    all_overall_metrics = []
    
    for cohort_name, split_val in cohorts.items():
        if isinstance(split_val, list):
            cohort_data = wide[wide["Split"].isin(split_val)]
        else:
            cohort_data = wide[wide["Split"] == split_val]
        
        y_true = cohort_data["TrueLabel"].values
        n_samples = len(y_true)
        prevalence = y_true.mean()
        
        strategies_for_plot = {}
        
        # Stepwise strategies
        for strategy_name, (step1_tag, step2_tag) in strategies.items():
            if step1_tag not in thresholds or step2_tag not in thresholds:
                continue
            
            result = stepwise_two_stage(
                y_true=y_true,
                p_step1=cohort_data[step1_tag].values,
                thr_step1=thresholds[step1_tag],
                p_step2=cohort_data[step2_tag].values,
                thr_step2=thresholds[step2_tag]
            )
            
            all_overall_metrics.append({
                "Cohort": cohort_name,
                "Strategy": strategy_name,
                "N": n_samples,
                "Prevalence": prevalence,
                "Step2_Entry_N": result.n_pass_step1,
                "Step2_Entry_Rate": result.n_pass_step1 / n_samples,
                "Referral_Rate": result.final_metrics["Positive_Rate"],
                **{k: v for k, v in result.final_metrics.items() 
                   if k not in ["N", "Prevalence", "Positive_Rate"]}
            })
            
            strategies_for_plot[strategy_name] = result.final_metrics
            
            # Plot Sankey
            safe_name = strategy_name.replace(" ", "_").replace("→", "to").replace("(", "").replace(")", "")
            plot_sankey_matplotlib(
                result.zone_counts,
                f"{strategy_name}\n{cohort_name}",
                f"{cfg.output_dir}/sankey/sankey_{cohort_name}_{safe_name}.png"
            )
        
        # Single models
        for tag in cfg.all_model_tags:
            if tag not in thresholds:
                continue
            
            y_prob = cohort_data[tag].values
            y_pred = (y_prob >= thresholds[tag]).astype(int)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            
            display_name = f"Single ({cfg.model_display_names.get(tag, tag)})"
            
            all_overall_metrics.append({
                "Cohort": cohort_name,
                "Strategy": display_name,
                "N": n_samples,
                "Prevalence": prevalence,
                **{k: v for k, v in metrics.items() if k not in ["N", "Prevalence"]}
            })
            
            strategies_for_plot[display_name] = metrics
        
        # PPV/NPV curves
        plot_ppv_npv_vs_prevalence(
            strategies_for_plot,
            f"PPV and NPV vs Disease Prevalence\n{cohort_name} (Custom Thresholds)",
            f"{cfg.output_dir}/curves/ppv_npv_vs_prevalence_{cohort_name}.png"
        )
    
    # Save results
    overall_df = pd.DataFrame(all_overall_metrics)
    overall_df.to_csv(f"{cfg.output_dir}/tables/overall_metrics_custom.csv", index=False)
    
    print(f"\n✔ Analysis with custom thresholds complete!")
    print(f"    Results saved to: {cfg.output_dir}")
    
    # Restore original output dir
    cfg.output_dir = original_output
    
    return overall_df


def load_custom_predictions(
    csv_path: str,
    model_tag: str,
    cfg: Config
) -> pd.DataFrame:
    """
    Load a CSV with manually adjusted PredProb/PredLabel for a specific model.
    
    Expected CSV format:
        PatientID, PredProb, PredLabel (optional)
    
    This will merge with the main predictions file.
    """
    # Load original wide data
    wide = load_predictions(cfg.pred_path, cfg)
    
    # Load custom predictions
    custom = pd.read_csv(csv_path)
    
    # Merge by PatientID
    if "PredProb" in custom.columns:
        prob_map = dict(zip(custom["PatientID"], custom["PredProb"]))
        wide[model_tag] = wide.apply(
            lambda row: prob_map.get(row["PatientID"], row[model_tag]),
            axis=1
        )
    
    return wide


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # You can modify the config here
    config = Config(
        pred_path="all_models_sample_predictions.csv",
        output_dir="stepwise_outputs_v3",
    )
    
    # Run main analysis
    overall_results, step_results = run_analysis(config)
    
    # Example: Run with custom thresholds (uncomment to use)
    # custom_thresholds = {
    #     config.tag_m1: 0.05,    # Echo-Net
    #     config.tag_m2: 0.60,    # Echo-Net+A
    #     config.tag_m3: 0.55,    # Echo-Net+All
    #     config.tag_m4: 0.40,    # Clinical-A
    #     config.tag_m5: 0.45,    # Clinical-Base
    # }
    # run_with_custom_thresholds(config, custom_thresholds, "_adjusted")
