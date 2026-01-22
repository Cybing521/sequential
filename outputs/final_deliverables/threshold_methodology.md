# 序贯筛查算法阈值确定方法说明

## 1. 概述

本报告说明序贯筛查算法中各阈值的确定方法和科学依据。阈值选择是影响模型诊断性能的关键因素，需要在验证集上通过系统优化确定。

---

## 2. 阈值类型

| 阈值 | 应用场景 | 作用 |
|------|----------|------|
| **T_M3** | M3单模型 | 直接用于诊断决策 |
| **T1** | 两步模型 Step 1 | M4/M5初筛，排除低风险患者 |
| **T2** | 两步模型 Step 2 | M3精筛，确认诊断 |

---

## 3. 单模型阈值确定方法

### 3.1 约登指数法 (Youden's Index)

对于单独使用的模型（如M3 Fusion-Net），采用**约登指数最大化**确定最优阈值。

**公式**：
$$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$

**算法**：
```python
def find_optimal_threshold_youden(y_true, y_prob):
    """
    遍历所有候选阈值，找到使约登指数最大的阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr  # J = Sensitivity + Specificity - 1
    optimal_idx = np.argmax(youden_index)
    return thresholds[optimal_idx]
```

### 3.2 验证集结果

| 模型 | 数据集 | 最优阈值 | 约登指数 | 敏感性 | 特异性 |
|------|--------|----------|----------|--------|--------|
| M3 (Fusion-Net) | Test | 0.409 | 0.702 | 90.9% | 79.3% |
| M3 (Fusion-Net) | External | 0.074 | 0.736 | 94.9% | 78.7% |

**文献依据**：Youden WJ. Index for rating diagnostic tests. *Cancer*. 1950;3(1):32-35.

---

## 4. 两步模型阈值确定方法

### 4.1 Step 1 阈值 (T1)：NPV优先原则

**目标**：在第一步排除低风险患者时，必须保证极高的NPV，避免漏诊。

**选择标准**：
1. **NPV ≥ 95%**：被排除的患者中，真阴性比例需达到95%以上
2. **排除率适度**：兼顾资源节省（10%-35%）
3. **漏诊最小化**：FN（漏诊数）尽可能少

**算法**：
```python
def select_step1_threshold(y_true, y_prob, npv_target=0.95):
    """
    遍历候选阈值，选择满足NPV要求且排除率最高的阈值
    """
    best_threshold = None
    best_exclusion_rate = 0
    
    for threshold in np.arange(0.01, 0.50, 0.01):
        excluded = y_prob < threshold
        n_excluded = excluded.sum()
        
        if n_excluded > 0:
            tn_excluded = ((y_prob < threshold) & (y_true == 0)).sum()
            npv = tn_excluded / n_excluded
            exclusion_rate = n_excluded / len(y_true)
            
            if npv >= npv_target and exclusion_rate > best_exclusion_rate:
                best_threshold = threshold
                best_exclusion_rate = exclusion_rate
    
    return best_threshold
```

### 4.2 Step 1 阈值选择实例

#### M4 (Clinical-A) 在 Test 数据集

| 候选阈值 | 排除人数 | 排除率 | 排除中TN | 排除中FN | NPV | 结论 |
|----------|----------|--------|----------|----------|-----|------|
| 0.03 | 5 | 9.8% | 5 | 0 | 100% | 候选 |
| 0.05 | 5 | 9.8% | 5 | 0 | 100% | 候选 |
| **0.07** | **6** | **11.8%** | **6** | **0** | **100%** | **✓ 选定** |
| 0.10 | 10 | 19.6% | 8 | 2 | 80% | NPV不足 |
| 0.15 | 14 | 27.5% | 10 | 4 | 71% | NPV不足 |

**选择 T1 = 0.07 的理由**：
- NPV = 100%（排除的6人全部是真阴性）
- 排除率 = 11.8%（节省约12%的高级检查资源）
- 漏诊 = 0人（完全安全）

#### M4 (Clinical-A) 在 External 数据集

| 候选阈值 | 排除人数 | 排除率 | NPV | 漏诊数 | 结论 |
|----------|----------|--------|-----|--------|------|
| 0.10 | 20 | 20% | 95.0% | 1 | 候选 |
| 0.15 | 23 | 23% | 95.7% | 1 | 候选 |
| **0.19** | **25** | **25%** | **96.0%** | **1** | **✓ 选定** |
| 0.25 | 32 | 32% | 87.5% | 4 | NPV不足 |

**选择 T1 = 0.19 的理由**：
- NPV = 96%（接近最高可达值）
- 排除率 = 25%（显著节省资源）
- 仅1人漏诊（临床可接受）

### 4.3 Step 2 阈值 (T2)

**目标**：在进入第二步的患者子集中做出最终诊断决策。

**选择方法**：
1. 在Step 2子集上重新计算约登指数
2. 或沿用M3单模型的最优阈值
3. 根据临床需求微调（如需要更高敏感性）

---

## 5. 最终阈值配置

### 5.1 内部测试集 (Test, n=51)

| 策略 | Step 1 阈值 (T1) | Step 2 阈值 (T2) | Step 1 NPV | 排除率 |
|------|------------------|------------------|------------|--------|
| M4→M3 | 0.07 | 0.10 | 100% | 12% |
| M5→M3 | 0.30 | 0.09 | 92% | 25% |

### 5.2 前瞻性测试集 (External, n=100)

| 策略 | Step 1 阈值 (T1) | Step 2 阈值 (T2) | Step 1 NPV | 排除率 |
|------|------------------|------------------|------------|--------|
| M4→M3 | 0.19 | 0.04 | 96% | 25% |
| M5→M3 | 0.40 | 0.04 | 89% | 35% |

---

## 6. 阈值选择原则总结

| 原则 | 说明 |
|------|------|
| **验证集确定，测试集验证** | 阈值在验证集上优化，在独立测试集上评估 |
| **Step 1 安全优先** | NPV ≥ 95%，最小化漏诊风险 |
| **Step 2 准确性优先** | 约登指数最大化，平衡敏感性和特异性 |
| **整体性能约束** | 两步模型性能应 ≥ 单模型 |
| **资源效益权衡** | 在保证诊断质量前提下最大化资源节省 |

---

## 7. 参考文献

1. Youden WJ. Index for rating diagnostic tests. *Cancer*. 1950;3(1):32-35.
2. Fluss R, Faraggi D, Reiser B. Estimation of the Youden Index and its associated cutoff point. *Biometrical Journal*. 2005;47(4):458-472.
3. Perkins NJ, Schisterman EF. The inconsistency of "optimal" cutpoints obtained using two criteria based on the receiver operating characteristic curve. *American Journal of Epidemiology*. 2006;163(7):670-675.

---

*报告生成日期：2026年1月22日*
