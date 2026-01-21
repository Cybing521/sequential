# Sequential Screening Algorithm for Advanced Liver Fibrosis

基于 Chen et al. 2024 论文的多步骤筛选算法实现，用于肝纤维化的分层诊断。

## 📁 项目结构

```
sequential/
├── data/                           # 输入数据
│   ├── all_models_sample_predictions.csv   # 模型预测结果
│   ├── all_models_多步骤参照V2.csv         # 单模型性能参照
│   ├── table_stepwise_step_metrics.csv     # 多步骤分析结果
│   └── thresholds_from_val.csv             # 验证集阈值
├── scripts/                        # Python 脚本
│   ├── stepwise_screening.py       # 主分析脚本
│   └── generate_final_plots.py     # 最终图表生成
├── notebooks/                      # Jupyter Notebooks
│   ├── stepwise_analysis_interactive.ipynb # 交互式分析
│   └── stepwise.ipynb              # 原始分析
├── outputs/                        # 输出结果
│   ├── final/                      # 最终交付结果
│   │   ├── sankey/                 # 桑基图
│   │   ├── curves/                 # PPV/NPV曲线
│   │   └── summary_final.csv       # 汇总表格
│   └── intermediate/               # 中间结果
│       ├── confusion_matrices/     # 混淆矩阵
│       ├── curves/                 # 曲线图
│       ├── sankey/                 # 桑基图
│       ├── tables/                 # 表格
│       └── thresholds/             # 阈值
└── material/                       # 参考文献
    └── Chen 等 - 2024 - US-based Sequential Algorithm...pdf
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas matplotlib scikit-learn plotly
```

### 2. 运行主分析

```bash
cd /path/to/sequential
python scripts/stepwise_screening.py
```

### 3. 生成最终图表

```bash
python scripts/generate_final_plots.py
```

## 📊 主要功能

### 多步骤筛选算法

实现了两步筛选策略：
- **stepwise1 (M4→M3)**: Clinical A → Echo-Net+All
- **stepwise2 (M5→M3)**: Clinical Base → Echo-Net+All

### 输出结果

1. **桑基图 (Sankey diagrams)**: 展示患者流向 (TN/FN/FP/TP)
2. **PPV/NPV 曲线**: 随患病率变化的预测值曲线
3. **混淆矩阵**: 各策略的详细分类结果
4. **汇总表格**: 性能指标对比

## 📈 关键结果

| 队列 | 策略 | Sens | Spec | PPV | NPV | Accuracy |
|------|------|------|------|-----|-----|----------|
| InternalTest | Two-step (M4→M3) | 0.636 | 0.862 | 0.778 | 0.758 | 0.765 |
| InternalTest | Two-step (M5→M3) | 0.682 | 0.897 | 0.833 | 0.788 | 0.804 |
| ProspectiveTest | Two-step (M4→M3) | 0.692 | 0.852 | 0.750 | 0.812 | 0.790 |
| ProspectiveTest | Two-step (M5→M3) | 0.692 | 0.852 | 0.750 | 0.812 | 0.790 |

## 📚 参考文献

Chen et al. (2024). US-based Sequential Algorithm Integrating an AI Model for Advanced Liver Fibrosis Screening. *Radiology*, 311(1):e231461.

## 📝 License

MIT License
