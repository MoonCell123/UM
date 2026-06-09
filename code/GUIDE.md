# UM (Uveal Melanoma) — Codebase Guide

> 项目根目录：`UM/code/`
> 编程语言：Python 3.9+
> 深度学习框架：PyTorch 2.0+
> 研究对象：葡萄膜黑色素瘤 (Uveal Melanoma, UM/UVM)

---

## 📂 目录结构总览

```
code/
├── config/                        # YAML 配置文件
├── architecture/                  # MIL 模型架构（22 种）
├── train/                         # 训练 & 评估主脚本
├── data_utils/                    # 数据加载工具
├── evaluation/                    # 分析与评估工具
├── visualization/                 # 可视化 & 出图
├── utils/                         # 工具脚本
├── modules/                       # 辅助模块（位置编码转发器）
└── GUIDE.md                       # 本文件
```

---

## 一、config/ — 配置文件

所有可配置参数通过 YAML 文件管理，无需修改代码即可调整实验。

| 文件 | 用途 | 对应主脚本 |
|------|------|-----------|
| `benchmark.yml` | **D3/M3 二分类横评** — 模型列表、数据路径、MIL 参数、训练超参、交叉验证折数 | `run_benchmark.py` |
| `cox.yml` | **Cox 生存回归** — 数据路径（HMU/温附一/福建协和）、模型架构、训练参数、运行模式 | `run_cox.py` |
| `roll_cox.yml` | **Cox 超参数网格遍历** — 搜索空间定义、并行数、基准配置 | `roll_cox.py` |

### 关键配置项速查

```yaml
# benchmark.yml 中定义模型列表（每个模型对应一个特征子目录）
models:
  - {name: conch_v1,    feat_subdir: 'features_conch_v1',    tcga_pretrained: false}
  - {name: uni_v2,      feat_subdir: 'features_uni_v2',      tcga_pretrained: false}
  - {name: phikon_v2,   feat_subdir: 'features_phikon_v2',   tcga_pretrained: true}
  
# 参数说明：
#   name            : 显示名（日志/汇总表用）
#   feat_subdir     : 特征子目录（相对于 feat_base）
#   tcga_pretrained : 预训练数据是否包含 TCGA 切片（用于数据污染分析）
```

---

## 二、architecture/ — MIL 模型架构（22 种）

所有模型以模块化的 `nn.Module` 实现，统一输入为 `[N_patches, D_feat]` 的 WSI 特征张量。

### 核心架构

| 文件 | 模型 | 特点 |
|------|------|------|
| `transformer.py` | **ACMIL_GA** ⭐ | **最常用** — 门控注意力 + 多分支 ACMIL（`n_token=1` 退化为标准 ABMIL） |
| `acmil_cox.py` | **ACMIL_Cox** ⭐ | Cox 生存版本 — 将分类头替换为单层线性 risk score |
| `abmil_cls.py` | **ABMIL_Cls** | 标准 Attention-Based MIL（四分类用） |
| `clam.py` | **CLAM** | 聚类约束注意力 |
| `attmil.py` | **ATT-MIL** | 基于注意力池化 |
| `dsmil.py` | **DS-MIL** | 双流 MIL |
| `transMIL.py` | **TransMIL** | 基于 Transformer 的 MIL |
| `bmil.py` | **Bayesian MIL** | 贝叶斯 MIL |
| `mhim.py` | **MHIM-MIL** | 多层级掩蔽交互 |
| `ilra.py` | **ILRA-MIL** | 跨尺度交互 |
| `ibmil.py` | **IB-MIL** | 信息瓶颈 MIL |
| `lbmil.py` | **LB-MIL** | 基于局部分支 |
| `dgrmil.py` | **DGR-MIL** | 基于图推理 |
| `rrt.py` | **RRT-MIL** | 区域递归 Transformer |
| `s4mil.py` / `S4MIL.py` | **S4-MIL** | 结构化状态空间序列模型 |
| `agata.py` | **AGATA** | 自适应门控注意力 |
| `hamitin_back.py` | **HIPT** | Hierarchical Image Pyramid Transformer |
| `attention.py` / `Attention.py` | **Attention baselines** | 多种注意力基线 |
| `mean_max.py` | **Mean-Max** | 均值/最大值池化基线 |
| `nystrom_attention.py` | **Nyström Attention** | 尼斯特罗姆近似注意力 |
| `rmsa.py` | **RMSA** | 区域多尺度注意力 |
| `datten.py` | **DAttention** | 可变形注意力 |
| `ips_net.py` | **IPS-Net** | 利用 timm 预训练 ViT 骨干 |
| `network.py` | **基础组件** | `DimReduction`（降维）、`Classifier_1fc`（分类头）、残差块 |

### 公共组件 (`network.py`)

```python
DimReduction(n_channels, m_dim)  # 线性降维：D_feat → D_inner
Classifier_1fc(n_channels, n_classes)  # 多层感知机分类头
```

### 位置编码 (`emb_position.py`)

| 模块 | 说明 |
|------|------|
| `PositionEmbedding` | 可学习位置编码（支持长度截断） |
| `PPEG` / `PEG` | 金字塔/金字塔位置编码（基于 2D 卷积） |
| `SINCOS` | 2D 正弦余弦绝对位置编码 |
| `APE` | 绝对位置编码 |
| `RPE` | 相对位置编码 |

---

## 三、train/ — 训练主脚本

所有脚本可直接从命令行运行。

### `run_benchmark.py` — D3/M3 二分类横评 🏆

**功能**：对 21 个 foundation model 的 WSI 特征，依次训练轻量 ABMIL 分类器，5 折交叉验证。

```bash
python train/run_benchmark.py                          # 跑全部模型
python train/run_benchmark.py --model uni_v2           # 只跑指定模型
python train/run_benchmark.py --folds 10               # 覆盖折数
python train/run_benchmark.py --seed 123               # 覆盖随机种子
```

**执行流程**：
1. 加载临床表 → 派生 D3/M3 标签（SCNA Cluster 1/2→0, 3/4→1）
2. 对每个模型：
   - 自动检测 `D_feat`（读第一个 .h5）
   - 设置 `D_inner = D_feat // 2`
   - 构建 `ACMIL_GA` 模型（n_token=1 = ABMIL）
   - 5 折 StratifiedKFold
   - 每折：AdamW + CosineAnnealingLR + 早停
3. 汇总输出 summary.csv（按 AUC 降序）

**输出目录**：`train/benchmark_output/{timestamp}/`

### `run_cox.py` — Cox 生存回归 🏥

**功能**：训练 ACMIL-Cox 模型，预测 CRC 患者 DFS/OS 风险。

```bash
python train/run_cox.py                                # 默认配置
python train/run_cox.py --config config/cox.yml        # 指定配置文件
```

**执行流程**：
1. 加载 HMU（训练+验证）、温附一（外部测试）、福建协和（外部测试）三个队列
2. HMU 7:3 分层划分（按 DFS_event 分层）
3. 训练 ACMIL-Cox 模型 → 早停
4. 训练集阈值搜索（Youden / Log-rank 两种策略）
5. 各队列 KM 曲线 + Time-dependent ROC + DCA
6. OS 终点评估 + 未化疗亚组评估

### `roll_cox.py` — Cox 超参数网格遍历 🔁

**功能**：自动遍历超参数组合，并行执行多个 `run_cox.py` 实验。

```bash
python train/roll_cox.py                               # 串行执行
python train/roll_cox.py --workers 8                   # 8 进程并行
```

### `cls_train.py` — 二分类训练引擎

被 `run_benchmark.py` 调用，包含：
- `train_one_epoch()` — 梯度累积训练，ACMIL 官方损失公式
- `evaluate_cls()` — AUC / Acc / F1 / Kappa
- `train_cls_model()` — 完整训练流程（含早停）

### `cox_train.py` — Cox 训练引擎

被 `run_cox.py` 调用，包含：
- `cox_loss()` — 负对数部分似然损失
- `diff_loss()` — 注意力多样性正则
- `train_one_epoch()` / `evaluate()` / `train_cox_model()`

---

## 四、data_utils/ — 数据加载

### `cls_dataset.py` — 二分类数据集

```python
ClsDataset(slide_ids, feat_dir, clinical_df, label_col="subtype")
```

- 输入：`.h5` 文件（内含 `feats` 或 `features` 数据集）
- 标签：自动从 `SCNA Cluster No.` 列派生 D3(0)/M3(1)
- `load_uvm_data()` — 加载 UVM 临床表的统一入口

### `survival_dataset.py` — 生存分析数据集

```python
SurvivalDataset(slide_ids, feat_dir, clinical_df)
```

- 输入：`.pth` 文件（内含 `feats` 键，numpy/tensor `[N, 512]`）
- 标签：`DFS_month` + `DFS_event`
- `load_cohort()` — 加载单个队列
- `split_hmu_train_val()` — HMU 7:3 分层划分

### 文件格式对照

| 任务 | 特征格式 | 关键 key | 工具函数 |
|------|---------|---------|---------|
| 二分类 (D3/M3) | `.h5` | `/feats` 或 `/features` | `load_uvm_data()` |
| 生存分析 (Cox) | `.pth` | `feats` (dict key) | `load_cohort()` |

---

## 五、evaluation/ — 分析与评估

### `analyze_results.py` — Benchmark 分组分析 📊

**功能**：读取 `run_benchmark.py` 输出的 summary.csv，回答两个科学问题。

```bash
python evaluation/analyze_results.py --results train/benchmark_output/20260331_171049
```

**统计方法**：
- **Q1**（病理模型 vs ResNet-50）：符号检验（二项检验），不依赖模型间独立性
- **Q2**（TCGA 组 vs 私有组）：置换检验（permutation test）

### `cox_evaluate.py` — Cox 评估套件

被 `run_cox.py` 调用，提供：

| 函数 | 功能 |
|------|------|
| `search_optimal_threshold()` | 多策略阈值搜索（Youden / Log-rank） |
| `evaluate_cohort()` | 完整队列评估（KM + C-index + HR） |
| `plot_km_curve()` | 绘制 Kaplan-Meier 生存曲线 |
| `compute_time_dependent_auc()` | 时间依赖 AUC（Uno's estimator） |
| `plot_time_dependent_roc()` | 时间依赖 ROC 曲线 |
| `compute_dca()` / `plot_dca_curves()` | 决策曲线分析 (DCA) |
| `save_results_report()` | 保存评估报告（CSV + 文本） |

---

## 六、visualization/ — 可视化与出图

### `visualize_results.py` ⭐

**功能**：生成 publication-quality 图表（三线表 + 组图）。

**执行**：
```bash
python visualization/visualize_results.py
```

**产出**（以 `20260331_171049` 结果为例）：

| 文件 | 内容 |
|------|------|
| `fig3_q1_transfer_value.pdf/png` | 全模型 AUC 柱状图（Q1：迁移价值） |
| `table4_q1_transfer_value.png/xlsx` | 三线表（各模型 vs ResNet-50） |
| `fig6_q2_tcga_overlap.pdf/png` | TCGA vs 私有组箱线图+散点图（Q2：数据污染） |
| `table5_q2_tcga_overlap.png/xlsx` | 三线表（两组对比） |
| `fig8_q3_kaiko_factorial.pdf/png` | Kaiko 2×2 析因设计交互图（模型容量×补丁大小） |
| `table6_q3_kaiko_factorial.png/xlsx` | 三线表（Kaiko 析因分析） |

### `visualize_extended.py`

**功能**：补充可视化（扩展分析图表，包含临床统计相关性等）。

### `draw_abmil_heatmap.py` / `heatmap_ljh_v6.py`

**功能**：注意力热图可视化，将注意力权重叠加到 WSI 原图上，直观展示模型关注的区域。

**依赖**：OpenSlide（读取 `.svs` / `.tiff` 全切片图像）

### `plot_all21_model_diagnostics.py`

**功能**：21 个模型的诊断图（Cox 相关），含 KM 曲线和 AUC 对比。

### `generate_captions_doc.py`

**功能**：生成 Word 文档（.docx），为每张图自动生成中英文图注。

---

## 七、modules/ — 辅助模块

| 文件 | 说明 |
|------|------|
| `emb_position.py` | 位置编码转发器（兼容旧 import 路径） |
| `__init__.py` | 空包文件 |

---

## 八、utils/ — 工具函数

| 文件 | 说明 |
|------|------|
| `generate_captions_doc.py` | 为可视化结果生成 Word 文档图注 |

---

## 九、实验流程速查

### D3/M3 二分类横评（完整流程）

```bash
# 1. 准备数据：确保临床表 + 各模型 .h5 特征在指定路径
# 2. 修改 config/benchmark.yml 中的路径
# 3. 运行横评
cd train
python run_benchmark.py

# 4. 分组统计分析
cd ../evaluation
python analyze_results.py --results ../train/benchmark_output/{timestamp}

# 5. 出图
cd ../visualization
python visualize_results.py
```

### Cox 生存分析（完整流程）

```bash
# 1. 准备数据：确保临床表 + .pth 特征在指定路径
# 2. 修改 config/cox.yml 中的路径
# 3. 运行 Cox 回归
cd train
python run_cox.py

# 4. （可选）超参数网格搜索
python roll_cox.py --workers 8
```

---

## 十、快速参考

### 常用命令行

```bash
# 只跑一个模型
python run_benchmark.py --model uni_v2

# 覆盖配置文件
python run_benchmark.py --config my_config.yml

# 覆盖折数
python run_benchmark.py --folds 10

# 覆盖种子（用于测试稳定性）
python run_benchmark.py --seed 99
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_token` | 1 | 1=ABMIL（轻量）；>1=ACMIL（多分支多样性正则） |
| `D_attn` | 128 | 注意力投影维度 |
| `droprate` | 0.25 | Dropout 比例 |
| `n_classes` | 2 | 分类数（D3/M3 二分类） |
| `n_folds` | 5 | 交叉验证折数 |
| `max_epochs` | 100 | 最大训练轮数 |
| `patience` | 20 | 早停耐心值 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 1e-3 | 权重衰减 |

### 依赖清单

详见根目录 `requirements.txt`。核心依赖：

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
h5py>=3.8.0
einops>=0.7.0
timm>=0.9.0
lifelines>=0.27.0
scikit-survival>=0.22.0
openpyxl>=3.1.0
openslide-python>=1.3.0
Pillow>=10.0.0
python-docx>=1.0.0
PyYAML>=6.0
```

---

> **注意**：代码中的硬编码绝对路径（如 `D:\Datas of lab\...`）为原始实验环境所用。在新环境中，请通过 `config/benchmark.yml`、`config/cox.yml` 等配置文件修改路径，**请勿直接修改 Python 代码**。
