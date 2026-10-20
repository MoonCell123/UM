# 四状态 interaction 诊断

用途：把交互幅度还原为可解释的边际作用变化，区分正向作用增强、收益递减、
作用反转和负向作用减轻。分析冻结参考模型，不改变训练、路由或原有预测。

## 1. 导出四种状态的 logits

继续使用 compute_pairwise_interactions.py 原命令。每次运行会在时间戳目录中，
除原有文件外新增 **pairwise_state_logits.csv**。现有 workflow 调用该脚本时也
自动得到新 CSV，不需要新增开关。旧 interaction CSV 无法恢复四种输出，需要重跑。

从项目根目录运行，替换下列占位路径，使用原训练环境的 Python：

~~~powershell
python code/evaluation/compute_pairwise_interactions.py --fold-dir "output/GME/<run>/fold_1" --manifest "output/Manifests/<manifest_run>/fusion_manifest.csv" --split val --output-dir "output/Pairwise_Interactions/fold_1"
~~~

checkpoint、clinical_path、baseline 等沿用原脚本的解析方式，也可按原 CLI 显式指定。
使用 run_gme_workflow.py 的 analysis_only 模式同样可以重算，无需重新训练。

### 状态位的顺序是 (encoder_i, encoder_j)

| 状态 | encoder_i | encoder_j | 其他 FM |
|---|---|---|---|
| 00 | baseline 替换 | baseline 替换 | 保持原样 |
| 10 | 保留 projected embedding | baseline 替换 | 保持原样 |
| 01 | baseline 替换 | 保留 projected embedding | 保持原样 |
| 11 | 保留 projected embedding | 保留 projected embedding | 保持原样 |

**0 不表示把 tensor 设成零，也不表示删掉 encoder。** 只有 zero 策略才是零替换。
mean、gaussian 保持各自原定义，mean fusion 的分母仍是完整 encoder 数量。

CSV 每行是一个 WSI + pair，包含：

- slide_id、label、fold、split、encoder_i、encoder_j；
- logit_00_class_0、logit_00_class_1 等列，10/01/11 状态同理；支持 C>=2；
- interaction、target_class：原始 interaction 及其解释类别；
- checkpoint、baseline_file、replacement_strategy、gaussian_std_scale：来源；
- reference_forward=mean_fusion、schema_version=1、n_classes。

导出通过 intervention_attribution 的可选 coalition_logits_out 参数读取现有缓存；
原返回值仍然是三个 tensor。不额外调用 classifier，也不重抽 Gaussian 替换值。
logits 使用 17 位有效数字保存，减少舍入对差分的影响。

Gaussian 模式沿用旧逻辑：不同 coalition 可能使用不同随机替换样本。导出能复现
当次 interaction，但单次随机结果不能当作稳健的机制证据。

## 2. 运行新分析脚本

### Workflow 自动执行（推荐）

run_gme_workflow.py 在 fusion_and_analysis 和 analysis_only 模式下，完成所有折的
interaction 导出后，会自动运行本脚本，无需手动执行第二条命令。默认合并本次
各折 val 数据、分析全部 pairs、使用真实类别 margin，并输出全部 CSV 及 PNG/PDF/SVG。

~~~powershell
python code/train/run_gme_workflow.py --config code/config/UVM/gme.yml --workflow-mode fusion_and_analysis
~~~

输出位置：output/Analysis_Result/<source_run>/pairwise_state_analysis/<timestamp>/。
本次 interaction 导出位于 interactions/<timestamp>/fold_<n>/<export_timestamp>/。
重复运行会新建目录，仅使用本次生成的文件，避免旧结果重复计数或覆盖诊断结果。
workflow_manifest.json 会记录具体输入文件、输出目录和目标类别策略；CKA 四象限图
同样只读取本次 interaction 批次。fusion_only 模式不执行任何分析。

### 单独运行（可选）

分析单个文件的全部 pairs：

~~~powershell
python code/evaluation/analyze_pairwise_states.py --input-files "output/Pairwise_Interactions/fold_1/<timestamp>/pairwise_state_logits.csv" --output-dir "output/Pairwise_State_Analysis/<new_run>"
~~~

合并同一实验的 held-out folds，仅分析 H0/H1：

~~~powershell
python code/evaluation/analyze_pairwise_states.py --input-root "output/Analysis_Result/<run>/interactions" --pair features_hoptimus0 features_hoptimus1 --output-dir "output/Pairwise_State_Analysis/<new_run>"
~~~

--pair 使用 CSV 中精确 encoder 名称，顺序不限，可重复指定。--input-root 递归
搜索 pairwise_state_logits.csv；若包含多次重跑，改用 --input-files file1 file2 ...。
脚本拒绝重复 WSI/pair、同折混入不同 checkpoint/baseline、混合类别数或替换设置。
输出目录必须不存在或为空，避免新旧图表混杂。

可选参数：

- --margin-target true（默认）：每张 WSI 的真实类别 margin。
- --margin-target exported：原始 target_class，用于对照旧图。此时正作用仅代表
  提高目标类别 margin，不一定提高正确类别 margin。
- --split val（默认）、train、all：按 split 分析，汇总和图形不混合 splits。
- --effect-epsilon 1e-6：近零效应容差，不是显著性检验阈值。
- --no-plots：只输出 CSV/metadata，不需要 matplotlib。
- --formats png pdf svg、--dpi 300：输出格式及位图分辨率。

分析脚本需要 numpy、pandas，绘图额外需要 matplotlib；不依赖 torch 或 GPU。
导出脚本仍需原训练环境及 checkpoint/H5/baseline。

## 3. 计算公式

固定目标类别 c，每个状态的 margin = z_c - logsumexp(z_other_classes)。
默认 c 是真实标签。二分类就是 z_y-z_(1-y)；多分类从全部 logits 直接计算。
四种状态始终使用同一目标类别。

~~~text
delta_i_partner_replaced = m10 - m00
delta_i_partner_retained = m11 - m01
delta_j_partner_replaced = m01 - m00
delta_j_partner_retained = m11 - m10

interaction_analysis = m11 - m10 - m01 + m00
joint_gain_vs_best_single (G) = m11 - max(m10, m01)
pair_gain_vs_both_replaced = m11 - m00
~~~

G>0：当前冻结模型中，同时保留两者比任一单替换状态的目标 margin 更高。
G<0：至少一种单替换状态具有更高 margin。
**不是比较独立训练的单 FM 模型，不代表 AUC/AUPRC 增益。**

脚本从 logits 重算原 target_class 的 interaction，与导出值进行误差检查；容许
原 float32 前向和 float64 重算的舍入差异，不一致则报错。每个状态也输出真实
类别概率、交叉熵和预测类别，供辅助核查，不参与在线路由。

## 4. 输出与读图

| 输出 | 内容及用途 |
|---|---|
| pairwise_state_diagnostics.csv | 每个 WSI/pair 的 logits、四个 margin、双向边际作用、I、G、作用类型、校验误差 |
| pair_summary.csv | pair/split 的均值、中位数、正/负/近零比例、作用反转比例 |
| pair_summary_by_label.csv | 按真实标签分层，避免类别构成掩盖差异 |
| pair_summary_by_fold.csv | 按 fold 分层，检查尺度与方向是否跨折一致 |
| marginal_transition_summary.csv | 双向边际作用变化的类型、数量和比例 |
| figures/*.png/pdf/svg | 每个 pair/split 一张四面板图，PDF/SVG 文本可编辑 |
| metadata.json | 输入来源、公式、目标类别策略、容差、解释边界及输出路径 |

四面板图：a 是 encoder_i 的边际作用，b 是 encoder_j 的对称分析；横轴表示
partner 被替换，纵轴表示 partner 保留。c 是 signed interaction 分布，d 是 G 分布。
a/b 每个点是一个 WSI，颜色表示真实类别，零轴区分正负作用，对角线 y=x 是加性
预期，纵轴减横轴恰好等于 interaction。所有点保留，不裁剪异常值。

类型含义（默认 true margin）：

- positive_enhanced：正作用增强；positive_diminished：仍有正作用，但收益递减。
- negative_attenuated：仍有负作用，但损害减轻；negative_amplified：负作用加重。
- positive_to_negative / negative_to_positive：作用反转。
- near_zero_* / *_unchanged：涉及容差内近零作用或近零变化。

## 5. 解释边界

- 这是诊断，不自动判定协同、冗余、因果机制、显著性或子集最优性。
- 统计单位是 WSI，不假设同患者多张 WSI 独立；不输出 p 值或置信区间。
- pooled 表按 WSI 等权汇总。不同 fold 的 margin 尺度可能不同，应结合分折表。
- abs_interaction_analysis_mean 是先逐样本取绝对值再平均，区别于 signed mean。
- 真实标签只用于离线分析，不得直接作为测试 WSI 在线选择的输入。
- baseline 替换可能造成输入分布变化；冻结模型诊断不同于子集重训练收益。
- 不要把 high |I|、正 I 或 G>0 自动描述为测试集 AUC 提升。

## 6. 验证

~~~powershell
python tests/test_pairwise_state_analysis.py -v
python tests/test_pairwise_cache.py -v
~~~

前者需要 numpy/pandas/matplotlib，测试公式、class-0 方向、多分类数值稳定性、
输入校验、CSV 和三种绘图格式。后者需要 torch/numpy，通过从生产文件抽取原方法
和极小参考头测试状态映射、前向次数、Gaussian 缓存一致性、旧 API 兼容性与异常
后的模式恢复。它不执行完整的 MIL/临床数据依赖链。
所有测试数据都是明确标记的合成 fixture，不是实验结果。

若 Torch 和分析依赖分属两个 Python 环境，还可执行跨环境端到端检查：

~~~powershell
<torch-python> tests/check_pairwise_roundtrip.py --analysis-python <numpy-pandas-python>
~~~

该检查将生产缓存及序列化函数产生的三组 logits 写入真实 CSV，再调用新脚本，
验证导出到读取之间的状态映射、interaction 重建及 class-0 符号转换。
