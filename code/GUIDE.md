## 端到端训练
流程是：

5-fold manifest
↓
每一折：
  raw embedding
    ↓
  ProjectionHead + Router + ABMIL
    ↓
    train split -> projected embedding -> build replacement baselines
    ↓
    对每个 WSI:
      Projection
      ↓
      replacement-based intervention scores
      ↓
      Intervention attribution
      ↓
      Routing 计算 final weight
      ↓
      final fusion
      ↓
      ABMIL
    ↓
    输出 AUC / AUPRC / Sensitivity / Specificity

对应代码位置：

- CLI 入口：`train/train_gme.py`（保持原命令兼容）
- CLI 参数与 YAML/JSON 配置：`train/gme/config.py`
- GME 网络、Attribution、Routing 与 joint fusion：`architecture/gme_model.py`
- 多编码器 H5 数据读取与 patch 对齐：`data_utils/gme_dataset.py`
- FLOPs、参数量和推理耗时：`train/gme/profiling.py`
- GME 训练阶段与 5-fold 实验编排：`train/gme/experiment.py`

I:\Anaconda\anaconda3\envs\Pytorch\python.exe code\train\train_gme.py --config code\config\gme.yml
I:\Anaconda\anaconda3\envs\Pytorch\python.exe code\train\train_offline_fusion_baselines.py --config code\config\offline_fusion_baselines.yml
I:\Anaconda\anaconda3\envs\Pytorch\python.exe .\code\train\run_gme_workflow.py --config .\code\config\gme.yml

## 多卡协同训练
跑GME方法：
I:\Anaconda\anaconda3\envs\Pytorch\python.exe code\train\launch_fold_parallel.py --config code\config\fold_parallel.yml
同时写好 fold_parallel.yml 和 gme.yml

跑对照实验：
I:\Anaconda\anaconda3\envs\Pytorch\python.exe code\train\launch_fold_parallel.py --config code\config\fold_parallel_offline.yml
同时写好 fold_parallel.yml 和 offline_fusion_baselines.yml

## ELF方法
官方权重地址：
contrast_method\ELF\checkpoints\elf_slide_encoder.pth
依次提取五组ELF slide embedding:
powershell -ExecutionPolicy Bypass -File .\contrast_method\ELF\feature_extract.ps1
然后直接评估：
i:\Anaconda\anaconda3\envs\Pytorch\python.exe .\contrast_method\ELF\evaluation\uvm\evaluate_elf_uvm.py --config .\code\config\UVM\elf.yml

## CKA图
i:\Anaconda\anaconda3\envs\Pytorch\python.exe `
.\code\evaluation\plot_pairwise_linear_cka_matrix.py `
--cka-file .\output\Analysis_Result\foundation_model_clustering\20260806_all13\pairwise_cka_by_slide.csv `
--output-dir $env:TEMP\cka_heatmap_reproduce `
--output-name foundation_model_cka_heatmap `
--order average_linkage `
--title "Mean Coordinate-Aligned Linear CKA"

## dendrogram图
i:\Anaconda\anaconda3\envs\Pytorch\python.exe `
.\code\evaluation\plot_linear_cka_dendrogram.py `
--cka-file .\output\Analysis_Result\foundation_model_clustering\20260806_all13\pairwise_cka_by_slide.csv `
--output-dir $env:TEMP\foundation_model_clustering `
--output-name foundation_model_dendrogram `
--title "Foundation Model Clustering from 1 - Linear CKA"
