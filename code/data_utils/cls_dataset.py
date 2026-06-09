"""
cls_dataset.py
UVM D3/M3 二分类任务数据集：从 .h5 特征文件和临床表构建分类数据。

h5 文件结构（支持两种常见命名）:
    /feats  or  /features : [N_patches, D_feat]   (必须)
    /coords               : [N_patches, 2]          (可选)

临床表必须包含列:
    slide_id        : WSI 标识符
    SCNA Cluster No.: 整数 1-4（自动派生为 d3m3 二分类标签）
                      D3 (Disomy 3) = Cluster 1/2 → 0
                      M3 (Monosomy 3) = Cluster 3/4 → 1
"""

import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ClsDataset(Dataset):
    """
    D3/M3 二分类 WSI 数据集。

    每个样本包含:
    - feats : [N_patches, D_feat] 特征张量
    - label : int (0=D3, 1=M3)
    - slide_id : str
    """

    def __init__(self, slide_ids, feat_dir, clinical_df, label_col="subtype"):
        """
        Args:
            slide_ids  : 要包含的 slide_id 列表
            feat_dir   : .h5 特征文件目录
            clinical_df: 包含 slide_id 和 label_col 列的 DataFrame
            label_col  : 标签列名（默认 'subtype'）
        """
        self.feat_dir = feat_dir
        self.label_col = label_col
        self.clinical_df = clinical_df.set_index("slide_id")
        self.slide_ids = []
        self.skipped = []

        for sid in slide_ids:
            sid_str = str(sid)
            h5_path = os.path.join(feat_dir, f"{sid_str}.h5")
            if not os.path.exists(h5_path):
                self.skipped.append(sid_str)
                continue
            if sid_str not in self.clinical_df.index:
                self.skipped.append(sid_str)
                continue
            row = self.clinical_df.loc[sid_str]
            if pd.isna(row[label_col]):
                self.skipped.append(sid_str)
                continue
            self.slide_ids.append(sid_str)

        if self.skipped:
            print(f"  跳过 {len(self.skipped)} 个样本（缺少特征文件或标签）")

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        h5_path = os.path.join(self.feat_dir, f"{sid}.h5")

        with h5py.File(h5_path, "r") as f:
            # 支持两种常见 key 命名
            if "feats" in f:
                feats = f["feats"][:]
            elif "features" in f:
                feats = f["features"][:]
            else:
                # 取第一个非 coords 的数据集
                keys = [k for k in f.keys() if k != "coords"]
                if not keys:
                    raise KeyError(f"{h5_path}: 未找到特征数据集，可用 key: {list(f.keys())}")
                feats = f[keys[0]][:]

        feats = torch.from_numpy(feats.astype(np.float32))

        row = self.clinical_df.loc[sid]
        label = int(row[self.label_col])

        return feats, label, sid


def load_uvm_data(clinical_path, label_col="d3m3", cohort_name="UVM"):
    """
    加载 UVM 临床表，派生 D3/M3 二分类标签并返回。

    D3/M3 标签由 'SCNA Cluster No.' 列自动派生：
        Cluster 1 or 2  →  D3 (Disomy 3)   →  label 0
        Cluster 3 or 4  →  M3 (Monosomy 3) →  label 1
        其他/缺失       →  丢弃

    Args:
        clinical_path : CSV 路径（需含 slide_id 和 SCNA Cluster No. 列）
        label_col     : 派生后的标签列名（默认 'd3m3'，无需手动修改）
        cohort_name   : 用于日志输出的队列名称

    Returns:
        clinical_df   : 含 d3m3 列的 DataFrame（仅含有效行）
        slide_ids     : 有效 slide_id 列表 (str)
        labels        : 对应整数标签列表（0=D3, 1=M3）
    """
    if clinical_path.endswith(".csv"):
        df = pd.read_csv(clinical_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(clinical_path)

    if "slide_id" not in df.columns:
        raise ValueError(
            f"临床表缺少 'slide_id' 列。当前列: {df.columns.tolist()}"
        )

    scna_col = "SCNA Cluster No."
    if scna_col not in df.columns:
        raise ValueError(
            f"临床表缺少 '{scna_col}' 列。当前列: {df.columns.tolist()}"
        )

    df["slide_id"] = df["slide_id"].astype(str)

    # 将 SCNA Cluster No. 转为数值，无法转换的变为 NaN
    df[scna_col] = pd.to_numeric(df[scna_col], errors="coerce")

    # 派生 D3/M3 二分类标签
    def _to_d3m3(cluster):
        if cluster in (1, 2):
            return 0   # D3
        elif cluster in (3, 4):
            return 1   # M3
        return np.nan

    df["d3m3"] = df[scna_col].apply(_to_d3m3)

    # 过滤无效行
    valid = df[df["d3m3"].notna()].copy()
    valid["d3m3"] = valid["d3m3"].astype(int)

    slide_ids = valid["slide_id"].tolist()
    labels = valid["d3m3"].tolist()

    n_d3 = labels.count(0)
    n_m3 = labels.count(1)
    print(f"[{cohort_name}] 临床表: {len(df)} 例，有效: {len(valid)} 例")
    print(f"  D3 (label=0): {n_d3}  |  M3 (label=1): {n_m3}")

    return valid, slide_ids, labels
