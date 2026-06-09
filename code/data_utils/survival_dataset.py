"""
survival_dataset.py
Cox生存分析数据集：从.pth特征文件和临床表构建生存数据。
支持HMU 7:3分层划分（按DFS_event分层）。
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SurvivalDataset(Dataset):
    """
    Cox生存分析数据集。

    每个样本包含:
    - feats: [N_patches, 512] CONCH特征
    - time: DFS_month (float)
    - event: DFS_event (0或1)
    - slide_id: 患者标识
    """

    def __init__(self, slide_ids, feat_dir, clinical_df):
        """
        Args:
            slide_ids: 要包含的slide_id列表
            feat_dir: .pth特征文件目录
            clinical_df: 包含 slide_id, DFS_month, DFS_event 列的DataFrame
        """
        self.feat_dir = feat_dir
        self.clinical_df = clinical_df.set_index("slide_id")
        self.slide_ids = []
        self.skipped = []

        for sid in slide_ids:
            sid_str = str(sid)
            pth_path = os.path.join(feat_dir, f"{sid_str}.pth")
            if not os.path.exists(pth_path):
                self.skipped.append(sid_str)
                continue
            if sid_str not in self.clinical_df.index:
                self.skipped.append(sid_str)
                continue
            row = self.clinical_df.loc[sid_str]
            if pd.isna(row["DFS_month"]) or pd.isna(row["DFS_event"]):
                self.skipped.append(sid_str)
                continue
            self.slide_ids.append(sid_str)

        if self.skipped:
            print(f"  跳过 {len(self.skipped)} 个样本（缺少特征或DFS数据）")

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        pth_path = os.path.join(self.feat_dir, f"{sid}.pth")

        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        feats = data["feats"]  # numpy array [N, 512]
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).float()
        else:
            feats = feats.float()

        row = self.clinical_df.loc[sid]
        time = float(row["DFS_month"])
        event = int(row["DFS_event"])

        return feats, time, event, sid


def collate_survival(batch):
    """
    自定义collate函数：每个slide的patch数不同，不能stack。
    返回列表形式的feats，和tensor形式的time/event。
    """
    feats_list = [item[0] for item in batch]
    times = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    events = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    slide_ids = [item[3] for item in batch]
    return feats_list, times, events, slide_ids


def load_cohort(clinical_path, feat_dir, cohort_name=""):
    """
    加载单个队列的临床数据和特征文件。

    Returns:
        clinical_df: 包含 slide_id, DFS_month, DFS_event 的DataFrame
        feat_dir: 特征文件目录
        available_ids: 同时有临床数据和特征文件的slide_id列表
    """
    df = pd.read_excel(clinical_path)

    # 确保有统一的 slide_id 列
    # prepare_dfs.py 输出的 _withDFS.xlsx 已统一为 slide_id
    if "slide_id" not in df.columns:
        raise ValueError(
            f"临床表缺少 'slide_id' 列，请先运行 prepare_dfs.py 生成统一格式的表。"
            f"\n当前列名: {df.columns.tolist()}"
        )

    df["slide_id"] = df["slide_id"].astype(str)

    # 过滤有效DFS数据
    valid = df[df["DFS_month"].notna() & df["DFS_event"].notna()].copy()

    # 检查哪些有对应的特征文件
    available_ids = []
    for sid in valid["slide_id"]:
        if os.path.exists(os.path.join(feat_dir, f"{sid}.pth")):
            available_ids.append(sid)

    print(f"[{cohort_name}] 临床表: {len(df)}人, 有效DFS: {len(valid)}人, "
          f"有特征文件: {len(available_ids)}人")

    return valid, feat_dir, available_ids


def split_hmu_train_val(slide_ids, clinical_df, val_ratio=0.3, random_state=42):
    """
    按DFS_event分层，将HMU队列7:3划分为训练集和验证集。

    Args:
        slide_ids: HMU队列的slide_id列表
        clinical_df: 临床数据DataFrame
        val_ratio: 验证集比例（默认0.3）
        random_state: 随机种子

    Returns:
        train_ids, val_ids: 训练集和验证集的slide_id列表
    """
    # 排序确保输入顺序与 Excel 行序无关，相同 random_state 产生相同切分
    slide_ids = sorted(slide_ids)

    df = clinical_df.set_index("slide_id")
    events = [int(df.loc[sid, "DFS_event"]) for sid in slide_ids]

    # 防御性检查：确保两个类别至少各有 2 个样本以支持分层划分
    n_pos = sum(events)
    n_neg = len(events) - n_pos
    min_required = max(2, int(1.0 / min(val_ratio, 1 - val_ratio)) + 1)
    if n_pos < min_required or n_neg < min_required:
        raise ValueError(
            f"分层划分失败: DFS_event=1 有 {n_pos} 人, DFS_event=0 有 {n_neg} 人, "
            f"val_ratio={val_ratio} 至少需要每类 {min_required} 人"
        )

    train_ids, val_ids = train_test_split(
        slide_ids,
        test_size=val_ratio,
        stratify=events,
        random_state=random_state,
    )

    train_events = sum(1 for sid in train_ids if int(df.loc[sid, "DFS_event"]) == 1)
    val_events = sum(1 for sid in val_ids if int(df.loc[sid, "DFS_event"]) == 1)

    print(f"  训练集: {len(train_ids)}人 (复发: {train_events})")
    print(f"  验证集: {len(val_ids)}人 (复发: {val_events})")

    return list(train_ids), list(val_ids)
