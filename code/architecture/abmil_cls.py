"""
abmil_cls.py
基于 Attention-Based MIL (ABMIL) 的四分类模型。

用于大模型横评：对所有特征提取器统一使用相同的 MIL 聚合头，
确保模型性能差异仅来自特征提取器的预训练策略，而非聚合架构。

输入 : [N_patches, D_feat]   patch 特征（单个 WSI）
输出 : [1, n_classes]        各类别 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.network import DimReduction


class Attention_Gated(nn.Module):
    """
    门控注意力机制（与 acmil_cox.py 一致）。

    V 分支用 Softplus（保正），U 分支用 Tanh（门控），
    逐元素相乘后线性投影到 K 个注意力头。
    """

    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Softplus(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
        )
        self.attention_weights = nn.Linear(D, K)

    def forward(self, x):
        # x: [N, L]
        A_V = self.attention_V(x)          # [N, D]
        A_U = self.attention_U(x)          # [N, D]
        A = self.attention_weights(A_V * A_U)  # [N, K]
        return A.transpose(1, 0)           # [K, N]


class ABMIL_Cls(nn.Module):
    """
    Attention-Based MIL 四分类模型。

    Args:
        D_feat   : 输入特征维度（随模型不同而变化，如 512/1024/1536/2048）
        D_inner  : 降维后的隐层维度（对所有模型统一，默认 256）
        D_attn   : 注意力投影维度（默认 128）
        n_classes: 分类数（默认 4）
        droprate : Dropout 比例（默认 0.25）
    """

    def __init__(self, D_feat=512, D_inner=256, D_attn=128, n_classes=4, droprate=0.25):
        super().__init__()

        # 1. 降维：D_feat → D_inner（统一后续注意力和分类头的维度）
        self.dimreduction = DimReduction(D_feat, D_inner)

        # 2. Feature-level dropout（防止对单个维度过拟合）
        self.feat_drop = nn.Dropout(droprate)

        # 3. 门控注意力（单头）
        self.attention = Attention_Gated(D_inner, D_attn, K=1)

        # 4. 分类头（单层线性）
        self.classifier = nn.Linear(D_inner, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x : [N, D_feat] 或 [1, N, D_feat] 或 list[tensor]

        Returns:
            logits : [1, n_classes]
            attn   : [1, N]  注意力权重（归一化后）
        """
        # 兼容多种输入格式
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 3:
            x = x[0]   # [1, N, D] → [N, D]

        # 降维
        x = self.dimreduction(x)    # [N, D_inner]
        x = self.feat_drop(x)

        # 注意力权重
        A = self.attention(x)       # [1, N]
        A_soft = F.softmax(A, dim=1)  # [1, N]

        # 加权聚合
        bag = torch.mm(A_soft, x)   # [1, D_inner]

        # 分类
        logits = self.classifier(bag)  # [1, n_classes]

        return logits, A_soft
