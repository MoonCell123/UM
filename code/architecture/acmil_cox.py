"""
acmil_cox.py
基于ACMIL_GA架构的Cox回归模型。
将分类头替换为Cox head（输出单个risk score）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.network import DimReduction


class Attention_Gated(nn.Module):
    """门控注意力机制（与transformer.py中一致）。"""

    def __init__(self, L=512, D=128, K=1):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Softplus(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        # x: N x L
        A_V = self.attention_V(x)  # N x D
        A_U = self.attention_U(x)  # N x D
        A = self.attention_weights(A_V * A_U)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        return A


class CoxHead(nn.Module):
    """Cox回归头：单层线性，输出单个risk score（无激活函数）。

    简化说明：去掉两层FC+ReLU，改为单层线性映射。
    - ReLU会截断负risk score，限制Cox回归的表达能力
    - 两层FC对极少事件数（~20个）过度参数化
    - 单层线性: 仅 n_channels+1 个参数（vs 原来的 ~33K）
    """

    def __init__(self, n_channels, droprate=0.0):
        super().__init__()
        self.drop = nn.Dropout(droprate)
        self.out = nn.Linear(n_channels, 1)

    def forward(self, x):
        x = self.drop(x)
        x = self.out(x)  # 无激活函数，直接线性映射
        return x.squeeze(-1)  # [batch] or scalar


class ACMIL_Cox(nn.Module):
    """
    ACMIL-Cox模型：基于ACMIL_GA的门控注意力多实例学习架构，
    用于Cox比例风险回归。

    输入: [N_patches, D_feat] CONCH特征
    输出: (sub_risks, slide_risk, attn)
        - sub_risks: [n_token] 各子分支的risk score
        - slide_risk: [1] 全局bag的risk score
        - attn: [1, n_token, N_patches] 注意力权重
    """

    def __init__(self, D_feat=512, D_inner=128, D_attn=64,
                 n_token=1, n_masked_patch=0, mask_drop=0, droprate=0.5):
        super().__init__()
        self.dimreduction = DimReduction(D_feat, D_inner)
        # Feature-level dropout: 防止对降维后的单个维度过拟合
        self.feat_drop = nn.Dropout(droprate)
        self.attention = Attention_Gated(D_inner, D_attn, n_token)

        # Sub-branch Cox heads
        self.sub_cox_heads = nn.ModuleList()
        for _ in range(n_token):
            self.sub_cox_heads.append(CoxHead(D_inner, droprate))

        # Slide-level Cox head
        self.slide_cox_head = CoxHead(D_inner, droprate)

        self.n_token = n_token
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

    def forward(self, x):
        """
        Args:
            x: [N, D_feat] patch特征（单个slide）

        Returns:
            sub_risks: [n_token] 子分支risk scores
            slide_risk: scalar, 全局risk score
            attn: [1, n_token, N] 注意力权重
        """
        # 兼容多种输入格式：list/tuple取第一个，3D tensor取第一个batch
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.dim() == 3:
            x = x[0]  # [1, N, D] → [N, D]

        # 1. 降维: [N, D_feat] → [N, D_inner]
        x = self.dimreduction(x)

        # 1.5 Feature-level dropout: 防止对降维后的单个维度过拟合
        x = self.feat_drop(x)

        # 2. 门控注意力: → [n_token, N]
        A = self.attention(x)

        # 3. 训练时掩蔽
        if self.n_masked_patch > 0 and self.training:
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(
                torch.rand(*indices.shape, device=A.device), dim=-1
            )[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[
                torch.arange(indices.shape[0], device=A.device).unsqueeze(-1),
                rand_selected
            ]
            random_mask = torch.ones(k, n, device=A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A  # 保存原始注意力用于diff_loss

        # 4. Softmax + 加权求和
        A_soft = F.softmax(A, dim=1)  # [n_token, N]
        afeat = torch.mm(A_soft, x)   # [n_token, D_inner]

        # 5. 各子分支的risk score
        sub_risks = []
        for i, head in enumerate(self.sub_cox_heads):
            r = head(afeat[i].unsqueeze(0))  # [1]
            sub_risks.append(r.squeeze())     # scalar
        sub_risks = torch.stack(sub_risks, dim=0)  # [n_token]

        # 6. 全局bag特征 → slide risk
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)  # [1, N]
        bag_feat = torch.mm(bag_A, x)  # [1, D_inner]
        slide_risk = self.slide_cox_head(bag_feat).squeeze()  # scalar

        return sub_risks, slide_risk, A_out.unsqueeze(0)
