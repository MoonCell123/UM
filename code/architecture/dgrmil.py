import torch
import torch.nn as nn
import torch.nn.functional as F


class DGRMIL(nn.Module):
    """
    双粒度表征多实例学习模型 (Dual Granularity Representation Multiple Instance Learning)
    """

    def __init__(self, conf):
        super(DGRMIL, self).__init__()
        self.feat_size = conf.D_feat
        self.L = conf.D_inner
        self.n_lesion = conf.n_lesion
        self.dropout_node = conf.dropout_node
        self.dropout_patch = conf.dropout_patch

        # 特征转换层
        self.feat_transform = nn.Sequential(
            nn.Linear(self.feat_size, self.L),
            nn.ReLU(),
            nn.Dropout(self.dropout_node)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh(),
            nn.Linear(self.L, 1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 2)
        )

        # 正常中心表征
        self.normal_center = nn.Parameter(torch.randn(self.n_lesion, self.L))

        # 异常中心表征
        self.abnormal_center = nn.Parameter(torch.randn(self.n_lesion, self.L))

        # 病变表征生成器
        self.lesion_representation = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(self.dropout_node),
            nn.Linear(self.L, self.n_lesion * self.L)
        )

    def forward(self, x, bag_mode=None):
        """
        前向传播
        Args:
            x: 输入特征, 形状为 [B, N, D] 或 [N, D]
            bag_mode: 'normal' 或 'abnormal' 或 None (评估模式)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # [1, N, D]

        batch_size, n_instances, _ = x.shape

        # 特征转换
        h = self.feat_transform(x)  # [B, N, L]

        # 计算注意力分数
        a = self.attention(h)  # [B, N, 1]
        a = torch.softmax(a, dim=1)  # 注意力分数归一化

        # 加权聚合
        z = torch.sum(a * h, dim=1)  # [B, L]

        # 分类
        bag_prediction = self.classifier(z)  # [B, 1]

        # 如果提供了bag_mode，则计算病变表征
        if bag_mode:
            if bag_mode == 'normal':
                # 对于正常样本
                p_center = self.normal_center
                nc_center = self.abnormal_center
            else:  # abnormal
                # 对于异常样本
                p_center = self.abnormal_center
                nc_center = self.normal_center

            # 生成病变表征
            lesion_repr = self.lesion_representation(z)
            lesion_repr = lesion_repr.view(batch_size, self.n_lesion, self.L)

            return bag_prediction, a, h, p_center, nc_center, lesion_repr
        else:
            # 评估模式，仅返回基本输出
            return bag_prediction, a, h