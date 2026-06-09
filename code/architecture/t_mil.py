"""
修改后的T_MIL实现，不依赖DGL库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from einops import rearrange

class T_MIL(nn.Module):
    def __init__(self, n_classes, architecture='GA_MIL', feat_dim=1024, latent_dim=128, num_heads=1, depth=2):
        """
        Args:
          n_classes: 预测目标的数量
          architecture: 架构选择: LA_MIL, GA_MIL
          feat_dim: 特征提取后所有瓦片的输出维度
          latent_dim: 注意力模块的隐藏维度
          num_heads: 多头注意力的头数
          depth: 注意力层的数量
        """
        super().__init__()
        self.n_classes = n_classes
        self.architecture = architecture
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.depth = depth

        self.fc1 = nn.Linear(self.feat_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList([])

        self.current_epoch = 0
        self.tau_init = 1.0
        self.tau_min = 0.1
        self.anneal_rate = 0.0001
        self.eval_tau = 0.1
        if self.architecture == 'LA_MIL':
            # 替换为自定义的图注意力层
            for _ in range(self.depth):
                self.layers.append(
                    CustomGraphAttentionLayer(latent_dim, out_dim=self.latent_dim, num_heads=self.num_heads)
                )

        if self.architecture == 'GA_MIL':
            for _ in range(self.depth):
                self.layers.append(
                    TransformerLayer(latent_dim, heads=self.num_heads, use_ff=False, use_norm=True)
                )

        self.mlp_head = nn.Linear(self.latent_dim, self.n_classes, bias=True)

    def forward(self, x, graphs=None, return_last_att=False, return_emb=False, epoch=None):
        # x的形状: [batch_size, n_instances, feat_dim]
        batch_size, n_instances = x.shape[0], x.shape[1]

        # 应用第一个线性层和激活函数
        x_flat = x.view(-1, self.feat_dim)
        x_flat = self.fc1(x_flat)
        x_flat = self.relu(x_flat)
        x = x_flat.view(batch_size, n_instances, -1)

        attention_scores = None

        if self.architecture == 'LA_MIL':
            # 对LA_MIL使用自定义的图注意力处理
            for layer in self.layers:
                x, att = layer(x)
                attention_scores = att  # 保存最后一层的注意力分数

        if self.architecture == 'GA_MIL':
            # 对每个样本单独处理
            for layer in self.layers:
                x_transformed = []
                all_att = []

                for i in range(batch_size):
                    sample_x = x[i]  # [n_instances, feat_dim]
                    sample_x, att = layer(sample_x.unsqueeze(0))
                    x_transformed.append(sample_x.squeeze(0))
                    all_att.append(att.squeeze(0))

                x = torch.stack(x_transformed)
                attention_scores = torch.stack(all_att)

        # 实例聚合 - 平均池化
        if self.architecture == 'LA_MIL':
            emb = x.mean(dim=1)  # [batch_size, latent_dim]
        elif self.architecture == 'GA_MIL':
            # if self.training:
            #     if epoch is not None:
            #         self.current_epoch = epoch
            #     tau = max(self.tau_min, self.tau_init * math.exp(-self.anneal_rate * self.current_epoch))
            # else:
            #     # 评估时使用固定温度
            #     tau = self.eval_tau
            emb = x.mean(dim=1)  # [batch_size, latent_dim]
            # attention_weights = F.gumbel_softmax(x, tau=tau, hard=False, dim=1)
            # emb = torch.sum(x * attention_weights, dim=1)  # [batch_size, latent_dim]

        # 最终分类器
        logits = self.mlp_head(emb)  # [batch_size, n_classes]

        # 准备输出
        out = [logits]

        # 返回嵌入向量（如果需要）
        if return_emb:
            out.append(emb.detach())

        # 返回注意力分数（如果需要）
        if return_last_att:
            out.append(attention_scores)

        return out

class CustomGraphAttentionLayer(nn.Module):
    """
    自定义的图注意力层，替代DGL的GraphTransformerLayer
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout

        # 多头注意力的查询、键、值矩阵
        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)

        # 输出投影
        self.o_linear = nn.Linear(out_dim, out_dim)

        # LayerNorm 和 残差连接
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, n_instances, in_dim]
        Returns:
            output: 转换后的特征 [batch_size, n_instances, out_dim]
            attention: 注意力分数 [batch_size, num_heads, n_instances, n_instances]
        """
        batch_size, n_instances = x.shape[0], x.shape[1]

        # 计算查询、键、值
        q = self.q_linear(x)  # [batch_size, n_instances, out_dim]
        k = self.k_linear(x)  # [batch_size, n_instances, out_dim]
        v = self.v_linear(x)  # [batch_size, n_instances, out_dim]

        # 重塑为多头形式
        q = q.view(batch_size, n_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, n_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, n_instances, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 缩放点积注意力
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch_size, num_heads, n_instances, n_instances]

        # 应用softmax获取注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 获取带权值
        out = torch.matmul(attention, v)  # [batch_size, num_heads, n_instances, head_dim]

        # 重塑并合并头
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, n_instances, self.out_dim)

        # 应用输出投影
        out = self.o_linear(out)

        # 残差连接和层归一化
        out = x + out
        out = self.layer_norm(out)

        return out, attention

class Attention(nn.Module):
    def __init__(self, dim=512, heads=1, dim_head=None, dropout=0.1):
        super().__init__()
        dim_head = dim_head or dim // heads
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        b, n, _ = x.shape

        # 获取查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用softmax
        attn = F.softmax(dots, dim=-1)

        # 注意力加权求和
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn

class TransformerLayer(nn.Module):
    def __init__(self, dim=512, heads=8, use_ff=False, use_norm=True):
        super().__init__()
        self.attn = Attention(dim=dim, heads=heads)
        self.use_ff = use_ff
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(dim)

        if use_ff:
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        # 注意力层
        if self.use_norm:
            x_norm = self.norm(x)
            attn_out, attn = self.attn(x_norm)
        else:
            attn_out, attn = self.attn(x)

        x = x + attn_out

        # 前馈网络层
        if self.use_ff:
            if self.use_norm:
                x_norm = self.norm(x)
                x = x + self.ff(x_norm)
            else:
                x = x + self.ff(x)

        return x, attn