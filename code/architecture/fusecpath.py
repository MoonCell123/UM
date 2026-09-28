"""FuseCPath patch-level re-embedding and ABMIL head.

This implementation follows the paper's patch branch: concatenate aligned
multi-encoder patch features, project them to a shared space, perform two
cluster-local self-attention blocks followed by cross-cluster attention, and
aggregate the re-embedded patches with ABMIL.  Slide-level teacher embeddings
are intentionally optional and are not fabricated when the manifest contains
only patch-level foundation-model features.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from architecture.abmil_cls import ABMIL_Cls
from architecture.projection_head import initialize_projection_weights


class _AttentionBlock(nn.Module):
    """Pre-norm MSA residual block; FuseCPath keeps its MLP branch disabled."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return x + attended


class _CrossClusterAttention(nn.Module):
    """FuseCPath-style CC-MSA over learnable cluster summaries.

    Each selected-patch cluster is compressed to a small set of learnable
    summary tokens, globally attended, and dispatched back to every patch.
    This mirrors the paper's combine/dispatch construction rather than simply
    adding an average cluster token to each patch.
    """

    def __init__(self, dim: int, heads: int, summaries: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.assignment = nn.Linear(dim, summaries, bias=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [clusters, selected patches per cluster, feature dimension]
        normalized = self.norm(x)
        logits = self.assignment(normalized)
        combine = torch.softmax(logits.transpose(1, 2), dim=-1)
        summaries = torch.matmul(combine, normalized)
        contextual, _ = self.attn(summaries, summaries, summaries, need_weights=False)
        dispatch = torch.softmax(logits, dim=-1)
        dispatch_min = dispatch.amin(dim=-1, keepdim=True)
        dispatch_max = dispatch.amax(dim=-1, keepdim=True)
        dispatch = (dispatch - dispatch_min) / (dispatch_max - dispatch_min + 1e-8)
        return x + torch.matmul(dispatch, contextual)


class FuseCPathModel(nn.Module):
    """Patch-level FuseCPath baseline with clustered re-embedding."""

    def __init__(self, input_dims: Mapping[str, int], args) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("FuseCPath requires at least two patch-level encoders.")
        self.encoder_names = sorted(input_dims)
        self.input_dims = {name: int(input_dims[name]) for name in self.encoder_names}
        self.input_dim = sum(self.input_dims.values())
        self.fuse_dim = int(getattr(args, "fusecpath_dim", 512))
        self.cluster_size = max(1, int(getattr(args, "fusecpath_patches_per_cluster", 10)))
        heads = int(getattr(args, "fusecpath_heads", 8))
        if self.fuse_dim % heads:
            raise ValueError("fusecpath_dim must be divisible by fusecpath_heads.")
        dropout = float(getattr(args, "fusecpath_dropout", 0.1))
        self.patch_to_emb = nn.Sequential(
            nn.Linear(self.input_dim, self.fuse_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.peg = nn.Conv1d(self.fuse_dim, self.fuse_dim, kernel_size=7, padding=3, groups=self.fuse_dim)
        n_local = max(1, int(getattr(args, "fusecpath_local_layers", 2)))
        self.local_blocks = nn.ModuleList(
            [_AttentionBlock(self.fuse_dim, heads, dropout) for _ in range(n_local)]
        )
        self.cross_cluster = _CrossClusterAttention(
            self.fuse_dim,
            heads,
            summaries=max(1, int(getattr(args, "fusecpath_cross_cluster_summaries", 3))),
            dropout=dropout,
        )
        self.classifier = ABMIL_Cls(
            D_feat=self.fuse_dim,
            D_inner=args.d_inner,
            D_attn=args.d_attn,
            n_classes=args.n_classes,
            droprate=args.droprate,
        )
        self.apply(initialize_projection_weights)

    def forward(self, raw_features: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = []
        patch_count = None
        for name in self.encoder_names:
            if name not in raw_features:
                raise KeyError(f"Missing encoder {name}; got {list(raw_features)}")
            tensor = raw_features[name].float()
            if tensor.ndim != 2 or tensor.shape[-1] != self.input_dims[name]:
                raise ValueError(
                    f"{name}: expected [N, {self.input_dims[name]}], got {tuple(tensor.shape)}"
                )
            patch_count = tensor.shape[0] if patch_count is None else min(patch_count, tensor.shape[0])
            tensors.append(tensor)
        x = torch.cat([tensor[:patch_count] for tensor in tensors], dim=-1)
        x = self.patch_to_emb(x)
        x = x + self.peg(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)

        n_clusters = max(1, (x.shape[0] + self.cluster_size - 1) // self.cluster_size)
        padded = n_clusters * self.cluster_size - x.shape[0]
        if padded:
            x = torch.cat([x, x.new_zeros((padded, x.shape[-1]))], dim=0)
        x = x.view(n_clusters, self.cluster_size, self.fuse_dim)
        for block in self.local_blocks:
            x = block(x)

        x = self.cross_cluster(x)
        x = x.reshape(-1, self.fuse_dim)[:patch_count]
        return self.classifier(x)
