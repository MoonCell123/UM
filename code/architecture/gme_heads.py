"""Downstream heads used by the GME fusion model.

All heads accept a patch bag shaped ``[N, D]`` (or ``[B, N, D]``) and return
``(logits, attention)``.  The second value is ``None`` for heads that do not
produce patch attention, which keeps the existing GME training and profiling
paths unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.abmil_cls import ABMIL_Cls
from architecture.transMIL import TransMIL


HEAD_NAMES = ("ABMIL", "TransMIL", "GNN", "MLP")


def _as_bag_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize a patch bag to ``[batch, patches, features]``."""
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected patch features shaped [N, D] or [B, N, D], got {tuple(x.shape)}")


class TransMILHead(nn.Module):
    """Adapter around the repository's TransMIL implementation."""

    def __init__(
        self,
        d_feat: int,
        d_inner: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        config = SimpleNamespace(D_feat=int(d_feat), D_inner=int(d_inner), n_class=int(n_classes))
        self.model = TransMIL(config)

    def forward(self, x: torch.Tensor):
        logits = self.model(_as_bag_batch(x))
        return logits, None


class MLPHead(nn.Module):
    """Slide-level MLP head using mean pooling over the fused patch bag."""

    def __init__(
        self,
        d_feat: int,
        hidden_dim: int,
        n_classes: int,
        droprate: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(int(d_feat)),
            nn.Linear(int(d_feat), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(droprate)),
            nn.Linear(int(hidden_dim), int(n_classes)),
        )

    def forward(self, x: torch.Tensor):
        bag = _as_bag_batch(x)
        logits = self.network(bag.mean(dim=1))
        return logits, None


class LocalGraphConv(nn.Module):
    """A lightweight graph convolution over adjacent patch nodes.

    The current dataset exposes aligned patch features but not coordinates to
    the model.  We therefore use the deterministic patch order as a chain
    graph and aggregate each node with itself and its immediate neighbours.
    This keeps the GNN option linear in the number of patches and avoids a
    dense pairwise graph construction for large WSI bags.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(int(in_dim), int(out_dim))
        self.neighbor_proj = nn.Linear(int(in_dim), int(out_dim))
        self.norm = nn.LayerNorm(int(out_dim))

    @staticmethod
    def _chain_neighbor_mean(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        neighbour_sum = x.clone()
        counts = torch.ones(
            (x.shape[0], x.shape[1], 1),
            dtype=x.dtype,
            device=x.device,
        )
        neighbour_sum[:, 1:] += x[:, :-1]
        counts[:, 1:] += 1
        neighbour_sum[:, :-1] += x[:, 1:]
        counts[:, :-1] += 1
        return neighbour_sum / counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neighbours = self._chain_neighbor_mean(x)
        h = self.self_proj(x) + self.neighbor_proj(neighbours)
        return self.norm(F.gelu(h))


class GNNHead(nn.Module):
    """Patch-graph convolution followed by mean graph pooling and classification."""

    def __init__(
        self,
        d_feat: int,
        hidden_dim: int,
        n_classes: int,
        n_layers: int,
        droprate: float,
    ) -> None:
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError("gnn_layers must be at least 1")
        self.input_norm = nn.LayerNorm(int(d_feat))
        layers = []
        for layer_idx in range(int(n_layers)):
            layers.append(
                LocalGraphConv(
                    int(d_feat) if layer_idx == 0 else int(hidden_dim),
                    int(hidden_dim),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(float(droprate))
        self.classifier = nn.Linear(int(hidden_dim), int(n_classes))

    def forward(self, x: torch.Tensor):
        h = self.input_norm(_as_bag_batch(x))
        for layer in self.layers:
            h = self.dropout(layer(h))
        logits = self.classifier(h.mean(dim=1))
        return logits, None


def build_downstream_head(
    name: str,
    d_feat: int,
    d_inner: int,
    d_attn: int,
    n_classes: int,
    droprate: float,
    mlp_hidden_dim: int = 256,
    gnn_hidden_dim: int = 256,
    gnn_layers: int = 2,
) -> nn.Module:
    """Construct a downstream head using the common GME head interface."""
    canonical_name = str(name).strip().lower()
    if canonical_name == "abmil":
        return ABMIL_Cls(
            D_feat=int(d_feat),
            D_inner=int(d_inner),
            D_attn=int(d_attn),
            n_classes=int(n_classes),
            droprate=float(droprate),
        )
    if canonical_name == "transmil":
        return TransMILHead(d_feat=int(d_feat), d_inner=int(d_inner), n_classes=int(n_classes))
    if canonical_name == "gnn":
        return GNNHead(
            d_feat=int(d_feat),
            hidden_dim=int(gnn_hidden_dim),
            n_classes=int(n_classes),
            n_layers=int(gnn_layers),
            droprate=float(droprate),
        )
    if canonical_name == "mlp":
        return MLPHead(
            d_feat=int(d_feat),
            hidden_dim=int(mlp_hidden_dim),
            n_classes=int(n_classes),
            droprate=float(droprate),
        )
    valid = ", ".join(HEAD_NAMES)
    raise ValueError(f"Unsupported downstream_head={name!r}. Choose one of: {valid}.")
