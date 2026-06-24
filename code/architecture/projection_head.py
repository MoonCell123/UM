"""Projection heads for multi-encoder middle-fusion models."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


def initialize_projection_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


class ProjectionHead(nn.Module):
    """Learnable feature projection optimized by the downstream task loss.

    Maps patch embeddings from one foundation-model encoder into a shared
    target dimension:

        h_{i,m} = LayerNorm(W_m e_{i,m} + b_m)
    """

    def __init__(self, input_dim: int, target_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, target_dim), nn.LayerNorm(target_dim)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.proj = nn.Sequential(*layers)
        self.apply(initialize_projection_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiEncoderProjectionHead(nn.Module):
    """One ProjectionHead per foundation-model encoder."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        target_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.heads = nn.ModuleDict({
            encoder_name: ProjectionHead(input_dim, target_dim, dropout)
            for encoder_name, input_dim in input_dims.items()
        })

    def forward(self, features_by_encoder: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        for encoder_name, features in features_by_encoder.items():
            if encoder_name not in self.heads:
                raise KeyError(f"Unknown encoder: {encoder_name}. Known: {list(self.heads.keys())}")
            projected[encoder_name] = self.heads[encoder_name](features)
        return projected
