"""Attribution-only routing weights and fusion for GME middle fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import math


TensorDict = Mapping[str, torch.Tensor]


@dataclass
class RoutingOutput:
    """Container returned by DualConsistencyRouter.forward."""

    fused: torch.Tensor
    weights: torch.Tensor
    adaptive_fused: torch.Tensor
    combined_scores: torch.Tensor
    normalized_attribution: torch.Tensor
    attribution_range: torch.Tensor
    tau: torch.Tensor
    encoder_names: Sequence[str]


def minmax_with_global_stats(
    scores: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-max normalize scores with train-set-level scalar statistics."""
    minimum = minimum.to(device=scores.device, dtype=scores.dtype)
    maximum = maximum.to(device=scores.device, dtype=scores.dtype)
    return (scores - minimum) / (maximum - minimum).clamp_min(eps)


def stack_feature_dict(
    features_by_encoder: TensorDict,
    encoder_names: Iterable[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Stack encoder features into shape [..., M, D].

    Each encoder tensor must have the same shape [..., D]. The new encoder
    dimension is inserted before the feature dimension.
    """
    names = list(encoder_names) if encoder_names is not None else list(features_by_encoder.keys())
    if not names:
        raise ValueError("No encoder names provided.")

    missing = [name for name in names if name not in features_by_encoder]
    if missing:
        raise KeyError(f"Missing encoder features: {missing}")

    tensors = [features_by_encoder[name] for name in names]
    first_shape = tensors[0].shape
    for name, tensor in zip(names, tensors):
        if tensor.shape != first_shape:
            raise ValueError(
                "All encoder feature tensors must have the same shape after projection. "
                f"Expected {tuple(first_shape)}, got {tuple(tensor.shape)} for {name}."
            )
    return torch.stack(tensors, dim=-2), names


def stack_score_dict(
    scores_by_encoder: Mapping[str, torch.Tensor],
    encoder_names: Sequence[str],
) -> torch.Tensor:
    """Stack score tensors into shape [..., M] using encoder order."""
    missing = [name for name in encoder_names if name not in scores_by_encoder]
    if missing:
        raise KeyError(f"Missing encoder scores: {missing}")

    tensors = [scores_by_encoder[name] for name in encoder_names]
    first_shape = tensors[0].shape
    for name, tensor in zip(encoder_names, tensors):
        if tensor.shape != first_shape:
            raise ValueError(
                "All score tensors must have the same shape. "
                f"Expected {tuple(first_shape)}, got {tuple(tensor.shape)} for {name}."
            )
    return torch.stack(tensors, dim=-1)


def ensure_score_tensor(
    scores: torch.Tensor | Mapping[str, torch.Tensor],
    encoder_names: Sequence[str],
) -> torch.Tensor:
    """Accept either already-stacked [..., M] scores or encoder score dicts."""
    if isinstance(scores, torch.Tensor):
        if scores.shape[-1] != len(encoder_names):
            raise ValueError(
                f"Expected score last dim={len(encoder_names)}, got {scores.shape[-1]}"
            )
        return scores
    return stack_score_dict(scores, encoder_names)


def _expand_weights_for_stacked_features(weights: torch.Tensor, stacked_features: torch.Tensor) -> torch.Tensor:
    """Broadcast weights to match stacked feature shape [..., M, D]."""
    if weights.shape[-1] != stacked_features.shape[-2]:
        raise ValueError(
            f"Weight encoder dim={weights.shape[-1]} does not match feature encoder dim={stacked_features.shape[-2]}"
        )

    feature_prefix = stacked_features.shape[:-2]
    weight_prefix = weights.shape[:-1]

    if weight_prefix == feature_prefix:
        expanded = weights
    elif len(weight_prefix) <= len(feature_prefix) and weight_prefix == feature_prefix[:len(weight_prefix)]:
        view_shape = list(weight_prefix) + [1] * (len(feature_prefix) - len(weight_prefix)) + [weights.shape[-1]]
        expanded = weights.reshape(view_shape)
    else:
        raise ValueError(
            f"Cannot broadcast weight shape {tuple(weights.shape)} to feature shape {tuple(stacked_features.shape)}"
        )

    return expanded.unsqueeze(-1)


def dual_consistency_fusion(
    features_by_encoder: TensorDict,
    weights: torch.Tensor,
    encoder_names: Iterable[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Fuse projected encoder embeddings with attribution-derived weights."""
    stacked_features, names = stack_feature_dict(features_by_encoder, encoder_names)
    expanded_weights = _expand_weights_for_stacked_features(weights, stacked_features)

    adaptive_fused = torch.sum(expanded_weights * stacked_features, dim=-2)
    return adaptive_fused, adaptive_fused, names


class DualConsistencyRouter(nn.Module):
    """Fuse multi-encoder embeddings using attribution-only routing scores."""

    def __init__(
        self,
        routing_temperature: float = 0.5,
        routing_logit_scale: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive.")
        self.routing_temperature = float(routing_temperature)
        self.routing_logit_scale = float(routing_logit_scale)
        self.eps = float(eps)
        self.register_buffer("attribution_min", torch.tensor(0.0))
        self.register_buffer("attribution_max", torch.tensor(1.0))
        self.register_buffer("score_stats_count", torch.tensor(0, dtype=torch.long))

    def set_score_stats(
        self,
        attribution_min: torch.Tensor | float,
        attribution_max: torch.Tensor | float,
        count: int,
    ) -> None:
        """Store train-set-level statistics used to normalize routing scores."""
        device = self.attribution_min.device
        self.attribution_min.copy_(torch.as_tensor(attribution_min, device=device, dtype=self.attribution_min.dtype))
        self.attribution_max.copy_(torch.as_tensor(attribution_max, device=device, dtype=self.attribution_max.dtype))
        self.score_stats_count.copy_(torch.as_tensor(int(count), device=device, dtype=self.score_stats_count.dtype))

    def get_score_stats(self) -> Dict[str, float]:
        """Return train-set-level routing score normalization statistics."""
        return {
            "attribution_min": float(self.attribution_min.detach().cpu().item()),
            "attribution_max": float(self.attribution_max.detach().cpu().item()),
            "count": int(self.score_stats_count.detach().cpu().item()),
        }

    def get_routing_stats(self) -> Dict[str, float]:
        """Return scalar routing parameters for logging/visualization."""
        stats = self.get_score_stats()
        stats["routing_temperature"] = float(self.routing_temperature)
        stats["routing_logit_scale"] = float(self.routing_logit_scale)
        return stats

    def compute_weights(
        self,
        attribution_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights from attribution logits.
        """
        norm_attr = minmax_with_global_stats(
            attribution_scores,
            minimum=self.attribution_min,
            maximum=self.attribution_max,
            eps=self.eps,
        ).clamp(0.0, 1.0)
        c_range = (norm_attr.amax(dim=-1) - norm_attr.amin(dim=-1)).clamp(0.0, 1.0)
        tau_min = 0.3
        tau_max = 0.6
        # inverse num of tau
        tau = torch.clamp_min(tau_max / (1.0 + c_range), tau_min)
        combined = norm_attr / tau.unsqueeze(-1)
        weights = torch.softmax(combined, dim=-1)
        return weights, combined, norm_attr, c_range, tau

    def forward(
        self,
        features_by_encoder: TensorDict,
        attribution_scores: torch.Tensor | Mapping[str, torch.Tensor],
        encoder_names: Iterable[str] | None = None,
    ) -> RoutingOutput:
        """Compute routing weights and fused representation."""
        stacked_features, names = stack_feature_dict(features_by_encoder, encoder_names)
        del stacked_features

        attr = ensure_score_tensor(attribution_scores, names).to(
            device=next(iter(features_by_encoder.values())).device
        )
        weights, combined, norm_attr, c_range, tau = self.compute_weights(attr)
        fused, adaptive_fused, names = dual_consistency_fusion(
            features_by_encoder=features_by_encoder,
            weights=weights,
            encoder_names=names,
        )

        return RoutingOutput(
            fused=fused,
            weights=weights,
            adaptive_fused=adaptive_fused,
            combined_scores=combined,
            normalized_attribution=norm_attr,
            attribution_range=c_range,
            tau=tau,
            encoder_names=names,
        )


class EmbeddingStudentRouter(nn.Module):
    """Predict encoder fusion weights directly from projected slide embeddings.

    This router is intended to be supervised by frozen teacher-attribution
    targets during training.  It never consumes labels or intervention scores
    at inference time.
    """

    def __init__(
        self,
        encoder_names: Sequence[str],
        feature_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not encoder_names:
            raise ValueError("EmbeddingStudentRouter requires at least one encoder.")
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.encoder_names = list(encoder_names)
        self.temperature = float(temperature)
        self.score_net = nn.Sequential(
            nn.LayerNorm(2 * int(feature_dim)),
            nn.Linear(2 * int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.encoder_bias = nn.Parameter(torch.zeros(len(self.encoder_names)))

    def _slide_means(self, features_by_encoder: TensorDict) -> torch.Tensor:
        means = []
        for name in self.encoder_names:
            if name not in features_by_encoder:
                raise KeyError(f"Missing encoder features: {name}")
            features = features_by_encoder[name]
            means.append(features.reshape(-1, features.shape[-1]).mean(dim=0))
        return torch.stack(means, dim=0)

    def forward(self, features_by_encoder: TensorDict) -> RoutingOutput:
        means = self._slide_means(features_by_encoder)
        consensus = means.mean(dim=0, keepdim=True).expand_as(means)
        scores = self.score_net(torch.cat([means, consensus], dim=-1)).squeeze(-1)
        scores = scores + self.encoder_bias
        weights = torch.softmax(scores / self.temperature, dim=-1)
        fused, adaptive_fused, names = dual_consistency_fusion(
            features_by_encoder=features_by_encoder,
            weights=weights,
            encoder_names=self.encoder_names,
        )
        score_range = (scores.max() - scores.min()).unsqueeze(0)
        return RoutingOutput(
            fused=fused,
            weights=weights,
            adaptive_fused=adaptive_fused,
            combined_scores=scores,
            normalized_attribution=weights,
            attribution_range=score_range,
            tau=scores.new_tensor([self.temperature]),
            encoder_names=names,
        )
