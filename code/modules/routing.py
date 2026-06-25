"""Routing weights and dual-consistency fusion for CGME/GME middle fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


TensorDict = Mapping[str, torch.Tensor]


@dataclass
class RoutingOutput:
    """Container returned by DualConsistencyRouter.forward."""

    fused: torch.Tensor
    weights: torch.Tensor
    adaptive_fused: torch.Tensor
    mean_fused: torch.Tensor
    combined_scores: torch.Tensor
    normalized_attribution: torch.Tensor
    normalized_similarity: torch.Tensor
    lambda_similarity: torch.Tensor
    lambda_attribution: torch.Tensor
    gamma: torch.Tensor
    encoder_names: Sequence[str]


def _logit_from_probability(value: float, eps: float = 1e-6) -> float:
    value = min(max(float(value), eps), 1.0 - eps)
    return float(torch.logit(torch.tensor(value)).item())


def zscore_over_encoders(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize scores over the encoder dimension, i.e. the last dimension."""
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True, unbiased=False)
    return (scores - mean) / std.clamp_min(eps)


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
    gamma: torch.Tensor | float,
    encoder_names: Iterable[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Fuse projected encoder embeddings with adaptive and mean branches."""
    stacked_features, names = stack_feature_dict(features_by_encoder, encoder_names)
    expanded_weights = _expand_weights_for_stacked_features(weights, stacked_features)

    adaptive_fused = torch.sum(expanded_weights * stacked_features, dim=-2)
    mean_fused = stacked_features.mean(dim=-2)
    gamma_tensor = torch.as_tensor(gamma, device=stacked_features.device, dtype=stacked_features.dtype)
    fused = (1.0 - gamma_tensor) * adaptive_fused + gamma_tensor * mean_fused
    return fused, adaptive_fused, mean_fused, names


class DualConsistencyRouter(nn.Module):
    """Combine attribution/similarity scores and fuse multi-encoder embeddings."""

    def __init__(
        self,
        score_temperature: float = 1.0,
        theta_init: float = 0.0,
        gamma_init: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if score_temperature <= 0:
            raise ValueError("score_temperature must be positive.")
        self.score_temperature = float(score_temperature)
        self.eps = float(eps)
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))
        self.gamma_logit = nn.Parameter(torch.tensor(_logit_from_probability(gamma_init)))

    @property
    def lambda_similarity(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    @property
    def lambda_attribution(self) -> torch.Tensor:
        return 1.0 - self.lambda_similarity

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_logit)

    def get_routing_stats(self) -> Dict[str, float]:
        """Return scalar routing parameters for logging/visualization."""
        lambda_similarity = self.lambda_similarity.detach().cpu()
        lambda_attribution = self.lambda_attribution.detach().cpu()
        gamma = self.gamma.detach().cpu()
        theta = self.theta.detach().cpu()
        gamma_logit = self.gamma_logit.detach().cpu()
        return {
            "theta": float(theta.item()),
            "lambda_similarity": float(lambda_similarity.item()),
            "lambda_attribution": float(lambda_attribution.item()),
            "gamma_logit": float(gamma_logit.item()),
            "gamma": float(gamma.item()),
            "score_temperature": float(self.score_temperature),
        }

    def compute_weights(
        self,
        attribution_scores: torch.Tensor,
        similarity_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute alpha from attribution and Beacon-similarity scores."""
        if attribution_scores.shape != similarity_scores.shape:
            raise ValueError(
                "Attribution and similarity scores must have identical shapes. "
                f"Got {tuple(attribution_scores.shape)} and {tuple(similarity_scores.shape)}."
            )

        norm_attr = zscore_over_encoders(attribution_scores, eps=self.eps)
        norm_sim = zscore_over_encoders(similarity_scores, eps=self.eps)
        combined = self.lambda_attribution * norm_attr + self.lambda_similarity * norm_sim
        weights = torch.softmax(combined / self.score_temperature, dim=-1)
        return weights, combined, norm_attr, norm_sim

    def forward(
        self,
        features_by_encoder: TensorDict,
        attribution_scores: torch.Tensor | Mapping[str, torch.Tensor],
        similarity_scores: torch.Tensor | Mapping[str, torch.Tensor],
        encoder_names: Iterable[str] | None = None,
    ) -> RoutingOutput:
        """Compute routing weights and dual-consistency fused representation."""
        stacked_features, names = stack_feature_dict(features_by_encoder, encoder_names)
        del stacked_features

        attr = ensure_score_tensor(attribution_scores, names).to(
            device=next(iter(features_by_encoder.values())).device
        )
        sim = ensure_score_tensor(similarity_scores, names).to(attr.device)
        weights, combined, norm_attr, norm_sim = self.compute_weights(attr, sim)
        fused, adaptive_fused, mean_fused, names = dual_consistency_fusion(
            features_by_encoder=features_by_encoder,
            weights=weights,
            gamma=self.gamma,
            encoder_names=names,
        )

        return RoutingOutput(
            fused=fused,
            weights=weights,
            adaptive_fused=adaptive_fused,
            mean_fused=mean_fused,
            combined_scores=combined,
            normalized_attribution=norm_attr,
            normalized_similarity=norm_sim,
            lambda_similarity=self.lambda_similarity,
            lambda_attribution=self.lambda_attribution,
            gamma=self.gamma,
            encoder_names=names,
        )
