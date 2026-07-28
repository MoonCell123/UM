"""GME network, intervention attribution, and joint representation."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.abmil_cls import ABMIL_Cls
from architecture.projection_head import MultiEncoderProjectionHead
from modules.attribution import replace_encoder_embedding
from modules.routing import DualConsistencyRouter


def class_logit_margin(logits: torch.Tensor, class_index: int) -> torch.Tensor:
    """Return one-vs-rest logit margin for the selected class."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(f"Expected logits shaped [batch, classes>=2], got {tuple(logits.shape)}")
    if not 0 <= class_index < logits.shape[1]:
        raise IndexError(f"class_index={class_index} is out of range for {logits.shape[1]} classes")
    other_logits = torch.cat(
        [logits[:, :class_index], logits[:, class_index + 1:]],
        dim=1,
    )
    return logits[:, class_index] - torch.logsumexp(other_logits, dim=1)


class GMEModel(nn.Module):
    """ProjectionHead + attribution router + ABMIL classifier."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        target_dim: int = 512,
        projection_dropout: float = 0.0,
        d_inner: int = 256,
        d_attn: int = 128,
        n_classes: int = 2,
        droprate: float = 0.25,
        routing_temperature: float = 0.5,
        routing_logit_scale: float = 1.0,
        attribution_target: str = "predicted_class",
        interaction_pair_beta: float = 0.1,
        interaction_rms_clip: float = 3.0,
    ):
        super().__init__()
        self.encoder_names = sorted(input_dims.keys())
        ordered_dims = {name: int(input_dims[name]) for name in self.encoder_names}
        self.projection_heads = MultiEncoderProjectionHead(
            input_dims=ordered_dims,
            target_dim=target_dim,
            dropout=projection_dropout,
        )
        self.router = DualConsistencyRouter(
            routing_temperature=routing_temperature,
            routing_logit_scale=routing_logit_scale,
        )
        self.classifier = ABMIL_Cls(
            D_feat=target_dim,
            D_inner=d_inner,
            D_attn=d_attn,
            n_classes=n_classes,
            droprate=droprate,
        )
        self.target_dim = int(target_dim)
        if attribution_target not in {"predicted_class", "class_1"}:
            raise ValueError(
                "attribution_target must be 'predicted_class' or 'class_1', "
                f"got {attribution_target!r}"
            )
        if attribution_target == "class_1" and n_classes != 2:
            raise ValueError("attribution_target='class_1' requires binary classification.")
        self.attribution_target = attribution_target
        if n_classes != 2:
            raise ValueError("Interaction pairwise fusion currently requires binary classification.")
        self.interaction_pair_beta = float(interaction_pair_beta)
        self.interaction_rms_clip = float(interaction_rms_clip)
        self.interaction_pairs = [
            (i, j)
            for i in range(len(self.encoder_names))
            for j in range(i + 1, len(self.encoder_names))
        ]
        # One learnable value per projected feature dimension. Zero
        # initialization starts exactly at the attribution-only baseline.
        self.interaction_gate = nn.Parameter(torch.zeros(self.target_dim))
        self.register_buffer("interaction_mean", torch.zeros(len(self.interaction_pairs)))
        self.register_buffer("interaction_std", torch.ones(len(self.interaction_pairs)))
        self.register_buffer("interaction_rms", torch.ones(len(self.interaction_pairs)))
        self.register_buffer("interaction_stats_count", torch.tensor(0, dtype=torch.long))

    def project(self, raw_features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.projection_heads(raw_features)

    def mean_fuse(self, projected: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([projected[name] for name in self.encoder_names], dim=0)
        return stacked.mean(dim=0)

    def route_with_scores(
        self,
        projected: Mapping[str, torch.Tensor],
        attribution_scores: torch.Tensor,
    ):
        return self.router(
            features_by_encoder=projected,
            attribution_scores=attribution_scores,
            encoder_names=self.encoder_names,
        )

    def beacon_constraint_loss(
        self,
        projected: Mapping[str, torch.Tensor],
        beacon: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Constrain encoder-level slide means toward a detached global Beacon.

        Beacon is used only as a structural/global semantic prior here. It does
        not enter routing weights or classifier features.
        """
        first = next(iter(projected.values()))
        beacon = beacon.to(device=first.device, dtype=first.dtype).reshape(-1)
        if beacon.shape[0] != self.target_dim:
            raise ValueError(f"Expected beacon dim={self.target_dim}, got {beacon.shape[0]}")
        beacon = beacon / beacon.norm(p=2).clamp_min(eps)

        losses = []
        similarities: Dict[str, torch.Tensor] = {}
        for name in self.encoder_names:
            h = projected[name].reshape(-1, projected[name].shape[-1])
            h = h / h.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            slide_mean = h.mean(dim=0)
            slide_mean = slide_mean / slide_mean.norm(p=2).clamp_min(eps)
            sim = torch.sum(slide_mean * beacon)
            losses.append(1.0 - sim)
            similarities[name] = sim.detach()

        return torch.stack(losses, dim=0).mean(), similarities

    def encoder_consistency_loss(
        self,
        projected: Mapping[str, torch.Tensor],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Align paired patch embeddings across encoders in the projected space."""
        stacked = torch.stack([projected[name] for name in self.encoder_names], dim=0)
        normalized = stacked / stacked.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        consensus = normalized.mean(dim=0)
        consensus = consensus / consensus.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        similarity = torch.sum(normalized * consensus.unsqueeze(0), dim=-1)
        return (1.0 - similarity).mean()

    def forward_mean_fusion(self, projected: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, attn = self.classifier(self.mean_fuse(projected))
        return logits, attn

    def intervention_attribution(
        self,
        projected: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = True,
        interaction_target_class: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return attribution routing scores and optional signed interactions.

        Every masked coalition is evaluated once and cached. Coalition scores
        either explain the class predicted by the complete encoder set or use
        a fixed class-1 margin, according to ``self.attribution_target``.
        Pairwise interactions never modify the routing score or encoder
        weights; the fixed class-1 version may modulate the separate joint
        representation branch.
        """
        classifier_was_training = self.classifier.training
        router_was_training = self.router.training
        self.classifier.eval()
        self.router.eval()
        try:
            with torch.no_grad():
                full_logits, _ = self.forward_mean_fusion(projected)
                attribution_target_class = (
                    1
                    if self.attribution_target == "class_1"
                    else int(full_logits.argmax(dim=1)[0].item())
                )
                coalition_cache: Dict[frozenset[str], torch.Tensor] = {
                    frozenset(): full_logits
                }

                def coalition_logits(masked_names: frozenset[str]) -> torch.Tensor:
                    cached = coalition_cache.get(masked_names)
                    if cached is not None:
                        return cached
                    replaced = dict(projected)
                    for name in sorted(masked_names):
                        replaced = replace_encoder_embedding(
                            replaced,
                            encoder_name=name,
                            replacement_strategy=replacement_strategy,  # type: ignore[arg-type]
                            baselines=baselines,
                            gaussian_std_scale=gaussian_std_scale,
                        )
                    masked_logits, _ = self.forward_mean_fusion(replaced)
                    coalition_cache[masked_names] = masked_logits
                    return masked_logits

                def coalition_score(masked_names: frozenset[str], class_index: int) -> torch.Tensor:
                    return class_logit_margin(coalition_logits(masked_names), class_index)[0]

                full_score = coalition_score(frozenset(), attribution_target_class)
                single_scores = {
                    name: coalition_score(frozenset({name}), attribution_target_class)
                    for name in self.encoder_names
                }
                single_contributions = torch.stack(
                    [full_score - single_scores[name] for name in self.encoder_names]
                )
                interactions = full_score.new_zeros((len(self.encoder_names), len(self.encoder_names)))
                if compute_interactions:
                    pair_target_class = (
                        attribution_target_class
                        if interaction_target_class is None
                        else int(interaction_target_class)
                    )
                    pair_full_score = coalition_score(frozenset(), pair_target_class)
                    pair_single_scores = {
                        name: coalition_score(frozenset({name}), pair_target_class)
                        for name in self.encoder_names
                    }
                    for i, left in enumerate(self.encoder_names):
                        for j in range(i + 1, len(self.encoder_names)):
                            right = self.encoder_names[j]
                            double_score = coalition_score(
                                frozenset({left, right}), pair_target_class
                            )
                            value = (
                                pair_full_score
                                - pair_single_scores[left]
                                - pair_single_scores[right]
                                + double_score
                            )
                            interactions[i, j] = value
                            interactions[j, i] = value
        finally:
            self.classifier.train(classifier_was_training)
            self.router.train(router_was_training)
        return single_contributions, single_contributions, interactions

    def flatten_interactions(self, interactions: torch.Tensor) -> torch.Tensor:
        """Flatten the signed upper triangle in deterministic encoder order."""
        return torch.stack([interactions[i, j] for i, j in self.interaction_pairs], dim=-1)

    def set_interaction_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        rms: torch.Tensor,
        count: int,
    ) -> None:
        expected = len(self.interaction_pairs)
        if mean.numel() != expected or std.numel() != expected or rms.numel() != expected:
            raise ValueError("Interaction statistics do not match the number of encoder pairs.")
        self.interaction_mean.copy_(mean.to(self.interaction_mean).reshape_as(self.interaction_mean))
        self.interaction_std.copy_(std.to(self.interaction_std).reshape_as(self.interaction_std).clamp_min(1e-8))
        self.interaction_rms.copy_(rms.to(self.interaction_rms).reshape_as(self.interaction_rms).clamp_min(1e-8))
        self.interaction_stats_count.copy_(
            torch.as_tensor(int(count), device=self.interaction_stats_count.device)
        )

    def scale_interaction_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Scale without centering so the original interaction sign is preserved."""
        scaled = vector / self.interaction_rms.to(vector).clamp_min(1e-8)
        return scaled.clamp(
            min=-self.interaction_rms_clip,
            max=self.interaction_rms_clip,
        )

    def interaction_pair_residual_from_vector(
        self,
        projected: Mapping[str, torch.Tensor],
        routing_weights: torch.Tensor,
        interaction_vector: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a signed, attribution-gated joint representation residual.

        All encoder tensors must be patch aligned. Pairwise Hadamard products
        stay in the common projection space; a zero-initialized per-dimension
        gate selects stable joint features. Division by Z_q prevents growth
        with the number of encoder pairs.
        """
        tensors = [projected[name] for name in self.encoder_names]
        first_shape = tensors[0].shape
        for name, tensor in zip(self.encoder_names, tensors):
            if tensor.shape != first_shape:
                raise ValueError(
                    "Pairwise interaction fusion requires aligned projected tensors; "
                    f"expected {tuple(first_shape)}, got {tuple(tensor.shape)} for {name}."
                )
        weights = routing_weights.reshape(-1)
        if weights.numel() != len(self.encoder_names):
            raise ValueError(
                f"Expected {len(self.encoder_names)} routing weights, got {weights.numel()}."
            )
        vector = interaction_vector.reshape(-1)
        if vector.numel() != len(self.interaction_pairs):
            raise ValueError(
                f"Expected {len(self.interaction_pairs)} interactions, got {vector.numel()}."
            )

        scaled = self.scale_interaction_vector(vector)
        normalized = {
            name: F.layer_norm(projected[name], (self.target_dim,))
            for name in self.encoder_names
        }
        numerator = torch.zeros_like(tensors[0])
        z_q = weights.new_zeros(())
        for pair_index, (i, j) in enumerate(self.interaction_pairs):
            pair_gate = torch.sqrt((weights[i] * weights[j]).clamp_min(0.0))
            pair_feature = (
                normalized[self.encoder_names[i]]
                * normalized[self.encoder_names[j]]
            )
            numerator = numerator + pair_gate * scaled[pair_index] * pair_feature
            z_q = z_q + pair_gate
        pair_representation = numerator / z_q.clamp_min(eps)
        pair_residual = (
            self.interaction_pair_beta
            * torch.tanh(self.interaction_gate)
            * torch.tanh(pair_representation)
        )
        return pair_residual, scaled

    def interaction_pair_residual(
        self,
        projected: Mapping[str, torch.Tensor],
        routing_weights: torch.Tensor,
        interactions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.interaction_pair_residual_from_vector(
            projected,
            routing_weights,
            self.flatten_interactions(interactions),
        )

    def forward_stage2(
        self,
        raw_features: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = True,
        return_attribution_logits: bool = False,
    ):
        projected = self.project(raw_features)
        attr, single_attr, interactions = self.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=compute_interactions and self.interaction_pair_beta > 0.0,
            interaction_target_class=1,
        )
        routed = self.route_with_scores(projected, attribution_scores=attr)
        if compute_interactions and self.interaction_pair_beta > 0.0:
            pair_residual, scaled_interactions = self.interaction_pair_residual(
                projected,
                routed.weights,
                interactions,
            )
            final_fused = routed.fused + pair_residual
        else:
            pair_residual = torch.zeros_like(routed.fused)
            scaled_interactions = routed.fused.new_zeros(len(self.interaction_pairs))
            final_fused = routed.fused
            interactions = routed.fused.new_zeros(
                (len(self.encoder_names), len(self.encoder_names))
            )
        logits, attn = self.classifier(final_fused)
        if return_attribution_logits and self.interaction_pair_beta > 0.0:
            attribution_logits, _ = self.classifier(routed.fused)
        else:
            attribution_logits = logits.detach()
        return (
            logits,
            attn,
            routed,
            projected,
            attr,
            single_attr,
            interactions,
            attribution_logits,
            pair_residual,
            scaled_interactions,
        )


