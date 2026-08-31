"""GME network and intervention attribution."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn

from architecture.gme_heads import build_downstream_head
from architecture.projection_head import MultiEncoderProjectionHead
from modules.attribution import replace_encoder_embedding
from modules.routing import DualConsistencyRouter, EmbeddingStudentRouter


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
    """ProjectionHead + attribution router + configurable downstream head."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        target_dim: int = 512,
        projection_dropout: float = 0.0,
        d_inner: int = 256,
        d_attn: int = 128,
        n_classes: int = 2,
        droprate: float = 0.25,
        downstream_head: str = "ABMIL",
        mlp_hidden_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_layers: int = 2,
        routing_temperature: float = 0.5,
        routing_logit_scale: float = 1.0,
        attribution_target: str = "predicted_class",
        student_router_hidden_dim: int = 0,
        student_router_temperature: float = 1.0,
        student_router_use_consensus: bool = True,
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
        self.student_router = (
            EmbeddingStudentRouter(
                encoder_names=self.encoder_names,
                feature_dim=target_dim,
                hidden_dim=student_router_hidden_dim,
                temperature=student_router_temperature,
                use_consensus=student_router_use_consensus,
            )
            if student_router_hidden_dim > 0
            else None
        )
        self.downstream_head = str(downstream_head).strip().upper()
        self.classifier = build_downstream_head(
            name=self.downstream_head,
            d_feat=target_dim,
            d_inner=d_inner,
            d_attn=d_attn,
            n_classes=n_classes,
            droprate=droprate,
            mlp_hidden_dim=mlp_hidden_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
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
        self.interaction_pairs = [
            (i, j)
            for i in range(len(self.encoder_names))
            for j in range(i + 1, len(self.encoder_names))
        ]
        if n_classes != 2 and attribution_target == "class_1":
            raise ValueError("attribution_target='class_1' requires binary classification.")

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

    def route_with_student(self, projected: Mapping[str, torch.Tensor]):
        if self.student_router is None:
            raise RuntimeError("Student routing was requested but no student_router is configured.")
        return self.student_router(projected)

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

    def forward_mean_fusion(
        self, projected: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        logits, attn = self.classifier(self.mean_fuse(projected))
        return logits, attn

    def intervention_attribution(
        self,
        projected: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = False,
        interaction_target_class: int | None = None,
        target_class: int | None = None,
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
                    int(target_class)
                    if target_class is not None
                    else (
                        1
                        if self.attribution_target == "class_1"
                        else int(full_logits.argmax(dim=1)[0].item())
                    )
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

    def forward_stage2(
        self,
        raw_features: Mapping[str, torch.Tensor],
        baselines: Mapping[str, Mapping[str, torch.Tensor]],
        replacement_strategy: str = "mean",
        gaussian_std_scale: float = 1.0,
        compute_interactions: bool = False,
        return_attribution_logits: bool = False,
        use_student_router: bool = False,
    ):
        projected = self.project(raw_features)
        if use_student_router:
            routed = self.route_with_student(projected)
            logits, attn = self.classifier(routed.fused)
            zeros = logits.new_zeros(len(self.encoder_names))
            return (
                logits,
                attn,
                routed,
                projected,
                routed.combined_scores,
                zeros,
                logits.new_zeros((len(self.encoder_names), len(self.encoder_names))),
                logits.detach(),
                torch.zeros_like(routed.fused),
                logits.new_zeros(len(self.interaction_pairs)),
            )
        attr, single_attr, interactions = self.intervention_attribution(
            projected=projected,
            baselines=baselines,
            replacement_strategy=replacement_strategy,
            gaussian_std_scale=gaussian_std_scale,
            compute_interactions=compute_interactions,
            interaction_target_class=1,
        )
        routed = self.route_with_scores(projected, attribution_scores=attr)
        # Interactions are a post-hoc analysis quantity. They are never fused
        # into the representation or exposed to the optimizer/inference path.
        logits, attn = self.classifier(routed.fused)
        if not compute_interactions:
            interactions = logits.new_zeros((len(self.encoder_names), len(self.encoder_names)))
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
            torch.zeros_like(routed.fused),
            logits.new_zeros(len(self.interaction_pairs)),
        )


