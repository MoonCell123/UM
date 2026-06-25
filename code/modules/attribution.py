"""
Intervention-based attribution for multi-encoder middle fusion.

This module is intentionally model-agnostic: pass a forward function that
accepts the same feature mapping used by the fusion model.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Literal, Mapping, Sequence

import torch


TensorDict = Mapping[str, torch.Tensor]
MutableTensorDict = Dict[str, torch.Tensor]
ReplacementStrategy = Literal["zero", "mean", "gaussian"]


class EncoderBaselineAccumulator:
    """Streaming train-set baseline statistics for intervention replacement."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self._sum: Dict[str, torch.Tensor] = {}
        self._sum_sq: Dict[str, torch.Tensor] = {}
        self._count: Dict[str, int] = {}

    @torch.no_grad()
    def update(self, encoder_name: str, embeddings: torch.Tensor) -> None:
        """Accumulate projected embeddings with shape [..., D]."""
        x = embeddings.detach().reshape(-1, embeddings.shape[-1]).float().cpu()
        if encoder_name not in self._sum:
            self._sum[encoder_name] = torch.zeros(x.shape[-1], dtype=torch.float32)
            self._sum_sq[encoder_name] = torch.zeros(x.shape[-1], dtype=torch.float32)
            self._count[encoder_name] = 0
        self._sum[encoder_name] += x.sum(dim=0)
        self._sum_sq[encoder_name] += (x * x).sum(dim=0)
        self._count[encoder_name] += int(x.shape[0])

    def compute(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return {encoder_name: {'mean': [D], 'std': [D], 'count': scalar}}."""
        baselines: Dict[str, Dict[str, torch.Tensor]] = {}
        for encoder_name, count in self._count.items():
            if count <= 0:
                raise ValueError(f"{encoder_name}: cannot compute baseline with count={count}")
            mean = self._sum[encoder_name] / count
            second_moment = self._sum_sq[encoder_name] / count
            var = (second_moment - mean * mean).clamp_min(self.eps)
            baselines[encoder_name] = {
                "mean": mean,
                "std": torch.sqrt(var),
                "count": torch.tensor(count),
            }
        return baselines


def _as_tensor_output(output: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Use the first tensor when a model returns (logits, aux...)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Sequence) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Expected tensor output or tuple/list with tensor first element, got {type(output)}")


def select_prediction_score(
    output: torch.Tensor | Sequence[torch.Tensor],
    class_index: int | None = None,
    use_probability: bool = False,
) -> torch.Tensor:
    """Convert model output to one scalar score per sample."""
    logits = _as_tensor_output(output)
    if logits.ndim == 0:
        logits = logits.reshape(1)
    if logits.ndim == 1:
        score = torch.sigmoid(logits) if use_probability else logits
        return score
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B], [B,1], or [B,C], got {tuple(logits.shape)}")

    if logits.shape[1] == 1:
        values = logits[:, 0]
        return torch.sigmoid(values) if use_probability else values

    idx = class_index if class_index is not None else min(1, logits.shape[1] - 1)
    if use_probability:
        return torch.softmax(logits, dim=1)[:, idx]
    return logits[:, idx]


def _broadcast_baseline(baseline: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a baseline vector/tensor to the target embedding shape."""
    baseline = baseline.to(device=target.device, dtype=target.dtype)
    if baseline.shape == target.shape:
        return baseline
    if baseline.ndim == 1 and baseline.shape[0] == target.shape[-1]:
        view_shape = [1] * target.ndim
        view_shape[-1] = baseline.shape[0]
        return baseline.reshape(view_shape).expand_as(target)
    try:
        return baseline.expand_as(target)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot broadcast baseline shape {tuple(baseline.shape)} "
            f"to target shape {tuple(target.shape)}"
        ) from exc


def replace_encoder_embedding(
    features_by_encoder: TensorDict,
    encoder_name: str,
    replacement_strategy: ReplacementStrategy = "mean",
    baselines: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
    mask_value: float = 0.0,
    gaussian_std_scale: float = 1.0,
) -> MutableTensorDict:
    """Return a feature dict with one encoder embedding intervention.
    """
    if encoder_name not in features_by_encoder:
        raise KeyError(f"Unknown encoder: {encoder_name}. Known: {list(features_by_encoder.keys())}")

    target = features_by_encoder[encoder_name]
    replaced = dict(features_by_encoder)

    if replacement_strategy == "zero":
        replaced[encoder_name] = torch.full_like(target, fill_value=mask_value)
        return replaced

    if baselines is None or encoder_name not in baselines:
        raise ValueError(
            f"{replacement_strategy} replacement requires baselines['{encoder_name}']. "
            "Use train-set projected embeddings to compute these baselines."
        )

    baseline = baselines[encoder_name]
    if replacement_strategy == "mean":
        if isinstance(baseline, Mapping):
            if "mean" not in baseline:
                raise KeyError(f"baselines['{encoder_name}'] must contain key 'mean'.")
            baseline_tensor = baseline["mean"]
        else:
            baseline_tensor = baseline
        replaced[encoder_name] = _broadcast_baseline(baseline_tensor, target)
        return replaced

    if replacement_strategy == "gaussian":
        if not isinstance(baseline, Mapping) or "mean" not in baseline or "std" not in baseline:
            raise KeyError(
                f"Gaussian replacement requires baselines['{encoder_name}'] "
                "with keys 'mean' and 'std'."
            )
        mean = _broadcast_baseline(baseline["mean"], target)
        std = _broadcast_baseline(baseline["std"], target).clamp_min(1e-6)
        replaced[encoder_name] = mean + torch.randn_like(target) * std * gaussian_std_scale
        return replaced

    raise ValueError(f"Unknown replacement_strategy: {replacement_strategy}")


def mask_encoder_embedding(
    features_by_encoder: TensorDict,
    encoder_name: str,
    mask_value: float = 0.0,
) -> MutableTensorDict:
    """Backward-compatible zero-mask wrapper."""
    return replace_encoder_embedding(
        features_by_encoder=features_by_encoder,
        encoder_name=encoder_name,
        replacement_strategy="zero",
        mask_value=mask_value,
    )


def compute_intervention_attribution(
    forward_fn: Callable[[TensorDict], torch.Tensor | Sequence[torch.Tensor]],
    features_by_encoder: TensorDict,
    encoder_names: Iterable[str] | None = None,
    class_index: int | None = None,
    use_probability: bool = False,
    replacement_strategy: ReplacementStrategy = "mean",
    baselines: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
    mask_value: float = 0.0,
    gaussian_std_scale: float = 1.0,
    detach: bool = True,
) -> torch.Tensor:
    """Compute I_{i,m}=f(x_i)-f(x_i^{(-m)}) for each selected encoder.
    """
    names = list(encoder_names) if encoder_names is not None else list(features_by_encoder.keys())
    if not names:
        raise ValueError("No encoder names provided.")

    context = torch.no_grad() if detach else torch.enable_grad()
    with context:
        full_score = select_prediction_score(
            forward_fn(features_by_encoder),
            class_index=class_index,
            use_probability=use_probability,
        )
        values = []
        for encoder_name in names:
            masked_features = replace_encoder_embedding(
                features_by_encoder,
                encoder_name=encoder_name,
                replacement_strategy=replacement_strategy,
                baselines=baselines,
                mask_value=mask_value,
                gaussian_std_scale=gaussian_std_scale,
            )
            masked_score = select_prediction_score(
                forward_fn(masked_features),
                class_index=class_index,
                use_probability=use_probability,
            )
            values.append(full_score - masked_score)

        attribution = torch.stack(values, dim=-1)
        return attribution.detach() if detach else attribution


def normalize_attribution_per_sample(attribution: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalize attribution over encoders for each sample."""
    mean = attribution.mean(dim=-1, keepdim=True)
    std = attribution.std(dim=-1, keepdim=True, unbiased=False)
    return (attribution - mean) / std.clamp_min(eps)
