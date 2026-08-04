"""Efficiency profiling helpers for GME models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from architecture.gme_model import GMEModel
from data_utils.gme_dataset import MultiEncoderSlideDataset


def move_features(features: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.float().to(device) for name, tensor in features.items()}


class GMEProfileWrapper(nn.Module):
    """Traceable online inference with attribution supplied offline."""

    def __init__(
        self,
        model: GMEModel,
        encoder_names: Sequence[str],
        use_student_router: bool = False,
    ):
        super().__init__()
        self.model = model
        self.encoder_names = list(encoder_names)
        self.use_student_router = bool(use_student_router)

    def forward(self, *feature_tensors: torch.Tensor) -> torch.Tensor:
        if self.use_student_router:
            raw_features = {
                name: tensor
                for name, tensor in zip(self.encoder_names, feature_tensors)
            }
            projected = self.model.project(raw_features)
            routed = self.model.route_with_student(projected)
            logits, _ = self.model.classifier(routed.fused)
            return logits
        attribution_scores = feature_tensors[-1]
        raw_features = {
            name: tensor for name, tensor in zip(self.encoder_names, feature_tensors[:-1])
        }
        projected = self.model.project(raw_features)
        routed = self.model.route_with_scores(projected, attribution_scores)
        logits, _ = self.model.classifier(routed.fused)
        return logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def cuda_synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def collect_profile_inputs(
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    max_samples: int,
) -> List[Dict[str, torch.Tensor]]:
    samples = []
    for idx in range(min(max(int(max_samples), 0), len(dataset))):
        raw_features, _, _ = dataset[idx]
        samples.append(move_features(raw_features, device))
    return samples


def profile_sample_tuples(
    samples: Sequence[Mapping[str, torch.Tensor]],
    encoder_names: Sequence[str],
    attribution_scores: Sequence[torch.Tensor] | None = None,
) -> List[Tuple[torch.Tensor, ...]]:
    feature_tuples = [
        tuple(raw_features[name] for name in encoder_names)
        for raw_features in samples
    ]
    if attribution_scores is None:
        return feature_tuples
    if len(feature_tuples) != len(attribution_scores):
        raise ValueError("Profile samples and offline routing inputs must have the same length.")
    return [
        features + (scores,)
        for features, scores in zip(feature_tuples, attribution_scores)
    ]


def estimate_fvcore_flops(wrapper: nn.Module, samples: Sequence[Tuple[torch.Tensor, ...]]) -> float:
    if not samples:
        return float("nan")
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[Warning] fvcore is not installed; FLOPs will be written as NaN.")
        return float("nan")

    values = []
    was_training = wrapper.training
    wrapper.eval()
    first_error = None
    try:
        for feature_tensors in samples:
            try:
                analysis = FlopCountAnalysis(wrapper, feature_tensors)
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                analysis.tracer_warnings("none")
                values.append(float(analysis.total()))
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                values.append(float("nan"))
    finally:
        wrapper.train(was_training)

    finite = [value for value in values if np.isfinite(value)]
    if not finite and first_error is not None:
        print(f"[Warning] fvcore FLOPs tracing failed; FLOPs will be written as NaN. First error: {first_error}")
    return float(np.mean(finite)) if finite else float("nan")


@torch.no_grad()
def measure_forward_time_seconds(
    wrapper: nn.Module,
    samples: Sequence[Tuple[torch.Tensor, ...]],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    if not samples or repeat <= 0:
        return float("nan")
    was_training = wrapper.training
    wrapper.eval()
    try:
        for _ in range(max(int(warmup), 0)):
            for feature_tensors in samples:
                wrapper(*feature_tensors)
        cuda_synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(int(repeat)):
            for feature_tensors in samples:
                wrapper(*feature_tensors)
        cuda_synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    finally:
        wrapper.train(was_training)
    return float(elapsed / (int(repeat) * len(samples)))


def profile_gme_efficiency(
    model: GMEModel,
    dataset: MultiEncoderSlideDataset,
    device: torch.device,
    baselines: Mapping[str, Mapping[str, torch.Tensor]],
    replacement_strategy: str,
    gaussian_std_scale: float,
    profile_samples: int,
    profile_warmup: int,
    profile_repeat: int,
    use_student_router: bool = False,
) -> Dict[str, float]:
    parameters = count_parameters(model)
    raw_samples = collect_profile_inputs(dataset, device, profile_samples)
    # Attribution is an offline preprocessing product and is deliberately
    # computed before FLOPs/timing.  The reported inference_time measures the
    # same online boundary as fusion baselines: projection + fusion/router +
    # classifier.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        offline_scores = []
        if not use_student_router:
            for raw_features in raw_samples:
                projected = model.project(raw_features)
                routing_scores, _, _ = model.intervention_attribution(
                    projected=projected,
                    baselines=baselines,
                    replacement_strategy=replacement_strategy,
                    gaussian_std_scale=gaussian_std_scale,
                    compute_interactions=False,
                )
                offline_scores.append(routing_scores.detach())
    model.train(was_training)
    profile_inputs = (
        profile_sample_tuples(raw_samples, model.encoder_names)
        if use_student_router
        else profile_sample_tuples(raw_samples, model.encoder_names, offline_scores)
    )
    wrapper = GMEProfileWrapper(
        model, model.encoder_names, use_student_router=use_student_router
    ).to(device)
    flops = estimate_fvcore_flops(wrapper, profile_inputs)
    inference_time = measure_forward_time_seconds(wrapper, profile_inputs, device, profile_warmup, profile_repeat)
    return {
        "parameters": round(float(parameters) / 1_000_000.0, 2),
        "flops": round(float(flops) / 1_000_000_000.0, 2),
        "inference_time": round(float(inference_time) * 1000.0, 3),
    }


