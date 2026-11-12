"""Shared, baseline-adjusted inference memory profiling for GME and baselines."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn


def model_storage_bytes(model: nn.Module, device: torch.device) -> int:
    """Count parameter/buffer storage once, including tied tensors and views."""
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device == device:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


@torch.no_grad()
def measure_peak_gpu_memory(
    wrapper: nn.Module,
    samples: Sequence[Tuple[torch.Tensor, ...]],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> Dict[str, float]:
    """Estimate isolated per-slide allocated peaks in MiB (mean and maximum).

    Each measurement starts before copying one slide's inputs to CUDA. Subtract
    the resident allocation baseline, then add parameter/buffer storage bytes.
    This excludes optimizer state, gradients, checkpoints and other loaded
    slides without changing the caller's training objects. The result includes
    model storage, current inputs and peak forward allocations, but excludes
    allocator-reserved memory and CUDA context overhead. Model storage is
    counted in tensor bytes rather than allocator-rounded block sizes.

    CPU runs, empty samples and disabled repeats produce NaN, not zero.
    """
    unavailable = {"peak_gpu_memory": float("nan"), "peak_gpu_memory_max": float("nan")}
    if device.type != "cuda" or not samples or repeat <= 0:
        return unavailable
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    resident_model_bytes = model_storage_bytes(wrapper, device)
    training_states = [(module, module.training) for module in wrapper.modules()]
    peaks = []
    wrapper.eval()
    try:
        for sample in samples:
            # A fresh transfer makes the input footprint independent of whether
            # the caller has already preloaded profiling samples on the GPU.
            cpu_inputs = tuple(tensor.detach().cpu() for tensor in sample)
            gpu_inputs = tuple(tensor.to(device) for tensor in cpu_inputs)
            try:
                for _ in range(max(int(warmup), 0)):
                    wrapper(*gpu_inputs)
            finally:
                del gpu_inputs
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_inputs = tuple(tensor.to(device) for tensor in cpu_inputs)
            try:
                for _ in range(int(repeat)):
                    wrapper(*gpu_inputs)
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                peaks.append(resident_model_bytes + max(0, peak - baseline))
            finally:
                del gpu_inputs
    finally:
        for module, was_training in training_states:
            module.training = was_training
    mib = 1024.0 ** 2
    return {
        "peak_gpu_memory": round(sum(peaks) / len(peaks) / mib, 3),
        "peak_gpu_memory_max": round(max(peaks) / mib, 3),
    }
