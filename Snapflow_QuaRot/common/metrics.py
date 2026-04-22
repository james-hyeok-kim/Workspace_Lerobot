"""Latency measurement via CUDA events and quant info helpers."""

from __future__ import annotations

import logging
import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


class LatencyTimer:
    """Measures per-select_action latency using CUDA events."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._use_cuda = "cuda" in device and torch.cuda.is_available()
        self._times_ms: list[float] = []
        self._handle = None

    def attach(self, policy: nn.Module):
        orig_select = policy.select_action

        timer = self

        def timed_select(obs):
            if timer._use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = orig_select(obs)
                end.record()
                torch.cuda.synchronize()
                timer._times_ms.append(start.elapsed_time(end))
            else:
                import time as _time
                t0 = _time.perf_counter()
                out = orig_select(obs)
                timer._times_ms.append((_time.perf_counter() - t0) * 1000)
            return out

        policy.select_action = timed_select
        self._policy = policy
        self._orig_select = orig_select

    def detach(self):
        if hasattr(self, "_policy") and hasattr(self, "_orig_select"):
            self._policy.select_action = self._orig_select

    def summary(self) -> dict:
        if not self._times_ms:
            return {"p50_ms": None, "p95_ms": None, "mean_ms": None, "n": 0}
        arr = np.array(self._times_ms)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "n": len(arr),
        }


def count_quant_bits(policy: nn.Module) -> dict:
    """Estimate effective weight bits from the policy module."""
    results = {"fp32": 0, "bf16": 0, "fp16": 0, "int8": 0, "int4": 0, "other": 0}
    for _, module in policy.named_modules():
        if not list(module.children()):  # leaf only
            w = getattr(module, "weight", None)
            if w is None:
                continue
            dt = w.dtype
            if dt == torch.float32:
                results["fp32"] += 1
            elif dt == torch.bfloat16:
                results["bf16"] += 1
            elif dt == torch.float16:
                results["fp16"] += 1
            elif dt == torch.int8:
                results["int8"] += 1
            elif dt == torch.int32:
                # packed INT4 is often stored as int32
                results["int4"] += 1
            else:
                results["other"] += 1
    return results
