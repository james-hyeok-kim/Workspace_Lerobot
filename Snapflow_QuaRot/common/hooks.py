"""Forward hook helpers for activation capture."""

from __future__ import annotations

import torch
from torch import nn


class _OnlineStats:
    """Welford online algorithm for mean, variance, and 4th central moment (kurtosis)."""

    __slots__ = ("n", "mean", "M2", "M4", "max_abs")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M4 = 0.0
        self.max_abs = 0.0

    def update(self, x: torch.Tensor):
        x = x.float().flatten()
        n_new = x.numel()
        if n_new == 0:
            return
        xmax = x.abs().max().item()
        if xmax > self.max_abs:
            self.max_abs = xmax

        # Batch update via combined-statistics formula (Chan et al.)
        new_mean = x.mean().item()
        new_M2 = x.var(unbiased=False).item() * n_new  # sum of squared deviations
        # 4th central moment sum from batch
        mu = x.mean()
        new_M4 = ((x - mu) ** 4).sum().item()

        n_a, n_b = self.n, n_new
        n_ab = n_a + n_b
        delta = new_mean - self.mean
        delta2 = delta * delta

        # Combined mean
        combined_mean = (n_a * self.mean + n_b * new_mean) / n_ab

        # Combined M2
        combined_M2 = self.M2 + new_M2 + delta2 * n_a * n_b / n_ab

        # Combined M4 (approximate; good for large n)
        combined_M4 = (
            self.M4
            + new_M4
            + delta2 * delta2 * (n_a * n_b * (n_a**2 - n_a * n_b + n_b**2)) / (n_ab**3)
            + 6 * delta2 * (n_a**2 * new_M2 + n_b**2 * self.M2) / (n_ab**2)
        )

        self.n = n_ab
        self.mean = combined_mean
        self.M2 = combined_M2
        self.M4 = combined_M4

    def finalize(self) -> dict:
        if self.n < 2:
            return {"max_abs": self.max_abs, "kurtosis": 0.0,
                    "mean": self.mean, "std": 0.0, "n_samples": self.n}
        var = self.M2 / self.n
        std = var ** 0.5
        # Excess kurtosis (Fisher)
        m4 = self.M4 / self.n
        kurt = (m4 / (var ** 2) - 3.0) if var > 1e-12 else 0.0
        return {
            "max_abs": self.max_abs,
            "kurtosis": kurt,
            "mean": self.mean,
            "std": std,
            "n_samples": self.n,
        }


class OnlineStatsStore:
    """Computes per-layer activation statistics online — no tensor accumulation."""

    def __init__(self):
        self._handles: list = []
        self._stats: dict[str, _OnlineStats] = {}

    def register(self, model: nn.Module, module_types=(nn.Linear,)):
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                self._stats[name] = _OnlineStats()
                handle = module.register_forward_pre_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, args):
            x = args[0].detach()
            self._stats[name].update(x)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def data(self) -> dict[str, dict]:
        return {name: s.finalize() for name, s in self._stats.items()}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


class ActivationStore:
    """Accumulates per-layer activation tensors from forward_pre_hooks.

    Warning: accumulates all tensors in RAM — use OnlineStatsStore for large runs.
    """

    def __init__(self):
        self._handles: list = []
        self.data: dict[str, list[torch.Tensor]] = {}

    def register(self, model: nn.Module, module_types=(nn.Linear,)):
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                self.data[name] = []
                handle = module.register_forward_pre_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, args):
            x = args[0].detach().cpu()
            self.data[name].append(x)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


class OutputStore:
    """Accumulates per-layer output tensors from forward hooks."""

    def __init__(self):
        self._handles: list = []
        self.data: dict[str, list[torch.Tensor]] = {}

    def register(self, model: nn.Module, module_types=(nn.Linear,)):
        for name, module in model.named_modules():
            if isinstance(module, module_types):
                self.data[name] = []
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, args, output):
            self.data[name].append(output.detach().cpu())
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()
