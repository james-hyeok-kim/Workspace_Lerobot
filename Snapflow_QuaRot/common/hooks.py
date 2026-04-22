"""Forward hook helpers for activation capture."""

from __future__ import annotations

import torch
from torch import nn


class ActivationStore:
    """Accumulates per-layer activation tensors from forward_pre_hooks."""

    def __init__(self):
        self._handles: list = []
        self.data: dict[str, list[torch.Tensor]] = {}

    def register(self, model: nn.Module, module_types=(nn.Linear,)):
        """Attach hooks to all modules matching module_types."""
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
