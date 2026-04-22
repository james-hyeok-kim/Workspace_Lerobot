"""Online (runtime) Hadamard wrapper for R4 at down_proj inputs.

R4 is applied to activations just before down_proj to smooth outliers
in the MLP bottleneck.  Unlike R1-R3 (offline weight rotation), R4
must be applied at runtime because the activations are input-dependent.

Implementation: wrap nn.Linear with a pre-Hadamard transform.
ModelOpt's TensorQuantizer with RotateConfig also supports this path;
this module provides a standalone Python wrapper as a fallback.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from quarot.rotations import hadamard_transform


class RotatedLinear(nn.Module):
    """Drop-in replacement for nn.Linear that applies Hadamard before the matmul.

    Used for R4: down_proj inputs are Hadamard-transformed at runtime.
    """

    def __init__(self, linear: nn.Linear, rotate_fp32: bool = True):
        super().__init__()
        self.linear = linear
        self.rotate_fp32 = rotate_fp32

    def forward(self, x: Tensor) -> Tensor:
        x = hadamard_transform(x, rotate_fp32=self.rotate_fp32)
        return self.linear(x)

    # Proxy weight/bias for compatibility with existing code that reads .weight
    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features


def wrap_down_proj_with_r4(model: nn.Module, scope: str = "llm") -> int:
    """Replace every mlp.down_proj with RotatedLinear(down_proj).

    Returns the number of layers wrapped.
    """
    n_wrapped = 0

    for name, module in model.named_modules():
        type_name = type(module).__name__
        if "DecoderLayer" not in type_name and "GemmaDecoderLayer" not in type_name:
            continue

        if scope == "llm" and "expert" in name.lower():
            continue

        mlp = getattr(module, "mlp", None)
        if mlp is None:
            continue

        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is None or not isinstance(down_proj, nn.Linear):
            continue

        mlp.down_proj = RotatedLinear(down_proj)
        n_wrapped += 1

    return n_wrapped
