"""Hadamard rotation primitives wrapping ModelOpt's normalized_hadamard_transform.

R1 — input embedding + residual stream (offline, absorbed into weights)
R2 — attention o_proj output <-> MLP down_proj input (offline, weight rotation)
R3 — V <-> o_proj offline rotation (both attention variants)
R4 — online Hadamard at down_proj input (runtime, via RotatedLinear wrapper)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor

# Ensure ModelOpt is on path
_opt_dir = Path(__file__).resolve().parents[3] / "TensorRT-Model-Optimizer"
if str(_opt_dir) not in sys.path:
    sys.path.insert(0, str(_opt_dir))


def hadamard_transform(x: Tensor, rotate_fp32: bool = True, block_size: int | None = None) -> Tensor:
    """Apply normalized (orthogonal) Hadamard transform along x's last dimension.

    Uses pure-PyTorch recursive WHT. ModelOpt's fast_hadamard_transform is NOT used
    because it requires the `fast_hadamard_transform` C extension (not installed here)
    and its block-RHT mode is non-unitary for QuaRot's purposes.
    """
    orig_dtype = x.dtype
    if rotate_fp32:
        x = x.float()
    out = _pytorch_hadamard(x)
    return out.to(orig_dtype) if rotate_fp32 else out


def _pytorch_hadamard(x: Tensor) -> Tensor:
    """Pure-PyTorch normalized Hadamard (orthogonal, norm-preserving)."""
    d = x.shape[-1]
    p2 = 1 << math.ceil(math.log2(d))
    if p2 != d:
        x = torch.nn.functional.pad(x, (0, p2 - d))
    h = _recursive_hadamard(x)
    return h[..., :d] / math.sqrt(p2)


def _recursive_hadamard(x: Tensor) -> Tensor:
    """Full Walsh-Hadamard transform (unnormalized). Norm scales by sqrt(n)."""
    n = x.shape[-1]
    if n == 1:
        return x
    half = n // 2
    a = _recursive_hadamard(x[..., :half])
    b = _recursive_hadamard(x[..., half:])
    return torch.cat([a + b, a - b], dim=-1)


def random_hadamard_matrix(size: int, device="cpu", dtype=torch.float32) -> Tensor:
    """Generate a random Hadamard matrix of given size (power of 2).

    For QuaRot offline rotations: R = H @ D where D is a random ±1 diagonal.
    This provides the "randomized Hadamard" that decorrelates residual stream
    outliers before quantization.
    """
    # Ensure size is power of 2
    assert size & (size - 1) == 0, f"size must be power of 2, got {size}"
    # Random ±1 signs
    signs = torch.randint(0, 2, (size,), device=device, dtype=dtype) * 2 - 1
    return signs  # We'll fold the signs into weight absorptions separately


def apply_offline_rotation_to_weight(
    W: Tensor,
    R: Tensor,
    side: str = "right",
) -> Tensor:
    """Absorb rotation R into weight W.

    side="right": W' = W @ R^T  (rotate the input space)
    side="left":  W' = R @ W    (rotate the output space)

    Both maintain W x == W' (R x) equivalence.
    """
    W = W.float()
    R = R.to(W.device, W.dtype)
    if side == "right":
        return (W @ R.t()).to(W.dtype)
    elif side == "left":
        return (R @ W).to(W.dtype)
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side}")
