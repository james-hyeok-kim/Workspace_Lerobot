"""Offline (static) weight rotations for QuaRot.

R1 — rotate the residual stream: absorbed into embed tokens, final LN, and
     all pre/post-norm Linear weights.
R2 — rotate attention o_proj output <-> MLP down_proj input.
     Applied to both sides so the computation is unchanged in FP.
R3 — rotate V <-> o_proj (attention internal).
     Applied as a linked pair; for cross-attention this requires the same
     rotation on Q of the attending module (see rotate_pi05.py).

All rotations use Hadamard (R = H @ diag(s), s random ±1) which is fast to
apply and produces near-orthogonal matrices with good outlier-smoothing
properties (see QuaRot paper).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import Tensor, nn

from quarot.rotations import hadamard_transform, apply_offline_rotation_to_weight

log = logging.getLogger(__name__)


class RotationState(NamedTuple):
    """Stores the random signs used to build each rotation matrix.

    Signs are saved so the same R can be re-applied (e.g., loading from checkpoint).
    """
    hidden_signs: Tensor    # R1 signs, shape [hidden_dim]
    head_signs: Tensor      # R3 signs, shape [head_dim]


def _make_signs(size: int, device, generator=None) -> Tensor:
    return torch.randint(0, 2, (size,), device=device, generator=generator) * 2 - 1


def _apply_r1_to_layer(layer: nn.Module, signs: Tensor) -> None:
    """Apply R1 (residual stream rotation) to one decoder layer.

    R1 rotates the residual stream dimension.  To keep outputs identical:
    - Weights reading from residual (q/k/v, gate/up): rotate columns (input side) → W' = W @ R^T
    - Weights writing to residual (o_proj, down_proj): rotate rows (output side) → W' = R @ W

    Because R = diag(signs) @ H, applying R^T = H^T @ diag(signs) to columns
    is equivalent to: W @ diag(signs) @ H (since H^T = H for normalized Hadamard, it's self-inverse).
    We implement this as:
      1. Multiply columns by signs: W' = W * signs[None, :]
      2. Apply Hadamard to rows: W' = H(W'.T).T  (i.e., apply along in_features)

    For output-side (R @ W): apply H along out_features, then scale rows by signs.
    """
    D = signs.to(layer.self_attn.q_proj.weight.device)

    def _rotate_input_side(linear: nn.Linear):
        W = linear.weight.data.float()
        W = W * D[None, :]
        W = hadamard_transform(W.T, rotate_fp32=True).T
        linear.weight.data = W.to(linear.weight.dtype)

    def _rotate_output_side(linear: nn.Linear):
        W = linear.weight.data.float()
        W = hadamard_transform(W, rotate_fp32=True)
        W = W * D[:, None]
        linear.weight.data = W.to(linear.weight.dtype)

    attn = layer.self_attn
    mlp = layer.mlp

    for proj in ["q_proj", "k_proj", "v_proj"]:
        m = getattr(attn, proj, None)
        if m is not None:
            _rotate_input_side(m)

    for proj in ["gate_proj", "up_proj"]:
        m = getattr(mlp, proj, None)
        if m is not None:
            _rotate_input_side(m)

    for proj in ["o_proj"]:
        m = getattr(attn, proj, None)
        if m is not None:
            _rotate_output_side(m)

    for proj in ["down_proj"]:
        m = getattr(mlp, proj, None)
        if m is not None:
            _rotate_output_side(m)


def _apply_r2_to_layer(layer: nn.Module, signs: Tensor) -> None:
    """Apply R2: rotate o_proj output <-> down_proj input.

    This effectively diagonalizes the residual addition before the MLP block,
    reducing outliers in the MLP input activations.
    R2 is applied BETWEEN o_proj and the subsequent add+norm+down_proj.
    - o_proj output side: rotate rows → o_proj.W' = R2 @ o_proj.W
    - down_proj input side: rotate columns → down_proj.W' = down_proj.W @ R2^T

    With R2 = diag(signs) @ H, same decomposition as R1.
    """
    D = signs.to(layer.self_attn.o_proj.weight.device)
    attn = layer.self_attn
    mlp = layer.mlp

    if attn.o_proj is not None:
        W = attn.o_proj.weight.data.float()
        W = hadamard_transform(W, rotate_fp32=True)
        W = W * D[:, None]
        attn.o_proj.weight.data = W.to(attn.o_proj.weight.dtype)

    if mlp.down_proj is not None:
        W = mlp.down_proj.weight.data.float()
        W = W * D[None, :]
        W = hadamard_transform(W.T, rotate_fp32=True).T
        mlp.down_proj.weight.data = W.to(mlp.down_proj.weight.dtype)


def _apply_r3_to_layer(layer: nn.Module, head_signs: Tensor) -> None:
    """Apply R3: rotate V <-> o_proj (within attention head sub-space).

    R3 is applied per-head in the head_dim subspace.
    - v_proj output (per-head columns): rotate → v_proj.W' = R3 @ v_proj.W (per head block)
    - o_proj input (per-head rows): rotate → o_proj.W' = o_proj.W @ R3^T (per head block)

    For cross-attention (LLM KV cache read by expert Q): the expert's q_proj
    must receive the INVERSE rotation. This is handled in rotate_pi05.py.
    """
    attn = layer.self_attn
    if not hasattr(attn, "v_proj") or not hasattr(attn, "o_proj"):
        return

    num_heads = getattr(attn, "num_heads", None) or getattr(attn.config, "num_attention_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if num_heads is None or head_dim is None:
        log.warning("Cannot determine num_heads/head_dim for R3 — skipping layer.")
        return

    D = head_signs.to(attn.v_proj.weight.device)  # [head_dim]

    # v_proj: [num_heads * head_dim, hidden_dim] — rotate output rows per head
    v_W = attn.v_proj.weight.data.float()
    v_W = v_W.view(num_heads, head_dim, -1)
    v_W = hadamard_transform(v_W, rotate_fp32=True)
    v_W = v_W * D[None, :, None]
    attn.v_proj.weight.data = v_W.view(num_heads * head_dim, -1).to(attn.v_proj.weight.dtype)

    # o_proj: [hidden_dim, num_heads * head_dim] — rotate input cols per head
    o_W = attn.o_proj.weight.data.float()
    o_W = o_W.view(-1, num_heads, head_dim)
    o_W = o_W * D[None, None, :]
    o_W = hadamard_transform(o_W.transpose(-1, -2), rotate_fp32=True).transpose(-1, -2)
    attn.o_proj.weight.data = o_W.view(-1, num_heads * head_dim).to(attn.o_proj.weight.dtype)


def apply_r1r2r3(
    decoder_layers: list[nn.Module],
    r1: bool = True,
    r2: bool = True,
    r3: bool = False,
    hidden_dim: int | None = None,
    head_dim: int | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> RotationState:
    """Apply R1, R2, R3 to a list of decoder layers.

    Returns the RotationState (signs) so the same rotation can be saved/restored.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    # Infer dims from first layer
    if hidden_dim is None or head_dim is None:
        first = decoder_layers[0]
        attn = first.self_attn
        if hidden_dim is None:
            q = getattr(attn, "q_proj", None)
            hidden_dim = q.weight.shape[1] if q is not None else 2048
        if head_dim is None:
            head_dim = getattr(attn, "head_dim", 256)

    hidden_signs = _make_signs(hidden_dim, device=device, generator=gen)
    head_signs = _make_signs(head_dim, device=device, generator=gen)

    for i, layer in enumerate(decoder_layers):
        if r1:
            _apply_r1_to_layer(layer, hidden_signs)
        if r2:
            _apply_r2_to_layer(layer, hidden_signs)
        if r3:
            _apply_r3_to_layer(layer, head_signs)
        if (i + 1) % 10 == 0:
            log.info(f"  ...rotated {i+1}/{len(decoder_layers)} layers")

    return RotationState(hidden_signs=hidden_signs, head_signs=head_signs)
