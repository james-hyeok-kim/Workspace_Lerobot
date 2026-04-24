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
        # W' = W @ R^T = W @ H @ diag(D)  (H on input cols = last dim)
        W = linear.weight.data.float()
        W = hadamard_transform(W, rotate_fp32=True)  # W @ H
        W = W * D[None, :]                            # W @ H @ diag(D)
        linear.weight.data = W.to(linear.weight.dtype)

    def _rotate_output_side(linear: nn.Linear):
        # W' = R @ W = diag(D) @ H @ W  (H on output rows)
        W = linear.weight.data.float()
        W = hadamard_transform(W.T, rotate_fp32=True).T  # H @ W
        W = W * D[:, None]                                # diag(D) @ H @ W
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
        # Rotate output rows (hidden_dim side) — same direction as o_proj.
        # down_proj.weight shape: (hidden_dim, intermediate_size).
        # D has size hidden_dim, so apply to output rows ([:, None] pattern),
        # NOT input cols (which have size intermediate_size != hidden_dim).
        W = mlp.down_proj.weight.data.float()
        W = hadamard_transform(W, rotate_fp32=True)
        W = W * D[:, None]
        mlp.down_proj.weight.data = W.to(mlp.down_proj.weight.dtype)


def _apply_r3_to_layer(layer: nn.Module, head_signs: Tensor) -> None:
    """Apply R3: rotate V <-> o_proj within the head_dim subspace.

    Correct for GQA (grouped-query attention):
    - v_proj has shape [num_kv_heads * head_dim, hidden]. Apply H (head_dim × head_dim)
      to the head_dim rows of each KV head: V' = V @ H → W_v' = H @ W_v
      (transpose trick: hadamard on W_v.T along last dim, then transpose back).
    - o_proj has shape [hidden, num_q_heads * head_dim]. Apply same H to each Q head's
      head_dim columns: W_o' = W_o @ H. Since H is symmetric, this equals
      hadamard_transform(o_W) directly (applied to last dim = head_dim).
    The product V'_out @ W_o'^T = (V @ H)(H @ W_o)^T = V @ H^2 @ W_o^T = V @ W_o^T ✓
    """
    attn = layer.self_attn
    if not hasattr(attn, "v_proj") or not hasattr(attn, "o_proj"):
        return

    cfg = getattr(attn, "config", None)
    num_q_heads = getattr(attn, "num_heads", None) or (cfg and getattr(cfg, "num_attention_heads", None))
    num_kv_heads = (getattr(attn, "num_key_value_heads", None)
                    or (cfg and getattr(cfg, "num_key_value_heads", None))
                    or num_q_heads)
    head_dim = getattr(attn, "head_dim", None)
    if num_q_heads is None or head_dim is None:
        log.warning("Cannot determine num_heads/head_dim for R3 — skipping layer.")
        return

    D = head_signs.to(attn.v_proj.weight.device)  # [head_dim]

    # v_proj: [num_kv * head_dim, hidden] — apply H to head_dim (row) subspace per KV head
    # W_v' = H @ W_v: use transpose trick since hadamard_transform acts on last dim.
    v_W = attn.v_proj.weight.data.float().contiguous()
    v_W = v_W.reshape(num_kv_heads, head_dim, -1)   # [kv, head_dim, hidden]
    v_W = hadamard_transform(v_W.transpose(-1, -2), rotate_fp32=True).transpose(-1, -2)
    v_W = v_W * D[None, :, None]                     # diag(D) @ H @ W_v[kv]
    attn.v_proj.weight.data = v_W.reshape(num_kv_heads * head_dim, -1).to(attn.v_proj.weight.dtype)

    # o_proj: [hidden, num_q * head_dim] — absorb R3^T = H @ diag(D) per Q head block.
    # V was rotated by R3 = diag(D) @ H, so attention output = attn_orig @ R3^T.
    # o_proj must absorb R3^T: W_o' = W_o @ R3^T = W_o @ H @ diag(D).
    # Apply H first (last dim = head_dim), then scale by D.
    o_W = attn.o_proj.weight.data.float().contiguous()
    o_W = o_W.reshape(-1, num_q_heads, head_dim)      # [hidden, q, head_dim]
    o_W = hadamard_transform(o_W, rotate_fp32=True)   # W_o @ H (H on last dim = head_dim)
    o_W = o_W * D[None, None, :]                      # W_o @ H @ diag(D) = W_o @ R3^T ✓
    attn.o_proj.weight.data = o_W.reshape(-1, num_q_heads * head_dim).to(attn.o_proj.weight.dtype)


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
