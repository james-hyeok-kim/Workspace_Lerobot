"""Fuse RMSNorm gain into the downstream Linear weight.

After fusion the norm.weight is set to all-ones (identity gain) so that
applying the rotation R to downstream weights keeps the computation equivalent.

QuaRot requires this because:
  y = RMSNorm(x) @ W = (x / rms(x) * g) @ W
If we absorb g into W: W' = diag(g) @ W, then
  y = (x / rms(x)) @ W'
This is lossless and lets us treat the residual stream as uniformly scaled,
making the Hadamard rotation effective.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)

# Gemma / Gemma2 RMSNorm class names to detect
_RMSNORM_NAMES = {"GemmaRMSNorm", "Gemma2RMSNorm", "RMSNorm", "LlamaRMSNorm", "T5LayerNorm"}


def is_rmsnorm(module: nn.Module) -> bool:
    return type(module).__name__ in _RMSNORM_NAMES


def fuse_rmsnorm_into_linear(norm: nn.Module, linear: nn.Module) -> None:
    """Absorb norm.weight (gain) into linear.weight in-place.

    Assumes norm applies: y = x / rms(x) * weight, then linear maps y.
    After fusion: linear.weight *= norm.weight (broadcast), norm.weight = 1.
    """
    if not hasattr(norm, "weight"):
        return
    if norm.weight is None:
        return

    gain = norm.weight.data.float()  # [hidden_dim]

    with torch.no_grad():
        # linear.weight shape: [out_features, in_features]
        # Multiply each column (input dim) by the corresponding gain
        linear.weight.data = (linear.weight.data.float() * gain[None, :]).to(linear.weight.dtype)

        if linear.bias is not None:
            # bias is added after the linear; it doesn't interact with the norm gain
            pass

        # Reset norm gain to identity (1s)
        norm.weight.data.fill_(1.0)


def fuse_all_rmsnorms(model: nn.Module, scope: str = "llm") -> int:
    """Walk the model and fuse every RMSNorm into the immediately following projections.

    For Gemma-based models the pattern is:
      input_layernorm -> [q_proj, k_proj, v_proj]   (attention input norm)
      post_feedforward_layernorm -> [gate_proj, up_proj]  (MLP input norm)

    Returns the number of fusions performed.
    """
    n_fused = 0

    def _process_decoder_layer(layer):
        nonlocal n_fused
        # Attention pre-norm -> Q, K, V
        for norm_attr, proj_attrs in [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("pre_feedforward_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("post_feedforward_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ]:
            norm = _get_nested(layer, norm_attr)
            if norm is None or not is_rmsnorm(norm):
                continue
            for proj_attr in proj_attrs:
                proj = _get_nested(layer, proj_attr)
                if proj is not None and isinstance(proj, nn.Linear):
                    fuse_rmsnorm_into_linear(norm, proj)
                    n_fused += 1

    # Locate decoder layers
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if "DecoderLayer" in type_name or "GemmaDecoderLayer" in type_name:
            if scope == "llm" and "expert" in name.lower():
                continue
            _process_decoder_layer(module)

    log.info(f"fuse_all_rmsnorms: fused {n_fused} norm→linear pairs (scope={scope})")
    return n_fused


def _get_nested(module: nn.Module, attr: str):
    """Traverse dot-separated attribute path."""
    parts = attr.split(".")
    obj = module
    for p in parts:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj
