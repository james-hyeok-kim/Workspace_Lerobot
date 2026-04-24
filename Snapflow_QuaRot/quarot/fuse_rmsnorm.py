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
# PiGemmaRMSNorm uses gain = (1 + weight); needs different fuse logic
_PIGEMMARMSNORM_NAMES = {"PiGemmaRMSNorm"}


def is_rmsnorm(module: nn.Module) -> bool:
    name = type(module).__name__
    return name in _RMSNORM_NAMES or name in _PIGEMMARMSNORM_NAMES


def fuse_rmsnorm_into_linear(norm: nn.Module, linear: nn.Module) -> None:
    """Absorb norm.weight (gain) into linear.weight in-place.

    Handles two variants:
    - Standard RMSNorm: y = (x/rms(x)) * weight  → gain = weight, reset to 1.0
    - PiGemmaRMSNorm: y = (x/rms(x)) * (1+weight) → gain = 1+weight, reset weight to 0.0
    """
    if not hasattr(norm, "weight"):
        return  # adaptive mode (cond_dim set) — no static weight to fuse
    if norm.weight is None:
        return

    is_pi_norm = type(norm).__name__ in _PIGEMMARMSNORM_NAMES

    with torch.no_grad():
        if is_pi_norm:
            gain = (1.0 + norm.weight.data.float())  # effective gain = 1 + w
        else:
            gain = norm.weight.data.float()

        # linear.weight shape: [out_features, in_features]
        # Multiply each input column by the corresponding gain
        linear.weight.data = (linear.weight.data.float() * gain[None, :]).to(linear.weight.dtype)

        # Reset norm gain to identity
        if is_pi_norm:
            norm.weight.data.fill_(0.0)  # 1 + 0 = 1 (identity)
        else:
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
        for norm_attr, proj_attrs in [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("pre_feedforward_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("post_feedforward_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
            ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ]:
            norm = _get_nested(layer, norm_attr)
            if norm is None or not is_rmsnorm(norm):
                continue

            # Compute gain once from the current norm weight
            with torch.no_grad():
                is_pi_norm = type(norm).__name__ in _PIGEMMARMSNORM_NAMES
                if not hasattr(norm, "weight") or norm.weight is None:
                    continue
                gain = (1.0 + norm.weight.data.float()) if is_pi_norm else norm.weight.data.float()

            # Absorb gain into ALL downstream linears (using the SAME gain)
            for proj_attr in proj_attrs:
                proj = _get_nested(layer, proj_attr)
                if proj is not None and isinstance(proj, nn.Linear):
                    with torch.no_grad():
                        proj.weight.data = (proj.weight.data.float() * gain[None, :]).to(proj.weight.dtype)
                    n_fused += 1

            # Reset norm to identity ONCE after all linears are fused
            with torch.no_grad():
                if is_pi_norm:
                    norm.weight.data.fill_(0.0)
                else:
                    norm.weight.data.fill_(1.0)

    def _is_decoder_layer(module: nn.Module) -> bool:
        """Detect decoder layers by attribute presence (robust across custom class names)."""
        type_name = type(module).__name__
        # Standard name check
        if "DecoderLayer" in type_name or "GemmaDecoderLayer" in type_name:
            return True
        # Attribute-based fallback: pi0.5 uses _PiGemmaDecoderLayerBase etc.
        return (hasattr(module, "input_layernorm") and
                hasattr(module, "self_attn") and
                hasattr(module, "mlp"))

    # Locate decoder layers
    for name, module in model.named_modules():
        if not _is_decoder_layer(module):
            continue
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
