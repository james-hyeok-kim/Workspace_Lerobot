"""Bridge between QuaRot pipeline and ModelOpt MTQ quantization.

Calls mtq.quantize(policy, config, forward_loop) with the W4A4 config
(optionally including OHB and R4 online Hadamard).

Also handles torch.compile cleanup (MTQ replaces Linear layers with
QuantizedLinear which is incompatible with existing compiled callables).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch import nn

log = logging.getLogger(__name__)


def _ensure_modelopt():
    """Import modelopt.torch.quantization, raising a clear error if not found."""
    try:
        import modelopt.torch.quantization as mtq
        return mtq
    except ImportError:
        raise ImportError(
            "modelopt not found. Add TensorRT-Model-Optimizer to PYTHONPATH:\n"
            "  export MODEL_OPT_DIR=/home/jovyan/workspace/Workspace_Lerobot/TensorRT-Model-Optimizer\n"
            "  export PYTHONPATH=$MODEL_OPT_DIR:$PYTHONPATH"
        )


def apply_w4a4(policy: nn.Module, w4a4_cfg, env_cfg=None):
    """Apply W4A4 quantization with optional R4 and OHB to policy in-place.

    Args:
        policy:   pi0.5 policy (may already have QuaRot rotations applied).
        w4a4_cfg: W4A4Config from recipe.
        env_cfg:  LiberoEnv config for calibration. If None, calibration is skipped.
    """
    from quant.w4a4_recipe import build_w4a4_config
    from quant.calib_quantize import make_calibration_loop

    mtq = _ensure_modelopt()

    ohb_manifest = getattr(policy, "_ohb_manifest", None)
    if ohb_manifest:
        log.info(f"OHB manifest found: {len(ohb_manifest)} layers will stay FP16.")

    quant_config = build_w4a4_config(
        group_size=w4a4_cfg.group_size,
        online_hadamard=w4a4_cfg.online_hadamard,
        ohb_manifest=ohb_manifest,
    )

    if env_cfg is not None:
        log.info("Building calibration loop for MTQ...")
        forward_loop = make_calibration_loop(
            policy=policy,
            env_cfg=env_cfg,
            n_chunks=64,
            device=w4a4_cfg.__dict__.get("device", "cuda"),
        )
    else:
        forward_loop = None
        log.warning("No env_cfg provided; MTQ calibration will be skipped (use amax=1 fallback).")

    log.info("Applying W4A4 quantization via ModelOpt MTQ...")
    mtq.quantize(policy, config=quant_config, forward_loop=forward_loop)
    log.info("MTQ quantization complete.")

    _disable_torch_compile(policy)


def _disable_torch_compile(policy: nn.Module):
    """Reset torch.compile state — MTQ-patched QuantizedLinear is incompatible with dynamo."""
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
    except Exception:
        pass

    inner = getattr(policy, "model", None)
    if inner is not None:
        for attr in ("sample_actions", "forward"):
            fn = getattr(inner, attr, None)
            if fn is None:
                continue
            orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
            if orig is not None:
                setattr(inner, attr, orig)
                log.info(f"torch.compile disabled for inner_model.{attr}")


def save_quant_state(policy: nn.Module, path: str):
    """Save ModelOpt quantization state dict for later restoration."""
    mtq = _ensure_modelopt()
    from pathlib import Path as P
    P(path).parent.mkdir(parents=True, exist_ok=True)
    state = mtq.modelopt_state(policy)
    torch.save(state, path)
    log.info(f"Quant state saved to {path}")


def load_quant_state(policy: nn.Module, path: str):
    """Restore ModelOpt quantization state from saved file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Quant state not found: {path}. Run Stage 5 first.")
    mtq = _ensure_modelopt()
    state = torch.load(path, weights_only=False)
    mtq.restore_from_modelopt_state(policy, state)
    log.info(f"Quant state loaded from {path}")
