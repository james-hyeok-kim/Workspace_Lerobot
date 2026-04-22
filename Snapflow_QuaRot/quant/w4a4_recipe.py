"""W4A4 quantization recipe builder for ModelOpt MTQ.

Builds on top of ModelOpt's NVFP4_KV_ROTATE_CFG structure but targets
INT4 weight + INT4 activation (per-tensor dynamic).

The RotateConfig entries enable the online Hadamard (R4) at down_proj inputs
via the existing ModelOpt TensorQuantizer path:
  tensor_quantizer.py:961 — calls normalized_hadamard_transform before quantize
  config.py:555-675       — NVFP4_KV_ROTATE_CFG shows the rotate: {enable: true} pattern

OHB (Outlier Head Bypass): layers listed in policy._ohb_manifest get
  {"*": {"enable": False}} to keep them in FP16.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_w4a4_config(
    group_size: int = 128,
    online_hadamard: bool = True,
    ohb_manifest: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build ModelOpt MTQ quantization config for W4A4.

    Returns a config dict compatible with mtq.quantize(model, config=...).
    """
    # Base quantizer templates
    w4_quantizer = {
        "num_bits": 4,
        "axis": 1,                  # per-group along output dim
        "block_sizes": {-1: group_size},
        "enable": True,
    }
    a4_quantizer = {
        "num_bits": 4,
        "axis": None,               # per-tensor dynamic
        "enable": True,
    }
    a4_rotate = {
        "num_bits": 4,
        "axis": None,
        "enable": True,
        "rotate": {
            "enable": online_hadamard,
            "rotate_fp32": True,
        },
    }
    fp16_quantizer = {"enable": False}

    # Default: quantize all Linear layers with W4A4
    config = {
        "quant_cfg": {
            "*weight_quantizer": w4_quantizer,
            "*input_quantizer": a4_quantizer,
            # KV quantization (attention BMMs)
            "*bmm_quantizer": a4_quantizer,
        },
        "algorithm": "max",
    }

    # Online Hadamard at down_proj (R4)
    if online_hadamard:
        config["quant_cfg"]["*down_proj*input_quantizer"] = a4_rotate

    # OHB: keep outlier layers in FP16
    if ohb_manifest:
        for layer_name in ohb_manifest:
            # ModelOpt uses module-path patterns; convert layer name to wildcard pattern
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16_quantizer
            config["quant_cfg"][f"*{layer_name}*input_quantizer"] = fp16_quantizer

    return config


def build_int8_config() -> dict[str, Any]:
    """W8A8 config — useful as intermediate step between FP16 and W4A4."""
    return {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0, "enable": True},
            "*input_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        },
        "algorithm": "max",
    }
