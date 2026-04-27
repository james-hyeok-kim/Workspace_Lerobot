"""W4A4 quantization recipe builder for ModelOpt MTQ.

개선사항 (논문 기반):
  - per-token activation 양자화 (axis=0) — 논문: "per-token symmetric"
  - GPTQ 알고리즘 지원 — 논문: GPTQ가 RTN보다 ~2pp 낫다
  - Activation clipping_ratio=0.9 — 논문 실험 설정

W4 weight: per-group-128, symmetric
W4A4 activation: per-token (axis=0), symmetric, clipping_ratio=0.9

OHB (Outlier Head Bypass): 상위 K% kurtosis 레이어 → FP16 유지
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def build_w4a4_config(
    group_size: int = 128,
    online_hadamard: bool = False,
    ohb_manifest: dict[str, bool] | None = None,
    algorithm: str = "max",
) -> dict[str, Any]:
    """W4A4 config: per-group weight + per-token activation.

    Args:
        group_size: weight 양자화 그룹 크기 (논문: 128)
        online_hadamard: R4 online Hadamard (fast_hadamard_transform 필요)
        ohb_manifest: 보호할 레이어 이름 집합
        algorithm: "max" (RTN) or "gptq" (더 정확하지만 느림)
    """
    w4_quantizer = {
        "num_bits": 4,
        "block_sizes": {-1: group_size},
        "enable": True,
    }
    # per-token (axis=0) — 논문: "per-token symmetric quantization (a single scale for every row)"
    a4_quantizer = {
        "num_bits": 4,
        "axis": 0,              # per-token (행별 scale)
        "enable": True,
        "clip_ratio": 0.9,      # 논문: "constant clipping ratio of 0.9 in all our experiments"
    }
    a4_rotate = {
        "num_bits": 4,
        "axis": 0,
        "enable": True,
        "clip_ratio": 0.9,
        "rotate": {
            "enable": online_hadamard,
            "rotate_fp32": True,
        },
    }
    fp16_quantizer = {"enable": False}

    config = {
        "quant_cfg": {
            "*weight_quantizer": w4_quantizer,
            "*input_quantizer": a4_quantizer,
            "nn.BatchNorm1d": {"*": fp16_quantizer},
            "nn.BatchNorm2d": {"*": fp16_quantizer},
            "nn.LayerNorm":   {"*": fp16_quantizer},
            "*lm_head*":      fp16_quantizer,
            "*proj_out.*":    fp16_quantizer,
            "default":        fp16_quantizer,
        },
        "algorithm": algorithm,
    }

    if online_hadamard:
        config["quant_cfg"]["*down_proj*input_quantizer"] = a4_rotate

    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16_quantizer
            config["quant_cfg"][f"*{layer_name}*input_quantizer"] = fp16_quantizer

    return config


def build_w4_weight_only_config(
    group_size: int = 128,
    ohb_manifest: dict[str, bool] | None = None,
    algorithm: str = "max",
) -> dict[str, Any]:
    """W4 weight-only quantization (activation은 FP16 유지).

    Stage 5 baseline: weight만 INT4, activation은 FP16.
    W4A4 full quantization 전 단계로 안정성 확인용.
    """
    w4_quantizer = {
        "num_bits": 4,
        "block_sizes": {-1: group_size},
        "enable": True,
    }
    fp16_quantizer = {"enable": False}

    config = {
        "quant_cfg": {
            "*weight_quantizer": w4_quantizer,
            "*input_quantizer": fp16_quantizer,
            "default": fp16_quantizer,
        },
        "algorithm": algorithm,
    }

    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16_quantizer

    return config


def build_llm_only_w4a4_config(
    ohb_manifest: dict | None = None,
    group_size: int = 128,
    algorithm: str = "max",
) -> dict[str, Any]:
    """Stage 5b: LLM/VisionTower/MMP만 W4A4, DiT(gemma_expert)는 FP16.

    per-token activation (axis=0) + clipping_ratio=0.9.
    """
    fp16 = {"enable": False}
    w4 = {"num_bits": 4, "block_sizes": {-1: group_size}, "enable": True}
    a4 = {"num_bits": 4, "axis": 0, "enable": True, "clip_ratio": 0.9}

    config = {
        "quant_cfg": {
            "*language_model*weight_quantizer": w4,
            "*language_model*input_quantizer": a4,
            "*vision_tower*weight_quantizer": w4,
            "*vision_tower*input_quantizer": a4,
            "*multi_modal_projector*weight_quantizer": w4,
            "*multi_modal_projector*input_quantizer": a4,
            "*gemma_expert*weight_quantizer": fp16,
            "*gemma_expert*input_quantizer": fp16,
            "*action_in_proj*": fp16,
            "*action_out_proj*": fp16,
            "*lm_head*": fp16,
            "default": fp16,
        },
        "algorithm": algorithm,
    }

    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16
            config["quant_cfg"][f"*{layer_name}*input_quantizer"] = fp16

    return config


def build_int4_weight_only_config(
    group_size: int = 128,
    ohb_manifest: dict[str, bool] | None = None,
    algorithm: str = "max",
) -> dict[str, Any]:
    """INT4 weight-only with configurable group size."""
    w4 = {"num_bits": 4, "block_sizes": {-1: group_size}, "enable": True}
    fp16 = {"enable": False}
    config = {
        "quant_cfg": {
            "*weight_quantizer": w4,
            "*input_quantizer": fp16,
            "*lm_head*": fp16,
            "default": fp16,
        },
        "algorithm": algorithm,
    }
    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16
    return config


def build_mxfp4_weight_only_config(
    ohb_manifest: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """MXFP4 weight-only: E1M2, block=32, E8M0 scale (MX standard).

    num_bits=(2,1): 2 mantissa bits + 1 exponent bit = 4-bit float.
    block_sizes scale_bits=(8,0): per-block E8M0 FP8 scale.
    """
    mxfp4_w = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
        "enable": True,
    }
    fp16 = {"enable": False}
    config = {
        "quant_cfg": {
            "*weight_quantizer": mxfp4_w,
            "*input_quantizer": fp16,
            "*lm_head*": fp16,
            "default": fp16,
        },
        "algorithm": None,
    }
    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16
    return config


def build_nvfp4_weight_only_config(
    ohb_manifest: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """NVFP4 weight-only: E1M2, block=16, E4M3 scale (NVIDIA FP4).

    num_bits=(2,1): same 4-bit float as MXFP4.
    block_sizes scale_bits=(4,3): finer-grained E4M3 FP8 scale per block-16.
    """
    nvfp4_w = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "enable": True,
    }
    fp16 = {"enable": False}
    config = {
        "quant_cfg": {
            "*weight_quantizer": nvfp4_w,
            "*input_quantizer": fp16,
            "*lm_head*": fp16,
            "default": fp16,
        },
        "algorithm": None,
    }
    if ohb_manifest:
        for layer_name in ohb_manifest:
            config["quant_cfg"][f"*{layer_name}*weight_quantizer"] = fp16
    return config


def build_int8_config() -> dict[str, Any]:
    """W8A8 — W4A4 전 중간 단계로 안정성 확인용."""
    return {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 8, "axis": 0, "enable": True},
            "*input_quantizer": {"num_bits": 8, "axis": 0, "enable": True},
        },
        "algorithm": "max",
    }
