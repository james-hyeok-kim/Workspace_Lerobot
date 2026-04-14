"""
Per-Layer Mixed-Precision FP Quantization for LeRobot pi05.

레이어별 distribution 통계(kurtosis / CV)를 기반으로 각 Linear 레이어에
다른 FP format 및 block size를 자동 배정하는 mixed-precision FP 양자화.

Format pool (bit-group 별 default):
  8-bit : MXFP8  (E4M3, E8M0 scale)  /  NVFP8  (E4M3, FP16 scale)
  6-bit : MXFP6_E2M3 (E2M3, max=7.5)  /  MXFP6_E3M2 (E3M2, max=28)
  4-bit : NVFP4  (non-uniform {0,±0.5,...,±6})  /  MXFP4 (E2M1, E8M0 scale)
  3-bit : FP3_E1M1 ({0,1,2,3})  /  FP3_E2M0 ({0,1,2,4})  /  FP3_E0M2 ({0,0.25,0.5,0.75})

Quantization simulates FP format (dequant stored — no int packing).

Usage:
    # 표만 보기 (모델 로드 없음)
    python eval_mixed_fp_quant.py --table_only --target all

    # 테스트: libero_10, 5 episodes
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_mixed_fp_quant.py \\
        --task libero_10 --n_episodes 5 --batch_size 5 --target all

    # LM만 NVFP4 minimum
    python eval_mixed_fp_quant.py --task libero_10 --n_episodes 5 \\
        --target lm --lm_min_bits 4
"""

import argparse
import gc
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _p in [str(_root / "src"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# 상수 & 레이어 분류 패턴
# ══════════════════════════════════════════════════════════════════════════════

DIST_STATS_PATH = _root / "logs/dist_analysis_v4/dist_stats.json"

LM_PATTERNS = [
    "paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
]
DIT_PATTERNS = [
    "paligemma_with_expert.gemma_expert",
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
]
SKIP_PATTERNS = ["vision_tower", "embed_tokens", "lm_head"]

# FP format → bit width (for min-format enforcement)
FORMAT_BITS = {
    "MXFP8":     8, "NVFP8":     8,
    "MXFP6_E2M3":6, "MXFP6_E3M2":6,
    "NVFP4":     4, "MXFP4":     4,
    "FP3_E1M1":  3, "FP3_E2M0":  3, "FP3_E0M2":  3,
}

# bit group → default format (best-known sub-format for pi05 weights)
_DEFAULT_FMT = {8: "MXFP8", 6: "MXFP6_E2M3", 4: "NVFP4", 3: "FP3_E1M1"}


# ══════════════════════════════════════════════════════════════════════════════
# FP Quantization functions
# (ported from pixart_layer_sensitivity.py — self-contained, no pixart dep)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_scale_dtype(scale: torch.Tensor, scale_dtype: str) -> torch.Tensor:
    """Simulate scale storage precision."""
    if scale_dtype == "FP32":
        return scale.float()
    elif scale_dtype == "FP16":
        return scale.half().float()
    elif scale_dtype == "BF16":
        return scale.bfloat16().float()
    elif scale_dtype == "NVFP8":
        return scale.to(torch.float8_e4m3fn).float()
    elif scale_dtype == "MXFP8":
        safe = scale.clamp(min=2 ** -127)
        return 2.0 ** torch.floor(torch.log2(safe))
    else:
        raise ValueError(f"Unknown scale_dtype: {scale_dtype}")


_GRID_CACHE: dict = {}


def _build_mxfp_grid(exp_bits: int, man_bits: int) -> torch.Tensor:
    key = (exp_bits, man_bits)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    if exp_bits == 0:
        vals = {m / (1 << man_bits) for m in range(1 << man_bits)}
        grid = torch.tensor(sorted(vals), dtype=torch.float32)
        _GRID_CACHE[key] = grid
        return grid
    exp_bias = (1 << (exp_bits - 1)) - 1
    max_exp  = (1 << exp_bits) - 1
    vals = {0.0}
    if man_bits > 0:
        for m in range(1, 1 << man_bits):
            vals.add((m / (1 << man_bits)) * (2.0 ** (1 - exp_bias)))
    for e in range(1, max_exp + 1):
        if man_bits == 0:
            vals.add(2.0 ** (e - exp_bias))
        else:
            for m in range(1 << man_bits):
                vals.add((1.0 + m / (1 << man_bits)) * (2.0 ** (e - exp_bias)))
    grid = torch.tensor(sorted(vals), dtype=torch.float32)
    _GRID_CACHE[key] = grid
    return grid


def _snap_to_grid(abs_flat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    pos  = torch.searchsorted(grid, abs_flat)
    lo   = (pos - 1).clamp(min=0)
    hi   = pos.clamp(max=grid.numel() - 1)
    v_lo = grid[lo]
    v_hi = grid[hi]
    return torch.where((abs_flat - v_lo) > (v_hi - abs_flat), v_hi, v_lo)


def _quant_mxfp(w: torch.Tensor, block_size: int, scale_dtype: str,
                exp_bits: int, man_bits: int) -> torch.Tensor:
    grid     = _build_mxfp_grid(exp_bits, man_bits).to(w.device)
    grid_max = grid[-1].item()
    orig     = w.shape
    wf       = w.reshape(-1, block_size).float()
    amax     = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale    = _apply_scale_dtype(amax / grid_max, scale_dtype)
    w_norm   = wf / scale.clamp(min=1e-12)
    sign     = w_norm.sign()
    snapped  = _snap_to_grid(w_norm.abs().reshape(-1), grid).reshape(w_norm.abs().shape)
    return (sign * snapped * scale).reshape(orig).to(w.dtype)


def _quantize_nvfp4(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """NVFP4: non-uniform 4-bit {0,±0.5,±1,±1.5,±2,±3,±4,±6}."""
    if not hasattr(_quantize_nvfp4, "_cache"):
        _quantize_nvfp4._cache = {}
    if w.device not in _quantize_nvfp4._cache:
        _quantize_nvfp4._cache[w.device] = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device, dtype=torch.float32
        )
    levels = _quantize_nvfp4._cache[w.device]
    orig   = w.shape
    wf     = w.reshape(-1, block_size).float()
    amax   = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale  = _apply_scale_dtype(amax / 6.0, scale_dtype)
    w_norm = wf.abs() / scale.clamp(min=1e-12)
    snapped = _snap_to_grid(w_norm.reshape(-1), levels).reshape(w_norm.shape)
    return (wf.sign() * snapped * scale).reshape(orig).to(w.dtype)


def _quantize_mxfp4(w, bs, sd):   return _quant_mxfp(w, bs, sd, exp_bits=2, man_bits=1)
def _quantize_mxfp6_e2m3(w, bs, sd): return _quant_mxfp(w, bs, sd, exp_bits=2, man_bits=3)
def _quantize_mxfp6_e3m2(w, bs, sd): return _quant_mxfp(w, bs, sd, exp_bits=3, man_bits=2)
def _quantize_fp3_e1m1(w, bs, sd): return _quant_mxfp(w, bs, sd, exp_bits=1, man_bits=1)
def _quantize_fp3_e2m0(w, bs, sd): return _quant_mxfp(w, bs, sd, exp_bits=2, man_bits=0)
def _quantize_fp3_e0m2(w, bs, sd): return _quant_mxfp(w, bs, sd, exp_bits=0, man_bits=2)


def _quantize_mxfp8(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    orig   = w.shape
    wf     = w.reshape(-1, block_size).float()
    q_max  = 448.0
    amax   = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale  = _apply_scale_dtype(amax / q_max, scale_dtype)
    w_fp8  = (wf / scale.clamp(min=1e-12)).to(torch.float8_e4m3fn).float()
    return (w_fp8 * scale).reshape(orig).to(w.dtype)


def _quantize_nvfp8(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    # Same data format as MXFP8; scale_dtype distinguishes NV (FP16) vs MX (E8M0)
    return _quantize_mxfp8(w, block_size, scale_dtype)


QUANT_FNS: dict = {
    "MXFP8":      _quantize_mxfp8,
    "NVFP8":      _quantize_nvfp8,
    "MXFP6_E2M3": _quantize_mxfp6_e2m3,
    "MXFP6_E3M2": _quantize_mxfp6_e3m2,
    "NVFP4":      _quantize_nvfp4,
    "MXFP4":      _quantize_mxfp4,
    "FP3_E1M1":   _quantize_fp3_e1m1,
    "FP3_E2M0":   _quantize_fp3_e2m0,
    "FP3_E0M2":   _quantize_fp3_e0m2,
}

ALL_FORMATS = list(QUANT_FNS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# FP Activation Quantization
# ══════════════════════════════════════════════════════════════════════════════

def quant_act_fp(x: torch.Tensor, fmt: str, block_size: int = 16) -> torch.Tensor:
    """Dynamic FP activation quantization.

    Reshapes x to [*, block_size], applies QUANT_FNS[fmt] per block,
    restores original shape.  Supports all 9 FP formats.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d  = x.reshape(-1, orig_shape[-1]).float()   # [tokens, hidden]
    tokens, hidden = x_2d.shape
    pad = (-hidden) % block_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad))
    # QUANT_FNS expects [N, block_size]: treat tokens as "out_features", blocks as "in_features"
    x_q = QUANT_FNS[fmt](x_2d, block_size, "FP16")  # FP16 scale for activations
    x_q = x_q[:, :hidden]
    return x_q.reshape(orig_shape).to(orig_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# LayerConfig dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FPLayerConfig:
    full_name:  str
    component:  str          # "lm" | "dit"
    layer_type: str
    attn_mlp:   str
    w_fmt:      str          # e.g. "MXFP8", "NVFP4", "FP3_E1M1"
    a_fmt:      str | None   # None → weight-only
    block_size: int          # 16 | 32 | 64
    scale_dtype: str         # "FP16" | "BF16" | "FP32" | "NVFP8" | "MXFP8"
    kurtosis_w: float
    cv_w:       float
    cv_act:     float | None


# ══════════════════════════════════════════════════════════════════════════════
# dist_stats 로드 & 분류
# ══════════════════════════════════════════════════════════════════════════════

def load_dist_stats(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    return data["weight_stats"], data["activation_stats"]


def _classify_component(full_name: str) -> str:
    for pat in SKIP_PATTERNS:
        if pat in full_name:
            return "skip"
    if any(pat in full_name for pat in LM_PATTERNS):
        return "lm"
    if any(pat in full_name for pat in DIT_PATTERNS):
        return "dit"
    return "skip"


def _assign_block_size(cv_w: float, cv_act: float | None) -> int:
    """CV 기반 block size 선택 (큰 CV → 작은 block)."""
    cv = max(cv_w, cv_act if cv_act is not None else 0.0)
    if cv > 0.5:   return 16
    if cv > 0.25:  return 32
    return 64


# ══════════════════════════════════════════════════════════════════════════════
# Format assignment
# ══════════════════════════════════════════════════════════════════════════════

def _bits_from_thresh(kurt: float, cv: float, thresh: dict) -> int:
    """기존 INT bit-width 임계값 로직을 재사용해 bit 그룹 결정.
    FP는 최소 3-bit이므로 INT2/INT1 → 3으로 올림.
    """
    if kurt > thresh["int8_kurt"] or cv > thresh["int8_cv"]:
        return 8
    elif kurt > thresh["int6_kurt"] or cv > thresh["int6_cv"]:
        return 6
    elif kurt > thresh["int4_kurt"] or cv > thresh["int4_cv"]:
        return 4
    return 3  # INT3/2/1 모두 FP3 그룹으로


def _assign_w_format(kurt: float, cv: float, thresh: dict, min_bits: int,
                     fmt_map: dict | None = None) -> str:
    """Weight kurtosis/CV → FP format.
    min_bits: 이 값 이상의 bit 그룹 사용 (예: 4 → FP3 불가, 최소 NVFP4)
    fmt_map: {8: fmt, 6: fmt, 4: fmt, 3: fmt} — None이면 _DEFAULT_FMT 사용
    """
    fmap = fmt_map if fmt_map is not None else _DEFAULT_FMT
    bits = _bits_from_thresh(kurt, cv, thresh)
    bits = max(bits, min_bits)
    bits = min(bits, 8)
    # 유효 그룹으로 정규화 (3, 4, 6, 8 중 하나)
    if bits > 6:   bits = 8
    elif bits > 4: bits = 6
    elif bits > 3: bits = 4
    return fmap[bits]


def _assign_a_format(kurt: float, cv: float, thresh_a: dict,
                     min_bits: int, max_bits: int,
                     fmt_map: dict | None = None,
                     force: bool = False) -> str | None:
    """Activation kurtosis/CV → FP format.
    NaN stats → None (weight-only), unless force=True.
    fmt_map: {8: fmt, 6: fmt, 4: fmt, 3: fmt} — None이면 _DEFAULT_FMT 사용
    """
    fmap = fmt_map if fmt_map is not None else _DEFAULT_FMT
    if math.isnan(kurt) or math.isnan(cv):
        if force:
            bits = min_bits
        else:
            return None
    else:
        bits = _bits_from_thresh(kurt, cv, thresh_a)
        bits = max(bits, min_bits)
        bits = min(bits, max_bits)
        bits = min(bits, 8)

    if bits > 6:   bits = 8
    elif bits > 4: bits = 6
    elif bits > 3: bits = 4
    return fmap[bits]


# ══════════════════════════════════════════════════════════════════════════════
# build_layer_configs
# ══════════════════════════════════════════════════════════════════════════════

def build_layer_configs(
    weight_stats: dict,
    activation_stats: dict,
    target: str,
    enable_act_quant: bool,
    scale_dtype: str,
    thresh: dict,
    thresh_a: dict,
    lm_min_bits:   int = 3,
    dit_min_bits:  int = 3,
    lm_a_min_bits: int = 3,
    dit_a_min_bits: int = 3,
    lm_a_max_bits: int = 8,
    dit_a_max_bits: int = 8,
    force_act_quant: bool = False,
    fmt_map: dict | None = None,
) -> dict[str, FPLayerConfig]:
    configs: dict[str, FPLayerConfig] = {}

    for full_name, wstat in weight_stats.items():
        comp = _classify_component(full_name)
        if comp == "skip":
            continue
        if target == "lm"  and comp != "lm":
            continue
        if target == "dit" and comp != "dit":
            continue

        astat  = activation_stats.get(full_name, {})
        kurt_a_raw = astat.get("kurtosis", float("nan"))
        cv_a_raw   = astat.get("per_token_absmax_cv", float("nan"))
        try:    kurt_a = float(kurt_a_raw)
        except: kurt_a = float("nan")
        try:    cv_a   = float(cv_a_raw)
        except: cv_a   = float("nan")

        kurt_w = float(wstat.get("kurtosis", 0.0))
        cv_w   = float(wstat.get("per_channel_absmax_cv", 0.0))
        cv_act = cv_a if not math.isnan(cv_a) else None

        w_min  = lm_min_bits  if comp == "lm" else dit_min_bits
        w_fmt  = _assign_w_format(kurt_w, cv_w, thresh, min_bits=w_min, fmt_map=fmt_map)

        if enable_act_quant:
            a_min = lm_a_min_bits  if comp == "lm" else dit_a_min_bits
            a_max = lm_a_max_bits  if comp == "lm" else dit_a_max_bits
            a_fmt = _assign_a_format(
                kurt_a, cv_a, thresh_a,
                min_bits=a_min, max_bits=a_max,
                fmt_map=fmt_map,
                force=force_act_quant,
            )
        else:
            a_fmt = None

        block_size = _assign_block_size(cv_w, cv_act)

        configs[full_name] = FPLayerConfig(
            full_name=full_name,
            component=comp,
            layer_type=wstat.get("layer_type", "unknown"),
            attn_mlp=wstat.get("attn_mlp", "unknown"),
            w_fmt=w_fmt,
            a_fmt=a_fmt,
            block_size=block_size,
            scale_dtype=scale_dtype,
            kurtosis_w=kurt_w,
            cv_w=cv_w,
            cv_act=cv_act,
        )

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# MixedFPQuantizedLinear
# ══════════════════════════════════════════════════════════════════════════════

class MixedFPQuantizedLinear(nn.Module):
    """FP format quantized Linear — stores FP-quantized (dequantized) weight.

    Weight is pre-quantized at replacement time:
        W_fp = QUANT_FNS[w_fmt](W_original, block_size, scale_dtype)
    Stored as model_dtype (BF16/FP16) — simulation, no int packing.

    Forward:
        1. Cast x to compute_dtype (prevents float32 propagation from LayerNorm)
        2. Optionally: activation FP quantization
        3. F.linear(x, W_fp, bias)
    """

    def __init__(
        self,
        w_quantized: torch.Tensor,   # FP-quantized weight, shape [out_f, in_f]
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        w_fmt: str,
        a_fmt: str | None,
        block_size: int,
        scale_dtype: str,
        model_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.register_buffer("w_quantized", w_quantized)
        _bd = model_dtype if model_dtype is not None else torch.float16
        self.bias = nn.Parameter(bias.to(_bd), requires_grad=False) if bias is not None else None
        self._in_features  = in_features
        self._out_features = out_features
        self.w_fmt      = w_fmt
        self.a_fmt      = a_fmt
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        self.model_dtype = model_dtype

    @property
    def in_features(self)  -> int: return self._in_features
    @property
    def out_features(self) -> int: return self._out_features
    @property
    def weight(self) -> torch.Tensor: return self.w_quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self.model_dtype if self.model_dtype is not None else x.dtype
        x_in = x.to(compute_dtype)
        if self.a_fmt is not None:
            x_in = quant_act_fp(x_in, self.a_fmt, self.block_size)
        W    = self.w_quantized.to(compute_dtype)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x_in, W, bias)


# ══════════════════════════════════════════════════════════════════════════════
# 모듈 교체
# ══════════════════════════════════════════════════════════════════════════════

def _replace_linear_fp_recursive(
    module: nn.Module,
    layer_configs: dict[str, FPLayerConfig],
    report: list[dict],
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            cfg = layer_configs.get(full_name)
            if cfg is None:
                continue

            out_f, in_f = child.weight.shape
            model_dtype  = child.weight.dtype
            W_fp32 = child.weight.data.float()

            # FP quantize weight (simulate)
            W_q = QUANT_FNS[cfg.w_fmt](W_fp32, cfg.block_size, cfg.scale_dtype)
            bias = child.bias.data if child.bias is not None else None

            new_layer = MixedFPQuantizedLinear(
                w_quantized = W_q.to(model_dtype),
                bias        = bias,
                in_features = in_f,
                out_features= out_f,
                w_fmt       = cfg.w_fmt,
                a_fmt       = cfg.a_fmt,
                block_size  = cfg.block_size,
                scale_dtype = cfg.scale_dtype,
                model_dtype = model_dtype,
            )
            setattr(module, name, new_layer)
            report.append({
                "layer":      full_name,
                "w_fmt":      cfg.w_fmt,
                "a_fmt":      cfg.a_fmt,
                "block_size": cfg.block_size,
                "scale_dtype": cfg.scale_dtype,
                "shape":      [out_f, in_f],
            })

        else:
            _replace_linear_fp_recursive(child, layer_configs, report, prefix=full_name)


def apply_mixed_fp_quant(
    policy: nn.Module,
    layer_configs: dict[str, FPLayerConfig],
) -> list[dict]:
    report: list[dict] = []
    _replace_linear_fp_recursive(policy, layer_configs, report, prefix="")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 표 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_layer_table(layer_configs: dict[str, FPLayerConfig]) -> None:
    COL = 70
    HEADER = (
        f"{'Layer':<{COL}} {'Comp':<5} {'Type':<14}"
        f" {'W_fmt':<12} {'A_fmt':<12} {'Blk':<5} {'Scale':<7}"
        f" {'Kurt_W':>8} {'CV_W':>6} {'CV_Act':>7}"
    )
    SEP = "─" * len(HEADER)
    print(SEP)
    print(HEADER)
    print(SEP)

    for comp in ("lm", "dit"):
        group = [(k, v) for k, v in layer_configs.items() if v.component == comp]
        if not group:
            continue
        print(f"  ── {comp.upper()} ({len(group)} layers) ──")
        for full_name, cfg in group:
            short = ("…" + full_name[-(COL - 1):]) if len(full_name) > COL else full_name
            a_str    = cfg.a_fmt if cfg.a_fmt is not None else "WO"
            cv_str   = f"{cfg.cv_act:.3f}" if cfg.cv_act is not None else "  NaN"
            print(
                f"{short:<{COL}} {comp:<5} {cfg.layer_type:<14}"
                f" {cfg.w_fmt:<12} {a_str:<12} {cfg.block_size:<5} {cfg.scale_dtype:<7}"
                f" {cfg.kurtosis_w:>8.2f} {cfg.cv_w:>6.3f} {cv_str:>7}"
            )

    print(SEP)
    total = len(layer_configs)

    # Weight format distribution
    w_cnt: dict[str, int] = {}
    for cfg in layer_configs.values():
        w_cnt[cfg.w_fmt] = w_cnt.get(cfg.w_fmt, 0) + 1
    print(f"\nWeight format  (total {total} layers):")
    for fmt in ALL_FORMATS:
        n = w_cnt.get(fmt, 0)
        if n == 0:
            continue
        bar = "█" * int(30 * n / total)
        print(f"  {fmt:<12}: {n:4d} ({100*n/total:5.1f}%)  {bar}")

    # Activation format distribution
    a_cnt: dict = {}
    for cfg in layer_configs.values():
        a_cnt[cfg.a_fmt] = a_cnt.get(cfg.a_fmt, 0) + 1
    print(f"\nActivation format  (total {total} layers):")
    for fmt in ALL_FORMATS + [None]:
        n = a_cnt.get(fmt, 0)
        if n == 0:
            continue
        label = fmt if fmt is not None else "weight-only"
        bar   = "█" * int(30 * n / total)
        print(f"  {label:<12}: {n:4d} ({100*n/total:5.1f}%)  {bar}")

    # Block size distribution
    blk_cnt: dict[int, int] = {}
    for cfg in layer_configs.values():
        blk_cnt[cfg.block_size] = blk_cnt.get(cfg.block_size, 0) + 1
    print(f"\nBlock size  (total {total} layers):")
    for b in sorted(blk_cnt):
        n = blk_cnt[b]
        bar = "█" * int(30 * n / total)
        print(f"  blk{b:<3}: {n:4d} ({100*n/total:5.1f}%)  {bar}")
    print()


def save_layer_table_csv(layer_configs: dict[str, FPLayerConfig], out_path: Path) -> None:
    import csv as _csv

    total = len(layer_configs)
    rows  = []
    for full_name, cfg in layer_configs.items():
        rows.append({
            "layer_name":  full_name,
            "short_name":  ("…" + full_name[-49:]) if len(full_name) > 50 else full_name,
            "component":   cfg.component,
            "layer_type":  cfg.layer_type,
            "attn_mlp":    cfg.attn_mlp,
            "w_fmt":       cfg.w_fmt,
            "w_bits":      FORMAT_BITS[cfg.w_fmt],
            "a_fmt":       cfg.a_fmt if cfg.a_fmt is not None else "weight-only",
            "a_bits":      FORMAT_BITS[cfg.a_fmt] if cfg.a_fmt is not None else "",
            "block_size":  cfg.block_size,
            "scale_dtype": cfg.scale_dtype,
            "kurtosis_w":  round(cfg.kurtosis_w, 4),
            "cv_w":        round(cfg.cv_w, 4),
            "cv_act":      round(cfg.cv_act, 4) if cfg.cv_act is not None else "",
        })

    w_cnt: dict[str, int]  = {}
    a_cnt: dict            = {}
    blk_cnt: dict[int, int] = {}
    for cfg in layer_configs.values():
        w_cnt[cfg.w_fmt] = w_cnt.get(cfg.w_fmt, 0) + 1
        a_cnt[cfg.a_fmt] = a_cnt.get(cfg.a_fmt, 0) + 1
        blk_cnt[cfg.block_size] = blk_cnt.get(cfg.block_size, 0) + 1

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        f.write("\n")
        f.write("# SUMMARY\n")
        f.write("stat_category,label,w_bits,count,pct,bar\n")

        f.write(f"[weight_fmt total={total}],,,,,\n")
        for fmt in ALL_FORMATS:
            n = w_cnt.get(fmt, 0)
            if n == 0:
                continue
            bar = "█" * int(20 * n / total)
            f.write(f"weight_fmt,{fmt},{FORMAT_BITS[fmt]},{n},{100*n/total:.1f},{bar}\n")

        f.write(f"[activation_fmt total={total}],,,,,\n")
        for fmt in ALL_FORMATS + [None]:
            n = a_cnt.get(fmt, 0)
            if n == 0:
                continue
            label = fmt if fmt is not None else "weight-only"
            bits  = FORMAT_BITS[fmt] if fmt is not None else ""
            bar   = "█" * int(20 * n / total)
            f.write(f"activation_fmt,{label},{bits},{n},{100*n/total:.1f},{bar}\n")

        f.write(f"[block_size total={total}],,,,,\n")
        for b in sorted(blk_cnt):
            n = blk_cnt[b]
            bar = "█" * int(20 * n / total)
            f.write(f"block_size,blk{b},,{n},{100*n/total:.1f},{bar}\n")

    print(f"[SAVED] {out_path}  ({len(rows)} layers + summary)")


# ══════════════════════════════════════════════════════════════════════════════
# torch.compile 비활성화
# ══════════════════════════════════════════════════════════════════════════════

def disable_torch_compile(policy: nn.Module) -> None:
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is None:
        return
    for attr in ("sample_actions", "forward"):
        fn  = getattr(inner, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
        if orig is not None:
            setattr(inner, attr, orig)
            print(f"[INFO] torch.compile disabled for model.{attr}")


# ══════════════════════════════════════════════════════════════════════════════
# Coverage report
# ══════════════════════════════════════════════════════════════════════════════

def build_quant_report(policy: nn.Module) -> dict:
    total = 0
    quantized = 0
    for _name, mod in policy.named_modules():
        if list(mod.children()):
            continue
        if isinstance(mod, (nn.Linear, MixedFPQuantizedLinear)):
            total += 1
            if isinstance(mod, MixedFPQuantizedLinear):
                quantized += 1
    return {"total_linear": total, "quantized": quantized, "unquantized": total - quantized}


# ══════════════════════════════════════════════════════════════════════════════
# 단일 실험 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    pretrained_path: str,
    task: str,
    n_episodes: int,
    batch_size: int,
    device_str: str,
    target: str,
    enable_act_quant: bool,
    layer_configs: dict[str, FPLayerConfig],
    env_cfg,
    envs_dict: dict,
    use_amp: bool = False,
) -> dict:
    print(f"\n[RUN] target={target}  act_quant={enable_act_quant}")
    print(f"      configs={len(layer_configs)} layers")

    device = torch.device(device_str)

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = device_str
    policy_cfg.use_amp = use_amp

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device_str}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    layer_report = apply_mixed_fp_quant(policy, layer_configs)
    disable_torch_compile(policy)
    coverage = build_quant_report(policy)
    print(f"[INFO] Quantized: {coverage['quantized']} / {coverage['total_linear']}")
    print(f"[INFO] layer_report entries: {len(layer_report)}")

    suite_name = next(iter(envs_dict))
    task_id    = next(iter(envs_dict[suite_name]))
    env        = envs_dict[suite_name][task_id]

    with torch.no_grad(), (
        torch.autocast(device_type=device.type) if use_amp else nullcontext()
    ):
        eval_info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
        )

    agg        = eval_info.get("aggregated", {})
    pc_success = agg.get("pc_success", float("nan"))
    avg_reward = agg.get("avg_sum_reward", float("nan"))
    print(f"[RESULT] success={pc_success:.1f}%  reward={avg_reward:.4f}")

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "target":           target,
        "enable_act_quant": enable_act_quant,
        "layer_report":     layer_report,
        "coverage":         coverage,
        "eval_results":     eval_info,
        "pc_success":       pc_success,
        "avg_sum_reward":   avg_reward,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer mixed-precision FP quantization for LeRobot pi05"
    )
    # Model & experiment
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task",       type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="logs/mixed_fp_quant")
    parser.add_argument("--use_amp",    action="store_true")

    # Quantization options
    parser.add_argument("--target", type=str, default="all",
                        choices=["lm", "dit", "all"])
    parser.add_argument("--no_act_quant", action="store_true",
                        help="Weight-only mode (no activation quant)")
    parser.add_argument("--force_act_quant", action="store_true",
                        help="Force act quant even for NaN-stats layers (uses a_min format)")
    parser.add_argument("--scale_dtype", type=str, default="FP16",
                        choices=["FP16", "BF16", "FP32", "NVFP8", "MXFP8"],
                        help="Scale storage precision (FP16=NV-style, MXFP8=MX E8M0 style)")
    parser.add_argument("--dist_stats", type=str, default=str(DIST_STATS_PATH))

    # Minimum format (as bit-depth floor: 3, 4, 6, 8)
    parser.add_argument("--lm_min_bits",   type=int, default=3,
                        choices=[3, 4, 6, 8],
                        help="LM  weight minimum bit group (3→FP3, 4→NVFP4, 6→MXFP6, 8→MXFP8)")
    parser.add_argument("--dit_min_bits",  type=int, default=3,
                        choices=[3, 4, 6, 8],
                        help="DiT weight minimum bit group")
    parser.add_argument("--lm_a_min_bits", type=int, default=3,
                        choices=[3, 4, 6, 8],
                        help="LM  activation minimum bit group")
    parser.add_argument("--dit_a_min_bits",type=int, default=3,
                        choices=[3, 4, 6, 8],
                        help="DiT activation minimum bit group")
    parser.add_argument("--lm_a_max_bits", type=int, default=8,
                        choices=[3, 4, 6, 8],
                        help="LM  activation maximum bit group (default 8 = MXFP8)")
    parser.add_argument("--dit_a_max_bits",type=int, default=8,
                        choices=[3, 4, 6, 8],
                        help="DiT activation maximum bit group")

    # Weight bit-assignment thresholds (reuse INT thresholds — same logic)
    parser.add_argument("--thresh_w_int8_kurt", type=float, default=50.0)
    parser.add_argument("--thresh_w_int8_cv",   type=float, default=0.7)
    parser.add_argument("--thresh_w_int6_kurt", type=float, default=10.0)
    parser.add_argument("--thresh_w_int6_cv",   type=float, default=0.45)
    parser.add_argument("--thresh_w_int4_kurt", type=float, default=3.0)
    parser.add_argument("--thresh_w_int4_cv",   type=float, default=0.25)
    parser.add_argument("--thresh_w_int3_kurt", type=float, default=1.5)
    parser.add_argument("--thresh_w_int3_cv",   type=float, default=0.12)

    # Activation bit-assignment thresholds
    parser.add_argument("--thresh_a_int8_kurt", type=float, default=50.0)
    parser.add_argument("--thresh_a_int8_cv",   type=float, default=0.3)
    parser.add_argument("--thresh_a_int6_kurt", type=float, default=10.0)
    parser.add_argument("--thresh_a_int6_cv",   type=float, default=0.15)
    parser.add_argument("--thresh_a_int4_kurt", type=float, default=3.0)
    parser.add_argument("--thresh_a_int4_cv",   type=float, default=0.08)
    parser.add_argument("--thresh_a_int3_kurt", type=float, default=1.5)
    parser.add_argument("--thresh_a_int3_cv",   type=float, default=0.04)

    # Sub-format selection per bit group
    parser.add_argument("--fmt_8bit", type=str, default="MXFP8",
                        choices=["MXFP8", "NVFP8"],
                        help="Sub-format for 8-bit group (default: MXFP8)")
    parser.add_argument("--fmt_6bit", type=str, default="MXFP6_E2M3",
                        choices=["MXFP6_E2M3", "MXFP6_E3M2"],
                        help="Sub-format for 6-bit group (default: MXFP6_E2M3)")
    parser.add_argument("--fmt_4bit", type=str, default="NVFP4",
                        choices=["NVFP4", "MXFP4"],
                        help="Sub-format for 4-bit group (default: NVFP4)")
    parser.add_argument("--fmt_3bit", type=str, default="FP3_E1M1",
                        choices=["FP3_E1M1", "FP3_E2M0", "FP3_E0M2"],
                        help="Sub-format for 3-bit group (default: FP3_E1M1)")

    # Mode
    parser.add_argument("--table_only", action="store_true",
                        help="Print layer table only (no model load)")

    args = parser.parse_args()

    # ── Threshold dicts ────────────────────────────────────────────────────────
    thresh = {
        "int8_kurt": args.thresh_w_int8_kurt, "int8_cv": args.thresh_w_int8_cv,
        "int6_kurt": args.thresh_w_int6_kurt, "int6_cv": args.thresh_w_int6_cv,
        "int4_kurt": args.thresh_w_int4_kurt, "int4_cv": args.thresh_w_int4_cv,
        "int3_kurt": args.thresh_w_int3_kurt, "int3_cv": args.thresh_w_int3_cv,
    }
    thresh_a = {
        "int8_kurt": args.thresh_a_int8_kurt, "int8_cv": args.thresh_a_int8_cv,
        "int6_kurt": args.thresh_a_int6_kurt, "int6_cv": args.thresh_a_int6_cv,
        "int4_kurt": args.thresh_a_int4_kurt, "int4_cv": args.thresh_a_int4_cv,
        "int3_kurt": args.thresh_a_int3_kurt, "int3_cv": args.thresh_a_int3_cv,
    }

    # ── Load dist_stats & build layer configs ─────────────────────────────────
    dist_stats_path = Path(args.dist_stats)
    if not dist_stats_path.exists():
        print(f"[ERROR] dist_stats not found: {dist_stats_path}")
        sys.exit(1)

    print(f"[INFO] Loading dist_stats: {dist_stats_path}")
    weight_stats, activation_stats = load_dist_stats(dist_stats_path)

    # ── fmt_map 구성 (sub-format 선택) ────────────────────────────────────────
    fmt_map = {8: args.fmt_8bit, 6: args.fmt_6bit, 4: args.fmt_4bit, 3: args.fmt_3bit}
    fmt_map_defaults = {8: "MXFP8", 6: "MXFP6_E2M3", 4: "NVFP4", 3: "FP3_E1M1"}
    non_default_fmts = [v for k, v in fmt_map.items() if v != fmt_map_defaults[k]]
    if non_default_fmts:
        print(f"[INFO] Sub-format overrides: {fmt_map}")

    layer_configs = build_layer_configs(
        weight_stats=weight_stats,
        activation_stats=activation_stats,
        target=args.target,
        enable_act_quant=not args.no_act_quant,
        scale_dtype=args.scale_dtype,
        thresh=thresh,
        thresh_a=thresh_a,
        lm_min_bits=args.lm_min_bits,
        dit_min_bits=args.dit_min_bits,
        lm_a_min_bits=args.lm_a_min_bits,
        dit_a_min_bits=args.dit_a_min_bits,
        lm_a_max_bits=args.lm_a_max_bits,
        dit_a_max_bits=args.dit_a_max_bits,
        force_act_quant=args.force_act_quant,
        fmt_map=fmt_map,
    )
    print(f"[INFO] layer_configs built: {len(layer_configs)} layers  (target={args.target})")

    # ── exp key ────────────────────────────────────────────────────────────────
    act_label  = "wo" if args.no_act_quant else "fp_act"
    w_label    = f"_lmmin{args.lm_min_bits}_ditmin{args.dit_min_bits}"
    fa_suffix  = "_fa" if args.force_act_quant else ""
    sd_label   = f"_sd{args.scale_dtype}"
    fmt_suffix = ("_" + "_".join(non_default_fmts)) if non_default_fmts else ""
    exp_key    = f"mixed_fp_{args.target}{w_label}{fa_suffix}{fmt_suffix}_{act_label}{sd_label}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Table only ─────────────────────────────────────────────────────────────
    if args.table_only:
        print_layer_table(layer_configs)
        csv_path = output_dir / f"table_only_{exp_key}.csv"
        save_layer_table_csv(layer_configs, csv_path)
        return

    # ── Run experiment ─────────────────────────────────────────────────────────
    print(f"[INFO] Creating LIBERO env: task={args.task}, n_envs={args.batch_size}")
    env_cfg    = LiberoEnv(task=args.task)
    envs_dict  = make_env(env_cfg, n_envs=args.batch_size)
    suite_name = next(iter(envs_dict))
    task_id    = next(iter(envs_dict[suite_name]))
    print(f"[INFO] suite={suite_name}  task_id={task_id}")

    result = run_experiment(
        pretrained_path=args.pretrained_path,
        task=args.task,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        device_str=args.device,
        target=args.target,
        enable_act_quant=not args.no_act_quant,
        layer_configs=layer_configs,
        env_cfg=env_cfg,
        envs_dict=envs_dict,
        use_amp=args.use_amp,
    )

    # ── Serialize layer_configs ────────────────────────────────────────────────
    configs_serializable = {
        k: {
            "component":   v.component,
            "layer_type":  v.layer_type,
            "attn_mlp":    v.attn_mlp,
            "w_fmt":       v.w_fmt,
            "w_bits":      FORMAT_BITS[v.w_fmt],
            "a_fmt":       v.a_fmt,
            "a_bits":      FORMAT_BITS[v.a_fmt] if v.a_fmt else None,
            "block_size":  v.block_size,
            "scale_dtype": v.scale_dtype,
            "kurtosis_w":  round(v.kurtosis_w, 4),
            "cv_w":        round(v.cv_w, 4),
            "cv_act":      round(v.cv_act, 4) if v.cv_act is not None else None,
        }
        for k, v in layer_configs.items()
    }

    out_data = {
        "config": {
            "pretrained_path": args.pretrained_path,
            "task":           args.task,
            "suite_name":     suite_name,
            "task_id":        task_id,
            "n_episodes":     args.n_episodes,
            "batch_size":     args.batch_size,
            "target":         args.target,
            "enable_act_quant": not args.no_act_quant,
            "force_act_quant": args.force_act_quant,
            "scale_dtype":    args.scale_dtype,
            "lm_min_bits":    args.lm_min_bits,
            "dit_min_bits":   args.dit_min_bits,
            "lm_a_min_bits":  args.lm_a_min_bits,
            "dit_a_min_bits": args.dit_a_min_bits,
            "lm_a_max_bits":  args.lm_a_max_bits,
            "dit_a_max_bits": args.dit_a_max_bits,
            "thresh":         thresh,
            "thresh_a":       thresh_a,
        },
        "layer_configs":  configs_serializable,
        "pc_success":     result["pc_success"],
        "avg_sum_reward": result["avg_sum_reward"],
        "coverage":       result["coverage"],
        "layer_report":   result["layer_report"],
        "eval_results":   result["eval_results"],
    }

    out_path = output_dir / f"{exp_key}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")

    # Table (txt + csv)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_layer_table(layer_configs)
    table_txt = output_dir / f"{exp_key}_layer_table.txt"
    table_txt.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[SAVED] {table_txt}")

    csv_path = output_dir / f"{exp_key}_layer_table.csv"
    save_layer_table_csv(layer_configs, csv_path)

    print(f"\n{'='*60}")
    print(f"  Mixed FP Quantization Result")
    print(f"  Target     : {args.target}")
    print(f"  Act quant  : {'disabled' if args.no_act_quant else 'enabled'}")
    print(f"  Scale dtype: {args.scale_dtype}")
    print(f"  Layers     : {len(layer_configs)} quantized")
    print(f"  Success    : {result['pc_success']:.1f}%")
    print(f"  AvgReward  : {result['avg_sum_reward']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
