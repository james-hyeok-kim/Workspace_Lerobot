"""
Per-Layer Mixed-Precision INT Quantization for LeRobot pi05.

레이어별 distribution 통계(kurtosis / CV)를 기반으로 각 Linear 레이어에
다른 bit-width 및 block size를 자동 배정하는 mixed-precision INT 양자화.

저장 포맷:
  - Weight : torch.int8  (INT3~INT8 값을 int8에 저장)
  - Scale  : torch.int32 (고정소수점 2^24 기반)
  - Block  : 16 / 32 / 64 (레이어별 자동 선택)

Usage:
    # 표만 보기 (모델 로드 없음, ~1초)
    python eval_mixed_int_quant.py --table_only --target all

    # LM만 quantize 후 eval
    python eval_mixed_int_quant.py --task libero_spatial --n_episodes 20 --target lm

    # DiT만, weight-only
    python eval_mixed_int_quant.py --task libero_spatial --n_episodes 20 --target dit --no_act_quant

    # 임계값 조정하여 표 확인
    python eval_mixed_int_quant.py --table_only --thresh_w_int8_kurt 30 --thresh_w_int8_cv 0.5
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

SCALE_SHIFT = 24  # INT32 고정소수점: scale_fp × 2^24

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
# lm_head: 257152×N embedding 헤드 제외
SKIP_PATTERNS = ["vision_tower", "embed_tokens", "lm_head"]


# ══════════════════════════════════════════════════════════════════════════════
# LayerConfig 데이터클래스
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerConfig:
    full_name: str
    component: str        # "lm" | "dit"
    layer_type: str       # "q_proj", "gate_proj", …
    attn_mlp: str         # "attn" | "mlp" | "action_head" | "other"
    w_bits: int           # 3–8
    a_bits: int | None    # None = weight-only
    block_size: int       # 16 | 32 | 64
    kurtosis_w: float
    cv_w: float
    cv_act: float | None


# ══════════════════════════════════════════════════════════════════════════════
# 분포 통계 로드 & 설정 배정
# ══════════════════════════════════════════════════════════════════════════════

def load_dist_stats(path: Path) -> tuple[dict, dict]:
    """dist_stats.json → (weight_stats, activation_stats)."""
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
    return "skip"  # 보수적으로 skip


def _assign_w_bits(kurt: float, cv: float, thresh: dict) -> int:
    if kurt > thresh["int8_kurt"] or cv > thresh["int8_cv"]:
        return 8
    if kurt > thresh["int6_kurt"] or cv > thresh["int6_cv"]:
        return 6
    if kurt > thresh["int4_kurt"] or cv > thresh["int4_cv"]:
        return 4
    if kurt > thresh["int3_kurt"] or cv > thresh["int3_cv"]:
        return 3
    if kurt > thresh["int2_kurt"] or cv > thresh["int2_cv"]:
        return 2
    return 1


def _assign_a_bits(kurt: float, cv: float, thresh_a: dict | None = None) -> int | None:
    """activation bit 배정. thresh_a가 None이면 기본 임계값 사용.
    NaN stats (미수집 레이어) → None (weight-only). force_act_quant 플래그는
    build_layer_configs에서 None을 a_min_bits로 교체한다.
    """
    if math.isnan(kurt) or math.isnan(cv) or kurt > 200 or cv > 0.6:
        return None
    t = thresh_a or {}
    if kurt > t.get("int8_kurt", 50) or cv > t.get("int8_cv", 0.3):
        return 8
    if kurt > t.get("int6_kurt", 10) or cv > t.get("int6_cv", 0.15):
        return 6
    if kurt > t.get("int4_kurt", 3) or cv > t.get("int4_cv", 0.08):
        return 4
    if kurt > t.get("int3_kurt", 1.5) or cv > t.get("int3_cv", 0.04):
        return 3
    if kurt > t.get("int2_kurt", 0.5) or cv > t.get("int2_cv", 0.02):
        return 2
    return 1


def _assign_block_size(cv_w: float, cv_act: float | None) -> int:
    cv_max = cv_w
    if cv_act is not None and not math.isnan(cv_act):
        cv_max = max(cv_max, cv_act)
    if cv_max > 0.5:
        return 16
    if cv_max > 0.25:
        return 32
    return 64


def build_layer_configs(
    weight_stats: dict,
    activation_stats: dict,
    target: str,
    enable_act_quant: bool,
    thresh: dict,
    lm_min_bits: int = 3,
    dit_min_bits: int = 3,
    lm_a_min_bits: int = 1,
    dit_a_min_bits: int = 1,
    lm_a_max_bits: int = 16,
    dit_a_max_bits: int = 16,
    thresh_a: dict | None = None,
    force_act_quant: bool = False,
) -> dict[str, LayerConfig]:
    """
    dist_stats에서 레이어별 LayerConfig 생성.

    Args:
        target: "lm" | "dit" | "all"
        lm_min_bits:   LM weight 최소 bit-width (1~16)
        dit_min_bits:  DiT weight 최소 bit-width (1~16)
        lm_a_min_bits: LM activation 최소 bit-width (1~16)
        dit_a_min_bits: DiT activation 최소 bit-width (1~16)
        lm_a_max_bits: LM activation 최대 bit-width cap (1~16)
        dit_a_max_bits: DiT activation 최대 bit-width cap (1~16)
        enable_act_quant: False → 모든 레이어 weight-only
        thresh: weight bit 배정 임계값 dict
        thresh_a: activation bit 배정 임계값 dict (None→기본값)
        force_act_quant: True → NaN stats 레이어도 weight-only 대신 a_min_bits 사용
    """
    configs: dict[str, LayerConfig] = {}
    for full_name, wstat in weight_stats.items():
        comp = _classify_component(full_name)
        if comp == "skip":
            continue
        if target == "lm" and comp != "lm":
            continue
        if target == "dit" and comp != "dit":
            continue

        astat = activation_stats.get(full_name, {})
        kurt_a_raw = astat.get("kurtosis", float("nan"))
        cv_a_raw   = astat.get("per_token_absmax_cv", float("nan"))

        try:
            kurt_a = float(kurt_a_raw)
        except (TypeError, ValueError):
            kurt_a = float("nan")
        try:
            cv_a = float(cv_a_raw)
        except (TypeError, ValueError):
            cv_a = float("nan")

        cv_act = cv_a if not math.isnan(cv_a) else None

        kurt_w = float(wstat.get("kurtosis", 0.0))
        cv_w   = float(wstat.get("per_channel_absmax_cv", 0.0))

        w_bits    = _assign_w_bits(kurt_w, cv_w, thresh)
        # 컴포넌트별 최소 weight bit-width 강제
        w_min = lm_min_bits if comp == "lm" else dit_min_bits
        w_bits = max(w_bits, w_min)

        if enable_act_quant:
            a_bits = _assign_a_bits(kurt_a, cv_a, thresh_a)
            if a_bits is None and force_act_quant:
                # NaN stats 레이어도 강제 activation 양자화 (weight-only 방지)
                a_bits = lm_a_min_bits if comp == "lm" else dit_a_min_bits
            if a_bits is not None:
                # 컴포넌트별 min/max activation bit-width 강제
                a_min = lm_a_min_bits if comp == "lm" else dit_a_min_bits
                a_max = lm_a_max_bits if comp == "lm" else dit_a_max_bits
                a_bits = max(min(a_bits, a_max), a_min)
        else:
            a_bits = None
        block_size = _assign_block_size(cv_w, cv_act)

        configs[full_name] = LayerConfig(
            full_name=full_name,
            component=comp,
            layer_type=wstat.get("layer_type", "unknown"),
            attn_mlp=wstat.get("attn_mlp", "unknown"),
            w_bits=w_bits,
            a_bits=a_bits,
            block_size=block_size,
            kurtosis_w=kurt_w,
            cv_w=cv_w,
            cv_act=cv_act,
        )
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# INT 양자화 핵심 함수
# ══════════════════════════════════════════════════════════════════════════════

def quant_blockwise_int_storage(
    W: torch.Tensor,   # float32, [out_f, in_f]
    bits: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Block-wise symmetric INT 양자화 (INT1~INT16).

    INT1 : binary sign quantization  — scale = mean(|W_block|), W_q ∈ {-1, +1}
    INT2 : ternary                   — scale = max(|W_block|)/1, W_q ∈ {-1, 0, +1}
    INT3~INT8  : standard symmetric  — scale = max(|W_block|)/n_pos, stored as int8
    INT9~INT16 : standard symmetric  — scale = max(|W_block|)/n_pos, stored as int16

    Returns:
        W_int    : torch.int8 (bits≤8) or torch.int16 (bits 9~16)
                   shape [out_f, num_blocks * block_size]  (패딩 포함)
        scale_i32: torch.int32 [out_f, num_blocks]
        in_f     : original (패딩 전) in_features
    """
    assert 1 <= bits <= 16, f"bits={bits} must be 1~16"

    out_f, in_f = W.shape
    pad = (-in_f) % block_size
    W_p = F.pad(W, (0, pad)) if pad else W
    W_b = W_p.reshape(out_f, -1, block_size)   # [out_f, num_blocks, block_size]

    if bits == 1:
        # Binary: scale = mean(|W|) per block, W_q = sign(W)
        scale_fp = W_b.abs().mean(dim=-1).clamp(min=1e-8)     # [out_f, num_blocks]
        scale_i32 = (scale_fp * (2 ** SCALE_SHIFT)).round().to(torch.int32).clamp(min=1)
        W_sign = W_b.sign()
        W_sign[W_sign == 0] = 1   # 0 → +1
        W_int = W_sign.reshape(out_f, -1).to(torch.int8)
    else:
        # INT2~INT16: symmetric, clamp to [-n_pos, n_pos]
        # INT2:1, INT3:3, INT4:7, INT6:31, INT8:127, INT16:32767
        n_pos = 2 ** (bits - 1) - 1
        scale_fp = W_b.abs().amax(dim=-1).clamp(min=1e-8) / n_pos
        scale_i32 = (scale_fp * (2 ** SCALE_SHIFT)).round().to(torch.int32).clamp(min=1)
        scale_fp_recon = scale_i32.float() / (2 ** SCALE_SHIFT)
        W_q = (W_b / scale_fp_recon.unsqueeze(-1)).round().clamp(-n_pos, n_pos)
        # int8 for bits≤8, int16 for bits 9~16
        int_dtype = torch.int8 if bits <= 8 else torch.int16
        W_int = W_q.reshape(out_f, -1).to(int_dtype)

    return W_int, scale_i32, in_f


def _quant_act(x: torch.Tensor, bits: int, mode: str = "per_tensor", block_size: int = 64) -> torch.Tensor:
    """동적 활성화 양자화.

    mode:
        per_tensor  — 텐서 전체 단일 scale
        per_token   — 토큰(마지막 차원 제외) 별 scale
        per_block   — 마지막 차원을 block_size 단위로 나눠 블록별 scale

    bits: 1~16 (INT16: near-lossless, n_pos=32767)
    """
    assert 1 <= bits <= 16, f"bits={bits} must be 1~16"
    orig_dtype = x.dtype

    def _quant_block(t: torch.Tensor) -> torch.Tensor:
        """t: [..., block_size] → quantize and dequant in-place shape"""
        if bits == 1:
            scale = t.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            s = t.sign()
            s[s == 0] = 1
            return s * scale
        else:
            n_pos = 2 ** (bits - 1) - 1
            scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / n_pos
            return (t / scale).round().clamp(-n_pos, n_pos) * scale

    if mode == "per_token":
        orig_shape = x.shape
        x_2d = x.float().reshape(-1, orig_shape[-1])      # [tokens, in_f]
        if bits == 1:
            scale = x_2d.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            q = x_2d.sign(); q[q == 0] = 1; q = q * scale
        else:
            n_pos = 2 ** (bits - 1) - 1
            scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / n_pos
            q = (x_2d / scale).round().clamp(-n_pos, n_pos) * scale
        return q.reshape(orig_shape).to(orig_dtype)

    elif mode == "per_block":
        orig_shape = x.shape
        x_2d = x.float().reshape(-1, orig_shape[-1])       # [tokens, in_f]
        tokens, in_f = x_2d.shape
        pad = (-in_f) % block_size
        x_p = F.pad(x_2d, (0, pad)) if pad else x_2d       # [tokens, in_f_pad]
        x_b = x_p.reshape(tokens, -1, block_size)           # [tokens, n_blk, blk]
        q = _quant_block(x_b)
        return q.reshape(tokens, -1)[:, :in_f].reshape(orig_shape).to(orig_dtype)

    else:  # per_tensor
        if bits == 1:
            scale = x.float().abs().mean().clamp(min=1e-8)
            s = x.float().sign(); s[s == 0] = 1
            return (s * scale).to(orig_dtype)
        n_pos = 2 ** (bits - 1) - 1
        scale = x.float().abs().amax().clamp(min=1e-8) / n_pos
        return ((x.float() / scale).round().clamp(-n_pos, n_pos) * scale).to(orig_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# MixedIntQuantizedLinear
# ══════════════════════════════════════════════════════════════════════════════

class MixedIntQuantizedLinear(nn.Module):
    """
    INT-only 양자화 Linear 레이어.

    저장:
        W_int     : torch.int8 (bits≤8) / torch.int16 (bits 9~16)
                    [out_f, num_blocks * block_size]  (패딩 포함)
        scale_i32 : torch.int32 [out_f, num_blocks]
                    = round(scale_fp × 2^24), clamp(min=1)
        model_dtype: 원본 모델 weight dtype (BF16 등) — forward에서 compute dtype으로 사용

    Forward:
        1. scale_i32 → FP32 scale (/ 2^24)
        2. W_int × scale → model_dtype weight (패딩 strip)
        3. x를 model_dtype으로 캐스팅 (layer_norm float32 upcast 방지)
        4. (선택) activation 양자화
        5. F.linear
    """

    def __init__(
        self,
        W_int: torch.Tensor,
        scale_i32: torch.Tensor,
        in_features: int,
        block_size: int,
        bias: torch.Tensor | None,
        w_bits: int,
        a_bits: int | None,
        act_quant_mode: str = "per_tensor",
        model_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.register_buffer("W_int", W_int)
        self.register_buffer("scale_i32", scale_i32)
        self._in_features = in_features
        self.block_size = block_size
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_quant_mode = act_quant_mode
        # model_dtype: 원본 Linear weight의 dtype (e.g. bfloat16).
        # None이면 입력 dtype을 그대로 사용 (기존 동작).
        self.model_dtype = model_dtype
        _bias_dtype = model_dtype if model_dtype is not None else torch.float16
        self.bias = nn.Parameter(bias.to(_bias_dtype), requires_grad=False) if bias is not None else None

    @property
    def out_features(self) -> int:
        return self.W_int.shape[0]

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def weight(self) -> torch.Tensor:
        return self._dequantize()

    def _dequantize(self) -> torch.Tensor:
        """int8/int16 + int32 scale → model_dtype weight."""
        out_f = self.W_int.shape[0]
        scale_fp = self.scale_i32.float() / (2 ** SCALE_SHIFT)     # [out_f, num_blocks]
        W_b = self.W_int.float().reshape(out_f, -1, self.block_size)  # [out_f, num_blocks, block_size]
        W_dq = (W_b * scale_fp.unsqueeze(-1)).reshape(out_f, -1)    # [out_f, padded_in]
        W_stripped = W_dq[:, :self._in_features]
        target_dtype = self.model_dtype if self.model_dtype is not None else torch.float16
        return W_stripped.to(target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute dtype: model_dtype (e.g. bfloat16) if set, else x.dtype.
        # model_dtype 설정 시 float32 upcast (layer_norm 등)를 model dtype으로 돌려놓음.
        # → target=lm 실험에서 LM float32 출력이 unquantized DiT로 전파되는 문제 방지.
        compute_dtype = self.model_dtype if self.model_dtype is not None else x.dtype
        W = self._dequantize()                                          # already in compute_dtype
        x_in = x.to(compute_dtype)
        if self.a_bits is not None:
            x_in = _quant_act(x_in, self.a_bits, self.act_quant_mode, block_size=self.block_size)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x_in, W, bias)


# ══════════════════════════════════════════════════════════════════════════════
# 모듈 교체 (재귀 순회)
# ══════════════════════════════════════════════════════════════════════════════

def _replace_linear_mixed_recursive(
    module: nn.Module,
    layer_configs: dict[str, LayerConfig],
    act_quant_mode: str,
    report: list[dict],
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            cfg = layer_configs.get(full_name)
            if cfg is None:
                continue  # vision_tower, embed_tokens, lm_head, 범위 외 컴포넌트

            out_f, in_f = child.weight.shape
            W = child.weight.data.float()
            bias = child.bias.data.float() if child.bias is not None else None

            W_int, scale_i32, _ = quant_blockwise_int_storage(W, cfg.w_bits, cfg.block_size)
            new_layer = MixedIntQuantizedLinear(
                W_int=W_int,
                scale_i32=scale_i32,
                in_features=in_f,
                block_size=cfg.block_size,
                bias=bias,
                w_bits=cfg.w_bits,
                a_bits=cfg.a_bits,
                act_quant_mode=act_quant_mode,
                model_dtype=child.weight.dtype,
            )
            setattr(module, name, new_layer)
            report.append({
                "layer": full_name,
                "w_bits": cfg.w_bits,
                "a_bits": cfg.a_bits,
                "block_size": cfg.block_size,
                "shape": [out_f, in_f],
            })

        else:
            _replace_linear_mixed_recursive(
                child, layer_configs, act_quant_mode, report, prefix=full_name
            )


def apply_mixed_int_quant(
    policy: nn.Module,
    layer_configs: dict[str, LayerConfig],
    act_quant_mode: str = "per_tensor",
) -> list[dict]:
    """policy의 nn.Linear를 MixedIntQuantizedLinear로 교체."""
    report: list[dict] = []
    _replace_linear_mixed_recursive(policy, layer_configs, act_quant_mode, report, prefix="")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 표 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_layer_table(layer_configs: dict[str, LayerConfig]) -> None:
    """레이어별 양자화 설정을 표로 출력."""
    COL_LAYER = 70
    HEADER = (
        f"{'Layer':<{COL_LAYER}} {'Comp':<5} {'Type':<14} {'W':<5} {'A':<6}"
        f" {'Blk':<5} {'Kurt_W':>8} {'CV_W':>6} {'CV_Act':>7}"
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
            short = ("…" + full_name[-(COL_LAYER - 1):]) if len(full_name) > COL_LAYER else full_name
            a_str = f"INT{cfg.a_bits}" if cfg.a_bits is not None else "WO"
            cv_act_str = f"{cfg.cv_act:.3f}" if cfg.cv_act is not None else "  NaN"
            print(
                f"{short:<{COL_LAYER}} {comp:<5} {cfg.layer_type:<14}"
                f" INT{cfg.w_bits:<1} {a_str:<6} {cfg.block_size:<5}"
                f" {cfg.kurtosis_w:>8.2f} {cfg.cv_w:>6.3f} {cv_act_str:>7}"
            )

    print(SEP)
    total = len(layer_configs)

    # Weight bits 분포
    w_cnt: dict[int, int] = {}
    for cfg in layer_configs.values():
        w_cnt[cfg.w_bits] = w_cnt.get(cfg.w_bits, 0) + 1
    print(f"\nWeight bits  (total {total} layers):")
    for b in sorted(w_cnt):
        n = w_cnt[b]
        bar = "█" * int(30 * n / total)
        print(f"  INT{b}: {n:4d} ({100*n/total:5.1f}%)  {bar}")

    # Activation bits 분포
    a_cnt: dict = {}
    for cfg in layer_configs.values():
        a_cnt[cfg.a_bits] = a_cnt.get(cfg.a_bits, 0) + 1
    print(f"\nActivation bits  (total {total} layers):")
    for b in sorted(a_cnt.keys(), key=lambda x: (x is None, x)):
        n = a_cnt[b]
        label = f"INT{b}" if b is not None else "weight-only"
        bar = "█" * int(30 * n / total)
        print(f"  {label:<11}: {n:4d} ({100*n/total:5.1f}%)  {bar}")

    # Block size 분포
    blk_cnt: dict[int, int] = {}
    for cfg in layer_configs.values():
        blk_cnt[cfg.block_size] = blk_cnt.get(cfg.block_size, 0) + 1
    print(f"\nBlock size  (total {total} layers):")
    for b in sorted(blk_cnt):
        n = blk_cnt[b]
        bar = "█" * int(30 * n / total)
        print(f"  blk{b:<3}: {n:4d} ({100*n/total:5.1f}%)  {bar}")

    print()


def save_layer_table_csv(layer_configs: dict[str, LayerConfig], out_path: Path) -> None:
    """레이어별 양자화 설정을 CSV로 저장. 파일 끝에 분포 요약 섹션 추가."""
    import csv as _csv

    total = len(layer_configs)

    # ── per-layer 행 구성 ────────────────────────────────────────────────────
    rows = []
    for full_name, cfg in layer_configs.items():
        rows.append({
            "layer_name":  full_name,
            "short_name":  ("…" + full_name[-49:]) if len(full_name) > 50 else full_name,
            "component":   cfg.component,
            "layer_type":  cfg.layer_type,
            "attn_mlp":    cfg.attn_mlp,
            "w_bits":      cfg.w_bits,
            "w_label":     f"INT{cfg.w_bits}",
            "a_bits":      cfg.a_bits if cfg.a_bits is not None else "",
            "a_label":     f"INT{cfg.a_bits}" if cfg.a_bits is not None else "weight-only",
            "block_size":  cfg.block_size,
            "kurtosis_w":  round(cfg.kurtosis_w, 4),
            "cv_w":        round(cfg.cv_w, 4),
            "cv_act":      round(cfg.cv_act, 4) if cfg.cv_act is not None else "",
        })

    # ── 분포 통계 계산 ───────────────────────────────────────────────────────
    w_cnt: dict[int, int] = {}
    a_cnt: dict = {}
    blk_cnt: dict[int, int] = {}
    for cfg in layer_configs.values():
        w_cnt[cfg.w_bits] = w_cnt.get(cfg.w_bits, 0) + 1
        a_cnt[cfg.a_bits] = a_cnt.get(cfg.a_bits, 0) + 1
        blk_cnt[cfg.block_size] = blk_cnt.get(cfg.block_size, 0) + 1

    # ── CSV 쓰기 ─────────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys()) if rows else [
        "layer_name","short_name","component","layer_type","attn_mlp",
        "w_bits","w_label","a_bits","a_label","block_size",
        "kurtosis_w","cv_w","cv_act",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        # ── 빈 행 + 요약 섹션 헤더 ─────────────────────────────────────────
        f.write("\n")
        f.write("# SUMMARY\n")
        f.write(f"stat_category,label,count,pct,bar\n")

        # Weight bits 분포
        f.write(f"[weight_bits total={total}],,,,\n")
        for b in sorted(w_cnt):
            n = w_cnt[b]
            bar = "█" * int(20 * n / total)
            f.write(f"weight_bits,INT{b},{n},{100*n/total:.1f},{bar}\n")

        # Activation bits 분포
        f.write(f"[activation_bits total={total}],,,,\n")
        for b in sorted(a_cnt.keys(), key=lambda x: (x is None, x or 0)):
            n = a_cnt[b]
            label = f"INT{b}" if b is not None else "weight-only"
            bar = "█" * int(20 * n / total)
            f.write(f"activation_bits,{label},{n},{100*n/total:.1f},{bar}\n")

        # Block size 분포
        f.write(f"[block_size total={total}],,,,\n")
        for b in sorted(blk_cnt):
            n = blk_cnt[b]
            bar = "█" * int(20 * n / total)
            f.write(f"block_size,blk{b},{n},{100*n/total:.1f},{bar}\n")

    print(f"[SAVED] {out_path}  ({len(rows)} layers + summary)")


# ══════════════════════════════════════════════════════════════════════════════
# torch.compile 비활성화 (eval_quant_sweep.py에서 복사)
# ══════════════════════════════════════════════════════════════════════════════

def disable_torch_compile(policy: nn.Module) -> None:
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is None:
        return
    for attr in ("sample_actions", "forward"):
        fn = getattr(inner, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
        if orig is not None:
            setattr(inner, attr, orig)
            print(f"[INFO] torch.compile disabled for model.{attr}")


# ══════════════════════════════════════════════════════════════════════════════
# 커버리지 리포트
# ══════════════════════════════════════════════════════════════════════════════

def build_quant_report(policy: nn.Module) -> dict:
    """양자화된/미양자화 Linear 레이어 수 집계."""
    total = 0
    quantized = 0
    for _name, mod in policy.named_modules():
        if list(mod.children()):
            continue  # non-leaf skip
        if isinstance(mod, (nn.Linear, MixedIntQuantizedLinear)):
            total += 1
            if isinstance(mod, MixedIntQuantizedLinear):
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
    act_quant_mode: str,
    layer_configs: dict[str, LayerConfig],
    env_cfg,
    envs_dict: dict,
    use_amp: bool = False,
) -> dict:
    """Mixed INT 양자화 후 LIBERO 평가."""
    print(f"\n[RUN] target={target}  act_quant={enable_act_quant}  mode={act_quant_mode}")
    print(f"      configs={len(layer_configs)} layers")

    device = torch.device(device_str)

    # 정책 로드
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

    # Mixed INT 양자화 적용
    layer_report = apply_mixed_int_quant(policy, layer_configs, act_quant_mode)
    disable_torch_compile(policy)

    coverage = build_quant_report(policy)
    print(f"[INFO] Quantized: {coverage['quantized']} / {coverage['total_linear']}")
    print(f"[INFO] layer_report entries: {len(layer_report)}")

    # 환경
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # 평가
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

    agg = eval_info.get("aggregated", {})
    pc_success = agg.get("pc_success", float("nan"))
    avg_reward = agg.get("avg_sum_reward", float("nan"))
    print(f"[RESULT] success={pc_success:.1f}%  reward={avg_reward:.4f}")

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "target": target,
        "enable_act_quant": enable_act_quant,
        "act_quant_mode": act_quant_mode,
        "layer_report": layer_report,
        "coverage": coverage,
        "eval_results": eval_info,
        "pc_success": pc_success,
        "avg_sum_reward": avg_reward,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 수치 정합성 테스트 (--selftest)
# ══════════════════════════════════════════════════════════════════════════════

def test_roundtrip():
    """INT 양자화 → dequant 왕복 오차 검증."""
    print("[TEST] quant_blockwise_int_storage roundtrip …")
    for bits in (3, 4, 6, 8):
        for blk in (16, 32, 64):
            W = torch.randn(64, 128)
            W_int, scale_i32, in_f = quant_blockwise_int_storage(W, bits=bits, block_size=blk)
            assert W_int.dtype == torch.int8, f"dtype mismatch: {W_int.dtype}"
            assert scale_i32.dtype == torch.int32, f"scale dtype: {scale_i32.dtype}"
            assert W_int.shape == (64, 128), f"shape: {W_int.shape}"
            assert scale_i32.min().item() >= 1, "scale_i32 min < 1"

            # dequant
            scale_fp = scale_i32.float() / (2 ** SCALE_SHIFT)
            W_b = W_int.float().reshape(64, -1, blk)
            W_dq = (W_b * scale_fp.unsqueeze(-1)).reshape(64, -1)[:, :in_f]
            max_err = (W - W_dq).abs().max().item()
            n_pos = 2 ** (bits - 1) - 1
            # 최대 오차 ≤ 0.5 * (max_scale / n_pos) * 2 (양자화 노이즈 상한)
            max_scale = scale_fp.max().item()
            expected_bound = max_scale  # scale 자체가 1 ULP
            print(f"  INT{bits} blk={blk:3d}: max_err={max_err:.6f}  bound≈{expected_bound:.6f}  {'OK' if max_err <= expected_bound * 1.01 else 'WARN'}")

    print("[TEST] MixedIntQuantizedLinear forward …")
    orig = nn.Linear(128, 64)
    W_int, scale_i32, in_f = quant_blockwise_int_storage(orig.weight.data.float(), 4, 16)
    layer = MixedIntQuantizedLinear(W_int, scale_i32, in_f, 16,
                                    orig.bias.data.float(), 4, 4)
    x = torch.randn(2, 10, 128)
    out = layer(x)
    assert out.shape == (2, 10, 64), f"output shape: {out.shape}"
    print(f"  forward OK: input {x.shape} → output {out.shape}")
    print("[TEST] All passed.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer mixed-precision INT quantization for LeRobot pi05"
    )
    # 모델 & 실험 설정
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task", type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1, help="n_envs (병렬 환경 수)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="logs/mixed_int_quant")
    parser.add_argument("--use_amp", action="store_true")

    # 양자화 옵션
    parser.add_argument(
        "--target", type=str, default="all",
        choices=["lm", "dit", "all"],
        help="양자화 대상 컴포넌트"
    )
    parser.add_argument(
        "--no_act_quant", action="store_true",
        help="활성화 양자화 비활성화 (weight-only 모드)"
    )
    parser.add_argument(
        "--force_act_quant", action="store_true",
        help="NaN activation stats 레이어도 weight-only 대신 a_min_bits로 강제 activation 양자화 (no weight-only layers)"
    )
    parser.add_argument(
        "--act_quant_mode", type=str, default="per_tensor",
        choices=["per_tensor", "per_token", "per_block"],
        help="활성화 양자화 방식 (per_block: 레이어별 block_size 단위 group-wise)"
    )
    parser.add_argument(
        "--dist_stats", type=str, default=str(DIST_STATS_PATH),
        help="dist_stats.json 경로"
    )

    # 모드
    parser.add_argument(
        "--table_only", action="store_true",
        help="레이어별 설정 표만 출력 (모델 로드 없음)"
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="수치 정합성 테스트 후 종료"
    )

    # Weight bit 배정 임계값 오버라이드
    parser.add_argument("--thresh_w_int8_kurt", type=float, default=50.0)
    parser.add_argument("--thresh_w_int8_cv",   type=float, default=0.7)
    parser.add_argument("--thresh_w_int6_kurt", type=float, default=10.0)
    parser.add_argument("--thresh_w_int6_cv",   type=float, default=0.45)
    parser.add_argument("--thresh_w_int4_kurt", type=float, default=3.0)
    parser.add_argument("--thresh_w_int4_cv",   type=float, default=0.25)
    parser.add_argument("--thresh_w_int3_kurt", type=float, default=1.5)
    parser.add_argument("--thresh_w_int3_cv",   type=float, default=0.12)
    parser.add_argument("--thresh_w_int2_kurt", type=float, default=0.5)
    parser.add_argument("--thresh_w_int2_cv",   type=float, default=0.05)

    # 컴포넌트별 최소 weight bit-width
    parser.add_argument("--lm_min_bits",   type=int, default=3,
                        help="LM  weight 최소 bit-width (1~16, default: 3)")
    parser.add_argument("--dit_min_bits",  type=int, default=3,
                        help="DiT weight 최소 bit-width (1~16, default: 3)")

    # 컴포넌트별 activation bit-width 범위
    parser.add_argument("--lm_a_min_bits",  type=int, default=1,
                        help="LM  activation 최소 bit-width (1~16, default: 1)")
    parser.add_argument("--dit_a_min_bits", type=int, default=1,
                        help="DiT activation 최소 bit-width (1~16, default: 1)")
    parser.add_argument("--lm_a_max_bits",  type=int, default=16,
                        help="LM  activation 최대 bit-width (cap, 1~16, default: 16)")
    parser.add_argument("--dit_a_max_bits", type=int, default=16,
                        help="DiT activation 최대 bit-width (cap, 1~16, default: 16)")

    # Activation bit 배정 임계값 오버라이드
    parser.add_argument("--thresh_a_int8_kurt", type=float, default=50.0)
    parser.add_argument("--thresh_a_int8_cv",   type=float, default=0.3)
    parser.add_argument("--thresh_a_int6_kurt", type=float, default=10.0)
    parser.add_argument("--thresh_a_int6_cv",   type=float, default=0.15)
    parser.add_argument("--thresh_a_int4_kurt", type=float, default=3.0)
    parser.add_argument("--thresh_a_int4_cv",   type=float, default=0.08)
    parser.add_argument("--thresh_a_int3_kurt", type=float, default=1.5)
    parser.add_argument("--thresh_a_int3_cv",   type=float, default=0.04)
    parser.add_argument("--thresh_a_int2_kurt", type=float, default=0.5)
    parser.add_argument("--thresh_a_int2_cv",   type=float, default=0.02)

    args = parser.parse_args()

    # ── 셀프 테스트 ──────────────────────────────────────────────────────────
    if args.selftest:
        test_roundtrip()
        return

    # ── 임계값 dict 구성 ────────────────────────────────────────────────────
    thresh = {
        "int8_kurt": args.thresh_w_int8_kurt,
        "int8_cv":   args.thresh_w_int8_cv,
        "int6_kurt": args.thresh_w_int6_kurt,
        "int6_cv":   args.thresh_w_int6_cv,
        "int4_kurt": args.thresh_w_int4_kurt,
        "int4_cv":   args.thresh_w_int4_cv,
        "int3_kurt": args.thresh_w_int3_kurt,
        "int3_cv":   args.thresh_w_int3_cv,
        "int2_kurt": args.thresh_w_int2_kurt,
        "int2_cv":   args.thresh_w_int2_cv,
    }
    thresh_a = {
        "int8_kurt": args.thresh_a_int8_kurt,
        "int8_cv":   args.thresh_a_int8_cv,
        "int6_kurt": args.thresh_a_int6_kurt,
        "int6_cv":   args.thresh_a_int6_cv,
        "int4_kurt": args.thresh_a_int4_kurt,
        "int4_cv":   args.thresh_a_int4_cv,
        "int3_kurt": args.thresh_a_int3_kurt,
        "int3_cv":   args.thresh_a_int3_cv,
        "int2_kurt": args.thresh_a_int2_kurt,
        "int2_cv":   args.thresh_a_int2_cv,
    }

    # ── dist_stats 로드 & layer_configs 빌드 ────────────────────────────────
    dist_stats_path = Path(args.dist_stats)
    if not dist_stats_path.exists():
        print(f"[ERROR] dist_stats not found: {dist_stats_path}")
        print("        먼저 analyze_distributions.py를 실행하세요.")
        sys.exit(1)

    print(f"[INFO] Loading dist_stats: {dist_stats_path}")
    weight_stats, activation_stats = load_dist_stats(dist_stats_path)
    layer_configs = build_layer_configs(
        weight_stats=weight_stats,
        activation_stats=activation_stats,
        target=args.target,
        enable_act_quant=not args.no_act_quant,
        thresh=thresh,
        lm_min_bits=args.lm_min_bits,
        dit_min_bits=args.dit_min_bits,
        lm_a_min_bits=args.lm_a_min_bits,
        dit_a_min_bits=args.dit_a_min_bits,
        lm_a_max_bits=args.lm_a_max_bits,
        dit_a_max_bits=args.dit_a_max_bits,
        thresh_a=thresh_a,
        force_act_quant=args.force_act_quant,
    )
    print(f"[INFO] layer_configs built: {len(layer_configs)} layers  (target={args.target})")

    # ── 표만 출력 ────────────────────────────────────────────────────────────
    if args.table_only:
        print_layer_table(layer_configs)
        # table_only 모드에서도 CSV 저장 (output_dir에)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        act_label_t = "wo" if args.no_act_quant else args.act_quant_mode
        w_label_t = f"_lm{args.lm_min_bits}_dit{args.dit_min_bits}" if (args.lm_min_bits != 3 or args.dit_min_bits != 3) else ""
        table_csv = output_dir / f"table_only_{args.target}{w_label_t}_{act_label_t}.csv"
        save_layer_table_csv(layer_configs, table_csv)
        return

    # ── 실험 실행 ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Creating LIBERO env: task={args.task}, n_envs={args.batch_size}")
    env_cfg = LiberoEnv(task=args.task)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    print(f"[INFO] suite={suite_name}  task_id={task_id}")

    act_label = "wo" if args.no_act_quant else args.act_quant_mode
    w_label = f"_lm{args.lm_min_bits}_dit{args.dit_min_bits}" if (args.lm_min_bits != 3 or args.dit_min_bits != 3) else ""
    if not args.no_act_quant:
        # 기본값 변경: min=1, max=16 (기존 max=8에서 변경됨)
        a_min_lm  = args.lm_a_min_bits  if args.lm_a_min_bits  != 1  else None
        a_max_lm  = args.lm_a_max_bits  if args.lm_a_max_bits  != 16 else None
        a_min_dit = args.dit_a_min_bits if args.dit_a_min_bits != 1  else None
        a_max_dit = args.dit_a_max_bits if args.dit_a_max_bits != 16 else None
        lm_a_str  = f"lm{'min'+str(a_min_lm) if a_min_lm else ''}{'max'+str(a_max_lm) if a_max_lm else ''}"
        dit_a_str = f"dit{'min'+str(a_min_dit) if a_min_dit else ''}{'max'+str(a_max_dit) if a_max_dit else ''}"
        has_a_label = bool(a_min_lm or a_max_lm or a_min_dit or a_max_dit)
        a_label = f"_a{lm_a_str}_{dit_a_str}" if has_a_label else ""
        force_suffix = "_fa" if args.force_act_quant else ""
    else:
        a_label = ""
        force_suffix = ""
    exp_key = f"mixed_int_{args.target}{w_label}{a_label}{force_suffix}_{act_label}"
    out_path = output_dir / f"{exp_key}.json"

    result = run_experiment(
        pretrained_path=args.pretrained_path,
        task=args.task,
        n_episodes=args.n_episodes,
        batch_size=args.batch_size,
        device_str=args.device,
        target=args.target,
        enable_act_quant=not args.no_act_quant,
        act_quant_mode=args.act_quant_mode,
        layer_configs=layer_configs,
        env_cfg=env_cfg,
        envs_dict=envs_dict,
        use_amp=args.use_amp,
    )

    # layer_configs를 직렬화 가능한 형태로 변환
    configs_serializable = {
        k: {
            "component": v.component,
            "layer_type": v.layer_type,
            "attn_mlp": v.attn_mlp,
            "w_bits": v.w_bits,
            "a_bits": v.a_bits,
            "block_size": v.block_size,
            "kurtosis_w": round(v.kurtosis_w, 4),
            "cv_w": round(v.cv_w, 4),
            "cv_act": round(v.cv_act, 4) if v.cv_act is not None else None,
        }
        for k, v in layer_configs.items()
    }

    out_data = {
        "config": {
            "pretrained_path": args.pretrained_path,
            "task": args.task,
            "suite_name": suite_name,
            "task_id": task_id,
            "n_episodes": args.n_episodes,
            "batch_size": args.batch_size,
            "target": args.target,
            "enable_act_quant": not args.no_act_quant,
            "force_act_quant": args.force_act_quant,
            "act_quant_mode": args.act_quant_mode,
            "lm_min_bits": args.lm_min_bits,
            "dit_min_bits": args.dit_min_bits,
            "lm_a_min_bits": args.lm_a_min_bits,
            "dit_a_min_bits": args.dit_a_min_bits,
            "lm_a_max_bits": args.lm_a_max_bits,
            "dit_a_max_bits": args.dit_a_max_bits,
            "device": args.device,
            "thresh": thresh,
            "scale_type": "INT32",
            "scale_shift": SCALE_SHIFT,
        },
        "layer_configs": configs_serializable,
        "pc_success": result["pc_success"],
        "avg_sum_reward": result["avg_sum_reward"],
        "coverage": result["coverage"],
        "layer_report": result["layer_report"],
        "eval_results": result["eval_results"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {out_path}")

    # 표도 함께 저장 (txt + csv)
    table_path = output_dir / f"{exp_key}_layer_table.txt"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_layer_table(layer_configs)
    table_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"[SAVED] {table_path}")

    csv_path = output_dir / f"{exp_key}_layer_table.csv"
    save_layer_table_csv(layer_configs, csv_path)

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"  Mixed INT Quantization Result")
    print(f"  Target     : {args.target}")
    print(f"  Act quant  : {'disabled' if args.no_act_quant else args.act_quant_mode}")
    print(f"  Scale type : INT32 (shift=2^{SCALE_SHIFT})")
    print(f"  Layers     : {len(layer_configs)} quantized")
    print(f"  Success    : {result['pc_success']:.1f}%")
    print(f"  AvgReward  : {result['avg_sum_reward']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
