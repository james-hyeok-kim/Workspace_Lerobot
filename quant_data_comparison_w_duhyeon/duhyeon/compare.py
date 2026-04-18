"""
Apply 두현's NVFP4 fake-quant to captured.pt and save compare_dh.pt.

Reads  : du_comparison/captured.pt   {layer_key, x, W, y_base}
Writes : du_comparison/compare_dh.pt {layer_key, x, W, y_fp, y_fq_dh}

Usage:
    python duhyeon/thanos/du_comparison/compare.py
"""

import os
import sys

import torch

# Import mx_block_quant from sibling e2e_quant.py
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from e2e_quant import mx_block_quant  # noqa: E402

# ============================================================
# Inlined from ../e2e_quant.py
# (originally imported via the lines above; copied here to keep this script self-contained)
# ============================================================

import math

def _fp_max_val(exp_bits: int, man_bits: int, nan_type: str = "none") -> float:
    """Max representable finite value for a FP format.

    nan_type:
      "none"     all bit patterns valid (E2M1/MXFP4, MXFP6).
                 fp_max = (2 - 2^-m) * 2^max_exp
      "max_man"  only (max_exp, all-ones mantissa) = NaN; all-ones exponent is otherwise valid.
                 OCP E4M3 / FP8-E4M3fn: max mantissa = (2^m - 2), fp_max = 1.75 * 2^8 = 448.
      "max_exp"  all-ones exponent = NaN/inf (IEEE-like; OCP E5M2).
                 fp_max = (2 - 2^-m) * 2^(max_exp - 1), e.g. 1.75 * 2^15 = 57344.
    """
    bias         = (1 << (exp_bits - 1)) - 1
    all_ones_exp = (1 << exp_bits) - 1  # biased

    if nan_type == "max_exp":
        # Reserve all-ones exponent; mantissa can be anything
        max_biased_exp = all_ones_exp - 1
        man_factor     = 2.0 - 2.0 ** (-man_bits)
    elif nan_type == "max_man":
        # All-ones exponent valid; only (max_exp, all-ones mantissa) = NaN
        max_biased_exp = all_ones_exp
        man_factor     = 1.0 + (2 ** man_bits - 2) / 2 ** man_bits  # = 2 - 2^(1-m)
    else:  # "none"
        max_biased_exp = all_ones_exp
        man_factor     = 2.0 - 2.0 ** (-man_bits)

    return man_factor * (2.0 ** (max_biased_exp - bias))


def _fp_quant(x: torch.Tensor, exp_bits: int, man_bits: int, fp_max: float) -> torch.Tensor:
    """Fake-quantize x to a low-precision FP format (normal + subnormal).

    fp_max is passed explicitly so callers control the NaN convention
    (see _fp_max_val for how to compute it per format).
    max_exp for log2 clamping is derived as floor(log2(fp_max)).
    """
    bias     = (1 << (exp_bits - 1)) - 1
    min_exp  = 1 - bias
    max_exp  = int(math.log2(fp_max))   # floor(log2(fp_max)); valid for all OCP formats
    min_norm = 2.0 ** min_exp
    sub_step = min_norm / (2 ** man_bits)

    x    = x.float()
    sign = x.sign()
    xabs = x.abs().clamp(max=fp_max)

    # Subnormal path
    xsub = (xabs / sub_step).round() * sub_step

    # Normal path
    log2f   = xabs.clamp(min=1e-38).log2().floor().clamp(min_exp, max_exp)
    man_ulp = (2.0 ** log2f) / (2 ** man_bits)
    xnorm   = ((xabs / man_ulp).round() * man_ulp).clamp(max=fp_max)

    result = torch.where(xabs >= min_norm, xnorm, xsub)
    result = torch.where(xabs == 0, torch.zeros_like(result), result)
    return (sign * result).to(x.dtype)


def _e8m0_scale(max_abs: torch.Tensor, fp_max: float) -> torch.Tensor:
    """MX E8M0 block scale: smallest power-of-2 >= max_abs / fp_max."""
    ratio = (max_abs / fp_max).clamp(min=2.0 ** -127)
    return 2.0 ** ratio.log2().ceil().clamp(-127, 127)


# FP8-E4M3fn constants (OCP / NVIDIA convention, nan_type="max_man")
#   fp_max = 448  (E=1111,M=111 is the single NaN; E=1111,M=110 = 448 is valid)
_FP8_E4M3FN_MAX      = _fp_max_val(4, 3, nan_type="max_man")  # 448.0
_FP8_E4M3FN_MIN_NORM = 2.0 ** (1 - 7)                         # 2^-6


def _fp8_e4m3fn_scale(max_abs: torch.Tensor, fp4_max: float,
                      x_amax: float) -> torch.Tensor:
    """NVFP4 block scale with tensor-level normalization (fouroversix spec).

    encode_scale = (fp4_max * fp8_max) / x_amax
    x_scales_hp  = (block_amax / fp4_max) * encode_scale
               = block_amax * fp8_max / x_amax   ← always in [0, fp8_max]
    x_scales     = E4M3fn(x_scales_hp)

    Ref: fouroversix quantize_to_nvfp4, ScaleType.nv → torch.float8_e4m3fn (fp_max=448).
    """
    x_scales_hp = (max_abs * _FP8_E4M3FN_MAX / x_amax).clamp(
        min=_FP8_E4M3FN_MIN_NORM, max=_FP8_E4M3FN_MAX
    )
    return _fp_quant(x_scales_hp, 4, 3, fp_max=_FP8_E4M3FN_MAX)


def mx_block_quant(x: torch.Tensor, fmt: dict, block_size: int, scale_type: str) -> torch.Tensor:
    """Block fake-quantize along the last dimension of x.

    fmt: element-format descriptor, e.g. {"exp_bits":2, "man_bits":1, "nan_type":"none"}
    """
    fp_max     = _fp_max_val(fmt["exp_bits"], fmt["man_bits"], fmt["nan_type"])
    orig_shape = x.shape
    K    = orig_shape[-1]
    x_2d = x.float().reshape(-1, K)
    T    = x_2d.shape[0]

    pad = (-K) % block_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad))

    n_blocks = x_2d.shape[1] // block_size
    x_blocks = x_2d.reshape(T, n_blocks, block_size)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
    if scale_type == "e8m0":
        scale = _e8m0_scale(max_abs, fp_max)
    else:  # "fp8_e4m3fn"  (NVFP4)
        x_amax       = x_2d.abs().max().clamp(min=1e-38).item()
        decode_scale = x_amax / (fp_max * _FP8_E4M3FN_MAX)   # x_amax / (6 * 448)
        x_scales     = _fp8_e4m3fn_scale(max_abs, fp_max, x_amax)
        # dequantize with decode_scale to recover absolute scale
        x_q  = _fp_quant(x_blocks / (decode_scale * x_scales),
                         fmt["exp_bits"], fmt["man_bits"], fp_max)
        x_dq = (x_q * decode_scale * x_scales).reshape(T, -1)[:, :K]
        return x_dq.reshape(orig_shape[:-1] + (K,))

    x_q  = _fp_quant(x_blocks / scale, fmt["exp_bits"], fmt["man_bits"], fp_max)
    x_dq = (x_q * scale).reshape(T, -1)[:, :K]
    return x_dq.reshape(orig_shape[:-1] + (K,))

# ============================================================
# End of inlined e2e_quant.py block
# ============================================================

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
IN_FILE  = os.path.join(OUT_DIR, "captured.pt")
OUT_FILE = os.path.join(OUT_DIR, "compare_dh.pt")

# NVFP4: E2M1/E2M1, block=16, FP8-E4M3fn scale
_E2M1   = {"exp_bits": 2, "man_bits": 1, "nan_type": "none"}
_NVFP4_W_FMT     = _E2M1
_NVFP4_A_FMT     = _E2M1
_NVFP4_BLOCK     = 16
_NVFP4_SCALE     = "fp8_e4m3fn"


def nvfp4_linear(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Fake-quantize x and W with NVFP4, return y_fq_dh.

    Matches the _make_hook_mx logic in e2e_quant.py exactly:
      W_fq = mx_block_quant(W.float(), w_fmt, block, scale).to(dtype)
      x_fq = mx_block_quant(x.float(), a_fmt, block, scale).to(dtype)
      y    = x_fq @ W_fq.T
    """
    dtype = x.dtype
    x2d   = x.reshape(-1, x.shape[-1])            # (T, in_features)

    W_fq  = mx_block_quant(W.float(),   _NVFP4_W_FMT, _NVFP4_BLOCK, _NVFP4_SCALE).to(dtype)
    x_fq  = mx_block_quant(x2d.float(), _NVFP4_A_FMT, _NVFP4_BLOCK, _NVFP4_SCALE).to(dtype)

    return (x_fq @ W_fq.T).reshape(x.shape[:-1] + (W.shape[0],))


def main():
    data      = torch.load(IN_FILE, map_location="cpu")
    layer_key = data["layer_key"]
    x         = data["x"]      # (B, T, in_features) or (T, in_features)
    W         = data["W"]      # (out_features, in_features)
    y_fp      = data["y_base"] # ground-truth FP output captured from model

    y_fq_dh = nvfp4_linear(x, W)

    torch.save({
        "layer_key": layer_key,
        "x":         x,
        "W":         W,
        "y_fp":      y_fp,
        "y_fq_dh":   y_fq_dh,
    }, OUT_FILE)

    print(f"Saved : {OUT_FILE}")
    print(f"  layer   : {layer_key}")
    print(f"  x       : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"  W       : {tuple(W.shape)}  dtype={W.dtype}")
    print(f"  y_fp    : {tuple(y_fp.shape)} dtype={y_fp.dtype}")
    print(f"  y_fq_dh : {tuple(y_fq_dh.shape)} dtype={y_fq_dh.dtype}")


if __name__ == "__main__":
    main()
