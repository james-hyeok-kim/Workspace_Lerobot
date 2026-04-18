"""
NVFP4 single-layer quantization comparison.

Loads captured.pt (layer_key, x, W) and quantizes input + weight
using the same NVFP4_DEFAULT_CFG settings (E2M1, block_size=16, FP8 per-block scale).
Saves FP and dequantized tensors + error stats to nvfp4_quant_result.pt.
"""

import os
import sys

import torch

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "TensorRT-Model-Optimizer")
sys.path.insert(0, os.path.abspath(REPO_ROOT))

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor  # noqa: E402
from modelopt.torch.quantization.utils.core_utils import reduce_amax  # noqa: E402

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(OUT_DIR, "captured.pt")
OUT_FILE = os.path.join(OUT_DIR, "nvfp4_quant_result.pt")

BLOCK_SIZE = 16  # NVFP4_DEFAULT_CFG: block_sizes={-1: 16}


# ── helpers ───────────────────────────────────────────────────────────────────

def quant_dequant(tensor: torch.Tensor, name: str):
    """Quantize tensor to NVFP4 (E2M1, block_size=16) and immediately dequantize.

    Returns (qtensor_fp4, weights_scaling_factor, weights_scaling_factor_2, dequant_tensor).
    """
    qtensor, sf, sf2 = NVFP4QTensor.quantize(
        tensor.float(),    # quantize expects float32 internally
        block_size=BLOCK_SIZE,
        try_tensorrt=False,
    )
    dequant = qtensor.dequantize(
        dtype=tensor.dtype,
        scale=sf,
        double_scale=sf2,
        block_sizes={-1: BLOCK_SIZE},
    )

    # ── stats ────────────────────────────────────────────────────────────────
    fp_f = tensor.float()
    dq_f = dequant.float()
    abs_err = (fp_f - dq_f).abs()
    rel_err = abs_err / (fp_f.abs().clamp(min=1e-6))
    mse = (abs_err ** 2).mean()
    snr_db = 10 * torch.log10(fp_f.pow(2).mean() / mse.clamp(min=1e-12))

    print(f"\n[{name}]")
    print(f"  shape       : {list(tensor.shape)}  dtype={tensor.dtype}")
    print(f"  global scale (sf2) : {sf2.item():.6e}")
    print(f"  per-block sf shape : {list(sf.shape)}  dtype={sf.dtype}")
    print(f"  abs_err  max={abs_err.max():.5f}  mean={abs_err.mean():.5f}")
    print(f"  rel_err  max={rel_err.max():.5f}  mean={rel_err.mean():.5f}")
    print(f"  MSE      : {mse.item():.6e}")
    print(f"  SNR      : {snr_db.item():.2f} dB")

    return qtensor, sf, sf2, dequant


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data = torch.load(IN_FILE, map_location="cpu")

    layer_key = data["layer_key"]
    x = data["x"]   # (B, T, in_features) or (T, in_features)
    W = data["W"]   # (out_features, in_features)

    print("=" * 60)
    print(f"layer : {layer_key}")
    print(f"x     : {list(x.shape)}  {x.dtype}")
    print(f"W     : {list(W.shape)}  {W.dtype}")
    print("=" * 60)

    # ── quantize ─────────────────────────────────────────────────────────────
    _, x_sf, x_sf2, x_dq = quant_dequant(x, "input (x)")
    _, w_sf, w_sf2, W_dq = quant_dequant(W, "weight (W)")

    # ── outputs ──────────────────────────────────────────────────────────────
    # x shape: (..., in_features), W shape: (out_features, in_features)
    y_fp    = x.float() @ W.float().T          # FP32 reference
    y_quant = x_dq.float() @ W_dq.float().T   # simulated NVFP4

    out_abs_err = (y_fp - y_quant).abs()
    out_rel_err = out_abs_err / (y_fp.abs().clamp(min=1e-6))
    out_mse     = (out_abs_err ** 2).mean()
    out_snr     = 10 * torch.log10(y_fp.pow(2).mean() / out_mse.clamp(min=1e-12))

    print("\n[output]")
    print(f"  y_fp    : {list(y_fp.shape)}")
    print(f"  y_quant : {list(y_quant.shape)}")
    print(f"  abs_err  max={out_abs_err.max():.5f}  mean={out_abs_err.mean():.5f}")
    print(f"  rel_err  max={out_rel_err.max():.5f}  mean={out_rel_err.mean():.5f}")
    print(f"  MSE      : {out_mse.item():.6e}")
    print(f"  SNR      : {out_snr.item():.2f} dB")

    # ── save ─────────────────────────────────────────────────────────────────
    result = {
        "layer_key": layer_key,
        # original tensors
        "x_fp": x,
        "W_fp": W,
        # dequantized (simulated FP4) tensors
        "x_dequant": x_dq.to(x.dtype),
        "W_dequant": W_dq.to(W.dtype),
        # scales
        "x_global_scale": x_sf2,
        "x_block_scale":  x_sf,
        "W_global_scale": w_sf2,
        "W_block_scale":  w_sf,
        # outputs
        "y_fp":    y_fp.to(x.dtype),
        "y_quant": y_quant.to(x.dtype),
        # error stats
        "stats": {
            "x_mse":   ((x.float() - x_dq.float()) ** 2).mean().item(),
            "W_mse":   ((W.float() - W_dq.float()) ** 2).mean().item(),
            "y_mse":   out_mse.item(),
            "y_snr_db": out_snr.item(),
        },
    }
    torch.save(result, OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
    print("Keys:", list(result.keys()))


if __name__ == "__main__":
    main()
