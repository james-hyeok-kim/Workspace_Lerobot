"""
4-way NVFP4 output comparison:
  M1. eval_nvfp4_mtq  (MTQ / NVFP4QTensor)   -- two-level scale, searchsorted cast
  M2. nvfp4_single_layer (mine)               -- same as M1, single-layer version
  M3. eval_mixed_fp_quant _quantize_nvfp4     -- single-level FP16 scale, snap-to-grid
  M4. Duhyeon compare.py  mx_block_quant      -- two-level scale, log2-based _fp_quant
"""

import os, sys, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
LEROBOT  = os.path.join(ROOT, "..", "lerobot")
MODELOPT = os.path.join(ROOT, "..", "TensorRT-Model-Optimizer")
sys.path.insert(0, os.path.abspath(MODELOPT))
sys.path.insert(0, os.path.abspath(LEROBOT))

# ── load data ─────────────────────────────────────────────────────────────────
dh = torch.load(os.path.join(ROOT, "duhyeon/output_compare_dh.pt"), map_location="cpu")
x      = dh["x"]           # [1, 50, 1024] bf16
W      = dh["W"]           # [4096, 1024] bf16
y_fp   = dh["y_fp"]        # [1, 50, 4096] bf16  — FP baseline
y_fq_dh = dh["y_fq_dh"]   # [1, 50, 4096] bf16  — Duhyeon

# ── M1/M2: NVFP4QTensor (MTQ / nvfp4_single_layer) ───────────────────────────
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

def quant_dequant_modelopt(t: torch.Tensor) -> torch.Tensor:
    qt, sf, sf2 = NVFP4QTensor.quantize(t.float(), block_size=16, try_tensorrt=False)
    return qt.dequantize(dtype=t.dtype, scale=sf, double_scale=sf2, block_sizes={-1: 16})

x_dq_mo = quant_dequant_modelopt(x)
W_dq_mo = quant_dequant_modelopt(W)
y_mo = (x_dq_mo.float().reshape(-1, x.shape[-1]) @ W_dq_mo.float().T
        ).reshape(y_fp.shape).to(x.dtype)

# ── M3: eval_mixed_fp_quant _quantize_nvfp4 ───────────────────────────────────
# weight: _quantize_nvfp4 (single-level FP16 scale, snap-to-grid)
# activation: quant_act_fp (same snap-to-grid, FP16 scale)

_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def _snap_to_grid(abs_flat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    pos  = torch.searchsorted(grid, abs_flat)
    lo   = (pos - 1).clamp(min=0)
    hi   = pos.clamp(max=grid.numel() - 1)
    v_lo = grid[lo];  v_hi = grid[hi]
    return torch.where((abs_flat - v_lo) > (v_hi - abs_flat), v_hi, v_lo)

def _apply_fp16_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.to(torch.float16).float()

def quant_nvfp4_mixed(t: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """eval_mixed_fp_quant._quantize_nvfp4 logic (weight + activation 동일 적용)."""
    grid   = _LEVELS.to(t.device)
    orig   = t.shape
    tf     = t.reshape(-1, block_size).float()
    amax   = tf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale  = _apply_fp16_scale(amax / 6.0)
    t_norm = tf.abs() / scale.clamp(min=1e-12)
    snapped = _snap_to_grid(t_norm.reshape(-1), grid).reshape(t_norm.shape)
    return (tf.sign() * snapped * scale).reshape(orig).to(t.dtype)

x_dq_mixed = quant_nvfp4_mixed(x)
W_dq_mixed = quant_nvfp4_mixed(W)
y_mixed = (x_dq_mixed.float().reshape(-1, x.shape[-1]) @ W_dq_mixed.float().T
           ).reshape(y_fp.shape).to(x.dtype)

# ── helpers ───────────────────────────────────────────────────────────────────
def err_stats(ref: torch.Tensor, pred: torch.Tensor) -> dict:
    ae  = (ref.float() - pred.float()).abs()
    mse = float((ae**2).mean())
    snr = 10 * math.log10(ref.float().pow(2).mean().item() / max(mse, 1e-12))
    per_token = ae.squeeze(0).mean(dim=-1).numpy()  # (T,)
    return dict(
        ae_mean=float(ae.mean()), ae_max=float(ae.max()),
        mse=mse, snr=snr, per_token=per_token,
        flat_err=(ref.float() - pred.float()).flatten().numpy(),
        flat_pred=pred.float().flatten().numpy(),
        flat_ref=ref.float().flatten().numpy(),
    )

METHODS = [
    ("M1/M2: MTQ / nvfp4_single_layer\n(two-level scale, searchsorted)",  y_mo,     "#4CAF50"),
    ("M3: eval_mixed_fp_quant\n(single-level FP16 scale, snap-to-grid)",  y_mixed,  "#FF9800"),
    ("M4: Duhyeon mx_block_quant\n(two-level scale, log2 _fp_quant)",     y_fq_dh,  "#9C27B0"),
]

stats = {name.split("\n")[0]: err_stats(y_fp, y) for name, y, _ in METHODS}
ref_flat = y_fp.float().flatten().numpy()

# ── print summary ─────────────────────────────────────────────────────────────
print(f"\n{'Method':<45} {'MSE':>10} {'SNR(dB)':>8} {'abs_mean':>10} {'abs_max':>10}")
print("-" * 85)
for name, y, _ in METHODS:
    k  = name.split("\n")[0]
    s  = stats[k]
    print(f"{k:<45} {s['mse']:>10.4e} {s['snr']:>8.2f} {s['ae_mean']:>10.5f} {s['ae_max']:>10.5f}")

# method-vs-method
print("\n=== Method vs Method (direct diff) ===")
pairs = [
    ("M1/M2 vs M3", y_mo,    y_mixed),
    ("M1/M2 vs M4", y_mo,    y_fq_dh),
    ("M3    vs M4", y_mixed, y_fq_dh),
]
print(f"{'Pair':<25} {'MSE':>12} {'SNR(dB)':>9} {'abs_mean':>12} {'abs_max':>12}")
print("-" * 72)
for pname, ya, yb in pairs:
    s = err_stats(ya, yb)
    print(f"{pname:<25} {s['mse']:>12.4e} {s['snr']:>9.2f} {s['ae_mean']:>12.6f} {s['ae_max']:>12.6f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    "NVFP4 4-Way Output Comparison  vs  FP baseline\n"
    "layer: model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj",
    fontsize=12, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.35)

for row, (title, y_pred, color) in enumerate(METHODS):
    k = title.split("\n")[0]
    s = stats[k]
    err   = s["flat_err"]
    ae    = np.abs(err)
    pred  = s["flat_pred"]
    tok   = s["per_token"]
    label = title.replace("\n", " | ")

    # col 0: value dist
    ax0 = fig.add_subplot(gs[row, 0])
    rng = (min(ref_flat.min(), pred.min()), max(ref_flat.max(), pred.max()))
    ax0.hist(ref_flat, bins=150, range=rng, alpha=0.45, color="#2196F3", label="FP baseline")
    ax0.hist(pred,     bins=150, range=rng, alpha=0.55, color=color,     label="quant")
    ax0.set_title(f"{title}\nValue Distribution", fontsize=8)
    ax0.set_xlabel("Value"); ax0.set_ylabel("Count"); ax0.legend(fontsize=7)
    ax0.text(0.97, 0.95, f"MSE={s['mse']:.2e}\nSNR={s['snr']:.1f} dB",
             transform=ax0.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # col 1: error distribution
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.hist(err, bins=150, color=color, alpha=0.75)
    ax1.axvline(0, color="black", lw=0.8, ls="--")
    ax1.set_title("Error Distribution  (FP − quant)", fontsize=8)
    ax1.set_xlabel("Error"); ax1.set_ylabel("Count")
    ax1.text(0.97, 0.95,
             f"mean={err.mean():.3e}\nstd={err.std():.3e}\nmax|e|={ae.max():.3e}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # col 2: scatter
    ax2 = fig.add_subplot(gs[row, 2])
    n   = len(ref_flat)
    idx = np.random.choice(n, min(8000, n), replace=False)
    ax2.scatter(ref_flat[idx], pred[idx], s=1.5, alpha=0.2, color=color, rasterized=True)
    lo, hi = min(ref_flat[idx].min(), pred[idx].min()), max(ref_flat[idx].max(), pred[idx].max())
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ideal")
    ax2.set_title("Scatter  (FP vs quant)", fontsize=8)
    ax2.set_xlabel("FP ref"); ax2.set_ylabel("quant pred"); ax2.legend(fontsize=7)

    # col 3: per-token MAE
    ax3 = fig.add_subplot(gs[row, 3])
    ax3.bar(np.arange(len(tok)), tok, color=color, alpha=0.8)
    ax3.axhline(tok.mean(), color="red", lw=1.0, ls="--",
                label=f"mean={tok.mean():.4f}")
    ax3.set_title("Mean |Error| per Token", fontsize=8)
    ax3.set_xlabel("Token index"); ax3.set_ylabel("Mean |error|"); ax3.legend(fontsize=7)

# bottom summary bar
summary = "  |  ".join(
    f"{name.split(chr(10))[0]}: MSE={stats[name.split(chr(10))[0]]['mse']:.3e} SNR={stats[name.split(chr(10))[0]]['snr']:.1f}dB"
    for name, _, _ in METHODS
)
fig.text(0.5, 0.005, summary, ha="center", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.9))

out = os.path.join(ROOT, "nvfp4_4methods_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close()
