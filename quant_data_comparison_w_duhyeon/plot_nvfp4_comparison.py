"""
NVFP4 quantization comparison plots.
한눈에 input / weight / output의 FP vs dequant 분포, 오차, scatter를 시각화.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE  = os.path.join(OUT_DIR, "nvfp4_quant_result.pt")
OUT_FILE = os.path.join(OUT_DIR, "nvfp4_comparison.png")

# ── load ──────────────────────────────────────────────────────────────────────
data = torch.load(IN_FILE, map_location="cpu")

x_fp    = data["x_fp"].float().flatten().numpy()
x_dq    = data["x_dequant"].float().flatten().numpy()
W_fp    = data["W_fp"].float().flatten().numpy()
W_dq    = data["W_dequant"].float().flatten().numpy()
y_fp    = data["y_fp"].float().flatten().numpy()
y_quant = data["y_quant"].float().flatten().numpy()
stats   = data["stats"]

layer_key = data["layer_key"]

# ── layout ────────────────────────────────────────────────────────────────────
# 3 rows (input / weight / output) × 4 cols (dist | error dist | scatter | abs-err heatmap/line)
fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    f"NVFP4 Quantization Comparison\n{layer_key}",
    fontsize=13, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

GROUPS = [
    ("Input (x)",  x_fp,  x_dq,  stats["x_mse"],  None),
    ("Weight (W)", W_fp,  W_dq,  stats["W_mse"],  None),
    ("Output (y)", y_fp,  y_quant, stats["y_mse"], stats["y_snr_db"]),
]

COLORS = {"fp": "#2196F3", "dq": "#FF5722", "err": "#9C27B0", "abs": "#4CAF50"}

for row, (title, fp, dq, mse, snr) in enumerate(GROUPS):
    err     = fp - dq
    abs_err = np.abs(err)
    n       = len(fp)

    # ── col 0: value distribution ─────────────────────────────────────────
    ax0 = fig.add_subplot(gs[row, 0])
    bins = min(200, max(50, n // 500))
    rng  = (min(fp.min(), dq.min()), max(fp.max(), dq.max()))
    ax0.hist(fp, bins=bins, range=rng, alpha=0.55, color=COLORS["fp"],  label="FP (orig)")
    ax0.hist(dq, bins=bins, range=rng, alpha=0.55, color=COLORS["dq"],  label="Dequant")
    ax0.set_title(f"{title} — Value Distribution", fontsize=9)
    ax0.set_xlabel("Value"); ax0.set_ylabel("Count")
    ax0.legend(fontsize=7)

    snr_str = f"  SNR={snr:.1f} dB" if snr is not None else ""
    ax0.text(0.97, 0.95, f"MSE={mse:.2e}{snr_str}",
             transform=ax0.transAxes, ha="right", va="top",
             fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── col 1: error distribution ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[row, 1])
    ebins = min(200, max(50, n // 500))
    ax1.hist(err, bins=ebins, color=COLORS["err"], alpha=0.75)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title(f"{title} — Error Distribution (FP − Dequant)", fontsize=9)
    ax1.set_xlabel("Error"); ax1.set_ylabel("Count")
    ax1.text(0.97, 0.95,
             f"mean={err.mean():.3e}\nstd={err.std():.3e}\nmax|e|={abs_err.max():.3e}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── col 2: scatter FP vs Dequant ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[row, 2])
    sample = min(8000, n)
    idx    = np.random.choice(n, sample, replace=False)
    ax2.scatter(fp[idx], dq[idx], s=1.5, alpha=0.25, color=COLORS["dq"], rasterized=True)
    lo = min(fp[idx].min(), dq[idx].min())
    hi = max(fp[idx].max(), dq[idx].max())
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="ideal")
    ax2.set_title(f"{title} — FP vs Dequant (scatter)", fontsize=9)
    ax2.set_xlabel("FP value"); ax2.set_ylabel("Dequant value")
    ax2.legend(fontsize=7)

    # ── col 3: absolute error profile ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[row, 3])
    if row < 2:
        # input / weight: abs error per element (sampled)
        max_pts = 2000
        step    = max(1, n // max_pts)
        xs      = np.arange(0, n, step)
        ax3.plot(xs, abs_err[::step], linewidth=0.6, color=COLORS["abs"], alpha=0.8)
        ax3.axhline(abs_err.mean(), color="red", linewidth=1.0, linestyle="--",
                    label=f"mean={abs_err.mean():.3e}")
        ax3.set_title(f"{title} — |Error| per Element", fontsize=9)
        ax3.set_xlabel("Element index"); ax3.set_ylabel("|FP − Dequant|")
        ax3.legend(fontsize=7)
    else:
        # output: abs error per output token (mean over out_features)
        y_fp_2d    = data["y_fp"].float().squeeze(0)      # (T, out_features)
        y_quant_2d = data["y_quant"].float().squeeze(0)
        per_token_mae = (y_fp_2d - y_quant_2d).abs().mean(dim=-1).numpy()
        ax3.bar(np.arange(len(per_token_mae)), per_token_mae,
                color=COLORS["abs"], alpha=0.8)
        ax3.axhline(per_token_mae.mean(), color="red", linewidth=1.0, linestyle="--",
                    label=f"mean={per_token_mae.mean():.3e}")
        ax3.set_title(f"{title} — Mean |Error| per Token", fontsize=9)
        ax3.set_xlabel("Token index"); ax3.set_ylabel("Mean |error|")
        ax3.legend(fontsize=7)

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
