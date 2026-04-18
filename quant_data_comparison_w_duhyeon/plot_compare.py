"""
Duhyeon vs NVFP4_DEFAULT_CFG output comparison plot.
"""

import os, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
d = torch.load(os.path.join(OUT_DIR, "compare_result.pt"), map_location="cpu")

y_fp    = d["y_fp"].float().squeeze(0)       # (50, 4096)
y_dh    = d["y_fq_dh"].float().squeeze(0)
y_mine  = d["y_mine"].float().squeeze(0)
layer_key = d["layer_key"]

def snr(ref, pred):
    mse = ((ref - pred)**2).mean()
    return 10 * math.log10(ref.pow(2).mean().item() / max(mse.item(), 1e-12))

# ── flat versions ─────────────────────────────────────────────────────────────
yf  = y_fp.flatten().numpy()
yd  = y_dh.flatten().numpy()
ym  = y_mine.flatten().numpy()

err_dh   = yf - yd
err_mine = yf - ym
diff_dm  = yd - ym          # Duhyeon - Mine

mse_dh   = float(((y_fp - y_dh)**2).mean())
mse_mine = float(((y_fp - y_mine)**2).mean())
mse_dm   = float(((y_dh - y_mine)**2).mean())
snr_dh   = snr(y_fp, y_dh)
snr_mine = snr(y_fp, y_mine)
snr_dm   = snr(y_dh, y_mine)

per_token_dh   = (y_fp - y_dh).abs().mean(dim=-1).numpy()
per_token_mine = (y_fp - y_mine).abs().mean(dim=-1).numpy()
per_token_dm   = (y_dh - y_mine).abs().mean(dim=-1).numpy()

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    f"NVFP4 Output Comparison: Duhyeon vs NVFP4_DEFAULT_CFG\n{layer_key}",
    fontsize=12, fontweight="bold", y=0.99,
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

C = {"fp": "#2196F3", "dh": "#FF5722", "mine": "#4CAF50", "diff": "#9C27B0"}
ROWS = [
    ("Duhyeon  vs  FP baseline",        yf, yd,   err_dh,   per_token_dh,   mse_dh,   snr_dh),
    ("NVFP4_DEFAULT_CFG  vs  FP baseline", yf, ym, err_mine, per_token_mine, mse_mine, snr_mine),
    ("Duhyeon  vs  NVFP4_DEFAULT_CFG",  yd, ym,   diff_dm,  per_token_dm,   mse_dm,   snr_dm),
]
COL_TITLES = [
    "Value Distribution",
    "Error Distribution",
    "Scatter (ref vs pred)",
    "Mean |Error| per Token",
]

for row, (title, ref, pred, err, per_tok, mse, snr_val) in enumerate(ROWS):
    abs_err = np.abs(err)
    color   = C["dh"] if row == 0 else (C["mine"] if row == 1 else C["diff"])

    # col 0: value dist
    ax0 = fig.add_subplot(gs[row, 0])
    bins = 150
    rng  = (min(ref.min(), pred.min()), max(ref.max(), pred.max()))
    ax0.hist(ref,  bins=bins, range=rng, alpha=0.5, color=C["fp"],   label="ref (FP)" if row < 2 else "Duhyeon")
    ax0.hist(pred, bins=bins, range=rng, alpha=0.5, color=color,     label="Duhyeon" if row == 0 else ("DEFAULT_CFG" if row == 1 else "DEFAULT_CFG"))
    ax0.set_title(f"{title}\n{COL_TITLES[0]}", fontsize=8)
    ax0.set_xlabel("Value"); ax0.set_ylabel("Count")
    ax0.legend(fontsize=7)
    ax0.text(0.97, 0.95, f"MSE={mse:.2e}\nSNR={snr_val:.1f} dB",
             transform=ax0.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # col 1: error dist
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.hist(err, bins=150, color=color, alpha=0.75)
    ax1.axvline(0, color="black", lw=0.8, ls="--")
    ax1.set_title(COL_TITLES[1], fontsize=8)
    ax1.set_xlabel("Error"); ax1.set_ylabel("Count")
    ax1.text(0.97, 0.95,
             f"mean={err.mean():.3e}\nstd={err.std():.3e}\nmax|e|={abs_err.max():.3e}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # col 2: scatter
    ax2 = fig.add_subplot(gs[row, 2])
    n   = len(ref)
    idx = np.random.choice(n, min(8000, n), replace=False)
    ax2.scatter(ref[idx], pred[idx], s=1.5, alpha=0.2, color=color, rasterized=True)
    lo, hi = min(ref[idx].min(), pred[idx].min()), max(ref[idx].max(), pred[idx].max())
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ideal")
    ax2.set_title(COL_TITLES[2], fontsize=8)
    ax2.set_xlabel("ref"); ax2.set_ylabel("pred")
    ax2.legend(fontsize=7)

    # col 3: per-token MAE
    ax3 = fig.add_subplot(gs[row, 3])
    ax3.bar(np.arange(len(per_tok)), per_tok, color=color, alpha=0.8)
    ax3.axhline(per_tok.mean(), color="red", lw=1.0, ls="--",
                label=f"mean={per_tok.mean():.4f}")
    ax3.set_title(COL_TITLES[3], fontsize=8)
    ax3.set_xlabel("Token index"); ax3.set_ylabel("Mean |error|")
    ax3.legend(fontsize=7)

# ── summary text ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
    f"Row 1 (Duhyeon vs FP):  MSE={mse_dh:.4e}  SNR={snr_dh:.2f} dB  │  "
    f"Row 2 (DEFAULT_CFG vs FP):  MSE={mse_mine:.4e}  SNR={snr_mine:.2f} dB  │  "
    f"Row 3 (Duhyeon vs DEFAULT_CFG):  MSE={mse_dm:.4e}  SNR={snr_dm:.2f} dB",
    ha="center", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", alpha=0.9))

out_path = os.path.join(OUT_DIR, "nvfp4_comparison_w_duhyeon.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()
