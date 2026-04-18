"""
compare_layerwise.py
Layer-wise step-by-step error comparison:
  FP baseline  vs  MTQ (NVFP4_DEFAULT_CFG, LM+DiT only)
              vs  Duhyeon (nvfp4_bmm, is_all + BMM)

Analysis:
  1. Structural diff — which layers each method quantizes
  2. Single forward pass (FP) with per-layer capture
  3. Isolated per-layer error: apply each quant to same {x, W}
  4. Per-layer MSE / SNR report and plots

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python compare_layerwise.py \\
        --pretrained_path lerobot/pi05_libero_finetuned \\
        --task_id 0 \\
        --output_dir compare_results
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "lerobot"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ModelOpt ──────────────────────────────────────────────────────────────────
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    print("[OK] ModelOpt loaded")
except ImportError as e:
    print(f"[ERROR] modelopt not found: {e}")
    sys.exit(1)

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# Duhyeon's NVFP4 fake-quant (from e2e_quant_0417.py)
# ══════════════════════════════════════════════════════════════════════════════

_FP8_E4M3FN_MAX      = 448.0
_FP8_E4M3FN_MIN_NORM = 2.0 ** (1 - 7)   # 2^-6


def _fp_quant_dh(x: torch.Tensor, exp_bits: int, man_bits: int, fp_max: float) -> torch.Tensor:
    """Duhyeon _fp_quant: log2-based normal + subnormal."""
    bias     = (1 << (exp_bits - 1)) - 1
    min_exp  = 1 - bias
    max_exp  = int(math.log2(fp_max))
    min_norm = 2.0 ** min_exp
    sub_step = min_norm / (2 ** man_bits)

    x    = x.float()
    sign = x.sign()
    xabs = x.abs().clamp(max=fp_max)

    xsub = (xabs / sub_step).round() * sub_step

    log2f   = xabs.clamp(min=1e-38).log2().floor().clamp(min_exp, max_exp)
    man_ulp = (2.0 ** log2f) / (2 ** man_bits)
    xnorm   = ((xabs / man_ulp).round() * man_ulp).clamp(max=fp_max)

    result = torch.where(xabs >= min_norm, xnorm, xsub)
    result = torch.where(xabs == 0, torch.zeros_like(result), result)
    return (sign * result)


def _fp8_e4m3fn_scale_dh(max_abs: torch.Tensor, fp4_max: float, x_amax: float) -> torch.Tensor:
    x_scales_hp = (max_abs * _FP8_E4M3FN_MAX / x_amax).clamp(
        min=_FP8_E4M3FN_MIN_NORM, max=_FP8_E4M3FN_MAX
    )
    return _fp_quant_dh(x_scales_hp, 4, 3, fp_max=_FP8_E4M3FN_MAX)


def duhyeon_nvfp4_fake_quant(t: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Duhyeon mx_block_quant with scale_type='fp8_e4m3fn', E2M1 format."""
    fp_max     = 6.0   # E2M1 fp_max
    exp_bits, man_bits = 2, 1
    orig_shape = t.shape
    K    = orig_shape[-1]
    x_2d = t.float().reshape(-1, K)
    T    = x_2d.shape[0]

    pad = (-K) % block_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad))

    n_blocks = x_2d.shape[1] // block_size
    x_blocks = x_2d.reshape(T, n_blocks, block_size)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
    x_amax  = x_2d.abs().max().clamp(min=1e-38).item()

    decode_scale = x_amax / (fp_max * _FP8_E4M3FN_MAX)
    x_scales     = _fp8_e4m3fn_scale_dh(max_abs, fp_max, x_amax)

    x_q  = _fp_quant_dh(x_blocks / (decode_scale * x_scales), exp_bits, man_bits, fp_max)
    x_dq = (x_q * decode_scale * x_scales).reshape(T, -1)[:, :K]
    return x_dq.reshape(orig_shape[:-1] + (K,)).to(t.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# MTQ fake-quant (NVFP4QTensor path)
# ══════════════════════════════════════════════════════════════════════════════

def mtq_nvfp4_fake_quant(t: torch.Tensor) -> torch.Tensor:
    """MTQ NVFP4 fake-quant: searchsorted + FP8-E4M3fn two-level scale."""
    qt, sf, sf2 = NVFP4QTensor.quantize(t.float(), block_size=16, try_tensorrt=False)
    return qt.dequantize(dtype=t.dtype, scale=sf, double_scale=sf2, block_sizes={-1: 16})


# ══════════════════════════════════════════════════════════════════════════════
# Layer coverage helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_qkv_proj(name: str) -> bool:
    return any(name.endswith(s) for s in (".q_proj", ".k_proj", ".v_proj"))


def get_all_linear_names(policy) -> set:
    return {name for name, m in policy.named_modules() if isinstance(m, nn.Linear)}


def get_mtq_quantized_names(policy) -> set:
    """Names of layers actually quantized by MTQ (have weight_quantizer)."""
    return {name for name, m in policy.named_modules() if hasattr(m, "weight_quantizer")}


# ══════════════════════════════════════════════════════════════════════════════
# FP baseline capture (read-only hooks)
# ══════════════════════════════════════════════════════════════════════════════

def register_fp_capture_hooks(policy):
    """Read-only hooks on all nn.Linear — capture first forward only."""
    captures = {}
    hooks    = []

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def _make_hook(layer_name, mod):
            def _hook(m, inp, out):
                if layer_name in captures:
                    return
                captures[layer_name] = {
                    "x":    inp[0].detach().cpu(),
                    "W":    mod.weight.detach().cpu(),
                    "y_fp": out.detach().cpu(),
                    "bias": mod.bias.detach().cpu() if mod.bias is not None else None,
                }
            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name, module)))

    return captures, hooks


def remove_hooks(hooks: list) -> None:
    for h in hooks:
        h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Isolated per-layer comparison
# ══════════════════════════════════════════════════════════════════════════════

def _stats(ref: torch.Tensor, pred: torch.Tensor) -> dict:
    ae  = (ref.float() - pred.float()).abs()
    mse = float((ae ** 2).mean())
    sig = ref.float().pow(2).mean().item()
    snr = 10 * math.log10(sig / max(mse, 1e-12)) if sig > 1e-12 else 0.0
    return {"mse": mse, "snr": snr, "ae_mean": float(ae.mean()), "ae_max": float(ae.max())}


def compute_isolated_comparison(
    fp_captures: dict,
    mtq_layers: set,
    dh_bmm_layers: set,
) -> dict:
    """
    For each captured layer compute MTQ and Duhyeon outputs from FP {x, W}.
    Returns per-layer stats dict (does NOT require model re-run).
    """
    results = {}

    for name, data in fp_captures.items():
        x    = data["x"]
        W    = data["W"]
        y_fp = data["y_fp"]
        bias = data["bias"]

        orig_shape  = y_fp.shape
        out_features = W.shape[0]

        # ── MTQ output ────────────────────────────────────────────────────────
        if name in mtq_layers:
            x_mtq = mtq_nvfp4_fake_quant(x)
            W_mtq = mtq_nvfp4_fake_quant(W)
        else:
            x_mtq, W_mtq = x, W

        x2d   = x_mtq.float().reshape(-1, x_mtq.shape[-1])
        y_mtq = (x2d @ W_mtq.float().T).reshape(orig_shape[:-1] + (out_features,))
        if bias is not None:
            y_mtq = y_mtq + bias.float()
        y_mtq = y_mtq.to(y_fp.dtype)

        # ── Duhyeon output (always quantizes W and x) ─────────────────────────
        x_dh = duhyeon_nvfp4_fake_quant(x)
        W_dh = duhyeon_nvfp4_fake_quant(W)

        x2d   = x_dh.float().reshape(-1, x_dh.shape[-1])
        y_dh  = (x2d @ W_dh.float().T).reshape(orig_shape[:-1] + (out_features,))
        if bias is not None:
            y_dh = y_dh + bias.float()
        y_dh = y_dh.to(y_fp.dtype)

        # BMM: Duhyeon also quantizes q/k/v output
        if name in dh_bmm_layers:
            y_dh = duhyeon_nvfp4_fake_quant(y_dh)

        # ── stats ─────────────────────────────────────────────────────────────
        s_mtq  = _stats(y_fp, y_mtq)
        s_dh   = _stats(y_fp, y_dh)
        s_diff = _stats(y_mtq, y_dh)

        results[name] = {
            "is_mtq_quantized": name in mtq_layers,
            "is_dh_only":       name not in mtq_layers,
            "has_bmm":          name in dh_bmm_layers,
            "mtq":              s_mtq,
            "dh":               s_dh,
            "diff":             s_diff,
            "W_shape":          list(W.shape),
            "x_shape":          list(x.shape),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Apply NVFP4 + torch.compile disable (from eval_nvfp4_lm_dit.py)
# ══════════════════════════════════════════════════════════════════════════════

def apply_nvfp4_lm_dit(policy) -> None:
    lm = policy.model.paligemma_with_expert.paligemma.model.language_model
    mtq.quantize(lm, config=NVFP4_DEFAULT_CFG)
    expert = policy.model.paligemma_with_expert.gemma_expert
    mtq.quantize(expert, config=NVFP4_DEFAULT_CFG)


def disable_torch_compile(policy) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_structural_report(all_linear: set, mtq_layers: set) -> None:
    dh_only  = all_linear - mtq_layers
    mtq_only = mtq_layers - all_linear   # shouldn't happen
    both     = all_linear & mtq_layers

    print(f"\n{'='*70}")
    print("  Structural Coverage Comparison")
    print(f"{'='*70}")
    print(f"  All nn.Linear in model     : {len(all_linear):>6}")
    print(f"  MTQ-quantized layers       : {len(mtq_layers):>6}  (LM + DiT, NVFP4_DEFAULT_CFG)")
    print(f"  Duhyeon-quantized layers   : {len(all_linear):>6}  (is_all — ALL nn.Linear)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Both MTQ + Duhyeon         : {len(both):>6}")
    print(f"  Duhyeon ONLY (not MTQ)     : {len(dh_only):>6}  ← vision encoder, embed, lm_head etc.")

    # Categorize dh-only layers
    vision   = [n for n in dh_only if "vision_tower" in n or "encoder.layers" in n or "siglip" in n.lower()]
    embed    = [n for n in dh_only if "embed" in n or "lm_head" in n]
    other    = [n for n in dh_only if n not in vision and n not in embed]
    print(f"\n  Duhyeon-only breakdown:")
    print(f"    Vision encoder / SigLIP  : {len(vision):>4}")
    print(f"    Embedding / lm_head      : {len(embed):>4}")
    print(f"    Other                    : {len(other):>4}")
    if other:
        for n in sorted(other)[:10]:
            print(f"      {n}")
        if len(other) > 10:
            print(f"      ... ({len(other)-10} more)")


def print_layer_table(results: dict, top_n: int = 30) -> None:
    """Print per-layer comparison table sorted by Duhyeon MSE (desc)."""
    rows = sorted(results.items(), key=lambda kv: kv[1]["dh"]["mse"], reverse=True)

    print(f"\n{'='*110}")
    print(f"  Per-layer MSE / SNR  (sorted by Duhyeon MSE, top {top_n})")
    print(f"  * = Duhyeon-only (not in MTQ)    B = has BMM output quantization")
    print(f"{'='*110}")
    hdr = f"  {'Layer name':<60} {'MTQ':>3} {'DH':>3} {'B':>2}  {'MTQ MSE':>10}  {'MTQ SNR':>8}  {'DH MSE':>10}  {'DH SNR':>8}  {'|MTQ-DH| SNR':>12}"
    print(hdr)
    print("  " + "-" * 106)

    for name, r in rows[:top_n]:
        is_dh_only = r["is_dh_only"]
        has_bmm    = r["has_bmm"]
        mtq_flag   = "  " if r["is_mtq_quantized"] else "* "
        bmm_flag   = "B" if has_bmm else " "
        mtq_q      = "Y" if r["is_mtq_quantized"] else "N"
        dh_q       = "Y"

        short_name = name[-60:] if len(name) > 60 else name
        print(f"  {mtq_flag}{short_name:<58} {mtq_q:>3} {dh_q:>3} {bmm_flag:>2}"
              f"  {r['mtq']['mse']:>10.3e}  {r['mtq']['snr']:>8.2f}"
              f"  {r['dh']['mse']:>10.3e}  {r['dh']['snr']:>8.2f}"
              f"  {r['diff']['snr']:>12.2f}")


def print_summary_stats(results: dict) -> None:
    mtq_q  = [r for r in results.values() if r["is_mtq_quantized"]]
    dh_all = list(results.values())
    dh_only_layers = [r for r in results.values() if r["is_dh_only"]]

    def agg(lst, key_path):
        vals = [v[key_path[0]][key_path[1]] for v in lst]
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals))}

    print(f"\n{'='*70}")
    print("  Summary Statistics")
    print(f"{'='*70}")

    s = agg(mtq_q, ("mtq", "mse"))
    print(f"  MTQ quantized layers  (n={len(mtq_q):3d}) : MSE mean={s['mean']:.3e}  max={s['max']:.3e}")

    s_d = agg(dh_all, ("dh", "mse"))
    print(f"  Duhyeon all layers    (n={len(dh_all):3d}) : MSE mean={s_d['mean']:.3e}  max={s_d['max']:.3e}")

    if dh_only_layers:
        s2 = agg(dh_only_layers, ("dh", "mse"))
        print(f"  Duhyeon-only layers   (n={len(dh_only_layers):3d}) : MSE mean={s2['mean']:.3e}  max={s2['max']:.3e}  (vision+embed etc.)")

    # Direct MTQ vs Duhyeon diff for commonly quantized layers
    both_q = [r for r in results.values() if r["is_mtq_quantized"]]
    if both_q:
        s3 = agg(both_q, ("diff", "mse"))
        print(f"\n  MTQ vs Duhyeon direct diff (same layers, n={len(both_q)}):")
        print(f"    MSE mean={s3['mean']:.3e}  max={s3['max']:.3e}")
        snr_vals = [r["diff"]["snr"] for r in both_q]
        print(f"    SNR mean={float(np.mean(snr_vals)):.1f} dB  min={float(np.min(snr_vals)):.1f} dB")
        print(f"    → If SNR >> 40dB: numerically equivalent per-layer")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_layerwise(results: dict, out_path: Path) -> None:
    names  = list(results.keys())
    n      = len(names)
    idx    = np.arange(n)

    mse_mtq = np.array([results[k]["mtq"]["mse"] for k in names])
    mse_dh  = np.array([results[k]["dh"]["mse"]  for k in names])
    snr_mtq = np.array([results[k]["mtq"]["snr"] for k in names])
    snr_dh  = np.array([results[k]["dh"]["snr"]  for k in names])
    snr_dif = np.array([results[k]["diff"]["snr"] for k in names])

    is_dh_only  = np.array([results[k]["is_dh_only"]  for k in names])
    has_bmm     = np.array([results[k]["has_bmm"]      for k in names])
    is_mtq_q    = np.array([results[k]["is_mtq_quantized"] for k in names])

    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(
        "Layer-wise Error: MTQ (LM+DiT only) vs Duhyeon (nvfp4_bmm, is_all)\n"
        "Isolated comparison — same FP {x, W} input, per-layer quant applied",
        fontsize=12, fontweight="bold",
    )

    # ── Row 0: MSE ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(idx, mse_mtq, lw=0.7, color="#4CAF50", label="MTQ MSE")
    ax.semilogy(idx, mse_dh,  lw=0.7, color="#FF9800", label="Duhyeon MSE", alpha=0.8)
    # Highlight Duhyeon-only layers
    if is_dh_only.any():
        ax.semilogy(idx[is_dh_only], mse_dh[is_dh_only], "r.", ms=3, alpha=0.6, label="Duhyeon-only layers")
    ax.set_title("Per-layer MSE (vs FP baseline)")
    ax.set_ylabel("MSE (log scale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 1: SNR ────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(idx, snr_mtq, lw=0.7, color="#4CAF50", label="MTQ SNR")
    ax.plot(idx, snr_dh,  lw=0.7, color="#FF9800", label="Duhyeon SNR", alpha=0.8)
    if is_dh_only.any():
        ax.plot(idx[is_dh_only], snr_dh[is_dh_only], "r.", ms=3, alpha=0.6, label="Duhyeon-only layers")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Per-layer SNR (vs FP baseline, higher is better)")
    ax.set_ylabel("SNR (dB)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 2: direct diff MTQ vs Duhyeon ────────────────────────────────────
    ax = axes[2]
    ax.plot(idx[is_mtq_q], snr_dif[is_mtq_q], lw=0.7, color="#9C27B0",
            label="MTQ vs Duhyeon SNR (same-layer direct diff)")
    ax.axhline(40, color="gray", lw=0.8, ls="--", alpha=0.5, label="40 dB (nearly identical)")
    ax.set_title("MTQ vs Duhyeon direct diff SNR (for layers quantized by BOTH methods)")
    ax.set_ylabel("SNR (dB)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Row 3: BMM highlight ──────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(idx, snr_dh, lw=0.6, color="#FF9800", alpha=0.5, label="Duhyeon SNR all")
    if has_bmm.any():
        ax.scatter(idx[has_bmm], snr_dh[has_bmm], s=15, color="red", zorder=5,
                   label=f"BMM layers (n={has_bmm.sum()}) — q/k/v output also quantized")
    ax.set_title("Duhyeon BMM layers (q/k/v output also quantized)")
    ax.set_xlabel("Layer index (sequential, all nn.Linear)")
    ax.set_ylabel("SNR (dB)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_component_breakdown(results: dict, out_path: Path) -> None:
    """MSE by component: vision / LM / DiT."""
    def component(name):
        if "vision_tower" in name or "encoder.layers" in name or "siglip" in name.lower():
            return "vision"
        if "language_model" in name:
            return "LM"
        if "gemma_expert" in name:
            return "DiT"
        return "other"

    from collections import defaultdict
    comp_mse_mtq = defaultdict(list)
    comp_mse_dh  = defaultdict(list)

    for name, r in results.items():
        c = component(name)
        comp_mse_mtq[c].append(r["mtq"]["mse"])
        comp_mse_dh[c].append(r["dh"]["mse"])

    comps     = ["vision", "LM", "DiT", "other"]
    mtq_means = [np.mean(comp_mse_mtq[c]) if comp_mse_mtq[c] else 0 for c in comps]
    dh_means  = [np.mean(comp_mse_dh[c])  if comp_mse_dh[c]  else 0 for c in comps]
    mtq_ns    = [len(comp_mse_mtq[c]) for c in comps]
    dh_ns     = [len(comp_mse_dh[c])  for c in comps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mean MSE per Model Component", fontsize=12, fontweight="bold")

    x = np.arange(len(comps))
    w = 0.35
    bars_mtq = ax1.bar(x - w/2, mtq_means, w, label="MTQ",     color="#4CAF50", alpha=0.8)
    bars_dh  = ax1.bar(x + w/2, dh_means,  w, label="Duhyeon", color="#FF9800", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={dh_ns[i]})" for i, c in enumerate(comps)])
    ax1.set_ylabel("Mean MSE")
    ax1.set_title("Mean per-layer MSE by component (linear scale)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    bars_mtq2 = ax2.bar(x - w/2, [max(v, 1e-15) for v in mtq_means], w,
                        label="MTQ", color="#4CAF50", alpha=0.8)
    bars_dh2  = ax2.bar(x + w/2, [max(v, 1e-15) for v in dh_means],  w,
                        label="Duhyeon", color="#FF9800", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{c}\n(n={dh_ns[i]})" for i, c in enumerate(comps)])
    ax2.set_ylabel("Mean MSE (log scale)")
    ax2.set_title("Mean per-layer MSE by component (log scale)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise MTQ vs Duhyeon error comparison"
    )
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task_id",         type=int, default=0)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--output_dir",      type=str, default="compare_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policy ───────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading policy: {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False

    env_cfg  = LiberoEnv(task="libero_10", task_ids=[args.task_id])
    envs_dict = make_env(env_cfg, n_envs=1)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][args.task_id]

    # ── Step 1: FP baseline capture (before any quantization) ─────────────────
    print("\n[INFO] Step 1: FP baseline capture ...")
    all_linear_names = get_all_linear_names(policy)
    print(f"  Total nn.Linear in model: {len(all_linear_names)}")

    fp_captures, fp_hooks = register_fp_capture_hooks(policy)

    with torch.no_grad():
        eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=1,
        )

    remove_hooks(fp_hooks)
    print(f"  Captured {len(fp_captures)} layers")

    # ── Step 2: Apply MTQ to identify quantized layers ────────────────────────
    print("\n[INFO] Step 2: Apply MTQ to determine quantized layer set ...")
    apply_nvfp4_lm_dit(policy)
    disable_torch_compile(policy)
    mtq_layers = get_mtq_quantized_names(policy)
    print(f"  MTQ quantized layers: {len(mtq_layers)}")

    # ── Step 3: Structural analysis ───────────────────────────────────────────
    print_structural_report(all_linear_names, mtq_layers)

    # Duhyeon BMM layers = q/k/v in ALL Linear
    dh_bmm_layers = {n for n in all_linear_names if is_qkv_proj(n)}
    print(f"\n  Duhyeon BMM layers (q/k/v — output also quantized): {len(dh_bmm_layers)}")

    # ── Step 4: Isolated per-layer comparison ─────────────────────────────────
    print("\n[INFO] Step 3: Running isolated per-layer comparison ...")
    # Only compare layers we actually captured (first-forward)
    captured_names  = set(fp_captures.keys())
    mtq_l_intersect = mtq_layers & captured_names
    dh_bmm_intersect = dh_bmm_layers & captured_names

    print(f"  Layers to compare: {len(captured_names)}")
    print(f"  MTQ quantized (captured): {len(mtq_l_intersect)}")
    print(f"  Duhyeon BMM (captured):   {len(dh_bmm_intersect)}")

    results = compute_isolated_comparison(
        fp_captures,
        mtq_l_intersect,
        dh_bmm_intersect,
    )

    # ── Step 5: Report ────────────────────────────────────────────────────────
    print_summary_stats(results)
    print_layer_table(results, top_n=40)

    # ── Step 6: Save results ──────────────────────────────────────────────────
    serializable = {}
    for name, r in results.items():
        serializable[name] = {
            "is_mtq_quantized": r["is_mtq_quantized"],
            "is_dh_only":       r["is_dh_only"],
            "has_bmm":          r["has_bmm"],
            "W_shape":          r["W_shape"],
            "mtq":              r["mtq"],
            "dh":               r["dh"],
            "diff":             r["diff"],
        }
    (out_dir / "layerwise_comparison.json").write_text(
        json.dumps(serializable, indent=2)
    )
    print(f"\n[SAVED] {out_dir}/layerwise_comparison.json")

    # ── Step 7: Plots ─────────────────────────────────────────────────────────
    plot_layerwise(results, out_dir / "layerwise_mse_snr.png")
    plot_component_breakdown(results, out_dir / "component_breakdown.png")

    print(f"\n[DONE] All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
