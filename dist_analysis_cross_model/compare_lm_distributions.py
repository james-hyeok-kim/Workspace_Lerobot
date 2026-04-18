"""
Task 1: VLA LM (pi0.5 PaliGemma Gemma-2B) vs Standalone Gemma-2B LLM
        Weight & Activation Distribution Comparison.

Weight stats: loaded from existing dist_stats.json (VLA LM)
              + freshly collected from google/gemma-2b (standalone)
Activation stats: VLA LM from existing dist_stats.json
                  Gemma-2B from WikiText-2 text prompts

Usage:
    # Default: compare VLA finetuned vs VLA base (lerobot/pi05_libero_base, cached)
    python compare_lm_distributions.py

    # With standalone Gemma-2B (requires HF token with gated access)
    python compare_lm_distributions.py --baseline_model google/gemma-2b
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_LEROBOT = _HERE.parent / "lerobot"
_STATS_JSON = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"
_OUT_DIR = _HERE / "plots"

# Add lerobot src to path
for _p in [str(_LEROBOT / "src"), str(_LEROBOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── layer classification ───────────────────────────────────────────────────────
ATTN_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_TYPES = {"gate_proj", "up_proj", "down_proj"}
ALL_TYPES = list(ATTN_TYPES) + list(MLP_TYPES)

# VLA LM key prefix
VLA_LM_PREFIX = "model.paligemma_with_expert.paligemma.model.language_model."

# Gemma-2B key prefix
GEMMA_PREFIX = "model."


def _layer_idx(name: str):
    m = re.search(r'layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def _layer_type(name: str):
    for t in ALL_TYPES:
        if name.endswith(t):
            return t
    return "other"


def _is_lm_quant_layer(name: str) -> bool:
    return _layer_type(name) != "other"


# ── VLA LM stats from existing json ──────────────────────────────────────────

def load_vla_lm_stats(stats_json: Path):
    with open(stats_json) as f:
        d = json.load(f)
    w = {k: v for k, v in d["weight_stats"].items()
         if v.get("component") == "lm" and v.get("layer_type") != "other"}
    a = {}
    for k, v in d["activation_stats"].items():
        if k in d["weight_stats"]:
            wi = d["weight_stats"][k]
            if wi.get("component") == "lm" and wi.get("layer_type") != "other":
                a[k] = {**v, "layer_type": wi["layer_type"]}
    return w, a


# ── Gemma-2B stats collection ─────────────────────────────────────────────────

def collect_gemma_weight_stats(model) -> dict:
    import torch
    import scipy.stats as ss

    stats = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, "weight") or mod.weight is None:
            continue
        if not isinstance(mod.weight, torch.Tensor):
            continue
        lt = _layer_type(name)
        if lt == "other":
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue

        W = mod.weight.detach().float()
        flat = W.flatten().cpu().numpy()
        if flat.size > 50_000:
            step = flat.size // 50_000
            flat = flat[::step][:50_000]

        per_ch = W.abs().amax(dim=1).cpu().numpy()
        stats[name] = {
            "layer_type": lt,
            "block_idx": idx,
            "mean": float(W.mean()),
            "std": float(W.std()),
            "abs_max": float(W.abs().max()),
            "kurtosis": float(ss.kurtosis(flat)),
            "per_channel_absmax_cv": float(per_ch.std() / (per_ch.mean() + 1e-8)),
            "per_channel_absmax_mean": float(per_ch.mean()),
            "percentiles": {
                f"p{int(q*100)}": float(v)
                for q, v in zip([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                                np.percentile(flat, [1, 5, 25, 50, 75, 95, 99]))
            },
        }
    return stats


def collect_gemma_act_stats(model, tokenizer, texts: list, device: str) -> dict:
    import torch
    import scipy.stats as ss

    raw = defaultdict(list)
    handles = []

    for name, mod in model.named_modules():
        lt = _layer_type(name)
        if lt == "other":
            continue
        if _layer_idx(name) is None:
            continue

        def make_hook(n, lt_):
            def hook(module, inp, out):
                x = inp[0].detach().float()
                flat = x.flatten().cpu().numpy()
                if flat.size > 50_000:
                    step = flat.size // 50_000
                    flat = flat[::step][:50_000]
                tok_absmax = x.abs().amax(dim=-1).flatten().cpu().numpy()
                pcts = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
                raw[n].append({
                    "layer_type": lt_,
                    "abs_max": float(x.abs().max()),
                    "mean": float(flat.mean()),
                    "kurtosis": float(ss.kurtosis(flat)),
                    "per_token_absmax_cv": float(tok_absmax.std() / (tok_absmax.mean() + 1e-8)),
                    "per_token_absmax_mean": float(tok_absmax.mean()),
                    "p1":  float(pcts[0]),
                    "p5":  float(pcts[1]),
                    "p25": float(pcts[2]),
                    "p50": float(pcts[3]),
                    "p75": float(pcts[4]),
                    "p95": float(pcts[5]),
                    "p99": float(pcts[6]),
                })
            return hook

        h = mod.register_forward_hook(make_hook(name, lt))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**enc)

    for h in handles:
        h.remove()

    # aggregate (average scalar stats across all text samples)
    agg = {}
    for name, steps in raw.items():
        if not steps:
            continue
        lt_ = steps[0]["layer_type"]
        scalar_keys = ["abs_max", "mean", "kurtosis", "per_token_absmax_cv",
                       "per_token_absmax_mean", "p1", "p5", "p25", "p50", "p75", "p95", "p99"]
        agg[name] = {"layer_type": lt_}
        for k in scalar_keys:
            vals = [s[k] for s in steps if k in s]
            if vals:
                agg[name][k] = float(np.mean(vals))
    return agg


def save_input_samples_csv(texts: list, out_path: Path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "length", "text"])
        writer.writeheader()
        for i, t in enumerate(texts):
            writer.writerow({"idx": i, "length": len(t), "text": t.replace("\n", " ")})
    print(f"[SAVED] {out_path}")


def load_wikitext2(n_samples: int = 50) -> list:
    """WikiText-2 에서 텍스트 샘플 로드. datasets 없으면 fallback 텍스트 사용."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [r["text"] for r in ds if len(r["text"]) > 50][:n_samples]
        print(f"[INFO] Loaded {len(texts)} WikiText-2 samples")
        return texts
    except Exception as e:
        print(f"[WARN] WikiText-2 load failed ({e}), using fallback texts")
        base = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was nothing, and then the universe began to expand.",
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
            "The mitochondria is the powerhouse of the cell and plays a crucial role in energy production.",
            "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean.",
        ]
        return (base * (n_samples // len(base) + 1))[:n_samples]


# ── normalization helper ───────────────────────────────────────────────────────

def normalize_vla_key(name: str) -> str:
    """VLA LM name → (block_idx, layer_type) 표준화 키."""
    name = name.replace(VLA_LM_PREFIX, "")
    return name


def to_block_type(name: str, prefix_strip: str = "") -> tuple:
    name2 = name.replace(prefix_strip, "")
    idx = _layer_idx(name2)
    lt = _layer_type(name2)
    return idx, lt


# ── plotting ───────────────────────────────────────────────────────────────────

COLORS = {"vla": "#D65F5F", "baseline": "#4878CF"}
LAYER_TYPE_COLORS = {
    "q_proj": "#1f77b4", "k_proj": "#aec7e8", "v_proj": "#ff7f0e", "o_proj": "#ffbb78",
    "gate_proj": "#2ca02c", "up_proj": "#98df8a", "down_proj": "#d62728",
}


def _extract_series(stats: dict, metric: str, prefix_strip: str = ""):
    """Returns dict {(block_idx, layer_type): value}."""
    out = {}
    for name, info in stats.items():
        lt = info.get("layer_type", "other")
        if lt == "other":
            continue
        idx = _layer_idx(name.replace(prefix_strip, ""))
        if idx is None:
            continue
        v = info.get(metric)
        if v is not None:
            out[(idx, lt)] = float(v)
    return out


def plot_comparison(vla_stats: dict, gemma_stats: dict,
                    metric: str, title: str, ylabel: str,
                    out_path: Path, gemma_prefix: str = "",
                    baseline_label_arg: str = "Baseline LM"):

    vla_series = _extract_series(vla_stats, metric, prefix_strip=VLA_LM_PREFIX)
    gem_series = _extract_series(gemma_stats, metric, prefix_strip=gemma_prefix)

    # Get union of block indices
    all_blocks = sorted(set(idx for idx, _ in vla_series) | set(idx for idx, _ in gem_series))

    fig, axes = plt.subplots(len(ALL_TYPES), 1, figsize=(14, 2.5 * len(ALL_TYPES)), sharex=True)
    if len(ALL_TYPES) == 1:
        axes = [axes]

    for ax, lt in zip(axes, ALL_TYPES):
        vla_vals = [vla_series.get((b, lt), np.nan) for b in all_blocks]
        gem_vals = [gem_series.get((b, lt), np.nan) for b in all_blocks]

        x = np.arange(len(all_blocks))
        ax.plot(x, vla_vals, "o-", color=COLORS["vla"], label="VLA LM (pi0.5)", linewidth=1.5, markersize=4)
        ax.plot(x, gem_vals, "s--", color=COLORS["baseline"], label=baseline_label_arg, linewidth=1.5, markersize=4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(lt, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in all_blocks], fontsize=7)

    axes[-1].set_xlabel("Block index")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_violin_comparison(vla_stats: dict, gemma_stats: dict,
                           metric: str, title: str, out_path: Path,
                           baseline_label_arg: str = "Baseline LM"):
    """Violin plot: distribution of metric across all layers, grouped by layer_type."""
    vla_by_type = defaultdict(list)
    gem_by_type = defaultdict(list)

    for name, info in vla_stats.items():
        lt = info.get("layer_type", "other")
        if lt == "other":
            continue
        v = info.get(metric)
        if v is not None:
            vla_by_type[lt].append(float(v))

    for name, info in gemma_stats.items():
        lt = info.get("layer_type", "other")
        if lt == "other":
            continue
        v = info.get(metric)
        if v is not None:
            gem_by_type[lt].append(float(v))

    common_types = [lt for lt in ALL_TYPES if vla_by_type[lt] and gem_by_type[lt]]
    if not common_types:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(common_types)
    x = np.arange(n)
    width = 0.35

    for i, lt in enumerate(common_types):
        vd = vla_by_type[lt]
        gd = gem_by_type[lt]
        vparts = ax.violinplot([vd], positions=[i - width / 2], widths=0.3, showmedians=True)
        gparts = ax.violinplot([gd], positions=[i + width / 2], widths=0.3, showmedians=True)
        for pc in vparts["bodies"]:
            pc.set_facecolor(COLORS["vla"])
            pc.set_alpha(0.6)
        for pc in gparts["bodies"]:
            pc.set_facecolor(COLORS["baseline"])
            pc.set_alpha(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(common_types, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["vla"], label="VLA LM (pi0.5)"),
                        Patch(color=COLORS["baseline"], label=baseline_label_arg)])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(vla_w, gemma_w, vla_a, gemma_a, out_path: Path):
    import csv
    rows = []
    for lt in ALL_TYPES:
        def vals(stats, metric):
            vs = [info[metric] for info in stats.values()
                  if info.get("layer_type") == lt and metric in info]
            return (np.mean(vs) if vs else float("nan"),
                    np.std(vs) if vs else float("nan"))

        vw_k, vw_k_s = vals(vla_w, "kurtosis")
        gw_k, gw_k_s = vals(gemma_w, "kurtosis")
        vw_cv, vw_cv_s = vals(vla_w, "per_channel_absmax_cv")
        gw_cv, gw_cv_s = vals(gemma_w, "per_channel_absmax_cv")
        va_k, va_k_s = vals(vla_a, "kurtosis")
        ga_k, ga_k_s = vals(gemma_a, "kurtosis")
        va_cv, va_cv_s = vals(vla_a, "per_token_absmax_cv")
        ga_cv, ga_cv_s = vals(gemma_a, "per_token_absmax_cv")

        rows.append({
            "layer_type": lt,
            "vla_w_kurtosis_mean": f"{vw_k:.2f}", "vla_w_kurtosis_std": f"{vw_k_s:.2f}",
            "gemma_w_kurtosis_mean": f"{gw_k:.2f}", "gemma_w_kurtosis_std": f"{gw_k_s:.2f}",
            "vla_w_cv_mean": f"{vw_cv:.4f}", "vla_w_cv_std": f"{vw_cv_s:.4f}",
            "gemma_w_cv_mean": f"{gw_cv:.4f}", "gemma_w_cv_std": f"{gw_cv_s:.4f}",
            "vla_a_kurtosis_mean": f"{va_k:.2f}", "gemma_a_kurtosis_mean": f"{ga_k:.2f}",
            "vla_a_cv_mean": f"{va_cv:.4f}", "gemma_a_cv_mean": f"{ga_cv:.4f}",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, default="lerobot/pi05_libero_base",
                        help="Baseline LM model to compare against VLA finetuned. "
                             "Default: lerobot/pi05_libero_base (cached, same arch before finetuning). "
                             "Alt: google/gemma-2b (requires gated HF access).")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of WikiText-2 samples for activation collection")
    parser.add_argument("--out_dir", type=str, default=str(_OUT_DIR))
    parser.add_argument("--weight_only", action="store_true",
                        help="Skip activation collection (faster)")
    parser.add_argument("--load_baseline_cache", type=str, default="",
                        help="Load pre-saved baseline stats JSON instead of re-running")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_label = args.baseline_model.replace("/", "_").replace("-", "_")

    # ── Step 1: Load VLA LM stats ──────────────────────────────────────────
    print("[INFO] Loading VLA LM stats from existing dist_stats.json ...")
    vla_w, vla_a = load_vla_lm_stats(_STATS_JSON)
    print(f"[INFO] VLA LM (finetuned): {len(vla_w)} weight layers, {len(vla_a)} activation layers")

    # ── Step 2: Baseline model stats ─────────────────────────────────────
    cache_path = (Path(args.load_baseline_cache) if args.load_baseline_cache
                  else out_dir / f"dist_stats_{baseline_label}.json")

    if cache_path.exists():
        print(f"[INFO] Loading baseline stats from cache: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        gemma_w = cached.get("weight_stats", {})
        gemma_a = cached.get("activation_stats", {})
        model_display = cached.get("model_name", args.baseline_model)
    else:
        import torch
        import gc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_display = args.baseline_model

        is_lerobot = args.baseline_model.startswith("lerobot/")
        is_paligemma = "paligemma" in args.baseline_model.lower()

        print(f"[INFO] Loading baseline model: {args.baseline_model} ...")

        if is_lerobot:
            # Load via LeRobot policy API and extract the LM submodule
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_policy
            from lerobot.envs.configs import LiberoEnv
            cfg = PreTrainedConfig.from_pretrained(args.baseline_model)
            env_cfg = LiberoEnv(task="libero_spatial")
            policy = make_policy(cfg, env_cfg=env_cfg)
            policy = policy.to(device)
            # Extract language model
            model = policy.model.paligemma_with_expert.paligemma.model.language_model
            tokenizer = None
        elif is_paligemma:
            from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
            full = PaliGemmaForConditionalGeneration.from_pretrained(
                args.baseline_model, torch_dtype=torch.bfloat16, device_map=device)
            model = full.language_model
            tokenizer = AutoProcessor.from_pretrained(args.baseline_model).tokenizer
            del full; gc.collect()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.baseline_model)
            model = AutoModelForCausalLM.from_pretrained(
                args.baseline_model, torch_dtype=torch.bfloat16, device_map=device)

        model.eval()
        print("[INFO] Collecting baseline weight stats ...")
        gemma_w = collect_gemma_weight_stats(model)
        print(f"[INFO] Baseline weight layers: {len(gemma_w)}")

        if not args.weight_only and tokenizer is not None:
            print("[INFO] Loading WikiText-2 texts ...")
            texts = load_wikitext2(args.n_samples)
            save_input_samples_csv(texts, out_dir / "gemma_input_samples.csv")
            print("[INFO] Collecting baseline activation stats ...")
            gemma_a = collect_gemma_act_stats(model, tokenizer, texts, device)
            print(f"[INFO] Baseline activation layers: {len(gemma_a)}")
        else:
            gemma_a = {}
            if not args.weight_only:
                print("[INFO] Skipping activation (no tokenizer for lerobot model)")

        with open(cache_path, "w") as f:
            json.dump({"model_name": model_display,
                       "weight_stats": gemma_w, "activation_stats": gemma_a}, f)
        print(f"[SAVED] {cache_path}")

        del model
        gc.collect()
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    bl_label = f"Baseline LM ({args.baseline_model.split('/')[-1]})"

    # ── Step 3: Generate plots ─────────────────────────────────────────────
    print("[INFO] Generating comparison plots ...")

    plot_comparison(vla_w, gemma_w, "kurtosis",
                    f"Weight Kurtosis: VLA LM (finetuned) vs {bl_label}",
                    "Kurtosis",
                    out_dir / "lm_weight_kurtosis_comparison.png",
                    baseline_label_arg=bl_label)

    plot_comparison(vla_w, gemma_w, "per_channel_absmax_cv",
                    f"Weight Per-Channel AbsMax CV: VLA LM vs {bl_label}",
                    "CV",
                    out_dir / "lm_weight_cv_comparison.png",
                    baseline_label_arg=bl_label)

    plot_comparison(vla_w, gemma_w, "abs_max",
                    f"Weight AbsMax: VLA LM vs {bl_label}",
                    "AbsMax",
                    out_dir / "lm_weight_absmax_comparison.png",
                    baseline_label_arg=bl_label)

    plot_violin_comparison(vla_w, gemma_w, "kurtosis",
                           f"Weight Kurtosis Distribution: VLA LM vs {bl_label}",
                           out_dir / "lm_weight_kurtosis_violin.png",
                           baseline_label_arg=bl_label)

    if vla_a and gemma_a:
        plot_comparison(vla_a, gemma_a, "kurtosis",
                        f"Activation Kurtosis: VLA LM vs {bl_label}",
                        "Kurtosis",
                        out_dir / "lm_act_kurtosis_comparison.png",
                        baseline_label_arg=bl_label)

        plot_comparison(vla_a, gemma_a, "per_token_absmax_cv",
                        f"Activation Per-Token CV: VLA LM vs {bl_label}",
                        "CV",
                        out_dir / "lm_act_cv_comparison.png",
                        baseline_label_arg=bl_label)

        plot_violin_comparison(vla_a, gemma_a, "kurtosis",
                               f"Activation Kurtosis Distribution: VLA LM vs {bl_label}",
                               out_dir / "lm_act_kurtosis_violin.png",
                               baseline_label_arg=bl_label)

    save_summary_csv(vla_w, gemma_w, vla_a, gemma_a,
                     out_dir / "lm_summary_stats.csv")

    print(f"\n[DONE] All LM comparison plots saved to {out_dir}")


if __name__ == "__main__":
    main()
