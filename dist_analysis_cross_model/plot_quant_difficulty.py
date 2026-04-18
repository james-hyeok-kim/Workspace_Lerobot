"""
Quantization Difficulty Scatter: Kurtosis vs CV (2D)

  X = per-channel absmax CV (weight) / per-token absmax CV (activation)
  Y = Kurtosis
  Low kurtosis + Low CV  → easy to quantize (bottom-left)
  High kurtosis + High CV → hard to quantize (top-right)

4 model types: LLM / VLA LM / Image DiT / VLA DiT
Separate figures for Weight and Activation.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE        = Path(__file__).resolve().parent
_LEROBOT     = _HERE.parent / "lerobot"
_VLA_STATS   = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"
_GEMMA_STATS = _HERE / "plots" / "dist_stats_google_gemma_2b.json"
_PIXART_STATS = (Path("/home/jameskimh/workspace/Workspace_DiT") /
                 "pixart_alpha" / "results" / "distribution_analysis" / "stats.json")
_OUT_DIR = _HERE / "plots"

MODEL_KEYS   = ["llm", "vla_lm", "image_dit", "vla_dit"]
MODEL_LABELS = {
    "llm":       "LLM (Gemma-2B)",
    "vla_lm":    "VLA LM (pi0.5)",
    "image_dit": "Image DiT (PixArt-α)",
    "vla_dit":   "VLA DiT (pi0.5)",
}
MODEL_COLORS = {
    "llm":       "#5B8DB8",
    "vla_lm":    "#D65F5F",
    "image_dit": "#2CA02C",
    "vla_dit":   "#FF7F0E",
}
MARKERS = {
    "llm": "o", "vla_lm": "s", "image_dit": "^", "vla_dit": "D",
}

ALL_LM_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
PIXART_TYPE_MAP = {
    "attn1.to_q": "q_proj", "attn1.to_k": "k_proj",
    "attn1.to_v": "v_proj", "attn1.to_out.0": "o_proj",
    "ff.net.0.proj": "gate_proj", "ff.net.2": "down_proj",
}

def _layer_type(name):
    for t in sorted(ALL_LM_TYPES, key=len, reverse=True):
        if name.endswith(t):
            return t
    return "other"

import re
def _layer_idx(name):
    m = re.search(r'layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None


# ── data loaders ──────────────────────────────────────────────────────────────

def load_lm_weight(path, component):
    with open(path) as f:
        d = json.load(f)
    ws = d.get("weight_stats", d)
    out = []
    for name, info in ws.items():
        if component is not None and "component" in info and info["component"] != component:
            continue
        lt = info.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        k = info.get("kurtosis")
        cv = info.get("per_channel_absmax_cv")
        if k is not None and cv is not None and not np.isnan(k) and not np.isnan(cv):
            out.append((float(cv), float(k), lt))
    return out  # list of (cv, kurtosis, layer_type)


def load_lm_act(path, component):
    with open(path) as f:
        d = json.load(f)
    ws   = d.get("weight_stats", {})
    acts = d.get("activation_stats", {})
    out = []
    for name, info in acts.items():
        wi = ws.get(name, {})
        if component is not None and "component" in wi and wi["component"] != component:
            continue
        lt = wi.get("layer_type", info.get("layer_type", _layer_type(name)))
        if lt == "other":
            continue
        k = info.get("kurtosis")
        cv = info.get("per_token_absmax_cv")
        if k is not None and cv is not None and not np.isnan(k) and not np.isnan(cv):
            out.append((float(cv), float(k), lt))
    return out


def load_pixart_weight(path):
    with open(path) as f:
        d = json.load(f)
    out = []
    for name, info in d["weight"].items():
        lt = next((v for k, v in PIXART_TYPE_MAP.items() if k in name), None)
        if lt is None:
            continue
        k  = info.get("kurtosis")
        cv = info.get("cv")
        if k is not None and cv is not None:
            out.append((float(cv), float(k), lt))
    return out


def load_pixart_act(path):
    with open(path) as f:
        d = json.load(f)
    out = []
    for name, info in d["dynamics"].items():
        lt = next((v for k, v in PIXART_TYPE_MAP.items() if k in name), None)
        if lt is None:
            continue
        act = info.get("act", {})
        if isinstance(act, str):
            try:
                act = eval(act)  # noqa: S307
            except Exception:
                continue
        agg = act.get("aggregated", {})
        k  = agg.get("kurtosis")
        cv = agg.get("cv")
        if k is not None and cv is not None and not np.isnan(float(k)) and not np.isnan(float(cv)):
            out.append((float(cv), float(k), lt))
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def _summary_stats(pts):
    """Return mean and std of (cv, kurtosis) for a list of (cv, k, lt) tuples."""
    if not pts:
        return None
    cvs = [p[0] for p in pts]
    ks  = [p[1] for p in pts]
    return np.mean(cvs), np.mean(ks), np.std(cvs), np.std(ks)


DIFFICULTY_LABELS = {
    # (high_k, high_cv) → label
    (False, False): ("Easy",     "#2ca02c"),
    (False, True):  ("Hard",     "#e07b00"),
    (True,  False): ("Moderate", "#1f6dbf"),
    (True,  True):  ("Hardest",  "#cc2222"),
}


def plot_scatter(weight_data, act_data, out_path, title):
    """
    2-panel figure: left=weight, right=activation.
    Each model: scatter + mean errorbar + difficulty label near the mean point.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (data_dict, cv_label, k_label, panel_title) in zip(axes, [
        (weight_data, "Per-Channel AbsMax CV (Weight)", "Kurtosis (Weight)", "Weight"),
        (act_data,    "Per-Token AbsMax CV (Activation)", "Kurtosis (Activation)", "Activation"),
    ]):
        # Collect means first to determine medians for quadrant boundaries
        means = {}
        for mk in MODEL_KEYS:
            pts = data_dict.get(mk, [])
            if not pts:
                continue
            cvs = np.array([p[0] for p in pts])
            ks  = np.array([p[1] for p in pts])
            means[mk] = (cvs.mean(), ks.mean(), cvs.std(), ks.std(), len(pts))

        all_cvs = [p[0] for pts in data_dict.values() for p in pts]
        all_ks  = [p[1] for pts in data_dict.values() for p in pts]
        if not all_cvs:
            continue
        med_cv = float(np.median(all_cvs))
        med_k  = float(np.median(all_ks))

        # Draw individual layer scatter + mean errorbar
        for mk in MODEL_KEYS:
            pts = data_dict.get(mk, [])
            if not pts:
                continue
            cvs = np.array([p[0] for p in pts])
            ks  = np.array([p[1] for p in pts])
            color  = MODEL_COLORS[mk]
            marker = MARKERS[mk]

            ax.scatter(cvs, ks, s=18, color=color, marker=marker,
                       alpha=0.35, linewidths=0)

            mx, mk_, sx, sk, n = means[mk]
            ax.errorbar(mx, mk_,
                        xerr=sx, yerr=sk,
                        fmt=marker, color=color, markersize=12,
                        markeredgecolor="white", markeredgewidth=1.2,
                        elinewidth=1.8, capsize=5,
                        label=f"{MODEL_LABELS[mk]} (n={n})",
                        zorder=5)

            # Difficulty label near the mean point
            high_k  = mk_ > med_k
            high_cv = mx  > med_cv
            dlabel, dcolor = DIFFICULTY_LABELS[(high_k, high_cv)]
            ax.annotate(dlabel, xy=(mx, mk_),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color=dcolor, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=dcolor, alpha=0.75, linewidth=0.8),
                        zorder=6)

        # Log scale
        ax.set_xscale("log")
        has_neg_k = any(v <= 0 for v in all_ks)
        ax.set_yscale("symlog", linthresh=1.0) if has_neg_k else ax.set_yscale("log")

        ax.set_xlabel(cv_label, fontsize=10)
        ax.set_ylabel(k_label, fontsize=10)
        ax.set_title(f"{panel_title} — Quantization Difficulty", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3, which="both")

        # Median divider lines
        ax.axvline(med_cv, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(med_k,  color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_scatter_by_type(weight_data, act_data, out_path, title):
    """
    Layer-type color coding within each model, 4 models as different marker shapes.
    Single-panel scatter for weight (left) and activation (right).
    """
    # Assign colors to layer types
    TYPE_COLORS = {
        "q_proj": "#1f77b4", "k_proj": "#aec7e8",
        "v_proj": "#ff7f0e", "o_proj": "#ffbb78",
        "gate_proj": "#2ca02c", "up_proj": "#98df8a", "down_proj": "#d62728",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (data_dict, cv_label, k_label, panel_title) in zip(axes, [
        (weight_data, "Per-Channel AbsMax CV (Weight)", "Kurtosis (Weight)", "Weight"),
        (act_data,    "Per-Token AbsMax CV (Activation)", "Kurtosis (Activation)", "Activation"),
    ]):
        plotted_types = set()
        plotted_models = set()

        for mk in MODEL_KEYS:
            pts = data_dict.get(mk, [])
            if not pts:
                continue
            marker = MARKERS[mk]
            for cv, k, lt in pts:
                color = TYPE_COLORS.get(lt, "#999999")
                ax.scatter(cv, k, s=30, color=color, marker=marker,
                           alpha=0.6, linewidths=0.3, edgecolors="white")
                plotted_types.add(lt)
                plotted_models.add(mk)

        # Legend: model shapes
        model_handles = [
            plt.scatter([], [], marker=MARKERS[mk], color="gray", s=50,
                        label=MODEL_LABELS[mk])
            for mk in MODEL_KEYS if mk in plotted_models
        ]
        type_handles = [
            mpatches.Patch(color=TYPE_COLORS.get(lt, "#999999"), label=lt)
            for lt in ALL_LM_TYPES if lt in plotted_types
        ]
        leg1 = ax.legend(handles=model_handles, title="Model", fontsize=7,
                         loc="upper left", framealpha=0.8)
        ax.add_artist(leg1)
        ax.legend(handles=type_handles, title="Layer Type", fontsize=7,
                  loc="upper right", framealpha=0.8)

        # Log scale
        all_cvs = [p[0] for pts in data_dict.values() for p in pts]
        all_ks  = [p[1] for pts in data_dict.values() for p in pts]
        ax.set_xscale("log")
        has_neg_k = any(v <= 0 for v in all_ks)
        ax.set_yscale("symlog", linthresh=1.0) if has_neg_k else ax.set_yscale("log")

        ax.set_xlabel(cv_label, fontsize=10)
        ax.set_ylabel(k_label, fontsize=10)
        ax.set_title(f"{panel_title} — by Layer Type", fontsize=11)
        ax.grid(alpha=0.3, which="both")

        # Quadrant dividers + per-model difficulty annotation
        if all_cvs and all_ks:
            med_cv = float(np.median(all_cvs))
            med_k  = float(np.median(all_ks))
            ax.axvline(med_cv, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(med_k,  color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

            # Annotate each model's mean with difficulty label
            for mk in MODEL_KEYS:
                pts = data_dict.get(mk, [])
                if not pts:
                    continue
                mx = float(np.mean([p[0] for p in pts]))
                my = float(np.mean([p[1] for p in pts]))
                high_k  = my > med_k
                high_cv = mx > med_cv
                dlabel, dcolor = DIFFICULTY_LABELS[(high_k, high_cv)]
                ax.annotate(dlabel, xy=(mx, my),
                            xytext=(8, 8), textcoords="offset points",
                            fontsize=7, color=dcolor, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=dcolor, alpha=0.75, linewidth=0.8),
                            zorder=6)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    out_dir = _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data ...")
    weight_data = {
        "llm":       load_lm_weight(_GEMMA_STATS, None),
        "vla_lm":    load_lm_weight(_VLA_STATS, "lm"),
        "image_dit": load_pixart_weight(_PIXART_STATS),
        "vla_dit":   load_lm_weight(_VLA_STATS, "dit"),
    }
    act_data = {
        "llm":       load_lm_act(_GEMMA_STATS, None),
        "vla_lm":    load_lm_act(_VLA_STATS, "lm"),
        "image_dit": load_pixart_act(_PIXART_STATS),
        "vla_dit":   load_lm_act(_VLA_STATS, "dit"),
    }

    for mk in MODEL_KEYS:
        print(f"  {mk:12s}: weight={len(weight_data[mk])} layers, act={len(act_data[mk])} layers")

    # Plot 1: model-color scatter (all layers, mean±std)
    plot_scatter(
        weight_data, act_data,
        out_path=out_dir / "quant_difficulty_scatter.png",
        title="Quantization Difficulty: Kurtosis vs CV — 4 Model Types",
    )

    # Plot 2: layer-type-color scatter
    plot_scatter_by_type(
        weight_data, act_data,
        out_path=out_dir / "quant_difficulty_by_type.png",
        title="Quantization Difficulty by Layer Type: Kurtosis vs CV",
    )

    print(f"\n[DONE] Saved to {out_dir}")


if __name__ == "__main__":
    main()
