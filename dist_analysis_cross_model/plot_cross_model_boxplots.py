"""
Cross-Model Boxplot: LLM / VLA LM / Image DiT / VLA DiT

Weight & Activation 분포를 4가지 모델 타입별로 side-by-side boxplot으로 시각화.

Data sources:
  LLM      : plots/dist_stats_google_gemma_2b.json
  VLA LM   : lerobot/logs/dist_analysis_v4/dist_stats.json  (component=lm)
  Image DiT: Workspace_DiT/pixart_alpha/results/distribution_analysis/stats.json
  VLA DiT  : lerobot/logs/dist_analysis_v4/dist_stats.json  (component=dit)

Usage:
    python plot_cross_model_boxplots.py [--out_dir plots/]
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_LEROBOT = _HERE.parent / "lerobot"
_VLA_STATS = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"
_GEMMA_STATS = _HERE / "plots" / "dist_stats_google_gemma_2b.json"
_PIXART_STATS = (Path("/home/jameskimh/workspace/Workspace_DiT") /
                 "pixart_alpha" / "results" / "distribution_analysis" / "stats.json")
_OUT_DIR = _HERE / "plots"

# ── model metadata ────────────────────────────────────────────────────────────
MODEL_KEYS  = ["llm", "vla_lm", "image_dit", "vla_dit"]
MODEL_LABELS = {
    "llm":       "LLM\n(Gemma-2B)",
    "vla_lm":    "VLA LM\n(pi0.5 LM)",
    "image_dit": "Image DiT\n(PixArt-α)",
    "vla_dit":   "VLA DiT\n(pi0.5 DiT)",
}
MODEL_COLORS = {
    "llm":       "#5B8DB8",
    "vla_lm":    "#D65F5F",
    "image_dit": "#2CA02C",
    "vla_dit":   "#FF7F0E",
}

# Layer type groups for filtering
ATTN_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_TYPES  = {"gate_proj", "up_proj", "down_proj"}
ALL_LM_TYPES = ATTN_TYPES | MLP_TYPES

PIXART_ATTN_KEYS = {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"}
PIXART_MLP_KEYS  = {"ff.net.0.proj", "ff.net.2"}
ALL_PIXART_TYPES = PIXART_ATTN_KEYS | PIXART_MLP_KEYS


def _layer_type(name: str) -> str:
    for t in sorted(ALL_LM_TYPES, key=len, reverse=True):
        if name.endswith(t):
            return t
    return "other"


def _pixart_group(name: str) -> str:
    for t in ALL_PIXART_TYPES:
        if t in name:
            return "attn" if "attn" in t else "mlp"
    return "other"


# ── data loading ──────────────────────────────────────────────────────────────

def load_lm_weight(stats_json: Path, component: str) -> dict[str, list]:
    """Returns {metric: [values across quant layers]}"""
    with open(stats_json) as f:
        d = json.load(f)
    ws = d.get("weight_stats", d)  # gemma cache has flat structure

    out = defaultdict(list)
    for name, info in ws.items():
        # VLA stats have 'component'; Gemma stats don't
        if "component" in info and info["component"] != component:
            continue
        lt = info.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        for metric in ("kurtosis", "per_channel_absmax_cv", "abs_max"):
            v = info.get(metric)
            if v is not None and not np.isnan(float(v)):
                out[metric].append(float(v))
    return dict(out)


def load_lm_act(stats_json: Path, component: str | None) -> dict[str, list]:
    with open(stats_json) as f:
        d = json.load(f)
    ws = d.get("weight_stats", {})
    acts = d.get("activation_stats", {})

    out = defaultdict(list)
    for name, info in acts.items():
        wi = ws.get(name, {})
        # component=None means no filter (e.g. standalone LLM cache)
        if component is not None and "component" in wi and wi["component"] != component:
            continue
        lt = wi.get("layer_type", info.get("layer_type", _layer_type(name)))
        if lt == "other":
            continue
        for metric in ("kurtosis", "per_token_absmax_cv", "abs_max"):
            v = info.get(metric)
            if v is not None and not np.isnan(float(v)):
                out[metric].append(float(v))
    return dict(out)


def load_pixart_weight(stats_json: Path) -> dict[str, list]:
    with open(stats_json) as f:
        d = json.load(f)
    out = defaultdict(list)
    for name, info in d["weight"].items():
        if _pixart_group(name) == "other":
            continue
        # kurtosis
        if "kurtosis" in info:
            out["kurtosis"].append(float(info["kurtosis"]))
        # cv (PixArt stores as per-channel std/mean ratio)
        if "cv" in info:
            out["per_channel_absmax_cv"].append(float(info["cv"]))
        if "abs_max" in info:
            out["abs_max"].append(float(info["abs_max"]))
    return dict(out)


def load_pixart_act(stats_json: Path) -> dict[str, list]:
    with open(stats_json) as f:
        d = json.load(f)
    out = defaultdict(list)
    for name, info in d["dynamics"].items():
        if _pixart_group(name) == "other":
            continue
        act_data = info.get("act", {})
        if isinstance(act_data, str):
            try:
                act_data = eval(act_data)  # noqa: S307
            except Exception:
                continue
        agg = act_data.get("aggregated", {})
        if not agg:
            continue
        if "kurtosis" in agg:
            out["kurtosis"].append(float(agg["kurtosis"]))
        if "cv" in agg:
            out["per_token_absmax_cv"].append(float(agg["cv"]))
        if "abs_max" in agg:
            out["abs_max"].append(float(agg["abs_max"]))
    return dict(out)


# ── boxplot helpers ────────────────────────────────────────────────────────────

def _boxplot_group(ax, data_dict: dict, models: list, metric: str,
                   ylabel: str, title: str, log_scale: bool = False):
    """data_dict: {model_key: {metric: [values]}}"""
    positions = []
    all_data = []
    colors = []
    labels = []

    for i, mk in enumerate(models):
        vals = data_dict.get(mk, {}).get(metric, [])
        if vals:
            positions.append(i)
            all_data.append(vals)
            colors.append(MODEL_COLORS[mk])
            labels.append(MODEL_LABELS[mk])

    if not all_data:
        return

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(list(range(len(MODEL_KEYS))))
    ax.set_xticklabels([MODEL_LABELS[mk] for mk in MODEL_KEYS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    if log_scale:
        # symlog handles negative values gracefully
        all_vals = [v for vals in all_data for v in vals]
        has_neg = any(v <= 0 for v in all_vals)
        if has_neg:
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_yscale("log")


# ── main plot function ─────────────────────────────────────────────────────────

def plot_boxplots(weight_data: dict, act_data: dict, metric_w: str, metric_a: str,
                  ylabel_w: str, ylabel_a: str, suptitle: str,
                  out_path: Path, log_scale: bool = False):
    """
    2-row figure: top = weight, bottom = activation
    4 groups per row: llm / vla_lm / image_dit / vla_dit
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _boxplot_group(axes[0], weight_data, MODEL_KEYS, metric_w,
                   ylabel_w, f"Weight — {ylabel_w}", log_scale=log_scale)
    _boxplot_group(axes[1], act_data, MODEL_KEYS, metric_a,
                   ylabel_a, f"Activation — {ylabel_a}", log_scale=log_scale)

    # Legend
    handles = [mpatches.Patch(color=MODEL_COLORS[mk], label=MODEL_LABELS[mk].replace("\n", " "))
               for mk in MODEL_KEYS]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(suptitle, fontsize=12, y=1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_per_type_boxplots(weight_data_by_type: dict, act_data_by_type: dict,
                            metric_w: str, metric_a: str,
                            ylabel: str, suptitle: str, out_path: Path,
                            log_scale: bool = False):
    """
    Layer-type-level boxplot: rows=layer_type, cols=weight/activation
    Each group shows 4 model-color boxes side by side.
    """
    # Collect all layer types
    all_types = sorted(set(
        lt for d in list(weight_data_by_type.values()) + list(act_data_by_type.values())
        for lt in d
    ))
    if not all_types:
        return

    n_types = len(all_types)
    fig, axes = plt.subplots(n_types, 2, figsize=(14, 2.8 * n_types), sharex=False)
    if n_types == 1:
        axes = axes.reshape(1, 2)

    for row, lt in enumerate(all_types):
        for col, (data_by_type, label_prefix) in enumerate([
            (weight_data_by_type, "Weight"),
            (act_data_by_type, "Activation"),
        ]):
            ax = axes[row, col]
            metric = metric_w if col == 0 else metric_a
            positions = []
            all_data = []
            valid_models = []

            for i, mk in enumerate(MODEL_KEYS):
                vals = data_by_type.get(mk, {}).get(lt, {}).get(metric, [])
                if vals:
                    positions.append(i)
                    all_data.append(vals)
                    valid_models.append(mk)

            if not all_data:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=9, color="#999999", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{label_prefix} — {lt}", fontsize=9)
                continue

            bp = ax.boxplot(
                all_data, positions=positions, widths=0.5,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                medianprops=dict(color="black", linewidth=1.5),
            )
            for patch, mk in zip(bp["boxes"], valid_models):
                patch.set_facecolor(MODEL_COLORS[mk])
                patch.set_alpha(0.75)

            ax.set_xticks(positions)
            ax.set_xticklabels([MODEL_LABELS[mk] for mk in valid_models], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{label_prefix} — {lt}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if log_scale:
                all_flat = [v for vals in all_data for v in vals]
                has_neg = any(v <= 0 for v in all_flat)
                ax.set_yscale("symlog", linthresh=1.0) if has_neg else ax.set_yscale("log")

    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── per-type data loader ───────────────────────────────────────────────────────

def load_lm_weight_by_type(stats_json: Path, component: str) -> dict[str, dict]:
    """Returns {layer_type: {metric: [values]}}"""
    with open(stats_json) as f:
        d = json.load(f)
    ws = d.get("weight_stats", d)

    out = defaultdict(lambda: defaultdict(list))
    for name, info in ws.items():
        if "component" in info and info["component"] != component:
            continue
        lt = info.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        for metric in ("kurtosis", "per_channel_absmax_cv", "abs_max"):
            v = info.get(metric)
            if v is not None and not np.isnan(float(v)):
                out[lt][metric].append(float(v))
    return {k: dict(v) for k, v in out.items()}


def load_lm_act_by_type(stats_json: Path, component: str | None) -> dict[str, dict]:
    with open(stats_json) as f:
        d = json.load(f)
    ws = d.get("weight_stats", {})
    acts = d.get("activation_stats", {})

    out = defaultdict(lambda: defaultdict(list))
    for name, info in acts.items():
        wi = ws.get(name, {})
        if component is not None and "component" in wi and wi["component"] != component:
            continue
        lt = wi.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        for metric in ("kurtosis", "per_token_absmax_cv", "abs_max"):
            v = info.get(metric)
            if v is not None and not np.isnan(float(v)):
                out[lt][metric].append(float(v))
    return {k: dict(v) for k, v in out.items()}


def load_pixart_weight_by_type(stats_json: Path) -> dict[str, dict]:
    with open(stats_json) as f:
        d = json.load(f)
    # Map PixArt names to canonical types
    type_map = {
        "attn1.to_q": "q_proj", "attn1.to_k": "k_proj",
        "attn1.to_v": "v_proj", "attn1.to_out.0": "o_proj",
        "attn2.to_q": "q_proj(cross)", "attn2.to_k": "k_proj(cross)",
        "attn2.to_v": "v_proj(cross)", "attn2.to_out.0": "o_proj(cross)",
        "ff.net.0.proj": "gate_proj", "ff.net.2": "down_proj",
    }
    out = defaultdict(lambda: defaultdict(list))
    for name, info in d["weight"].items():
        lt = next((v for k, v in type_map.items() if k in name), None)
        if lt is None:
            continue
        for dst, src in [("kurtosis", "kurtosis"),
                          ("per_channel_absmax_cv", "cv"),
                          ("abs_max", "abs_max")]:
            v = info.get(src)
            if v is not None:
                out[lt][dst].append(float(v))
    return {k: dict(v) for k, v in out.items()}


def load_pixart_act_by_type(stats_json: Path) -> dict[str, dict]:
    with open(stats_json) as f:
        d = json.load(f)
    type_map = {
        "attn1.to_q": "q_proj", "attn1.to_k": "k_proj",
        "attn1.to_v": "v_proj", "attn1.to_out.0": "o_proj",
        "attn2.to_q": "q_proj(cross)", "attn2.to_k": "k_proj(cross)",
        "attn2.to_v": "v_proj(cross)", "attn2.to_out.0": "o_proj(cross)",
        "ff.net.0.proj": "gate_proj", "ff.net.2": "down_proj",
    }
    out = defaultdict(lambda: defaultdict(list))
    for name, info in d["dynamics"].items():
        lt = next((v for k, v in type_map.items() if k in name), None)
        if lt is None:
            continue
        act_data = info.get("act", {})
        if isinstance(act_data, str):
            try:
                act_data = eval(act_data)  # noqa: S307
            except Exception:
                continue
        agg = act_data.get("aggregated", {})
        if not agg:
            continue
        for dst, src in [("kurtosis", "kurtosis"),
                          ("per_token_absmax_cv", "cv"),
                          ("abs_max", "abs_max")]:
            v = agg.get(src)
            if v is not None:
                out[lt][dst].append(float(v))
    return {k: dict(v) for k, v in out.items()}


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(_OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all-layer data (for aggregate boxplots) ───────────────────────
    print("[INFO] Loading data for all 4 model types ...")

    # Weight (all layers aggregated)
    weight_data = {
        "llm":       load_lm_weight(_GEMMA_STATS, "lm"),
        "vla_lm":    load_lm_weight(_VLA_STATS, "lm"),
        "image_dit": load_pixart_weight(_PIXART_STATS),
        "vla_dit":   load_lm_weight(_VLA_STATS, "dit"),
    }
    # Activation (all layers aggregated)
    act_data = {
        "llm":       load_lm_act(_GEMMA_STATS, None),
        "vla_lm":    load_lm_act(_VLA_STATS, "lm"),
        "image_dit": load_pixart_act(_PIXART_STATS),
        "vla_dit":   load_lm_act(_VLA_STATS, "dit"),
    }

    for mk in MODEL_KEYS:
        nw = len(weight_data[mk].get("kurtosis", []))
        na = len(act_data[mk].get("kurtosis", []))
        print(f"  {mk:12s}: weight={nw} layers, act={na} layers")

    # ── Plot 1: Kurtosis (weight + act) ───────────────────────────────────
    plot_boxplots(
        weight_data, act_data,
        metric_w="kurtosis", metric_a="kurtosis",
        ylabel_w="Kurtosis", ylabel_a="Kurtosis",
        suptitle="Kurtosis Distribution: LLM / VLA LM / Image DiT / VLA DiT",
        out_path=out_dir / "cross_model_kurtosis_boxplot.png",
        log_scale=True,
    )

    # ── Plot 2: CV (weight per-channel / act per-token) ───────────────────
    plot_boxplots(
        weight_data, act_data,
        metric_w="per_channel_absmax_cv", metric_a="per_token_absmax_cv",
        ylabel_w="Per-Channel AbsMax CV", ylabel_a="Per-Token AbsMax CV",
        suptitle="CV Distribution: LLM / VLA LM / Image DiT / VLA DiT",
        out_path=out_dir / "cross_model_cv_boxplot.png",
        log_scale=True,
    )

    # ── Plot 3: AbsMax ────────────────────────────────────────────────────
    plot_boxplots(
        weight_data, act_data,
        metric_w="abs_max", metric_a="abs_max",
        ylabel_w="AbsMax", ylabel_a="AbsMax",
        suptitle="AbsMax Distribution: LLM / VLA LM / Image DiT / VLA DiT",
        out_path=out_dir / "cross_model_absmax_boxplot.png",
        log_scale=True,
    )

    # ── Per-layer-type data ────────────────────────────────────────────────
    print("[INFO] Loading per-layer-type data ...")
    weight_by_type = {
        "llm":       load_lm_weight_by_type(_GEMMA_STATS, "lm"),
        "vla_lm":    load_lm_weight_by_type(_VLA_STATS, "lm"),
        "image_dit": load_pixart_weight_by_type(_PIXART_STATS),
        "vla_dit":   load_lm_weight_by_type(_VLA_STATS, "dit"),
    }
    act_by_type = {
        "llm":       load_lm_act_by_type(_GEMMA_STATS, None),
        "vla_lm":    load_lm_act_by_type(_VLA_STATS, "lm"),
        "image_dit": load_pixart_act_by_type(_PIXART_STATS),
        "vla_dit":   load_lm_act_by_type(_VLA_STATS, "dit"),
    }

    # ── Plot 4: Per-type kurtosis ─────────────────────────────────────────
    plot_per_type_boxplots(
        weight_by_type, act_by_type,
        metric_w="kurtosis", metric_a="kurtosis",
        ylabel="Kurtosis",
        suptitle="Kurtosis by Layer Type: LLM / VLA LM / Image DiT / VLA DiT",
        out_path=out_dir / "cross_model_kurtosis_by_type.png",
        log_scale=False,
    )

    # ── Plot 5: Per-type CV ───────────────────────────────────────────────
    plot_per_type_boxplots(
        weight_by_type, act_by_type,
        metric_w="per_channel_absmax_cv", metric_a="per_token_absmax_cv",
        ylabel="CV",
        suptitle="CV by Layer Type: LLM / VLA LM / Image DiT / VLA DiT",
        out_path=out_dir / "cross_model_cv_by_type.png",
        log_scale=False,
    )

    print(f"\n[DONE] Cross-model boxplots saved to {out_dir}")


if __name__ == "__main__":
    main()
