"""
Task 3: Heatmap 재플롯 — other 레이어 제거, quant 대상만.

기존 lerobot/logs/dist_analysis_v4/dist_stats.json 을 로드하고
layer_type != "other" 인 레이어만 필터링하여 heatmap / bar 플롯을 재생성.

Usage:
    python replot_heatmaps.py [--out_dir plots/]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_LEROBOT = _HERE.parent / "lerobot"
_STATS_JSON = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"

# ── constants ─────────────────────────────────────────────────────────────────
PREFERRED_ORDER = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "action_in_proj", "action_out_proj",
    "time_mlp_in", "time_mlp_out",
]

QUANT_TYPES = set(PREFERRED_ORDER)  # "other" 제외

COMP_COLOR = {"lm": "#4878CF", "dit": "#D65F5F"}
COMP_TITLE = {"lm": "VLA LM (PaliGemma Gemma-2B)", "dit": "VLA DiT (Gemma Expert + Head)"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _layer_idx(name: str):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def filter_quant(stats: dict) -> dict:
    """layer_type != 'other' 만 남긴다."""
    return {k: v for k, v in stats.items() if v.get("layer_type", "other") != "other"}


def get_layout(stats: dict, comp: str):
    """layer_types list, col_indices list, col_labels list 반환."""
    present_types = set()
    indexed_idxs = set()
    has_nonindexed = set()

    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        present_types.add(lt)
        idx = _layer_idx(name)
        if idx is not None:
            indexed_idxs.add(idx)
        else:
            has_nonindexed.add(lt)

    ordered = [t for t in PREFERRED_ORDER if t in present_types]
    extra = sorted(present_types - set(ordered))
    layer_types = ordered + extra

    sorted_idxs = sorted(indexed_idxs)
    n_indexed = max(sorted_idxs) + 1 if sorted_idxs else 0
    col_indices = list(range(n_indexed))
    col_labels = [str(i) for i in range(n_indexed)]

    if has_nonindexed:
        col_indices.append(n_indexed)
        col_labels.append("fixed")

    return layer_types, col_indices, col_labels


def build_matrix(stats: dict, comp: str, metric: str, layer_types, col_indices, col_labels):
    n_rows = len(layer_types)
    n_cols = len(col_indices)
    mat = np.full((n_rows, n_cols), np.nan)
    fixed_col = col_labels.index("fixed") if "fixed" in col_labels else None

    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        if lt not in layer_types:
            continue
        row = layer_types.index(lt)
        idx = _layer_idx(name)
        if idx is not None and idx < n_cols:
            v = info.get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mat[row, idx] = float(v)
        elif idx is None and fixed_col is not None:
            v = info.get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mat[row, fixed_col] = float(v)

    return mat


# ── plotting functions ─────────────────────────────────────────────────────────

def plot_heatmap(stats: dict, comp: str, metric: str, title: str, out_path: Path):
    layer_types, col_indices, col_labels = get_layout(stats, comp)
    if not layer_types:
        print(f"[WARN] No data for comp={comp}")
        return

    mat = build_matrix(stats, comp, metric, layer_types, col_indices, col_labels)
    if np.all(np.isnan(mat)):
        print(f"[WARN] All NaN for {comp}/{metric}, skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(col_indices) * 0.5), max(4, len(layer_types) * 0.7)))

    vmax = np.nanpercentile(mat, 95)
    vmin = np.nanmin(mat)
    im = ax.imshow(mat, aspect="auto", cmap="hot_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_yticks(range(len(layer_types)))
    ax.set_yticklabels(layer_types)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_title(f"{COMP_TITLE.get(comp, comp)} — {title}")
    ax.set_xlabel("Layer block index")
    ax.set_ylabel("Layer type")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_bar(stats: dict, comp: str, metric: str, title: str, out_path: Path):
    """layer_type별 mean 및 per-block 시각화 (bar chart, depth on x-axis)."""
    layer_types, col_indices, col_labels = get_layout(stats, comp)
    if not layer_types:
        return

    # collect {layer_type: list of (idx, val)}
    data = defaultdict(list)
    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        if lt not in layer_types:
            continue
        v = info.get(metric)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        idx = _layer_idx(name)
        pos = idx if idx is not None else -1
        data[lt].append((pos, float(v)))

    active = [lt for lt in layer_types if data[lt]]
    if not active:
        return

    n_types = len(active)
    fig, axes = plt.subplots(n_types, 1, figsize=(12, 2.5 * n_types), sharex=False)
    if n_types == 1:
        axes = [axes]

    color = COMP_COLOR.get(comp, "steelblue")

    for ax, lt in zip(axes, active):
        pts = sorted(data[lt])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.bar(range(len(xs)), ys, color=color, alpha=0.8)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=90, fontsize=7)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_title(f"{lt}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{COMP_TITLE.get(comp, comp)} — {title}", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_json", type=str, default=str(_STATS_JSON))
    parser.add_argument("--out_dir", type=str, default=str(_HERE / "plots"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.stats_json}")
    with open(args.stats_json) as f:
        d = json.load(f)

    raw_weight = d["weight_stats"]
    raw_act = d["activation_stats"]

    # Filter: quant targets only (no "other")
    w_stats = filter_quant(raw_weight)
    print(f"[INFO] Weight layers: {len(raw_weight)} total → {len(w_stats)} after filtering 'other'")

    # Build activation stats with component info from weight_stats
    # (activation keys match weight keys)
    a_stats = {}
    for k, v in raw_act.items():
        if k in raw_weight:
            winfo = raw_weight[k]
            if winfo.get("layer_type", "other") != "other":
                a_stats[k] = {**v, "component": winfo["component"], "layer_type": winfo["layer_type"]}
    print(f"[INFO] Activation layers: {len(raw_act)} total → {len(a_stats)} after filtering")

    for comp in ["lm", "dit"]:
        # Heatmaps
        plot_heatmap(w_stats, comp, "kurtosis",
                     "Weight Kurtosis",
                     out_dir / f"weight_{comp}_heatmap.png")
        plot_heatmap(w_stats, comp, "per_channel_absmax_cv",
                     "Weight Per-Channel AbsMax CV",
                     out_dir / f"weight_cv_{comp}_heatmap.png")
        plot_heatmap(a_stats, comp, "kurtosis",
                     "Activation Kurtosis",
                     out_dir / f"act_{comp}_heatmap.png")
        plot_heatmap(a_stats, comp, "per_token_absmax_cv",
                     "Activation Per-Token AbsMax CV",
                     out_dir / f"act_cv_{comp}_heatmap.png")

        # Bar plots
        plot_bar(w_stats, comp, "kurtosis",
                 "Weight Kurtosis",
                 out_dir / f"weight_kurtosis_{comp}.png")
        plot_bar(w_stats, comp, "per_channel_absmax_cv",
                 "Weight Per-Channel AbsMax CV",
                 out_dir / f"weight_abs_max_cv_{comp}.png")
        plot_bar(a_stats, comp, "kurtosis",
                 "Activation Kurtosis",
                 out_dir / f"act_kurtosis_{comp}.png")
        plot_bar(a_stats, comp, "per_token_absmax_cv",
                 "Activation Per-Token AbsMax CV",
                 out_dir / f"act_abs_max_cv_{comp}.png")

    # Also copy to lerobot/data_dist_analysis/
    lerobot_out = _LEROBOT / "data_dist_analysis"
    lerobot_out.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in out_dir.glob("*.png"):
        dst = lerobot_out / f.name
        shutil.copy2(f, dst)
    print(f"\n[DONE] All plots saved to {out_dir} and copied to {lerobot_out}")


if __name__ == "__main__":
    main()
