"""
Weight & Activation Value Distribution — Per-Layer Depth Box Plot

스타일: box=p25-p75 / whisker=p5-p95 / line=p50 / diamond=mean / dots=p1(yellow),p99(red)
X축 = Layer Index, Y축 = value

출력:
  value_weight_{model_key}.png   -- 모델별 weight per-layer boxplot (layer type grid)
  value_act_{model_key}.png      -- 모델별 activation per-layer boxplot
  value_weight_compare_{lt}.png  -- layer type별 4 모델 비교 (side-by-side depth)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_LEROBOT     = _HERE.parent / "lerobot"
_VLA_STATS   = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"
_GEMMA_STATS = _HERE / "plots" / "dist_stats_google_gemma_2b.json"
_PIXART_STATS = (Path("/home/jameskimh/workspace/Workspace_DiT") /
                 "pixart_alpha" / "results" / "distribution_analysis" / "stats.json")
_OUT_DIR = _HERE / "plots"

# ── model config ──────────────────────────────────────────────────────────────
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

ALL_LAYER_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
PIXART_TYPE_MAP = {
    "attn1.to_q":    "q_proj",
    "attn1.to_k":    "k_proj",
    "attn1.to_v":    "v_proj",
    "attn1.to_out.0":"o_proj",
    "ff.net.0.proj": "gate_proj",
    "ff.net.2":      "down_proj",
}


def _layer_type(name: str) -> str:
    for t in sorted(ALL_LAYER_TYPES, key=len, reverse=True):
        if name.endswith(t):
            return t
    return "other"


# ── per-layer stat record ─────────────────────────────────────────────────────
# Each layer = dict with keys: mean, p1, p5, p25, p50, p75, p95, p99
# (whiskers=p5/p95, box=p25/p75, median=p50, mean, outliers=p1/p99)

def _make_stat(mean, p1, p5, p25, p50, p75, p95, p99) -> dict:
    return dict(mean=mean, p1=p1, p5=p5, p25=p25,
                p50=p50, p75=p75, p95=p95, p99=p99)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_lm_weight(stats_json: Path, component: str | None) -> dict[str, list[dict]]:
    """Returns {layer_type: [stat_dict per layer in depth order]}"""
    with open(stats_json) as f:
        d = json.load(f)
    ws = d.get("weight_stats", d)

    # group by (layer_type, depth_idx) to preserve order
    grouped = defaultdict(dict)
    for name, info in ws.items():
        if component and "component" in info and info["component"] != component:
            continue
        lt = info.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        # determine depth index from layer name
        import re
        m = re.search(r'layers\.(\d+)\.', name)
        idx = int(m.group(1)) if m else 9999

        pct = info.get("percentiles", {})
        stat = _make_stat(
            mean=info.get("mean", 0),
            p1 =pct.get("p1",  pct.get("p5",  0)),
            p5 =pct.get("p5",  0),
            p25=pct.get("p25", 0),
            p50=pct.get("p50", 0),
            p75=pct.get("p75", 0),
            p95=pct.get("p95", 0),
            p99=pct.get("p99", pct.get("p95", 0)),
        )
        grouped[lt][idx] = stat

    # sort by depth
    return {lt: [grouped[lt][i] for i in sorted(grouped[lt])]
            for lt in grouped}


def load_lm_act(stats_json: Path, component: str | None) -> dict[str, list[dict]]:
    import re
    with open(stats_json) as f:
        d = json.load(f)
    ws   = d.get("weight_stats", {})
    acts = d.get("activation_stats", {})

    grouped = defaultdict(dict)
    for name, info in acts.items():
        wi = ws.get(name, {})
        if component and "component" in wi and wi["component"] != component:
            continue
        lt = wi.get("layer_type", _layer_type(name))
        if lt == "other":
            continue
        m = re.search(r'layers\.(\d+)\.', name)
        idx = int(m.group(1)) if m else 9999

        stat = _make_stat(
            mean=info.get("mean", 0),
            p1 =info.get("p5",  0),   # p1 not stored, use p5 as low outlier
            p5 =info.get("p5",  0),
            p25=info.get("p25", 0),
            p50=info.get("p50", 0),
            p75=info.get("p75", 0),
            p95=info.get("p95", 0),
            p99=info.get("p95", 0),   # p99 not stored, use p95 as high outlier
        )
        grouped[lt][idx] = stat

    return {lt: [grouped[lt][i] for i in sorted(grouped[lt])]
            for lt in grouped}


def load_pixart_weight() -> dict[str, list[dict]]:
    import re
    with open(_PIXART_STATS) as f:
        d = json.load(f)
    grouped = defaultdict(dict)
    for name, info in d["weight"].items():
        lt = next((v for k, v in PIXART_TYPE_MAP.items() if k in name), None)
        if lt is None:
            continue
        m = re.search(r'\.(\d+)\.', name)
        idx = int(m.group(1)) if m else 9999
        stat = _make_stat(
            mean=float(info.get("mean", 0)),
            p1 =float(info.get("q1",  info.get("q5", 0))),
            p5 =float(info.get("q5",  0)),
            p25=float(info.get("q25", 0)),
            p50=float(info.get("q50", 0)),
            p75=float(info.get("q75", 0)),
            p95=float(info.get("q95", 0)),
            p99=float(info.get("q99", info.get("q95", 0))),
        )
        grouped[lt][idx] = stat
    return {lt: [grouped[lt][i] for i in sorted(grouped[lt])]
            for lt in grouped}


def load_pixart_act() -> dict[str, list[dict]]:
    import re
    with open(_PIXART_STATS) as f:
        d = json.load(f)
    grouped = defaultdict(dict)
    for name, info in d["dynamics"].items():
        lt = next((v for k, v in PIXART_TYPE_MAP.items() if k in name), None)
        if lt is None:
            continue
        m = re.search(r'\.(\d+)\.', name)
        idx = int(m.group(1)) if m else 9999

        act_data = info.get("act", {})
        if isinstance(act_data, str):
            try:
                act_data = eval(act_data)  # noqa: S307
            except Exception:
                continue
        pct = act_data.get("percentiles", {})
        if not pct:
            continue
        stat = _make_stat(
            mean=float(act_data.get("aggregated", {}).get("mean", 0)),
            p1 =float(pct.get("q1",  pct.get("q5",  0))),
            p5 =float(pct.get("q5",  0)),
            p25=float(pct.get("q25", 0)),
            p50=float(pct.get("q50", 0)),
            p75=float(pct.get("q75", 0)),
            p95=float(pct.get("q95", 0)),
            p99=float(pct.get("q99", pct.get("q95", 0))),
        )
        grouped[lt][idx] = stat
    return {lt: [grouped[lt][i] for i in sorted(grouped[lt])]
            for lt in grouped}


# ── core drawing ──────────────────────────────────────────────────────────────

def draw_depth_boxplot(ax, stats: list[dict], color: str, label: str = ""):
    """
    Draw one subplot: per-layer depth boxplot.
    Box=p25-p75, whisker=p5-p95, median line=p50,
    mean=diamond marker, p1=yellow dot (low outlier), p99=red dot (high outlier).
    """
    n = len(stats)
    if n == 0:
        return

    xs = list(range(n))
    # Build bxp stat list
    bxp_stats = []
    for s in stats:
        bxp_stats.append({
            "med":    s["p50"],
            "q1":     s["p25"],
            "q3":     s["p75"],
            "whislo": s["p5"],
            "whishi": s["p95"],
            "mean":   s["mean"],
            "fliers": [],
        })

    bp = ax.bxp(
        bxp_stats,
        positions=xs,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        medianprops=dict(color="black",  linewidth=1.5),
        whiskerprops=dict(color="#555555", linewidth=1.0),
        capprops=dict(color="#555555",   linewidth=1.0),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4, zorder=5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # p1 = low outlier (yellow/gold dots)
    p1_vals  = [s["p1"]  for s in stats]
    p99_vals = [s["p99"] for s in stats]
    ax.scatter(xs, p1_vals,  s=10, color="#FFD700", zorder=4, alpha=0.8, label="p1")
    ax.scatter(xs, p99_vals, s=10, color="#CC3333", zorder=4, alpha=0.8, label="p99")

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer Index", fontsize=7)
    ax.set_ylabel("Weight value", fontsize=7)
    ax.set_xticks(xs[::max(1, n // 10)])
    ax.set_xticklabels([str(i) for i in xs[::max(1, n // 10)]], fontsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    if label:
        ax.set_title(label, fontsize=9)


# ── figure generators ─────────────────────────────────────────────────────────

def plot_model_grid(data_by_type: dict[str, list[dict]],
                    model_key: str, data_label: str,
                    out_path: Path):
    """
    One figure per model: grid of subplots by layer_type.
    Matches reference image style.
    """
    types_present = [lt for lt in ALL_LAYER_TYPES if data_by_type.get(lt)]
    # also include non-standard types (action head etc.)
    extra = [lt for lt in data_by_type if lt not in ALL_LAYER_TYPES]
    types_present = types_present + extra

    if not types_present:
        print(f"[SKIP] No data for {model_key} {data_label}")
        return

    n = len(types_present)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    color = MODEL_COLORS[model_key]

    for i, lt in enumerate(types_present):
        ax = axes_flat[i]
        stats = data_by_type.get(lt, [])
        draw_depth_boxplot(ax, stats, color=color, label=lt)

    # hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{MODEL_LABELS[model_key]} — {data_label} per Layer Box Plot\n"
        f"(box: p25-p75 / whisker: p5-p95 / diamond: mean / dots: p1=●yellow, p99=●red)",
        fontsize=10,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_compare_grid(all_data: dict[str, dict[str, list[dict]]],
                       data_label: str, out_path: Path):
    """
    Comparison figure: rows=layer_type, cols=4 models.
    Shows all 4 models side by side per layer type.
    """
    types_present = [lt for lt in ALL_LAYER_TYPES
                     if any(d.get(lt) for d in all_data.values())]
    if not types_present:
        return

    nrows = len(types_present)
    ncols = len(MODEL_KEYS)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.2 * nrows),
                              sharey="row")

    for row, lt in enumerate(types_present):
        for col, mk in enumerate(MODEL_KEYS):
            ax = axes[row, col] if nrows > 1 else axes[col]
            stats = all_data.get(mk, {}).get(lt, [])
            if not stats:
                # No data for this model/type combination — show N/A
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "N/A\n(arch diff)",
                        ha="center", va="center", fontsize=8,
                        color="#999999", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(MODEL_LABELS[mk], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{lt}\nValue", fontsize=8)
            else:
                draw_depth_boxplot(ax, stats, color=MODEL_COLORS[mk])
                if row == 0:
                    ax.set_title(MODEL_LABELS[mk], fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{lt}\nValue", fontsize=8)
                else:
                    ax.set_ylabel("")

    fig.suptitle(
        f"Cross-Model {data_label} Value Distribution by Layer Type\n"
        f"(box: p25-p75 / whisker: p5-p95 / diamond: mean / dots: p1●yellow, p99●red)",
        fontsize=10,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(_OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data ...")

    weight_all = {
        "llm":       load_lm_weight(_GEMMA_STATS, None),
        "vla_lm":    load_lm_weight(_VLA_STATS,   "lm"),
        "image_dit": load_pixart_weight(),
        "vla_dit":   load_lm_weight(_VLA_STATS,   "dit"),
    }
    act_all = {
        "llm":       load_lm_act(_GEMMA_STATS, None),
        "vla_lm":    load_lm_act(_VLA_STATS, "lm"),
        "image_dit": load_pixart_act(),
        "vla_dit":   load_lm_act(_VLA_STATS, "dit"),
    }

    for mk in MODEL_KEYS:
        nw = sum(len(v) for v in weight_all[mk].values())
        na = sum(len(v) for v in act_all[mk].values())
        print(f"  {mk:12s}: weight={nw} layers, act={na} layers")

    # ── Per-model weight grid ──────────────────────────────────────────
    for mk in MODEL_KEYS:
        plot_model_grid(weight_all[mk], mk, "Weight",
                        out_dir / f"value_weight_{mk}.png")

    # ── Per-model activation grid ──────────────────────────────────────
    for mk in MODEL_KEYS:
        if not any(act_all[mk].values()):
            print(f"[SKIP] {mk} activation — no data")
            continue
        plot_model_grid(act_all[mk], mk, "Activation",
                        out_dir / f"value_act_{mk}.png")

    # ── Cross-model comparison (weight) ───────────────────────────────
    plot_compare_grid(weight_all, "Weight",
                      out_dir / "value_weight_cross_model.png")

    # ── Cross-model comparison (activation) ───────────────────────────
    plot_compare_grid(act_all, "Activation",
                      out_dir / "value_act_cross_model.png")

    print(f"\n[DONE] All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
