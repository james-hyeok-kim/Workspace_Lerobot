"""
Task 2: VLA DiT (pi0.5 Gemma Expert) vs PixArt-Alpha DiT
         Weight & Activation Distribution Comparison + Timestep-wise Analysis.

Weight stats:     VLA DiT from dist_stats.json; PixArt from Workspace_DiT stats.json
Activation stats: VLA DiT from dist_stats.json (aggregated) + optional timestep hook
                  PixArt from stats.json per_timestep data

Layer mapping (depth-normalized, 18 vs 28 blocks):
  VLA DiT: self_attn.{q,k,v,o}_proj  ↔  PixArt: attn1.to_{q,k,v,out.0}
  VLA DiT: mlp.{gate,up,down}_proj   ↔  PixArt: ff.net.{0.proj,2}   (gate≈0.proj, down≈2)

Usage:
    # Weight + aggregated activation comparison only (fast, no LIBERO env)
    python compare_dit_distributions.py

    # Include VLA DiT timestep-wise analysis (requires LIBERO env)
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python compare_dit_distributions.py --collect_vla_timestep --task libero_spatial --n_episodes 2
"""

import argparse
import ast
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
_VLA_STATS_JSON = _LEROBOT / "logs" / "dist_analysis_v4" / "dist_stats.json"
_PIXART_STATS_JSON = (Path("/home/jameskimh/workspace/Workspace_DiT") /
                      "pixart_alpha" / "results" / "distribution_analysis" / "stats.json")
_OUT_DIR = _HERE / "plots"

for _p in [str(_LEROBOT / "src"), str(_LEROBOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── VLA DiT layer helpers ──────────────────────────────────────────────────────

VLA_TYPE_MAP = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
}
PIXART_TYPE_MAP = {
    "attn1.to_q": "q", "attn1.to_k": "k", "attn1.to_v": "v", "attn1.to_out.0": "o",
    "ff.net.0.proj": "gate", "ff.net.2": "down",
    # attn2 = cross-attn → skip (no VLA equivalent)
}

COMMON_TYPES = ["q", "k", "v", "o", "gate", "down"]  # common cross-arch types
TYPE_LABELS = {
    "q": "q_proj (self-attn Q)",
    "k": "k_proj (self-attn K)",
    "v": "v_proj (self-attn V)",
    "o": "o_proj (self-attn O)",
    "gate": "gate/ff.proj",
    "down": "down/ff.net.2",
}

COLORS = {"vla": "#D65F5F", "pixart": "#5B8DB8"}


def _layer_idx(name: str):
    m = re.search(r'\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def _vla_type(name: str) -> str | None:
    for suffix, canonical in VLA_TYPE_MAP.items():
        if name.endswith(suffix):
            return canonical
    return None


def _pixart_type(name: str) -> str | None:
    for pattern, canonical in PIXART_TYPE_MAP.items():
        if pattern in name:
            return canonical
    return None


# ── data loading ──────────────────────────────────────────────────────────────

def load_vla_dit_stats(stats_json: Path) -> tuple[dict, dict]:
    """Returns (weight_stats, act_stats) each as {(block_idx, type): {metric: val}}"""
    with open(stats_json) as f:
        d = json.load(f)

    w_stats = {}
    for name, info in d["weight_stats"].items():
        if info.get("component") != "dit":
            continue
        if info.get("layer_type") == "other":
            continue
        lt = _vla_type(name)
        if lt is None:
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue
        w_stats[(idx, lt)] = {
            "kurtosis": info["kurtosis"],
            "cv": info["per_channel_absmax_cv"],
            "abs_max": info["abs_max"],
            "layer_type": info["layer_type"],
        }

    a_stats = {}
    for name, info in d["activation_stats"].items():
        if name not in d["weight_stats"]:
            continue
        winfo = d["weight_stats"][name]
        if winfo.get("component") != "dit" or winfo.get("layer_type") == "other":
            continue
        lt = _vla_type(name)
        if lt is None:
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue
        a_stats[(idx, lt)] = {
            "kurtosis": info["kurtosis"],
            "cv": info["per_token_absmax_cv"],
            "abs_max": info["abs_max"],
        }

    return w_stats, a_stats


def load_pixart_stats(stats_json: Path) -> tuple[dict, dict, dict]:
    """Returns (weight_stats, act_agg_stats, act_per_timestep)
    Each indexed as {(block_idx, type): ...}
    act_per_timestep: {(block_idx, type): [list of {kurtosis, abs_max, cv} per timestep]}
    """
    with open(stats_json) as f:
        d = json.load(f)

    w_stats = {}
    for name, info in d["weight"].items():
        lt = _pixart_type(name)
        if lt is None:
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue
        w_stats[(idx, lt)] = {
            "kurtosis": float(info["kurtosis"]),
            "cv": float(info["cv"]) / 100.0 if float(info.get("cv", 0)) > 10 else float(info.get("cv", 0)),
            # PixArt cv is per-channel std/mean in percentage or ratio — normalize to same scale
            "abs_max": float(info["abs_max"]),
        }
        # Note: PixArt cv in stats.json seems to be per_channel std/mean (not %)
        # Re-compute as raw value
        w_stats[(idx, lt)]["cv"] = float(info["cv"])

    act_agg = {}
    act_per_ts = {}
    for name, info in d["dynamics"].items():
        lt = _pixart_type(name)
        if lt is None:
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue

        act_data = info.get("act", {})
        if isinstance(act_data, str):
            try:
                act_data = eval(act_data)  # noqa: S307 — internal data parsing
            except Exception:
                continue

        agg = act_data.get("aggregated", {})
        if agg:
            act_agg[(idx, lt)] = {
                "kurtosis": float(agg.get("kurtosis", float("nan"))),
                "cv": float(agg.get("cv", float("nan"))),
                "abs_max": float(agg.get("abs_max", float("nan"))),
            }

        per_ts = act_data.get("per_timestep", [])
        if per_ts:
            act_per_ts[(idx, lt)] = [
                {
                    "kurtosis": float(ts.get("kurtosis", float("nan"))),
                    "abs_max": float(ts.get("abs_max", float("nan"))),
                    "cv": float(ts.get("cv", float("nan"))),
                }
                for ts in per_ts
            ]

    return w_stats, act_agg, act_per_ts


# ── VLA DiT timestep-wise hook (optional) ─────────────────────────────────────

def collect_vla_dit_timestep_stats(task: str, n_episodes: int) -> dict:
    """Run pi0.5 denoising and capture per-timestep DiT activation stats.
    Returns {(block_idx, type): [list of {kurtosis, abs_max, cv} per denoising step]}
    """
    import torch
    import torch.nn as nn
    import scipy.stats as ss

    print(f"[INFO] Collecting VLA DiT timestep stats: task={task}, n_episodes={n_episodes}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    pretrained_path = "lerobot/pi05_libero_finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load policy (same pattern as analyze_distributions.py) ───────────────
    env_cfg = LiberoEnv(task=task)
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = device
    policy_cfg.use_amp = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    print("[INFO] Policy loaded")

    # ── Build env + processors ────────────────────────────────────────────────
    envs_dict = make_env(env_cfg, n_envs=1)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # ── Disable torch.compile (hooks don't fire through compiled code) ────────
    inner = policy.model
    for attr in ["sample_actions", "forward", "_forward", "denoise"]:
        fn = getattr(inner, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
        if orig is not None:
            setattr(inner, attr, orig)
            print(f"[INFO] torch.compile disabled for model.{attr}")

    # ── Step counter via select_action wrapper ────────────────────────────────
    # Each call to select_action triggers one denoising pass (10 steps).
    # We track which denoising step we're in by counting calls to a sentinel
    # linear layer in block 0 q_proj — the first linear called per denoising step.
    current_step = [-1]
    raw = defaultdict(lambda: defaultdict(list))  # {(idx, lt): {step_idx: [stats]}}

    # Find the first q_proj in gemma_expert (block 0) as step sentinel
    sentinel_mod = None
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "gemma_expert" not in name:
            continue
        if _vla_type(name) == "q" and _layer_idx(name) == 0:
            sentinel_mod = mod
            print(f"[INFO] Step sentinel: {name}")
            break

    step_handles = []
    if sentinel_mod is not None:
        def sentinel_hook(module, inp):
            current_step[0] += 1
        step_handles.append(sentinel_mod.register_forward_pre_hook(sentinel_hook))
    else:
        print("[WARN] Sentinel not found — step counter disabled, all data mapped to step 0")
        current_step[0] = 0

    # Linear layer hooks for per-layer stats
    layer_handles = []
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "gemma_expert" not in name:
            continue
        lt = _vla_type(name)
        if lt is None:
            continue
        idx = _layer_idx(name)
        if idx is None:
            continue

        def make_hook(idx_, lt_):
            def hook(module, inp, out):
                step = current_step[0]
                if step < 0:
                    return
                x = inp[0].detach().float()
                flat = x.reshape(-1).cpu().numpy()
                if flat.size > 20_000:
                    s = flat.size // 20_000
                    flat = flat[::s][:20_000]
                tok_absmax = x.abs().amax(dim=-1).flatten().cpu().numpy()
                raw[(idx_, lt_)][step].append({
                    "kurtosis": float(ss.kurtosis(flat)),
                    "abs_max":  float(x.abs().max()),
                    "cv":       float(tok_absmax.std() / (tok_absmax.mean() + 1e-8)),
                })
            return hook

        h = mod.register_forward_hook(make_hook(idx, lt))
        layer_handles.append(h)

    print(f"[INFO] Registered {len(layer_handles)} DiT layer hooks")

    # ── Run episodes ──────────────────────────────────────────────────────────
    suite = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite]))
    env = envs_dict[suite][task_id]

    with torch.no_grad():
        eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
        )

    for h in step_handles + layer_handles:
        h.remove()

    total_steps = current_step[0] + 1
    print(f"[INFO] Captured {total_steps} total denoising steps across {n_episodes} episodes")

    # ── Aggregate per denoising step index (mod n_denoise_steps) ─────────────
    # Each episode has the same number of denoising steps per action chunk call.
    # Infer n_steps from the data distribution.
    if not raw:
        return {}

    all_step_indices = sorted(set(s for step_data in raw.values() for s in step_data))
    if not all_step_indices:
        return {}

    # Normalize: group by (step_idx % n_denoise_steps) to average across episodes
    # Use the max number of distinct steps seen per layer as n_denoise_steps estimate
    sample_key = next(iter(raw))
    steps_per_layer = sorted(raw[sample_key].keys())
    # Find repeating period: the number of unique steps before the pattern repeats
    # Conservative: use all unique step indices as-is (already per-episode averaged)
    n_ts = max(all_step_indices) + 1

    result = {}
    for key, step_data in raw.items():
        ts_list = []
        for s in range(n_ts):
            entries = step_data.get(s, [])
            if entries:
                ts_list.append({
                    "kurtosis": float(np.mean([e["kurtosis"] for e in entries])),
                    "abs_max":  float(np.mean([e["abs_max"]  for e in entries])),
                    "cv":       float(np.mean([e["cv"]       for e in entries])),
                })
        if ts_list:
            result[key] = ts_list

    return result


# ── depth normalization ───────────────────────────────────────────────────────

def depth_normalize(stats: dict, n_blocks: int) -> dict:
    """Normalize block index to [0,1] for cross-architecture comparison."""
    norm = {}
    for (idx, lt), v in stats.items():
        norm_idx = idx / (n_blocks - 1) if n_blocks > 1 else 0.0
        norm[(norm_idx, lt)] = v
    return norm


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_weight_comparison(vla_w: dict, px_w: dict, metric: str,
                            ylabel: str, title: str, out_path: Path,
                            vla_n_blocks: int = 18, px_n_blocks: int = 28):
    """Depth-normalized line plot for weight metric comparison."""
    vla_norm = depth_normalize(vla_w, vla_n_blocks)
    px_norm = depth_normalize(px_w, px_n_blocks)

    n_types = len(COMMON_TYPES)
    fig, axes = plt.subplots(n_types, 1, figsize=(12, 2.5 * n_types), sharex=True)

    for ax, lt in zip(axes, COMMON_TYPES):
        vla_pts = sorted((k, v[metric]) for (k, t), v in vla_norm.items() if t == lt and metric in v)
        px_pts = sorted((k, v[metric]) for (k, t), v in px_norm.items() if t == lt and metric in v)

        if vla_pts:
            xs, ys = zip(*vla_pts)
            ax.plot(xs, ys, "o-", color=COLORS["vla"], label=f"VLA DiT (pi0.5, {vla_n_blocks}b)",
                    linewidth=1.5, markersize=4)
        if px_pts:
            xs, ys = zip(*px_pts)
            ax.plot(xs, ys, "s--", color=COLORS["pixart"], label=f"PixArt DiT (28b)",
                    linewidth=1.5, markersize=4)

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(TYPE_LABELS.get(lt, lt), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Normalized depth (0=input, 1=output)")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_act_comparison(vla_a: dict, px_a: dict, metric: str,
                         ylabel: str, title: str, out_path: Path,
                         vla_n_blocks: int = 18, px_n_blocks: int = 28):
    """Same as plot_weight_comparison but for activation stats."""
    plot_weight_comparison(vla_a, px_a, metric, ylabel, title, out_path,
                           vla_n_blocks, px_n_blocks)


def plot_timestep_heatmap(per_ts: dict, n_blocks: int, metric: str,
                           title: str, out_path: Path, model_name: str):
    """Heatmap: x=block depth, y=denoising timestep, value=metric."""
    if not per_ts:
        print(f"[WARN] No timestep data for {model_name}")
        return

    # Get max timesteps and all types
    all_types = sorted(set(lt for (_, lt) in per_ts))
    n_ts = max(len(v) for v in per_ts.values())

    n_types = len(all_types)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 6))
    if n_types == 1:
        axes = [axes]

    for ax, lt in zip(axes, all_types):
        # Collect blocks for this type
        keys = sorted((idx, lt2) for idx, lt2 in per_ts if lt2 == lt)
        if not keys:
            continue

        mat = np.full((n_ts, len(keys)), np.nan)
        for ci, key in enumerate(keys):
            ts_data = per_ts[key]
            for ti, ts in enumerate(ts_data[:n_ts]):
                v = ts.get(metric)
                if v is not None and not np.isnan(v):
                    mat[ti, ci] = float(v)

        if np.all(np.isnan(mat)):
            continue

        vmax = np.nanpercentile(mat, 95)
        vmin = np.nanmin(mat)
        im = ax.imshow(mat, aspect="auto", cmap="hot_r", vmin=vmin, vmax=vmax,
                       origin="upper")
        plt.colorbar(im, ax=ax, label=metric)
        ax.set_title(TYPE_LABELS.get(lt, lt), fontsize=9)
        ax.set_xlabel("Block depth")
        ax.set_ylabel("Denoising step")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([str(k[0]) for k in keys], rotation=90, fontsize=7)

    fig.suptitle(f"{title} — {model_name}", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_outlier_curve(vla_per_ts: dict | None, px_per_ts: dict,
                        metric: str, title: str, out_path: Path,
                        n_steps: int = None):
    """Mean outlier metric across all layers per denoising step.
    n_steps: None=전체, 정수=첫 n_steps만 (1 episode/sample).
    """
    def get_curve(per_ts):
        if not per_ts:
            return None
        n_ts_raw = max(len(v) for v in per_ts.values())
        n_ts = min(n_steps, n_ts_raw) if n_steps is not None else n_ts_raw
        curve = []
        for ti in range(n_ts):
            vals = []
            for ts_data in per_ts.values():
                if ti < len(ts_data):
                    v = ts_data[ti].get(metric)
                    if v is not None and not np.isnan(v):
                        vals.append(v)
            curve.append(np.mean(vals) if vals else np.nan)
        return curve

    px_curve = get_curve(px_per_ts)
    vla_curve = get_curve(vla_per_ts) if vla_per_ts else None

    step_label = f"Denoising step  (1 episode = {n_steps} steps)" if n_steps else "Denoising step"
    px_label = f"PixArt DiT  (1 prompt, {len(px_curve)} steps)" if px_curve else "PixArt DiT"
    vla_label = f"VLA DiT (pi0.5)  (1 episode, {len(vla_curve)} steps)" if vla_curve else "VLA DiT (pi0.5)"

    fig, ax = plt.subplots(figsize=(8, 4))
    if px_curve:
        ax.plot(range(len(px_curve)), px_curve, "s--", color=COLORS["pixart"],
                label=px_label, linewidth=1.5)
    if vla_curve:
        ax.plot(range(len(vla_curve)), vla_curve, "o-", color=COLORS["vla"],
                label=vla_label, linewidth=1.5)

    ax.set_xlabel(step_label)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_outlier_curve_by_type(
    vla_per_ts: dict | None, px_per_ts: dict,
    metric: str, out_path: Path,
    n_steps: int = None,
    layer_types: list = None,
):
    """layer_type별 분리된 abs_max curve.
    subplot 1개 per layer_type. 각 subplot: x=timestep, y=mean across blocks.
    PixArt(파랑 dashed) + VLA(빨강 solid) 두 선.
    """
    if layer_types is None:
        layer_types = ["q", "k", "v", "o", "gate", "up", "down"]

    LT_FULL = {
        "q": "q_proj (self-attn Q)",
        "k": "k_proj (self-attn K)",
        "v": "v_proj (self-attn V)",
        "o": "o_proj (self-attn O)",
        "gate": "gate_proj (MLP gate)",
        "up":   "up_proj (MLP up)",
        "down": "down_proj (MLP down)",
    }

    def get_curve_by_lt(per_ts, lt):
        if not per_ts:
            return None
        keys = [(blk, l) for (blk, l) in per_ts if l == lt]
        if not keys:
            return None
        n_ts_raw = max(len(per_ts[k]) for k in keys)
        n_ts = min(n_steps, n_ts_raw) if n_steps is not None else n_ts_raw
        curve = []
        for ti in range(n_ts):
            vals = [float(per_ts[k][ti].get(metric, float("nan")))
                    for k in keys if ti < len(per_ts[k])]
            vals = [v for v in vals if not np.isnan(v)]
            curve.append(np.mean(vals) if vals else np.nan)
        return curve

    n_lt = len(layer_types)
    ncols = 4
    nrows = (n_lt + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2),
                             sharex=False)
    axes = np.array(axes).flatten()

    for i, lt in enumerate(layer_types):
        ax = axes[i]
        px_curve = get_curve_by_lt(px_per_ts, lt)
        vla_curve = get_curve_by_lt(vla_per_ts, lt) if vla_per_ts else None

        if px_curve:
            ax.plot(range(len(px_curve)), px_curve, "s--",
                    color=COLORS["pixart"], linewidth=1.5, label="PixArt DiT")
        if vla_curve:
            ax.plot(range(len(vla_curve)), vla_curve, "o-",
                    color=COLORS["vla"], linewidth=1.5, label="VLA DiT (pi0.5)")

        ax.set_title(LT_FULL.get(lt, lt), fontsize=9)
        ax.set_xlabel("Denoising step", fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # 남은 subplot 숨기기
    for j in range(n_lt, len(axes)):
        axes[j].set_visible(False)

    step_note = f"  (1 episode/prompt = {n_steps} steps)" if n_steps else ""
    fig.suptitle(f"Mean {metric} per Denoising Step — by Layer Type{step_note}\n"
                 f"mean across all blocks per layer_type",
                 fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")




def extract_absmax_per_layer(per_ts: dict) -> dict:
    """
    입력: {(block_idx, layer_type): [ {abs_max, ...}, × n_global_ts ]}
    출력: {(block_idx, layer_type): np.array of abs_max over timesteps}
    """
    out = {}
    for (blk, lt), ts_list in per_ts.items():
        arr = np.array([
            float(e.get("abs_max", np.nan)) for e in ts_list
        ], dtype=np.float32)
        out[(blk, lt)] = arr
    return out


def plot_absmax_3d_per_layer(
    per_layer: dict, n_blocks: int, out_path: Path, model_name: str,
    layer_types: list = None,
    n_steps: int = None,   # None=전체, 정수=첫 n_steps만 사용 (1 episode)
):
    """
    3D axes: x=block_idx*7+layer_type_idx, y=global_timestep, z=abs_max.
    - 각 (block, layer_type) 조합마다 timestep-line 1개 → outlier의 timestep-wise 흐름이 보임
    - 각 line의 max 점은 별 마커로 강조
    - 색상 = layer_type (7 색)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.lines import Line2D

    if not per_layer:
        print(f"[WARN] No data for {model_name}, skipping")
        return

    if layer_types is None:
        layer_types = ["q", "k", "v", "o", "gate", "up", "down"]
    n_lt = len(layer_types)

    LT_FULL = {
        "q": "q_proj (self-attn Q)",
        "k": "k_proj (self-attn K)",
        "v": "v_proj (self-attn V)",
        "o": "o_proj (self-attn O)",
        "gate": "gate_proj (MLP gate)",
        "up": "up_proj (MLP up)",
        "down": "down_proj (MLP down)",
    }

    cmap = plt.get_cmap("tab10")
    lt_color = {lt: cmap(i) for i, lt in enumerate(layer_types)}

    # determine n_ts (전체 or 첫 episode만)
    n_ts_raw = max(len(v) for v in per_layer.values())
    n_ts = min(n_steps, n_ts_raw) if n_steps is not None else n_ts_raw

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection="3d")

    ts_axis = np.arange(n_ts)

    # 옅은 선: raw timestep trace per (block, layer_type) — 배경 역할
    for (blk, lt), arr in sorted(per_layer.items()):
        if lt not in layer_types:
            continue
        x_pos = blk * n_lt + layer_types.index(lt)
        arr_cut = arr[:n_ts]
        y = ts_axis[:len(arr_cut)]
        color = lt_color[lt]
        ax.plot3D([x_pos] * len(y), y, arr_cut,
                  color=color, alpha=0.35, linewidth=0.6)

    # 굵은 선: 각 layer_type별 timestep마다 block 간 max를 이은 peak trajectory
    for lt in layer_types:
        blocks_for_lt = sorted(b for (b, l) in per_layer if l == lt)
        if not blocks_for_lt:
            continue
        stack = np.full((len(blocks_for_lt), n_ts), np.nan, dtype=np.float32)
        for i, b in enumerate(blocks_for_lt):
            arr = per_layer[(b, lt)][:n_ts]
            stack[i, :len(arr)] = arr
        peak_per_ts = np.nanmax(stack, axis=0)
        argmax_blk = np.nanargmax(stack, axis=0)
        x_traj = np.array([blocks_for_lt[i] * n_lt + layer_types.index(lt)
                           for i in argmax_blk])
        valid = np.isfinite(peak_per_ts)
        ax.plot3D(x_traj[valid], ts_axis[valid], peak_per_ts[valid],
                  color=lt_color[lt], alpha=0.95, linewidth=2.2, zorder=8)

    # x tick: block 경계마다 라벨
    x_ticks = [b * n_lt + (n_lt // 2) for b in range(n_blocks)]
    x_labels = [f"B{b}" for b in range(n_blocks)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=6)

    # y tick: 10 step 간격
    y_ticks = list(range(0, n_ts, 10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=7)

    ax.set_xlabel(f"Block × LayerType  (x = block_idx × {n_lt} + layer_type_idx, "
                  f"{n_blocks}×{n_lt}={n_blocks*n_lt} lines)", fontsize=9, labelpad=10)
    ax.set_ylabel("Denoising step (global)", fontsize=9, labelpad=8)
    ax.set_zlabel("abs_max", fontsize=9)
    ax.view_init(elev=22, azim=-55)
    ax.set_title(f"Activation abs_max per (block_idx, layer_type) over timesteps — {model_name}",
                 fontsize=11, pad=12)

    # layer_type color legend
    legend_lines = [
        Line2D([0], [0], color=lt_color[lt], lw=2.5, label=LT_FULL.get(lt, lt))
        for lt in layer_types
    ]
    legend_lines.append(
        Line2D([0], [0], color="black", lw=2.2,
               label="굵은 선 = per-timestep peak across blocks (layer_type별)")
    )
    legend_lines.append(
        Line2D([0], [0], color="gray", lw=0.6, alpha=0.5,
               label=f"옅은 선 = 1 (block, layer_type) raw trace ({n_ts} steps)")
    )
    ax.legend(handles=legend_lines, loc="upper left",
              bbox_to_anchor=(0.0, 1.0), fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_violin_comparison(vla_w: dict, px_w: dict, metric: str,
                            title: str, out_path: Path):
    vla_by_type = defaultdict(list)
    px_by_type = defaultdict(list)

    for (_, lt), v in vla_w.items():
        if metric in v and not np.isnan(float(v[metric])):
            vla_by_type[lt].append(float(v[metric]))
    for (_, lt), v in px_w.items():
        if metric in v and not np.isnan(float(v[metric])):
            px_by_type[lt].append(float(v[metric]))

    common = [lt for lt in COMMON_TYPES if vla_by_type[lt] and px_by_type[lt]]
    if not common:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(common)
    for i, lt in enumerate(common):
        vd = vla_by_type[lt]
        pd_ = px_by_type[lt]
        vparts = ax.violinplot([vd], positions=[i * 2], widths=0.8, showmedians=True)
        pparts = ax.violinplot([pd_], positions=[i * 2 + 0.8], widths=0.8, showmedians=True)
        for pc in vparts["bodies"]:
            pc.set_facecolor(COLORS["vla"]); pc.set_alpha(0.6)
        for pc in pparts["bodies"]:
            pc.set_facecolor(COLORS["pixart"]); pc.set_alpha(0.6)

    ax.set_xticks([i * 2 + 0.4 for i in range(n)])
    ax.set_xticklabels([TYPE_LABELS.get(lt, lt) for lt in common], rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS["vla"], label="VLA DiT (pi0.5)"),
                        Patch(color=COLORS["pixart"], label="PixArt DiT")])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def save_summary_csv(vla_w, px_w, vla_a, px_a, out_path: Path):
    import csv

    rows = []
    for lt in COMMON_TYPES:
        def agg(d, metric):
            vals = [float(v[metric]) for (_, t), v in d.items()
                    if t == lt and metric in v and not np.isnan(float(v[metric]))]
            return (np.mean(vals) if vals else float("nan"),
                    np.std(vals) if vals else float("nan"))

        vw_k, vw_k_s = agg(vla_w, "kurtosis")
        pw_k, pw_k_s = agg(px_w, "kurtosis")
        vw_cv, _ = agg(vla_w, "cv")
        pw_cv, _ = agg(px_w, "cv")
        va_k, _ = agg(vla_a, "kurtosis")
        pa_k, _ = agg(px_a, "kurtosis")

        rows.append({
            "layer_type": lt,
            "vla_w_kurtosis_mean": f"{vw_k:.2f}", "vla_w_kurtosis_std": f"{vw_k_s:.2f}",
            "pixart_w_kurtosis_mean": f"{pw_k:.2f}", "pixart_w_kurtosis_std": f"{pw_k_s:.2f}",
            "vla_w_cv_mean": f"{vw_cv:.4f}",
            "pixart_w_cv_mean": f"{pw_cv:.4f}",
            "vla_a_kurtosis_mean": f"{va_k:.2f}",
            "pixart_a_kurtosis_mean": f"{pa_k:.2f}",
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
    parser.add_argument("--out_dir", type=str, default=str(_OUT_DIR))
    parser.add_argument("--collect_vla_timestep", action="store_true",
                        help="Collect VLA DiT per-timestep activation stats via LIBERO env")
    parser.add_argument("--task", type=str, default="libero_spatial",
                        help="LIBERO task for VLA timestep collection")
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--vla_ts_cache", type=str, default="",
                        help="Load pre-saved VLA timestep stats JSON")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load VLA DiT stats ────────────────────────────────────────
    print("[INFO] Loading VLA DiT stats ...")
    vla_w, vla_a = load_vla_dit_stats(_VLA_STATS_JSON)
    vla_n_blocks = max(idx for idx, _ in vla_w) + 1 if vla_w else 18
    print(f"[INFO] VLA DiT: {len(vla_w)} weight layers, {len(vla_a)} activation layers, {vla_n_blocks} blocks")

    # ── Step 2: Load PixArt stats ─────────────────────────────────────────
    print("[INFO] Loading PixArt-Alpha DiT stats ...")
    px_w, px_a, px_per_ts = load_pixart_stats(_PIXART_STATS_JSON)
    px_n_blocks = max(idx for idx, _ in px_w) + 1 if px_w else 28
    print(f"[INFO] PixArt DiT: {len(px_w)} weight layers, {len(px_a)} activation layers, {px_n_blocks} blocks")
    print(f"[INFO] PixArt per-timestep: {len(px_per_ts)} layers, "
          f"{max(len(v) for v in px_per_ts.values()) if px_per_ts else 0} timesteps")

    # ── Step 3: (Optional) VLA DiT timestep collection ───────────────────
    vla_per_ts = None
    ts_cache = Path(args.vla_ts_cache) if args.vla_ts_cache else out_dir / "vla_dit_timestep_stats.json"

    if ts_cache.exists():
        print(f"[INFO] Loading VLA DiT timestep cache: {ts_cache}")
        with open(ts_cache) as f:
            raw = json.load(f)
        vla_per_ts = {(int(k.split(",")[0]), k.split(",")[1]): v for k, v in raw.items()}
    elif args.collect_vla_timestep:
        vla_per_ts = collect_vla_dit_timestep_stats(args.task, args.n_episodes)
        # Save cache
        cache_data = {f"{k[0]},{k[1]}": v for k, v in vla_per_ts.items()}
        with open(ts_cache, "w") as f:
            json.dump(cache_data, f)
        print(f"[SAVED] VLA timestep cache → {ts_cache}")

    # ── Step 4: Generate plots ────────────────────────────────────────────
    print("[INFO] Generating DiT comparison plots ...")

    # Weight kurtosis
    plot_weight_comparison(
        vla_w, px_w, "kurtosis", "Kurtosis",
        "Weight Kurtosis: VLA DiT vs PixArt DiT (Depth-Normalized)",
        out_dir / "dit_weight_kurtosis_comparison.png",
        vla_n_blocks, px_n_blocks,
    )

    # Weight CV
    plot_weight_comparison(
        vla_w, px_w, "cv", "Per-Channel CV",
        "Weight Per-Channel CV: VLA DiT vs PixArt DiT (Depth-Normalized)",
        out_dir / "dit_weight_cv_comparison.png",
        vla_n_blocks, px_n_blocks,
    )

    # Activation kurtosis
    plot_act_comparison(
        vla_a, px_a, "kurtosis", "Kurtosis",
        "Activation Kurtosis: VLA DiT vs PixArt DiT (Depth-Normalized)",
        out_dir / "dit_act_kurtosis_comparison.png",
        vla_n_blocks, px_n_blocks,
    )

    # Activation CV
    plot_act_comparison(
        vla_a, px_a, "cv", "Per-Token AbsMax CV",
        "Activation Per-Token CV: VLA DiT vs PixArt DiT (Depth-Normalized)",
        out_dir / "dit_act_cv_comparison.png",
        vla_n_blocks, px_n_blocks,
    )

    # Violin
    plot_violin_comparison(vla_w, px_w, "kurtosis",
                           "Weight Kurtosis Distribution by Type: VLA DiT vs PixArt DiT",
                           out_dir / "dit_weight_kurtosis_violin.png")

    # PixArt timestep heatmap (always available)
    plot_timestep_heatmap(
        px_per_ts, px_n_blocks, "kurtosis",
        "Activation Kurtosis per Denoising Step",
        out_dir / "dit_timestep_heatmap_pixart.png",
        model_name="PixArt-Alpha DiT",
    )

    # VLA DiT timestep heatmap (if available)
    if vla_per_ts:
        plot_timestep_heatmap(
            vla_per_ts, vla_n_blocks, "kurtosis",
            "Activation Kurtosis per Denoising Step",
            out_dir / "dit_timestep_heatmap_vla.png",
            model_name="VLA DiT (pi0.5)",
        )

    # Outlier curve (1 episode/sample = 20 steps)
    plot_outlier_curve(
        vla_per_ts, px_per_ts, "abs_max",
        "Mean Activation AbsMax per Denoising Step: VLA DiT vs PixArt DiT",
        out_dir / "dit_timestep_outlier_curve.png",
        n_steps=20,
    )

    # Layer-type별 분리된 curve (1 episode = 20 steps)
    plot_outlier_curve_by_type(
        vla_per_ts, px_per_ts, "abs_max",
        out_dir / "dit_outlier_curve_by_type.png",
        n_steps=20,
    )

    # 3D per-layer: abs_max × (block × layer_type) × timestep  (1 episode = 20 steps)
    px_per_layer = extract_absmax_per_layer(px_per_ts)
    plot_absmax_3d_per_layer(
        px_per_layer, px_n_blocks,
        out_dir / "dit_3d_per_layer_absmax_pixart.png",
        model_name="PixArt-Alpha DiT  (1 prompt, 20 denoising steps)",
        n_steps=20,
    )
    if vla_per_ts:
        vla_per_layer = extract_absmax_per_layer(vla_per_ts)
        plot_absmax_3d_per_layer(
            vla_per_layer, vla_n_blocks,
            out_dir / "dit_3d_per_layer_absmax_vla.png",
            model_name="VLA DiT (pi0.5)  (1 episode, 20 denoising steps)",
            n_steps=20,
        )

    # Summary CSV
    save_summary_csv(vla_w, px_w, vla_a, px_a,
                     out_dir / "dit_summary_stats.csv")

    print(f"\n[DONE] All DiT comparison plots saved to {out_dir}")


if __name__ == "__main__":
    main()
