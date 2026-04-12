"""
LM vs DiT Weight & Activation Distribution Analysis for LeRobot pi05.

Weight distribution:  분석 시 env 불필요 (--weight_only)
Activation distribution: LIBERO env에서 forward hook으로 캡처

Usage:
    # Weight only (~1분)
    python analyze_distributions.py --weight_only

    # Weight + Activation (~5분)
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python analyze_distributions.py --task libero_spatial --n_steps 50
"""

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import scipy.stats
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("[WARN] scipy not available; kurtosis will be skipped")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ─────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _p in [str(_root / "src"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# 레이어 분류 (eval_quant_sweep.py 패턴 재사용)
# ══════════════════════════════════════════════════════════════════════════════

LM_PATTERNS = [
    "paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
]
DIT_PATTERNS = [
    "paligemma_with_expert.gemma_expert",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
]
SKIP_PATTERNS = ["vision_tower", "embed_tokens"]

ACTION_HEAD_SUFFIXES = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
LAYER_TYPE_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
]


def classify_layer(name: str) -> str:
    for p in SKIP_PATTERNS:
        if p in name:
            return "skip"
    if any(p in name for p in LM_PATTERNS):
        return "lm"
    if any(p in name for p in DIT_PATTERNS):
        return "dit"
    return "other"


def layer_type(name: str) -> str:
    for t in LAYER_TYPE_SUFFIXES:
        if name.endswith(t):
            return t
    return "other"


def attn_or_mlp(ltype: str) -> str:
    if ltype in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return "attn"
    if ltype in ("gate_proj", "up_proj", "down_proj"):
        return "mlp"
    if ltype in ACTION_HEAD_SUFFIXES:
        return "action_head"
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# Weight 통계
# ══════════════════════════════════════════════════════════════════════════════

PCTILE_Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
PCTILE_LABELS = ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "p999"]


def kurtosis(arr: np.ndarray) -> float:
    if _SCIPY:
        return float(scipy.stats.kurtosis(arr))
    # fallback: Fisher kurtosis
    m = arr.mean()
    s = arr.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


MAX_SAMPLE = 50_000  # 통계 계산 시 최대 원소 수 (속도 vs 정확도 트레이드오프)


def _subsample_tensor(t: torch.Tensor, max_n: int = MAX_SAMPLE) -> np.ndarray:
    """GPU 텐서에서 stride 서브샘플링 후 소량만 CPU로 전송."""
    flat = t.flatten()
    n = flat.numel()
    if n <= max_n:
        return flat.float().cpu().numpy()
    step = n // max_n
    return flat[::step][:max_n].float().cpu().numpy()


def analyze_weight(W: torch.Tensor) -> dict:
    """GPU 텐서에서 최소한만 CPU로 전송해 빠르게 통계 계산."""
    W_d = W.detach()
    shape = list(W_d.shape)

    with torch.no_grad():
        abs_max = float(W_d.abs().max())
        # mean/std: torch scalar, GPU에서 계산
        W_f32 = W_d.float()
        mean_v = float(W_f32.mean())
        std_v  = float(W_f32.std())

        # per-output-channel abs-max (GPU → tiny CPU vector)
        per_ch = W_d.abs().amax(dim=1).float()
        mean_ch = float(per_ch.mean().clamp(min=1e-8))
        per_ch_cpu = per_ch.cpu().numpy()  # shape: [out_features], max 16384 elements — 빠름

    # GPU에서 stride 샘플링 후 소량(50K)만 CPU 전송
    flat_s = _subsample_tensor(W_d)
    pct = np.percentile(flat_s, [q * 100 for q in PCTILE_Q]).tolist()
    kurt = kurtosis(flat_s)

    return {
        "mean": mean_v,
        "std":  std_v,
        "abs_max": abs_max,
        "percentiles": dict(zip(PCTILE_LABELS, [float(v) for v in pct])),
        "kurtosis": kurt,
        "per_channel_absmax_mean": float(per_ch_cpu.mean()),
        "per_channel_absmax_std":  float(per_ch_cpu.std()),
        "per_channel_absmax_cv":   float(per_ch_cpu.std() / max(per_ch_cpu.mean(), 1e-8)),
        "shape": shape,
    }


def collect_weight_stats(policy: nn.Module) -> dict[str, dict]:
    """모든 nn.Linear weight를 순회하며 통계 수집."""
    stats = {}
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        comp = classify_layer(name)
        if comp == "skip":
            continue
        ltype = layer_type(name)
        stats[name] = {
            "component": comp,
            "layer_type": ltype,
            "attn_mlp": attn_or_mlp(ltype),
            **analyze_weight(mod.weight),
        }
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Activation 통계 (forward pre-hook)
# ══════════════════════════════════════════════════════════════════════════════

_ACT_PCTILE_Q      = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
_ACT_PCTILE_LABELS = ["p5", "p25", "p50", "p75", "p95"]


def make_pre_hook(name: str, store: dict) -> callable:
    def hook(module, inp):
        x = inp[0].detach()
        with torch.no_grad():
            xf = x.float()
            abs_max = float(xf.abs().max())
            mean_v  = float(xf.mean())
            std_v   = float(xf.std())
            token_absmax = xf.abs().amax(dim=-1).flatten()
            mean_t  = float(token_absmax.mean().clamp(min=1e-8))
            tok_cv  = float(token_absmax.std() / mean_t)
            tok_mean = float(token_absmax.mean())
            # activation 값 분포 percentile (weight boxplot과 동일 방식)
            flat_gpu = xf.flatten()
            if flat_gpu.numel() > 50_000:
                step = flat_gpu.numel() // 50_000
                flat_gpu = flat_gpu[::step][:50_000]
            pct_vals = torch.quantile(flat_gpu, _ACT_PCTILE_Q.to(flat_gpu.device)).tolist()
        flat_s = _subsample_tensor(x)
        entry = {
            "abs_max": abs_max,
            "mean":    mean_v,
            "std":     std_v,
            "kurtosis": kurtosis(flat_s),
            "per_token_absmax_cv": tok_cv,
            "per_token_absmax_mean": tok_mean,
        }
        for label, val in zip(_ACT_PCTILE_LABELS, pct_vals):
            entry[label] = val
        store[name].append(entry)
    return hook


def register_hooks(policy: nn.Module, store: dict) -> list:
    handles = []
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        comp = classify_layer(name)
        if comp == "skip":
            continue
        store[name] = []
        handles.append(mod.register_forward_pre_hook(make_pre_hook(name, store)))
    return handles


def aggregate_act_stats(raw: dict[str, list]) -> dict[str, dict]:
    """step별 activation 통계를 레이어별로 평균."""
    agg = {}
    for name, steps in raw.items():
        if not steps:
            continue
        keys = steps[0].keys()
        agg[name] = {k: float(np.mean([s[k] for s in steps])) for k in keys}
        # worst-case: 모든 step 중 최대 abs_max (quantization clipping threshold 기준)
        agg[name]["max_abs_max"] = float(np.max([s["abs_max"] for s in steps]))
        # box plot용 raw 분포도 보존
        agg[name]["_raw"] = {k: [s[k] for s in steps] for k in keys}
    return agg


def disable_torch_compile(policy: nn.Module) -> None:
    """eval_quant_sweep.py 동일 패턴: torch.compile 비활성화."""
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is None:
        return
    for attr in ("sample_actions", "forward"):
        fn = getattr(inner, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
        if orig is not None:
            setattr(inner, attr, orig)
            print(f"[INFO] torch.compile disabled for model.{attr}")


def collect_activation_stats(
    policy: nn.Module,
    env_cfg,
    envs_dict: dict,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    n_episodes: int,
) -> dict[str, dict]:
    """eval_policy를 통해 n_episodes 동안 activation 캡처."""
    # torch.compile은 hook 내 numpy 변환과 충돌 → 비활성화
    disable_torch_compile(policy)

    raw: dict[str, list] = {}
    handles = register_hooks(policy, raw)
    print(f"[INFO] Registered {len(handles)} activation hooks")

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

    for h in handles:
        h.remove()

    print(f"[INFO] Captured {sum(len(v) for v in raw.values())} activation samples across {len(raw)} layers")
    return aggregate_act_stats(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

import re

COMP_COLOR = {"lm": "#4878CF", "dit": "#D65F5F"}
COMP_TITLE = {"lm": "LM (Gemma 2B)", "dit": "DiT (Expert+Head)"}

# layer_type 우선 표시 순서 (없는 타입은 알파벳 순으로 뒤에 추가)
PREFERRED_ORDER = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "action_in_proj", "action_out_proj",
    "time_mlp_in", "time_mlp_out",
    "other",
]

COLS_PER_CHUNK = 32  # 이 수 초과 시 파일 분할 (가독성 유지)


def _chunk_indices(all_idx: list, chunk_size: int = COLS_PER_CHUNK) -> list:
    """인덱스 목록을 chunk_size 단위로 분할."""
    return [all_idx[i:i + chunk_size] for i in range(0, len(all_idx), chunk_size)]


def _layer_idx(name: str):
    m = re.search(r'\.layers\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def _get_comp_layout(stats: dict, comp: str) -> tuple[list[str], list[int], list[str]]:
    """
    컴포넌트의 실제 데이터에서 동적으로 레이어 구성을 감지.

    Returns:
        layer_types: 표시할 layer_type 목록 (행)
        col_indices: heatmap 열 인덱스 목록 (정수 layer idx + 비인덱스 레이어용 가상 인덱스)
        col_labels:  heatmap 열 레이블 문자열
    """
    # 존재하는 layer_type 수집
    present_types: set[str] = set()
    indexed_idxs: set[int] = set()
    has_nonindexed: dict[str, bool] = {}  # layer_type → has non-indexed layers

    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        present_types.add(lt)
        idx = _layer_idx(name)
        if idx is not None:
            indexed_idxs.add(idx)
        else:
            has_nonindexed[lt] = True

    # layer_type 정렬 (PREFERRED_ORDER 우선, 나머지 알파벳)
    ordered = [t for t in PREFERRED_ORDER if t in present_types]
    extra   = sorted(present_types - set(ordered))
    layer_types = ordered + extra

    # 열: 인덱스 레이어 → 실제 인덱스 / 비인덱스 레이어 → 별도 열 (오른쪽)
    sorted_idxs = sorted(indexed_idxs)
    n_indexed = max(sorted_idxs) + 1 if sorted_idxs else 0
    col_indices = list(range(n_indexed))
    col_labels  = [str(i) for i in range(n_indexed)]

    # 비인덱스 레이어가 있는 layer_type의 경우 별도 열로 표시
    nonindexed_types = [lt for lt in layer_types if has_nonindexed.get(lt)]
    if nonindexed_types:
        col_indices.append(n_indexed)   # 가상 인덱스
        col_labels.append("fixed")      # 레이블

    return layer_types, col_indices, col_labels


def _build_matrix_dynamic(stats: dict, comp: str, metric: str,
                           layer_types: list[str], col_indices: list[int],
                           col_labels: list[str]) -> np.ndarray:
    """동적 레이아웃으로 2D 행렬 구성."""
    n_rows = len(layer_types)
    n_cols = len(col_indices)
    mat = np.full((n_rows, n_cols), np.nan)

    # "fixed" 열이 있으면 그 컬럼 인덱스 파악
    fixed_col = col_labels.index("fixed") if "fixed" in col_labels else None

    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        if lt not in layer_types:
            continue
        row = layer_types.index(lt)
        idx = _layer_idx(name)

        if idx is not None:
            # 인덱스 있는 레이어 → 해당 열에 직접
            if idx < n_cols:
                v = info.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mat[row, idx] = float(v)
        else:
            # 비인덱스 레이어 → "fixed" 열에 평균 누적 (여러 개면 마지막 값)
            if fixed_col is not None:
                v = info.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mat[row, fixed_col] = float(v)

    return mat


def _build_matrix_chunk(stats: dict, comp: str, metric: str,
                         layer_types: list, chunk_cols: list,
                         col_labels: list) -> np.ndarray:
    """청크된 열 범위로 2D 행렬 구성. actual layer idx → matrix col position 매핑."""
    n_rows = len(layer_types)
    n_cols = len(chunk_cols)
    mat = np.full((n_rows, n_cols), np.nan)

    idx_to_col = {actual: col for col, actual in enumerate(chunk_cols)}
    # "fixed" 열의 matrix position
    fixed_pos = next(
        (col for col, actual in enumerate(chunk_cols)
         if actual < len(col_labels) and col_labels[actual] == "fixed"),
        None
    )

    for name, info in stats.items():
        if info.get("component") != comp:
            continue
        lt = info.get("layer_type", "other")
        if lt not in layer_types:
            continue
        row = layer_types.index(lt)
        idx = _layer_idx(name)

        if idx is not None:
            if idx in idx_to_col:
                v = info.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mat[row, idx_to_col[idx]] = float(v)
        else:
            if fixed_pos is not None:
                v = info.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    mat[row, fixed_pos] = float(v)

    return mat


def _group_by(stats: dict, key: str) -> dict:
    grouped = defaultdict(list)
    for info in stats.values():
        grouped[info[key]].append(info)
    return dict(grouped)


def plot_weight_boxplot(weight_stats: dict, comp: str, out_dir: Path) -> None:
    """per-layer weight 분포를 percentile 기반 box plot으로 표시. COLS_PER_CHUNK 초과 시 파일 분할."""
    layer_types, col_indices, col_labels = _get_comp_layout(weight_stats, comp)
    color = COMP_COLOR[comp]

    # {layer_type: {col_pos: info_dict}}
    data: dict = {lt: {} for lt in layer_types}
    for name, info in weight_stats.items():
        if info["component"] != comp:
            continue
        lt = info["layer_type"]
        if lt not in layer_types:
            continue
        idx = _layer_idx(name)
        if idx is None:
            col_pos = col_labels.index("fixed") if "fixed" in col_labels else 0
        else:
            col_pos = idx
        data[lt][col_pos] = info

    active = [lt for lt in layer_types if data[lt]]
    if not active:
        return

    # 숫자 인덱스 vs fixed 분리 후 청크 분할
    numeric_cols = sorted(c for c in col_indices if c < len(col_labels) and col_labels[c] != "fixed")
    fixed_cols   = [c for c in col_indices if c < len(col_labels) and col_labels[c] == "fixed"]
    chunks = _chunk_indices(numeric_cols) if len(numeric_cols) > COLS_PER_CHUNK else [numeric_cols]
    if fixed_cols:
        chunks[-1] = list(chunks[-1]) + fixed_cols

    for part, chunk_cols in enumerate(chunks, start=1):
        part_label = f"p{part}" if len(chunks) > 1 else ""
        chunk_set  = set(chunk_cols)

        n_cols_fig = min(4, len(active))
        n_rows_fig = (len(active) + n_cols_fig - 1) // n_cols_fig
        fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                                  figsize=(5 * n_cols_fig, 4 * n_rows_fig), squeeze=False)
        axes_flat = [ax for row in axes for ax in row]

        for i, lt in enumerate(active):
            ax = axes_flat[i]
            idxs = [idx for idx in sorted(data[lt].keys()) if idx in chunk_set]
            if not idxs:
                ax.set_visible(False)
                continue
            box_stats = []
            outliers_p1   = []   # (pos, val)
            outliers_p99  = []
            outliers_p999 = []
            for pos, idx in enumerate(idxs):
                info = data[lt][idx]
                pct = info["percentiles"]
                box_stats.append({
                    "med":    pct["p50"],
                    "q1":     pct["p25"],
                    "q3":     pct["p75"],
                    "whislo": pct["p5"],
                    "whishi": pct["p95"],
                    "mean":   info["mean"],
                    "fliers": [],
                })
                outliers_p1.append((pos, pct["p1"]))
                outliers_p99.append((pos, pct["p99"]))
                outliers_p999.append((pos, pct["p999"]))

            bxp = ax.bxp(box_stats, positions=list(range(len(idxs))), widths=0.6,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markersize=3,
                                        markerfacecolor="white", markeredgecolor="k"))
            for patch in bxp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # outlier scatter: p1 / p99 / p999
            ax.scatter([p for p, _ in outliers_p1],   [v for _, v in outliers_p1],
                       marker="o", s=12, color="orange", alpha=0.8,
                       zorder=4, label="p1/p99")
            ax.scatter([p for p, _ in outliers_p99],  [v for _, v in outliers_p99],
                       marker="o", s=12, color="orange", alpha=0.8, zorder=4)
            ax.scatter([p for p, _ in outliers_p999], [v for _, v in outliers_p999],
                       marker="^", s=18, color="red", alpha=0.9,
                       zorder=5, label="p999")
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")

            ax.set_title(lt, fontsize=10, fontweight="bold")
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Weight value")
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(axis="y", alpha=0.3)
            x_labels = [col_labels[i] if i < len(col_labels) else str(i) for i in idxs]
            ax.set_xticks(range(len(idxs)))
            ax.set_xticklabels(x_labels, fontsize=8)

        for j in range(len(active), len(axes_flat)):
            axes_flat[j].set_visible(False)

        range_info = (f" [L{chunk_cols[0]}\u2013L{chunk_cols[-1]}]"
                      if part_label and chunk_cols else "")
        fig.suptitle(f"{COMP_TITLE[comp]} — Weight per Layer Box Plot{range_info}\n"
                     f"(box: p25\u2013p75 / whisker: p5\u2013p95 / diamond: mean)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        suffix = f"_{part_label}" if part_label else ""
        path = out_dir / f"weight_{comp}_boxplot{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {path}")


def _draw_heatmap_panel(ax, mat, layer_types, col_labels, title, cmap, fmt=".2f"):
    """heatmap 패널 공통 렌더링."""
    vmin = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0
    vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    # NaN → 회색
    nan_mask = np.isnan(mat).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    ax.imshow(nan_mask[:, :, np.newaxis] * np.array([[[0.75, 0.75, 0.75]]]),
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(layer_types)))
    ax.set_yticklabels(layer_types, fontsize=9)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_xlabel("Layer index")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=6, color="black")
    return im


def plot_weight_heatmap(weight_stats: dict, comp: str, out_dir: Path) -> None:
    """rows=layer_type, cols=layer_idx — 동적 레이아웃, COLS_PER_CHUNK 초과 시 파일 분할."""
    layer_types, col_indices, col_labels = _get_comp_layout(weight_stats, comp)
    metrics = [
        ("abs_max",               "Abs-Max (weight range)",       "YlOrRd"),
        ("kurtosis",              "Kurtosis (outlier intensity)",  "PuBu"),
        ("per_channel_absmax_cv", "Per-channel AbsMax CV",        "Greens"),
    ]

    numeric_cols = sorted(c for c in col_indices if c < len(col_labels) and col_labels[c] != "fixed")
    fixed_cols   = [c for c in col_indices if c < len(col_labels) and col_labels[c] == "fixed"]
    chunks = _chunk_indices(numeric_cols) if len(numeric_cols) > COLS_PER_CHUNK else [numeric_cols]
    if fixed_cols:
        chunks[-1] = list(chunks[-1]) + fixed_cols

    for part, chunk_cols in enumerate(chunks, start=1):
        part_label   = f"p{part}" if len(chunks) > 1 else ""
        chunk_labels = [col_labels[c] for c in chunk_cols]

        fig, axes = plt.subplots(len(metrics), 1,
                                  figsize=(max(10, len(chunk_cols) * 0.7), 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for ax, (metric, title, cmap) in zip(axes, metrics):
            mat = _build_matrix_chunk(weight_stats, comp, metric,
                                      layer_types, chunk_cols, col_labels)
            _draw_heatmap_panel(ax, mat, layer_types, chunk_labels, title, cmap)

        range_info = (f" [L{chunk_cols[0]}\u2013L{chunk_cols[-1]}]"
                      if part_label and chunk_cols else "")
        fig.suptitle(f"{COMP_TITLE[comp]} — Weight Heatmap{range_info}\n"
                     f"(rows=layer_type, cols=layer_idx  |  grey=없음  |  'fixed'=비인덱스 레이어)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        suffix = f"_{part_label}" if part_label else ""
        path = out_dir / f"weight_{comp}_heatmap{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {path}")


def plot_act_boxplot(act_stats: dict, weight_stats: dict,
                     comp: str, out_dir: Path) -> None:
    """레이어별 activation 값 분포 box plot (weight boxplot과 동일 방식).

    box  : mean(p25) ~ mean(p75)  — 전형적 분포 형태
    whisker: mean(p5) ~ mean(p95)
    빨간 ◆ : max_abs_max across all steps — quantization worst-case
    """
    layer_types, col_indices, col_labels = _get_comp_layout(weight_stats, comp)
    color = COMP_COLOR[comp]

    # {layer_type: {col_pos: act_info}}
    data: dict = {lt: {} for lt in layer_types}
    for name, info in act_stats.items():
        wi = weight_stats.get(name, {})
        if wi.get("component") != comp:
            continue
        lt = wi.get("layer_type", "other")
        if lt not in layer_types:
            continue
        idx = _layer_idx(name)
        col_pos = idx if idx is not None else (
            col_labels.index("fixed") if "fixed" in col_labels else 0
        )
        data[lt][col_pos] = info

    active = [lt for lt in layer_types if data[lt]]
    if not active:
        return

    # 청크 분할
    numeric_cols = sorted(c for c in col_indices if c < len(col_labels) and col_labels[c] != "fixed")
    fixed_cols   = [c for c in col_indices if c < len(col_labels) and col_labels[c] == "fixed"]
    chunks = _chunk_indices(numeric_cols) if len(numeric_cols) > COLS_PER_CHUNK else [numeric_cols]
    if fixed_cols:
        chunks[-1] = list(chunks[-1]) + fixed_cols

    for part, chunk_cols in enumerate(chunks, start=1):
        part_label = f"p{part}" if len(chunks) > 1 else ""
        chunk_set  = set(chunk_cols)

        n_cols_fig = min(4, len(active))
        n_rows_fig = (len(active) + n_cols_fig - 1) // n_cols_fig
        fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                                  figsize=(5 * n_cols_fig, 4 * n_rows_fig), squeeze=False)
        axes_flat = [ax for row in axes for ax in row]

        for i, lt in enumerate(active):
            ax = axes_flat[i]
            idxs = [idx for idx in sorted(data[lt].keys()) if idx in chunk_set]
            if not idxs:
                ax.set_visible(False)
                continue

            positions  = list(range(len(idxs)))
            box_stats  = []
            worst_vals = []
            for idx in idxs:
                info = data[lt][idx]
                box_stats.append({
                    "med":    info.get("p50", info.get("mean", 0.0)),
                    "q1":     info.get("p25", 0.0),
                    "q3":     info.get("p75", 0.0),
                    "whislo": info.get("p5",  0.0),
                    "whishi": info.get("p95", 0.0),
                    "mean":   info.get("mean", 0.0),
                    "fliers": [],
                })
                worst_vals.append(info.get("max_abs_max", info.get("abs_max", 0.0)))

            bxp = ax.bxp(box_stats, positions=positions, widths=0.6,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker="D", markersize=3,
                                        markerfacecolor="white", markeredgecolor="k"))
            for patch in bxp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # worst-case marker (빨간 ◆)
            ax.scatter(positions, worst_vals, marker="D", color="red",
                       s=18, zorder=5, label="max abs_max")
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")

            ax.set_title(lt, fontsize=10, fontweight="bold")
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Activation value")
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.grid(axis="y", alpha=0.3)
            x_labels = [col_labels[i] if i < len(col_labels) else str(i) for i in idxs]
            ax.set_xticks(positions)
            ax.set_xticklabels(x_labels, fontsize=8)

        for j in range(len(active), len(axes_flat)):
            axes_flat[j].set_visible(False)

        range_info = (f" [L{chunk_cols[0]}\u2013L{chunk_cols[-1]}]"
                      if part_label and chunk_cols else "")
        fig.suptitle(
            f"{COMP_TITLE[comp]} — Activation Value Distribution per Layer{range_info}\n"
            f"(box: mean p25\u2013p75 / whisker: mean p5\u2013p95 / \u25c6: worst-case abs_max)",
            fontsize=12, fontweight="bold")
        fig.tight_layout()
        suffix = f"_{part_label}" if part_label else ""
        path = out_dir / f"act_{comp}_boxplot{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {path}")


def plot_act_heatmap(act_stats: dict, weight_stats: dict,
                     comp: str, out_dir: Path) -> None:
    """activation 통계 heatmap. 동적 레이아웃, COLS_PER_CHUNK 초과 시 파일 분할."""
    # weight_stats 기반 동적 레이아웃 감지
    layer_types, col_indices, col_labels = _get_comp_layout(weight_stats, comp)

    # act_stats에 component/layer_type 보강
    augmented = {}
    for name, info in act_stats.items():
        wi = weight_stats.get(name, {})
        augmented[name] = {
            **{k: v for k, v in info.items() if k != "_raw"},
            "component": wi.get("component", "other"),
            "layer_type": wi.get("layer_type", "other"),
        }

    metrics = [
        ("abs_max",             "Activation Abs-Max",  "YlOrRd"),
        ("kurtosis",            "Activation Kurtosis", "PuBu"),
        ("per_token_absmax_cv", "Per-token AbsMax CV", "Greens"),
    ]

    numeric_cols = sorted(c for c in col_indices if c < len(col_labels) and col_labels[c] != "fixed")
    fixed_cols   = [c for c in col_indices if c < len(col_labels) and col_labels[c] == "fixed"]
    chunks = _chunk_indices(numeric_cols) if len(numeric_cols) > COLS_PER_CHUNK else [numeric_cols]
    if fixed_cols:
        chunks[-1] = list(chunks[-1]) + fixed_cols

    for part, chunk_cols in enumerate(chunks, start=1):
        part_label   = f"p{part}" if len(chunks) > 1 else ""
        chunk_labels = [col_labels[c] for c in chunk_cols]

        fig, axes = plt.subplots(len(metrics), 1,
                                  figsize=(max(10, len(chunk_cols) * 0.65), 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        for ax, (metric, title, cmap) in zip(axes, metrics):
            mat = _build_matrix_chunk(augmented, comp, metric,
                                      layer_types, chunk_cols, col_labels)
            _draw_heatmap_panel(ax, mat, layer_types, chunk_labels, title, cmap, fmt=".1f")

        range_info = (f" [L{chunk_cols[0]}\u2013L{chunk_cols[-1]}]"
                      if part_label and chunk_cols else "")
        fig.suptitle(f"{COMP_TITLE[comp]} — Activation Heatmap{range_info}\n"
                     f"(rows=layer_type, cols=layer_idx)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        suffix = f"_{part_label}" if part_label else ""
        path = out_dir / f"act_{comp}_heatmap{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {path}")


def plot_metric_bars(val_stats: dict, comp: str, metric: str,
                     ylabel: str, filename: str, out_dir: Path,
                     meta_stats: dict = None, raw_metric: str = None) -> None:
    """레이어별 단일 metric 값을 bar plot으로 표시.

    val_stats : 값을 읽을 stats dict (weight_stats 또는 augmented act_stats)
    meta_stats: component/layer_type 조회용 (None이면 val_stats 자체 사용)
    raw_metric: _raw에서 error bar용 std를 뽑을 key (None이면 error bar 없음)
    """
    meta = meta_stats if meta_stats is not None else val_stats
    layer_types, col_indices, col_labels = _get_comp_layout(meta, comp)
    color = COMP_COLOR[comp]

    # {layer_type: {col_pos: (value, err)}}
    data: dict = {lt: {} for lt in layer_types}
    for name, info in val_stats.items():
        mi = meta.get(name, info)
        if mi.get("component") != comp:
            continue
        lt = mi.get("layer_type", "other")
        if lt not in layer_types:
            continue
        val = info.get(metric)
        if val is None:
            continue
        idx = _layer_idx(name)
        col_pos = idx if idx is not None else (
            col_labels.index("fixed") if "fixed" in col_labels else 0
        )
        err = 0.0
        if raw_metric and "_raw" in info:
            raw_vals = info["_raw"].get(raw_metric, [])
            if len(raw_vals) > 1:
                err = float(np.std(raw_vals))
        data[lt][col_pos] = (float(val), err)

    active = [lt for lt in layer_types if data[lt]]
    if not active:
        return

    numeric_cols = sorted(c for c in col_indices if c < len(col_labels) and col_labels[c] != "fixed")
    fixed_cols   = [c for c in col_indices if c < len(col_labels) and col_labels[c] == "fixed"]
    chunks = _chunk_indices(numeric_cols) if len(numeric_cols) > COLS_PER_CHUNK else [numeric_cols]
    if fixed_cols:
        chunks[-1] = list(chunks[-1]) + fixed_cols

    for part, chunk_cols in enumerate(chunks, start=1):
        part_label = f"p{part}" if len(chunks) > 1 else ""
        chunk_set  = set(chunk_cols)

        n_cols_fig = min(4, len(active))
        n_rows_fig = (len(active) + n_cols_fig - 1) // n_cols_fig
        fig, axes = plt.subplots(n_rows_fig, n_cols_fig,
                                  figsize=(5 * n_cols_fig, 4 * n_rows_fig), squeeze=False)
        axes_flat = [ax for row in axes for ax in row]

        for i, lt in enumerate(active):
            ax = axes_flat[i]
            idxs = [idx for idx in sorted(data[lt].keys()) if idx in chunk_set]
            if not idxs:
                ax.set_visible(False)
                continue
            vals     = [data[lt][idx][0] for idx in idxs]
            errs     = [data[lt][idx][1] for idx in idxs]
            positions = list(range(len(idxs)))
            ax.bar(positions, vals, color=color, alpha=0.7, width=0.6)
            if any(e > 0 for e in errs):
                ax.errorbar(positions, vals, yerr=errs, fmt="none",
                            ecolor="k", capsize=3, linewidth=1)
            ax.set_title(lt, fontsize=10, fontweight="bold")
            ax.set_xlabel("Layer index")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
            x_labels = [col_labels[i] if i < len(col_labels) else str(i) for i in idxs]
            ax.set_xticks(positions)
            ax.set_xticklabels(x_labels, fontsize=8)

        for j in range(len(active), len(axes_flat)):
            axes_flat[j].set_visible(False)

        range_info = (f" [L{chunk_cols[0]}\u2013L{chunk_cols[-1]}]"
                      if part_label and chunk_cols else "")
        fig.suptitle(f"{COMP_TITLE[comp]} — {ylabel} per Layer{range_info}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        suffix = f"_{part_label}" if part_label else ""
        path = out_dir / f"{filename}_{comp}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 콘솔 요약표
# ══════════════════════════════════════════════════════════════════════════════

def print_weight_summary(weight_stats: dict) -> None:
    # group by (component, layer_type)
    groups: dict[tuple, list] = defaultdict(list)
    for info in weight_stats.values():
        groups[(info["component"], info["layer_type"])].append(info)

    W = 72
    print(f"\n{'='*W}")
    print("  Weight Distribution Summary (fp32)")
    print(f"  {'Comp':<6} {'LayerType':<16} {'n':>4}  {'AbsMax(μ)':>10} {'Kurt(μ)':>8}  {'ChCV(μ)':>8}")
    print(f"  {'-'*64}")
    for (comp, ltype), infos in sorted(groups.items()):
        if comp not in ("lm", "dit"):
            continue
        n = len(infos)
        ab = np.mean([x["abs_max"] for x in infos])
        ku = np.mean([x["kurtosis"] for x in infos])
        cv = np.mean([x["per_channel_absmax_cv"] for x in infos])
        print(f"  {comp:<6} {ltype:<16} {n:>4}  {ab:>10.4f} {ku:>8.2f}  {cv:>8.4f}")
    print(f"{'='*W}")


def print_act_summary(act_stats: dict, weight_stats: dict, n_steps: int) -> None:
    groups: dict[tuple, list] = defaultdict(list)
    for name, info in act_stats.items():
        comp = weight_stats.get(name, {}).get("component", "other")
        ltype = weight_stats.get(name, {}).get("layer_type", "other")
        groups[(comp, ltype)].append(info)

    W = 72
    print(f"\n{'='*W}")
    print(f"  Activation Distribution Summary ({n_steps} steps)")
    print(f"  {'Comp':<6} {'LayerType':<16} {'n':>4}  {'AbsMax(μ)':>10} {'Kurt(μ)':>8}  {'TokCV(μ)':>9}")
    print(f"  {'-'*64}")
    for (comp, ltype), infos in sorted(groups.items()):
        if comp not in ("lm", "dit"):
            continue
        n = len(infos)
        ab = np.mean([x["abs_max"] for x in infos])
        ku = np.mean([x["kurtosis"] for x in infos])
        cv = np.mean([x["per_token_absmax_cv"] for x in infos])
        print(f"  {comp:<6} {ltype:<16} {n:>4}  {ab:>10.4f} {ku:>8.2f}  {cv:>9.4f}")
    print(f"{'='*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LM vs DiT weight & activation distribution analysis"
    )
    parser.add_argument("--pretrained_path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task", default="libero_spatial",
                        help="LIBERO task (activation 수집 시 사용)")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="activation 캡처 episode 수")
    parser.add_argument("--output_dir", default="logs/dist_analysis")
    parser.add_argument("--weight_only", action="store_true",
                        help="activation 분석 skip")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ── Policy 로드 ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading policy from {args.pretrained_path}")
    env_cfg = LiberoEnv(task=args.task)
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    print("[INFO] Policy loaded")

    # ── Weight 분석 ────────────────────────────────────────────────────────────
    print("\n[INFO] Analyzing weight distributions...")
    weight_stats = collect_weight_stats(policy)
    print(f"[INFO] Collected weight stats for {len(weight_stats)} layers")
    print_weight_summary(weight_stats)

    # ── Activation 분석 ───────────────────────────────────────────────────────
    act_stats: dict = {}
    if not args.weight_only:
        print(f"\n[INFO] Creating LIBERO env (task={args.task}) for activation analysis...")
        envs_dict = make_env(env_cfg, n_envs=1)

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )

        print(f"[INFO] Collecting activations over {args.n_episodes} episodes...")
        act_stats = collect_activation_stats(
            policy=policy,
            env_cfg=env_cfg,
            envs_dict=envs_dict,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            n_episodes=args.n_episodes,
        )
        print(f"[INFO] Collected activation stats for {len(act_stats)} layers")
        print_act_summary(act_stats, weight_stats, args.n_episodes)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    print("\n[INFO] Generating plots...")
    for comp in ("lm", "dit"):
        plot_weight_boxplot(weight_stats, comp, out_dir)
        plot_weight_heatmap(weight_stats, comp, out_dir)
        plot_metric_bars(weight_stats, comp,
                         metric="kurtosis", ylabel="Kurtosis",
                         filename="weight_kurtosis", out_dir=out_dir)
        plot_metric_bars(weight_stats, comp,
                         metric="per_channel_absmax_cv", ylabel="Per-channel AbsMax CV",
                         filename="weight_abs_max_cv", out_dir=out_dir)

    if act_stats:
        # act_stats에 component/layer_type 보강 (plot_metric_bars가 meta 없이 사용할 수 있도록)
        act_augmented = {}
        for name, info in act_stats.items():
            wi = weight_stats.get(name, {})
            act_augmented[name] = {
                **{k: v for k, v in info.items()},
                "component":  wi.get("component", "other"),
                "layer_type": wi.get("layer_type", "other"),
            }

        for comp in ("lm", "dit"):
            plot_act_boxplot(act_stats, weight_stats, comp, out_dir)
            plot_act_heatmap(act_stats, weight_stats, comp, out_dir)
            plot_metric_bars(act_augmented, comp,
                             metric="kurtosis", ylabel="Kurtosis",
                             filename="act_kurtosis", out_dir=out_dir,
                             raw_metric="kurtosis")
            plot_metric_bars(act_augmented, comp,
                             metric="per_token_absmax_cv", ylabel="Per-token AbsMax CV",
                             filename="act_abs_max_cv", out_dir=out_dir,
                             raw_metric="per_token_absmax_cv")

    # ── JSON 저장 ──────────────────────────────────────────────────────────────
    stats_json = {
        "config": {
            "pretrained_path": args.pretrained_path,
            "task": args.task,
            "n_episodes": args.n_episodes,
            "weight_only": args.weight_only,
        },
        "weight_stats": weight_stats,
        "activation_stats": act_stats,
    }
    json_path = out_dir / "dist_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats_json, f, indent=2, default=str)
    print(f"[SAVED] {json_path}")
    print(f"\n[DONE] All outputs → {out_dir}/")


if __name__ == "__main__":
    main()
