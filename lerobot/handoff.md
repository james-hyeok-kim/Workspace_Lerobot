# Quantization Experiments — Handoff Notes

**Date**: 2026-04-12
**Branch**: main
**Working dir**: `/home/jameskimh/workspace/Workspace_Lerobot/lerobot`

---

## Completed Work

### 1. NVFP4 MTQ Experiment (`eval_nvfp4_mtq.py` + `run_nvfp4.sh`)

Applied NVIDIA ModelOpt (MTQ) `NVFP4_DEFAULT_CFG` to the pi05 policy.

**Key changes:**
- `eval_nvfp4_mtq.py`: added `torch.compile` disable block after MTQ quantize
  - Call `_dynamo.reset()` after `mtq.quantize()`
  - Restore `_torchdynamo_orig_callable` for `inner_model.sample_actions` / `.forward`
- `run_nvfp4.sh`:
  - `TEST_MODE=1` → libero_10, 1 episode
  - `FULL_MODE` → 4 suites (libero_10/spatial/object/goal), 50 episodes, batch=10
  - Result JSON: `logs/nvfp4_full/<suite>.json`

### 2. RPCA SVD Experiment (`eval_rpca_svd.py` + `run_rpca.sh`)

Weight decomposition via Robust PCA (Inexact ALM) followed by quantization.

**Algorithm**: W = L (low-rank) + S (sparse outlier)
- L is quantized; S is kept as fp16 residual
- Layers with dim > `max_rpca_dim=1024` skip RPCA and are quantized directly

**Supported schemes:**
| Scheme | Description |
|--------|-------------|
| `int8_w` | INT8 weight-only, per-channel symmetric |
| `int8_wa` | INT8 weight + activation |
| `int4_w` | INT4 weight-only, block-wise (block=16) |
| `int4_wa` | INT4 weight + activation |
| `int2_w` | INT2 weight-only, per-channel symmetric |
| `int2_wa` | INT2 weight + activation |
| `ternary_w` | Ternary weight {-α, 0, +α} (TWN) |
| `nvfp4_wa` | NVFP4 via ModelOpt MTQ, weight + activation |

**Custom modules:**
- `RPCAQuantizedLinear`: stores L_dequant + S as fp16 buffers; exposes `.weight`, `.out_features`, `.in_features` properties
- `RPCANvfp4Wrapper`: creates internal `nn.Linear(device=L.device, dtype=L.dtype)` then applies MTQ; exposes same properties

**Coverage report** (`build_quant_coverage` / `print_coverage_table`):
- By component: vision / llm / expert / action_head
- By role: attn_Q/K/V/O, mlp, proj, etc.
- Shows Direct vs RPCA breakdown

**run_rpca.sh:**
- `TEST_MODE=1` → schemes="int8_w nvfp4_wa", 1 episode
- `FULL_MODE` → 8 schemes, 50 episodes
- Override via `SCHEMES="int4_w"` env var

### 3. LeRobot Upstream Bug Fix

**File**: `src/lerobot/scripts/lerobot_eval.py:371`

```python
# Before (bug)
else:
    all_seeds.append(None)

# After (fix)
else:
    all_seeds.extend([None] * env.num_envs)
```

**Root cause**: `all_seeds` appended only 1 element per batch, causing a length mismatch with `sum_rewards`, `max_rewards`, `all_successes` (which have `num_envs` elements per batch) → `ValueError` from `zip(..., strict=True)`.
**Symptom**: `ValueError: zip() argument 4 is shorter than arguments 1-3`

---

## Known Issues / Caveats

1. **torch.compile conflict**: After MTQ `quantize()`, must call `_dynamo.reset()` and restore the original callable. pi05's `model.sample_actions` / `model.forward` are wrapped by `torch.compile`, which conflicts with `_FoldedCallback`.

2. **RPCA coverage**: With `max_rpca_dim=1024` and pi05's architecture:
   - Vision encoder (SigLIP, 1152-dim), LLM (Gemma, 2048-dim) → quantized directly
   - Expert k/v proj, action_head MLP → RPCA applied

3. **CUDA device for RPCANvfp4Wrapper**: Must pass `device=L.device, dtype=L.dtype` when creating the inner `nn.Linear`. Omitting this creates a CPU tensor and causes a CUDA assert.

4. **Coverage aggregation**: `RPCANvfp4Wrapper` is a non-leaf module with children, so `build_quant_coverage` handles it separately via `isinstance` check.

---

## Running Experiments

```bash
# NVFP4 TEST (1 episode)
TEST_MODE=1 bash run_nvfp4.sh

# NVFP4 FULL (4 suites x 50 episodes)
bash run_nvfp4.sh

# RPCA TEST (int8_w + nvfp4_wa, 1 episode)
TEST_MODE=1 bash run_rpca.sh

# RPCA FULL (8 schemes x 50 episodes)
bash run_rpca.sh

# RPCA with specific schemes
SCHEMES="int4_w int4_wa" bash run_rpca.sh
```

---

## File Index

| File | Description |
|------|-------------|
| `eval_nvfp4_mtq.py` | NVFP4 MTQ quantization + LIBERO eval |
| `run_nvfp4.sh` | NVFP4 experiment runner (TEST/FULL mode) |
| `eval_rpca_svd.py` | RPCA-based multi-scheme quantization + LIBERO eval |
| `run_rpca.sh` | RPCA experiment runner (TEST/FULL mode) |
| `check_pi05_precision.py` | Inspect weight dtype distribution of a HF Hub model |
| `run_check_pi05_precision.sh` | Runner for precision check |
| `src/lerobot/scripts/lerobot_eval.py` | Includes upstream bug fix |

---

## PixArt-Alpha DiT Distribution Analysis (2026-04-12)

**Working dir**: `/home/jameskimh/workspace/Workspace_DiT/pixart_alpha`
**Script**: `pixart_distribution_analysis.py`

### Overview

Analysis pipeline for extracting and visualizing weight/activation/output distribution statistics across 285 Linear layers in PixArt-Alpha DiT.

- **Setup**: 8 prompts x 20 diffusion steps
- **Stats output**: `results/distribution_analysis/stats.json` (already generated)
- **Plot output**: `results/distribution_analysis/plots/`

### Layer Structure

- Target: `transformer_blocks.{i}.{subtype}` (i=0..27, 28 blocks)
- Skipped: `x_embedder`, `t_embedder`, `proj_out`
- Subtypes: `attn1.to_q`, `attn1.to_k`, `attn1.to_v`, `attn1.to_out.0`, `attn2.to_q`, `attn2.to_k`, `attn2.to_v`, `attn2.to_out.0`, `ff.net.0.proj`, `ff.net.2`

### stats.json Schema

```json
{
  "weight": {
    "<layer_name>": {
      "mean", "std", "abs_max", "kurtosis", "outlier_ratio", "cv",
      "q1", "q5", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
      "shape", "layer_type"
    }
  },
  "dynamics": {
    "<layer_name>": {
      "act": {
        "per_timestep": [{"mean", "std", "abs_max", "kurtosis", "outlier_ratio", "cv"}, ...],
        "aggregated": {"mean", "std", "abs_max", "kurtosis", "outlier_ratio", "cv"},
        "percentiles": {"q1", "q5", ..., "q99"}
      },
      "out": { ... }
    }
  }
}
```

### Visualizations

**Boxplot** (separate for weight/activation/output, grouped by attn/mlp/other):
- Y: value distribution (Q5~Q95 whisker, Q25~Q75 box)
- Red dot: `abs_max` (extreme maximum)
- Gold dot: `q99` / `q1` (upper/lower outliers)
- X-axis: block_idx (0~27), one subplot per subtype

**Stats Heatmap** (`stats_heatmap_{metric}_part{n}.png`):
- Separate file per metric: `abs_max`, `kurtosis`, `cv`, `outlier_ratio`
- Each file has 3 source panels (Weight / Activation / Output)
- No cell text, p2/p98 color normalization

**Timestep Heatmap** (`{act|output}_timestep_heatmap_{subtype}.png`):
- Separate file per subtype (10 subtypes)
- Y=block_idx (0~27), X=timestep (t0~t19)
- No cell text

### Re-running

```bash
# Re-generate plots only (when stats.json already exists)
/home/jameskimh/.dit/bin/python pixart_distribution_analysis.py \
    --plot_only --output_dir results/distribution_analysis

# Full analysis (collect stats via forward hooks)
/home/jameskimh/.dit/bin/python pixart_distribution_analysis.py \
    --output_dir results/distribution_analysis
```

### Key Implementation Details

- `_draw_heatmap_panel`: p2/p98 normalization, no cell text, 45° xtick rotation
- `_bxp_data_weight` / `_bxp_data_dyn`: `fliers=[]` + private `_abs_max`, `_q99`, `_q1` fields
- `_plot_boxplot_chunk`: `ax.bxp(showfliers=False)` followed by manual scatter (2-color outliers)
- `plot_stats_heatmap`: split into 4 separate files by metric
- `plot_timestep_heatmap`: separate file per subtype, Y=block, X=timestep

---

## pi0.5 Weight/Activation Distribution Analysis (2026-04-12)

**Script**: `analyze_distributions.py`
**Output dir**: `logs/dist_analysis_v4/`

### Overview

Visualizes weight and activation value distributions across 296 Linear layers in pi0.5 (PaliGemma + DiT Action Expert). Layers are split into two components: LM (Gemma 2B) and DiT (Action Expert + Action Head).

- **Setup**: `libero_spatial`, 3 episodes
- **Forward pass count**: LM ~6 times, DiT ~60 times (action chunking x 10 denoising steps)
- **Output**: 17 files in `logs/dist_analysis_v4/`

### Layer Classification Patterns

```python
LM_PATTERNS  = ["model.language_model.model"]               # Gemma 2B transformer
DIT_PATTERNS = ["model.action_expert_model", "model.action_head"]  # DiT
```

- LM: q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj (x18 layers)
- DiT: to_q / to_k / to_v / to_out / ff_linear1 / ff_linear2 / etc.

### Generated Plots

| File | Description |
|------|-------------|
| `weight_lm_boxplot.png` | LM weight distribution per layer (with p1/p99/p999 outlier scatter) |
| `weight_dit_boxplot.png` | DiT weight distribution per layer |
| `weight_lm_heatmap.png` | LM weight heatmap (abs_max / kurtosis / CV) |
| `weight_dit_heatmap.png` | DiT weight heatmap |
| `weight_kurtosis_lm.png` | LM weight kurtosis bar plot |
| `weight_kurtosis_dit.png` | DiT weight kurtosis bar plot |
| `weight_abs_max_cv_lm.png` | LM weight per-channel AbsMax CV bar plot |
| `weight_abs_max_cv_dit.png` | DiT weight per-channel AbsMax CV bar plot |
| `act_lm_boxplot.png` | LM activation value distribution (mean percentile + red diamond worst-case) |
| `act_dit_boxplot.png` | DiT activation value distribution |
| `act_lm_heatmap.png` | LM activation heatmap (per-step metrics) |
| `act_dit_heatmap.png` | DiT activation heatmap |
| `act_kurtosis_lm.png` | LM activation kurtosis bar plot |
| `act_kurtosis_dit.png` | DiT activation kurtosis bar plot |
| `act_abs_max_cv_lm.png` | LM activation per-token AbsMax CV bar plot |
| `act_abs_max_cv_dit.png` | DiT activation per-token AbsMax CV bar plot |
| `dist_stats.json` | Full statistics JSON |

### Weight Boxplot Layout

- Box: p25~p75, whisker: p5~p95, median line
- Orange circle: p1 / p99 (outliers)
- Red triangle: p999 (extreme outlier)

### Activation Boxplot Layout (Method B — mean percentile)

- Box: mean(p25)~mean(p75) across all forward passes
- Whisker: mean(p5)~mean(p95)
- Median line: mean(p50)
- Red diamond: max_abs_max (worst-case across all timesteps)

### Key Functions

| Function | Approx. Line | Role |
|----------|-------------|------|
| `_get_comp_layout()` | ~100 | Dynamically detect layer_types and indices from data |
| `_chunk_indices()` | ~110 | Split indices into chunks of COLS_PER_CHUNK (=32) |
| `_build_matrix_chunk()` | ~120 | Build heatmap matrix with idx_to_col mapping |
| `make_pre_hook()` | ~200 | Forward pre-hook: collect p5/p25/p50/p75/p95 (50k subsample) |
| `aggregate_act_stats()` | ~230 | Compute per-step mean percentiles + max_abs_max |
| `plot_weight_boxplot()` | ~400 | Weight distribution bxp + outlier scatter |
| `plot_act_boxplot()` | ~618 | Activation value distribution bxp + worst-case scatter |
| `plot_metric_bars()` | ~700 | Generic bar plot for kurtosis / abs_max_cv |

### Re-running

```bash
cd /home/jameskimh/workspace/Workspace_Lerobot/lerobot
export PYTHONPATH="src:.:../TensorRT-Model-Optimizer:${PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

python analyze_distributions.py \
  --task libero_spatial --n_episodes 3 \
  --output_dir logs/dist_analysis_v4

ls logs/dist_analysis_v4/
```

### Design Decisions

1. **LM/DiT sampling asymmetry**: LM runs once per action chunk; DiT runs 10 denoising steps per chunk. This is intentional architecture behavior, not a bug.
2. **Dynamic layout detection**: `_get_comp_layout()` detects layer types and indices directly from data instead of hardcoded constants.
3. **Heatmap file splitting**: `COLS_PER_CHUNK=32`; with 18 layers currently outputs a single file per component.
4. **Activation subsampling**: If tensor size > 50k elements, stride-sample to 50k before computing quantiles — reduces memory and compute overhead.

---

## Suggested Next Steps

- Run FULL MODE to collect results across all 4 suites x 8 schemes
- Quantify accuracy drop vs. fp32/bf16 baseline
- Search RPCA rank / lambda hyperparameters (current defaults: rank=32, lam_scale=1.0)
- Extend same experiments to GROOT N1.5 model
