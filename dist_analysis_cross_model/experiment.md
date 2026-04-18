# Cross-Model Distribution Analysis

## 목적

LeRobot pi0.5의 구성 요소(VLA LM, VLA DiT)와 동일 아키텍처의 일반 모델(Gemma-2B LLM, PixArt-Alpha DiT)의
weight / activation 분포를 비교하여, VLA fine-tuning이 distribution에 미치는 영향을 정량화한다.

총 3가지 분석:

| # | Task | Script | 상태 |
|---|------|--------|------|
| 1 | VLA LM vs Gemma-2B LLM weight/act 분포 비교 | `compare_lm_distributions.py` | 예정 |
| 2 | VLA DiT vs PixArt-Alpha DiT + timestep-wise 분석 | `compare_dit_distributions.py` | 예정 |
| 3 | 기존 heatmap 재플롯 (other 제거, quant 대상만) | `replot_heatmaps.py` | 예정 |

---

## Task 1: VLA LM vs Gemma-2B LLM

### 목적

pi0.5의 PaliGemma LM (VLA-fine-tuned Gemma-2B)과 standalone Gemma-2B의 weight/activation 분포를 비교.
VLA fine-tuning이 kurtosis, CV, abs_max에 어떤 영향을 미치는지 파악.

### 모델

| 모델 | HF Hub ID | 역할 |
|------|-----------|------|
| pi0.5 LM | `lerobot/pi05_libero_finetuned` (기존 stats 재활용) | VLA LM |
| Gemma-2B | `google/gemma-2b` | 일반 LLM baseline |

### 데이터셋

| 모델 | 데이터 |
|------|--------|
| pi0.5 LM activation | 기존 `logs/dist_analysis_v4/dist_stats.json` 재활용 (LIBERO 3 episodes) |
| Gemma-2B activation | WikiText-2 (wikitext-2-raw-v1, validation split) 텍스트 프롬프트 50개 → forward → hook |

#### VLA LM Activation Input (LIBERO)

- **모델**: `lerobot/pi05_libero_finetuned`
- **Task suite**: `libero_spatial` (10개 task)
- **Episodes per task**: 3
- **Max steps per episode**: 280
- **Observation**: RGB 이미지 + language instruction
- **CSV**: `plots/vla_input_samples.csv` (30 rows: 10 tasks × 3 episodes)

| task_id | task_name |
|---------|-----------|
| 0 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate |
| 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate |
| 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate |
| 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate |
| 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate |
| 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate |
| 6 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate |
| 7 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate |
| 8 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate |
| 9 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate |

#### Gemma-2B Activation Input (WikiText-2)

- **Dataset**: HuggingFace `wikitext` / `wikitext-2-raw-v1`, split=`validation`
- **Samples**: 50개 (length > 50 chars 필터)
- **Tokenization**: max_length=256, truncation=True
- **CSV**: `plots/gemma_input_samples.csv` (50 rows: idx, length, text)
- **수집 항목**: abs_max, mean, kurtosis, per_token_absmax_cv, p1/p5/p25/p50/p75/p95/p99 (per layer)

### 레이어 매핑 (1:1 대응, 18 blocks × 7 types)

| pi0.5 LM key pattern | Gemma-2B key pattern | Type |
|----------------------|----------------------|------|
| `language_model.layers.{i}.self_attn.q_proj` | `model.layers.{i}.self_attn.q_proj` | q_proj |
| `language_model.layers.{i}.self_attn.k_proj` | `model.layers.{i}.self_attn.k_proj` | k_proj |
| `language_model.layers.{i}.self_attn.v_proj` | `model.layers.{i}.self_attn.v_proj` | v_proj |
| `language_model.layers.{i}.self_attn.o_proj` | `model.layers.{i}.self_attn.o_proj` | o_proj |
| `language_model.layers.{i}.mlp.gate_proj` | `model.layers.{i}.mlp.gate_proj` | gate_proj |
| `language_model.layers.{i}.mlp.up_proj` | `model.layers.{i}.mlp.up_proj` | up_proj |
| `language_model.layers.{i}.mlp.down_proj` | `model.layers.{i}.mlp.down_proj` | down_proj |

### 수집 항목

- **Weight**: abs_max, kurtosis, per_channel_absmax_cv, percentiles (p1~p99)
- **Activation**: abs_max, kurtosis, per_token_absmax_cv

### 출력 (plots/)

| 파일 | 내용 |
|------|------|
| `lm_weight_kurtosis_comparison.png` | Block-depth별 kurtosis: VLA LM vs Gemma-2B, layer type별 |
| `lm_weight_cv_comparison.png` | Block-depth별 CV: VLA LM vs Gemma-2B |
| `lm_weight_absmax_comparison.png` | Block-depth별 abs_max |
| `lm_act_kurtosis_comparison.png` | Activation kurtosis 비교 |
| `lm_act_cv_comparison.png` | Activation per_token CV 비교 |
| `lm_summary_stats.csv` | 집계 통계 (mean/std across layers) |
| `dist_stats_gemma2b.json` | Gemma-2B raw stats 저장 |

---

## Task 2: VLA DiT vs PixArt-Alpha DiT

### 목적

pi0.5의 Gemma Expert(DiT, 18 blocks)와 PixArt-Alpha DiT(28 blocks)의 weight/activation 분포 비교.
timestep(denoising step)별 outlier 강도 분석.

### 모델

| 모델 | 소스 | 역할 |
|------|------|------|
| pi0.5 DiT | `lerobot/pi05_libero_finetuned` (기존 stats 재활용) | VLA DiT |
| PixArt-Alpha DiT | 기존 `pixart_alpha/pixart_data_distributions/` | 이미지 생성 DiT |

### 데이터셋

| 모델 | 데이터 |
|------|--------|
| pi0.5 DiT activation | 기존 stats 재활용 + LIBERO env denoising hook |
| PixArt-Alpha | MJHQ dataset (xingjianleng/mjhq30k) 8 prompts, 20 inference steps |

#### VLA DiT Activation Input (LIBERO)

- **모델**: `lerobot/pi05_libero_finetuned`
- **Task suite**: `libero_spatial` (10개 task, VLA LM과 동일)
- **Episodes per task**: 3
- **Aggregated stats** (`dist_stats.json`): 전체 denoising step 평균
- **Timestep-wise stats** (`vla_dit_timestep_stats.json`): 총 60 denoising step 캡처 (3 episodes × 20 steps/episode)
- **CSV**: `plots/vla_input_samples.csv`, `plots/vla_dit_timestep_input_samples.csv`

#### PixArt-Alpha Activation Input (MJHQ)

- **Dataset**: `xingjianleng/mjhq30k`, split=`test`
- **Prompts**: 8개 (MJHQ 텍스트 캡션)
- **Inference steps**: 20 (DDPM scheduler)
- **Resolution**: 1024×1024
- **Per-timestep**: 160 timestep entries (8 prompts × 20 steps)
- **CSV**: `plots/pixart_input_samples.csv` (160 rows: prompt_idx, prompt, denoising_step)

### Timestep-wise 분석

- pi0.5: flow matching (time 1→0, 20 steps/episode) → sentinel hook (block 0 q_proj pre-hook)으로 step 카운트
- PixArt: DDPM (20 inference steps) → 기존 `per_timestep` 데이터 활용 (160 timesteps)

### 레이어 매핑 (depth-normalized 비교, 각각 28/18 blocks)

| pi0.5 DiT | PixArt-Alpha | Type |
|-----------|-------------|------|
| `gemma_expert.layers.{i}.self_attn.{q,k,v,o}_proj` | `transformer.transformer_blocks.{i}.attn1.to_{q,k,v,out.0}` | self-attn |
| `gemma_expert.layers.{i}.mlp.{gate,up,down}_proj` | `transformer.transformer_blocks.{i}.ff.net.{0.proj,2}` | mlp |

### 출력 (plots/)

| 파일 | 내용 |
|------|------|
| `dit_weight_kurtosis_comparison.png` | Depth-normalized kurtosis 비교 |
| `dit_weight_cv_comparison.png` | Depth-normalized CV 비교 |
| `dit_timestep_heatmap_vla.png` | pi0.5 DiT timestep × layer 히트맵 (kurtosis) |
| `dit_timestep_heatmap_pixart.png` | PixArt DiT timestep × layer 히트맵 |
| `dit_timestep_outlier_curve.png` | Timestep별 평균 outlier (abs_max) 곡선 |
| `dit_summary_stats.csv` | 집계 통계 |

---

## Task 3: Heatmap 재플롯 (other 제거)

### 목적

기존 `lerobot/data_dist_analysis/` 히트맵에 `layer_type="other"` (multi_modal_projector, lm_head 등 38개)가 포함되어
시각적으로 왜곡됨. Quantize 대상 레이어만 (q/k/v/o_proj, gate/up/down_proj) 필터링하여 재플롯.

### 입력

- `lerobot/logs/dist_analysis_v4/dist_stats.json` — pi0.5 전체 stats
- `layer_type != "other"` 필터

### 출력 (lerobot/data_dist_analysis/ 및 plots/)

| 파일 | 내용 |
|------|------|
| `weight_lm_heatmap.png` | LM weight kurtosis 히트맵 (quant 대상만) |
| `weight_dit_heatmap.png` | DiT weight kurtosis 히트맵 |
| `act_lm_heatmap.png` | LM activation kurtosis 히트맵 |
| `act_dit_heatmap.png` | DiT activation kurtosis 히트맵 |
| `weight_kurtosis_lm.png` | LM weight kurtosis bar (layer별) |
| `weight_kurtosis_dit.png` | DiT weight kurtosis bar |
| `weight_abs_max_cv_lm.png` | LM weight CV bar |
| `weight_abs_max_cv_dit.png` | DiT weight CV bar |
| `act_kurtosis_lm.png` | LM act kurtosis bar |
| `act_kurtosis_dit.png` | DiT act kurtosis bar |
| `act_abs_max_cv_lm.png` | LM act CV bar |
| `act_abs_max_cv_dit.png` | DiT act CV bar |

---

## 실행 순서

```bash
cd /home/jameskimh/workspace/Workspace_Lerobot/dist_analysis_cross_model

# Task 3: 가장 빠름 (~1분, 기존 데이터만 사용)
python replot_heatmaps.py

# Task 1: Gemma-2B weight+act 수집 (~10분)
python compare_lm_distributions.py

# Task 2: timestep-wise hook (~20분, LIBERO env 필요)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python compare_dit_distributions.py
```

## 핵심 참조 파일

| 파일 | 설명 |
|------|------|
| `../lerobot/analyze_distributions.py` | weight/act 통계 수집 + 플롯 함수 |
| `../lerobot/logs/dist_analysis_v4/dist_stats.json` | pi0.5 기존 stats (weight + act) |
| `../pixart_alpha/pixart_data_distributions/` | PixArt 기존 결과 (dyn_stats.json) |

---

*Created: 2026-04-14*
