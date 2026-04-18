# NVFP4 Quantization Comparison — Experiment Result

## 개요

`captured.pt`에 저장된 single linear layer의 input(`x`)과 weight(`W`)를
`NVFP4_DEFAULT_CFG` 설정으로 quantize하여 FP baseline과 비교한 실험.

- **스크립트**: `nvfp4_single_layer.py`, `plot_nvfp4_comparison.py`
- **결과 파일**: `nvfp4_quant_result.pt`, `nvfp4_comparison.png`

---

## 대상 레이어

```
model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj
```

pi0.5 모델의 Gemma Expert 첫 번째 레이어의 MLP gate projection.

---

## Quantization 설정 (NVFP4_DEFAULT_CFG)

| 항목 | 값 |
|------|----|
| Format | NVFP4 (E2M1) — `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` |
| Block size | **16** (last dim 기준, 16원소마다 scale 1개) |
| Scale 방식 | Two-level: global (float32) × per-block (FP8 E4M3) |
| Algorithm | max (amax calibration, SVD 없음) |
| Global scale 공식 | `amax / (6.0 × 448.0)` |
| Per-block scale 공식 | `block_amax / (6.0 × global_scale)` → FP8 E4M3으로 저장 |

---

## 텐서 정보

| | Input (x) | Weight (W) |
|---|---|---|
| Shape | `[1, 50, 1024]` | `[4096, 1024]` |
| dtype (원본) | bfloat16 | bfloat16 |
| FP 값 범위 | −6.81 ~ 15.06 | −0.336 ~ 0.223 |
| Dequant 값 범위 | −7.00 ~ 15.06 | −0.336 ~ 0.216 |

---

## Scale 정보

| | Input (x) | Weight (W) |
|---|---|---|
| Global scale | `5.604e-3` (float32) | `1.250e-4` (float32) |
| Per-block scale shape | `[1, 50, 64]` | `[4096, 64]` |
| Per-block scale dtype | float8_e4m3fn | float8_e4m3fn |
| Per-block scale range | 9.0 ~ 448.0 | 32.0 ~ 448.0 |
| Block 수 (per row) | 1024 / 16 = **64** | 1024 / 16 = **64** |

> per-block scale 최대값 **448.0** = FP8 E4M3 maxval → 일부 block이 saturation 범위에 해당.

---

## 오차 분석

### Input (x)

| 지표 | 값 |
|------|----|
| MSE | `4.907e-3` |
| SNR | **21.5 dB** |
| Abs error mean | 0.0444 |
| Abs error max | 0.5859 |
| Rel error mean | 0.369 |
| Rel error max | 1.000 |

### Weight (W)

| 지표 | 값 |
|------|----|
| MSE | `1.091e-5` |
| SNR | **20.3 dB** |
| Abs error mean | 0.00247 |
| Abs error max | 0.02734 |
| Rel error mean | 0.169 |
| Rel error max | 1.000 |

### Output (y = x @ W.T)

| 지표 | 값 |
|------|----|
| FP output 범위 | −7.50 ~ 7.66 |
| Quant output 범위 | −7.66 ~ 7.69 |
| MSE | `1.421e-2` |
| SNR | **23.4 dB** |
| Abs error mean | 0.0944 |
| Abs error max | 0.6602 |
| Rel error mean | 0.598 |

---

## 시각화

`nvfp4_comparison.png` — 3×4 레이아웃 (행: input / weight / output):

| Col | 내용 |
|-----|------|
| 1 | FP vs Dequant 값 분포 히스토그램 |
| 2 | 오차 분포 (FP − Dequant) |
| 3 | FP vs Dequant scatter plot |
| 4 | 절대오차 프로파일 (input/weight: 원소별, output: 토큰별) |

---

## 주요 관찰

- **Weight quantization quality > Input quantization quality**: weight의 값 범위(−0.34~0.22)가 input(−6.8~15.1)보다 훨씬 좁고 분포가 균일 → per-block scale이 더 정밀하게 동작.
- **Input rel_error mean 0.37**: activation range가 불규칙(outlier 존재)하여 일부 블록의 작은 값들이 FP4 표현 한계로 손실됨. per-block scale 최대값이 448.0(FP8 E4M3 maxval)에 도달한 블록 존재.
- **Output SNR 23.4 dB**: input과 weight 오차가 matmul에서 부분적으로 상쇄되어 오히려 개별 SNR(21.5, 20.3 dB)보다 높게 나옴.
- **Output rel_error max 7154**: 절대값이 0에 가까운 output 원소에서 상대오차 폭발 → 절대오차 기준(mean 0.094, max 0.66)으로 평가하는 것이 더 적절.
- **SVD correction 없음**: NVFP4_DEFAULT_CFG는 max calibration만 사용, low-rank 오차 보정 없음.

---

# 4-Way NVFP4 Comparison (Duhyeon vs MTQ vs eval_mixed)

## 비교 대상

| ID | 출처 | 핵심 방식 |
|----|------|----------|
| **M1/M2** | `eval_nvfp4_mtq.py` / `nvfp4_single_layer.py` | ModelOpt `NVFP4QTensor` (MTQ 내부 경로와 동일) |
| **M3** | `eval_mixed_fp_quant.py` `_quantize_nvfp4` | 직접 구현 (custom) |
| **M4** | Duhyeon `compare.py` `mx_block_quant` | 직접 구현 (custom) |

---

## 구현 차이 비교

### Scale 구조

| | M1/M2 (MTQ) | M3 (eval_mixed) | M4 (Duhyeon) |
|---|---|---|---|
| **구조** | **two-level** | **single-level** | **two-level** |
| **Global scale** | `amax / (6.0 × 448.0)` → float32 | 없음 | `amax / (6.0 × 448.0)` → float32 |
| **Per-block scale** | `block_amax / (6.0 × global)` → **FP8 E4M3fn** | `amax / 6.0` → **FP16** | `block_amax / (6.0 × global)` → **FP8 E4M3fn** |
| **Block size** | 16 | 16 | 16 |

### FP4 값 결정 방식

| | M1/M2 (MTQ) | M3 (eval_mixed) | M4 (Duhyeon) |
|---|---|---|---|
| **방법** | `searchsorted(e2m1_bounds)` + odd-index tie-breaking | `_snap_to_grid` nearest neighbor | log2 기반 `_fp_quant` (normal + subnormal) |
| **Tie-break** | 경계 `[0.75, 1.75, 2.5]`에서 올림 처리 | 단순 거리 비교 | 없음 (round 기반) |
| **FP4 값 집합** | `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` | 동일 | 동일 (E2M1 normal+subnormal) |

### Activation Quantization

| | M1/M2 (MTQ) | M3 (eval_mixed) | M4 (Duhyeon) |
|---|---|---|---|
| **적용** | weight와 동일 경로 (`NVFP4QTensor`) | **별도 함수** `quant_act_fp(x, "NVFP4")` | weight와 동일 경로 (`mx_block_quant`) |
| **scale_dtype** | FP8 E4M3fn | **FP16** | FP8 E4M3fn |

---

## 수치 결과 (vs FP baseline)

| Method | MSE | SNR (dB) | abs_err mean | abs_err max |
|--------|-----|----------|-------------|-------------|
| M1/M2: MTQ / nvfp4_single_layer | `1.4234e-2` | 23.38 | 0.09447 | 0.66016 |
| M3: eval_mixed_fp_quant | `1.3301e-2` | **23.67** | **0.09145** | **0.59375** |
| M4: Duhyeon mx_block_quant | `1.4234e-2` | 23.38 | 0.09447 | 0.66016 |

### Method 간 직접 차이

| 쌍 | MSE | SNR (dB) | 해석 |
|----|-----|----------|------|
| M1/M2 vs M4 (MTQ vs Duhyeon) | `1.79e-9` | **92.4** | 사실상 동일 — 다른 코드, 같은 알고리즘 |
| M1/M2 vs M3 (MTQ vs mixed) | `7.76e-3` | 26.0 | 명확히 다름 — scale 구조 차이 |
| M3 vs M4 (mixed vs Duhyeon) | `7.76e-3` | 26.0 | 명확히 다름 — scale 구조 차이 |

---

## 결론

- **M1/M2 ≡ M4**: MTQ(`NVFP4QTensor`)와 Duhyeon(`mx_block_quant`)은 구현 코드는 다르지만 two-level scaling (global float32 + per-block FP8 E4M3fn) + E2M1 quantization 로직이 동일하여 SNR 92.4 dB로 사실상 같은 결과.
- **M3 차이 원인**: `eval_mixed_fp_quant`는 single-level FP16 scale만 사용하여 scale 구조가 다름. 이 레이어에서는 MSE 기준 미세하게 유리(`1.330e-2` vs `1.423e-2`)하나, global scale 없이 per-block FP16 단일 스케일만 사용하는 구조적 차이가 있음.
- **내가 사용할 quant (`mtq.quantize(policy, NVFP4_DEFAULT_CFG`)**: M1/M2 경로와 동일. single layer + single forward pass에서 `NVFP4QTensor.quantize()`는 MTQ calibration과 같은 amax를 계산하므로 수치적으로 완전히 일치.

---

## 파일 목록

| 파일 | 설명 |
|------|------|
| `duhyeon/input_weight_captured.pt` | x, W 원본 텐서 |
| `duhyeon/output_compare_dh.pt` | x, W, y_fp, y_fq_dh (Duhyeon 결과) |
| `compare_result.pt` | y_fp, y_fq_dh, y_mine 2-way 비교 결과 |
| `compare_4methods.py` | 4-way 비교 스크립트 |
| `nvfp4_comparison_w_duhyeon.png` | M1/M2 vs M4 2-way 비교 그래프 |
| `nvfp4_4methods_comparison.png` | M1/M2, M3, M4 4-way 비교 그래프 |
