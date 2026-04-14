# Quantization Experiments — LeRobot pi0.5 (LIBERO)

> 모델: `lerobot/pi05_libero_finetuned` (BF16, ~4605 MB)  
> 벤치마크: LIBERO 4 tasks × 50 episodes  
> 스크립트: `eval_mixed_int_quant.py`, `eval_mixed_fp_quant.py`  
> 결과 경로: `logs/mixed_int_quant/`, `logs/mixed_fp_quant/`

---

## 모델 구조

| Component | 역할 | 레이어 수 | BF16 size |
|-----------|------|----------|-----------|
| LM (PaliGemma) | language/vision backbone | 127 Linear | ~3785 MB |
| DiT (Gemma Expert) | diffusion action head | 167 Linear | ~820 MB |
| 합계 (quantized 대상) | - | 294 Linear | ~4605 MB |
| Vision tower / embed / lm_head | skip (quantize 안 함) | 164 Linear | - |

---

## 실험 1 — Mixed INT Quantization

**스크립트:** `eval_mixed_int_quant.py`  
**방식:** per-layer kurtosis/CV → INT bit-width 자동 배정 (INT1~INT16)  
**Activation:** per-block symmetric INT quantization  
**결과 CSV:** `logs/mixed_int_quant/final_comparison_9configs_wide.csv`

### 배정 임계값

| Bit | Weight kurtosis | Weight CV | Act kurtosis | Act CV |
|-----|----------------|-----------|--------------|--------|
| INT8 | >50 | >0.7 | >50 | >0.3 |
| INT6 | >10 | >0.45 | >10 | >0.15 |
| INT4 | >3 | >0.25 | >3 | >0.08 |
| INT3↓ | else | else | else | else |

### 9-Config 결과 (50ep)

| Config | target | sp% | ob% | go% | l10% | avg% | avg_w | avg_a | qMB | x_BF16 | ops_save |
|--------|--------|-----|-----|-----|------|------|-------|-------|-----|--------|---------|
| INT8+INT16a all | all | 94 | 92 | 96 | 68 | **87.5** | 8.00 | 16.00 | 2667 | 1.73x | 87.5% |
| INT6/4 all | all | 94 | 82 | 94 | 50 | 80.0 | 5.16 | 6.64 | 2022 | 2.28x | 89.7% |
| INT1 all | all | 80 | 42 | 96 | 22 | 60.0 | 3.57 | 6.64 | 1356 | 3.39x | 93.5% |
| INT8+INT16a lm | lm | 98 | 94 | 94 | 68 | **88.5** | 8.00 | 16.00 | 2209 | 1.71x | 87.5% |
| INT6 lm | lm | 98 | 92 | 98 | 62 | 87.5 | 6.02 | 7.46 | 1737 | 2.18x | 89.2% |
| INT1 lm | lm | 90 | 72 | 82 | 70 | 78.5 | 3.46 | 7.46 | 1113 | 3.40x | 93.6% |
| INT8+INT16a dit | dit | 94 | 88 | 96 | 76 | **88.5** | 8.00 | 16.00 | 458 | 1.79x | 87.5% |
| INT4 dit | dit | 94 | 84 | 96 | 38 | 78.0 | 4.51 | 5.98 | 286 | 2.87x | 92.1% |
| INT1 dit | dit | 94 | 60 | 98 | 24 | 69.0 | 3.65 | 5.98 | 243 | 3.37x | 92.8% |

### 주요 관찰
- LM이 DiT보다 정확도에 훨씬 민감 (target=dit quant는 대부분 영향 적음)
- libero_10 (long-horizon)이 quantization에 가장 취약
- INT6 lm only (target=lm)에서 87.5% — BF16 대비 거의 손실 없이 2.18x 압축

---

## 실험 2 — Mixed FP Quantization

**스크립트:** `eval_mixed_fp_quant.py`  
**방식:** per-layer kurtosis/CV → FP format 자동 배정 (동일 임계값, bit group → sub-format 선택)  
**SVD:** 없음  
**결과 경로:** `logs/mixed_fp_quant/`

### FP Format Pool

| Bit group | Sub-format | 표현값 | Scale |
|-----------|-----------|--------|-------|
| 8-bit | **MXFP8** (E4M3) | non-uniform FP | E8M0 (8-bit) |
| 8-bit | NVFP8 (E4M3) | non-uniform FP | FP16 (16-bit) |
| 6-bit | **MXFP6_E2M3** | E2M3, max=7.5 | E8M0 |
| 6-bit | MXFP6_E3M2 | E3M2, max=28 | E8M0 |
| 4-bit | **NVFP4** | {0,±0.5,...,±6} | FP16 |
| 4-bit | MXFP4 | E2M1 | E8M0 |
| 3-bit | **FP3_E1M1** | {0,1,2,3} | FP16 |
| 3-bit | FP3_E2M0 | {0,1,2,4} | FP16 |
| 3-bit | FP3_E0M2 | {0,0.25,0.5,0.75} | FP16 |

Bold = 각 bit-group의 default sub-format.

### CLI 주요 인자

```bash
--target {all,lm,dit}
--lm_min_bits {3,4,6,8}      # LM weight 최소 bit group
--dit_min_bits {3,4,6,8}     # DiT weight 최소 bit group
--fmt_8bit {MXFP8,NVFP8}
--fmt_6bit {MXFP6_E2M3,MXFP6_E3M2}
--fmt_4bit {NVFP4,MXFP4}
--fmt_3bit {FP3_E1M1,FP3_E2M0,FP3_E0M2}
--force_act_quant             # NaN-stats 레이어도 강제 act quant
--scale_dtype {FP16,BF16,FP32,NVFP8,MXFP8}
```

### 결과 (50ep, all target)

| Config | sp% | ob% | go% | l10% | avg% | avg_w | avg_a | qMB | x_BF16 | ops_save |
|--------|-----|-----|-----|------|------|-------|-------|-----|--------|---------|
| MXFP8 all (no force_act) | 86 | 66 | 88 | 54 | **73.5** | 8.00 | 7.71 | 2394 | 1.92x | 47.5% |
| NVFP8 all+fa | 진행중 | | | | | 8.00 | 7.07 | 2485 | 1.85x | 50.0% |
| MXFP8 lm8/dit8+fa | 진행중 | | | | | 8.00 | 7.07 | 2394 | 1.92x | 50.0% |
| MXFP8 lm8/dit4+fa | 진행중 | | | | | 7.40 | 7.07 | 2228 | 2.07x | 51.9% |
| MXFP8 lm6/dit4+fa | 진행중 | | | | | 5.76 | 7.07 | 1756 | 2.62x | 54.8% |
| FP3_E1M1 lm3/dit3+fa | 진행중 | | | | | 3.47 | 7.07 | 1176 | 3.92x | 54.8% |
| FP3_E2M0 lm3/dit3+fa | 진행중 | | | | | 3.47 | 7.07 | 1176 | 3.92x | 54.8% |
| FP3_E0M2 lm3/dit3+fa | 진행중 | | | | | 3.47 | 7.07 | 1176 | 3.92x | 54.8% |

> 50ep 결과 완료 후 업데이트 예정: `logs/mixed_fp_quant/fp_vs_int_comparison_full.csv`

### 5ep 사전 관찰 (libero_10)

| Config | l10% (5ep) | 비고 |
|--------|-----------|------|
| MXFP8 all (no fa) | 80% | |
| NVFP8 all+fa | 40% | NaN/Inf 불안정 경고 |
| MXFP8 lm8/dit8+fa | 40% | force_act가 drop 유발 |
| MXFP8 lm8/dit4+fa | 40% | |
| MXFP8 lm6/dit4+fa | 60% | |
| FP3_E1M1 lm3/dit3+fa | 0% | 표현 범위 부족 |
| FP3_E2M0 lm3/dit3+fa | 60% | 3-bit 중 유일하게 생존 |
| FP3_E0M2 lm3/dit3+fa | 0% | E0M2={0,0.25,0.5,0.75} 범위 너무 좁음 |

### INT vs FP 비교 (8-bit 기준)

| | INT8+INT16a | MXFP8 (no fa) |
|-|-------------|--------------|
| avg accuracy (4 tasks) | **87.5%** | 73.5% |
| memory 절감 | 42% | **48%** | 
| ops 절감\* | 87.5% | **47.5%** |
| avg_a_bits | 16.0 | **7.71** |

\* 계산 기준 다름: INT ops = `max(w,a)/8`, FP ops = `max(w,a)/16` (BF16=1.0)

---

## 실험 메모

- **force_act_quant 주의**: NaN-stats 레이어 (주로 특수 projection) 강제 quantize 시 accuracy 크게 하락. WO(weight-only)가 더 안전.
- **libero_object 취약**: FP activation quantization 적용 시 특히 큰 drop (-26%). object recognition 특성상 activation precision에 민감.
- **DiT 영향 제한적**: DiT는 820MB로 전체의 18%. target=dit quant만으로는 전체 accuracy 거의 유지.
- **FP3_E0M2**: max값 0.75로 표현 범위 너무 좁아 weight 왜곡 → 0% 발산.
- **NVFP4**: 비균등 분포라 bell-curve 분포 weight에 INT4보다 이론상 유리하나, activation 적용 시 불안정.
