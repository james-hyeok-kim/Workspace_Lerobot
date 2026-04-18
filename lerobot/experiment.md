# Quantization Experiments — LeRobot pi0.5 (LIBERO)

> 모델: `lerobot/pi05_libero_finetuned` (BF16, ~4605 MB)  
> 벤치마크: LIBERO 4 tasks × 50 episodes (spatial / object / goal / libero_10)  
> 스크립트: `eval_mixed_int_quant.py`, `eval_mixed_fp_quant.py`  
> 결과 경로: `logs/mixed_int_quant/`, `logs/mixed_fp_quant/`

---

## 1. 모델 구조

| Component | 역할 | Linear layers | BF16 size |
|-----------|------|--------------|-----------|
| LM (PaliGemma backbone) | language / vision | 127 | ~3,785 MB |
| DiT (Gemma Expert) | diffusion action head | 167 | ~820 MB |
| **합계 (quantize 대상)** | | **294** | **~4,605 MB** |
| Vision tower / embed_tokens / lm_head | skip | 164 | - |

**Quantize 스킵 이유**: vision_tower는 image encoder로 별도 precision 유지, embed_tokens/lm_head는 vocabulary lookup이라 quantization 효과 낮음.

---

## 2. 공통 설계 원칙

### 민감도 기반 bit/format 배정

모든 실험에서 **per-layer 통계**를 사용:

| 통계 | 의미 | 계산 대상 |
|------|------|----------|
| Kurtosis | 분포의 뾰족함 (outlier 존재 여부) | Weight 행렬 전체 |
| CV (Coefficient of Variation) | absmax의 채널 간 변동성 | Weight: per-channel, Act: per-token |

높은 kurtosis / 높은 CV → outlier 많음 → 높은 bit 필요 → INT8 / MXFP8 배정  
낮은 kurtosis / 낮은 CV → 부드러운 분포 → 낮은 bit 가능 → INT3 / FP3 배정

통계 파일: `logs/dist_analysis_v4/dist_stats.json`

### 배정 임계값 (INT / FP 공통)

| Bit group | Weight kurt | Weight CV | Act kurt | Act CV |
|-----------|------------|-----------|---------|--------|
| 8-bit | >50 | >0.7 | >50 | >0.3 |
| 6-bit | >10 | >0.45 | >10 | >0.15 |
| 4-bit | >3 | >0.25 | >3 | >0.08 |
| 3-bit↓ | else | else | else | else |

### Block size 배정

| 조건 | Block size |
|------|-----------|
| CV_w > 0.5 또는 CV_act > 0.5 | 16 |
| CV_w > 0.25 또는 CV_act > 0.25 | 32 |
| else | 64 |

### Block size 분포 (294 layers 기준)

| Block size | 레이어 수 |
|-----------|---------|
| 16 | 59 |
| 32 | 83 |
| 64 | 152 |

---

## 3. 실험 1 — Mixed INT Quantization

**스크립트:** `eval_mixed_int_quant.py`  
**결과 CSV:** `logs/mixed_int_quant/final_comparison_9configs_wide.csv`  
**per-layer OPs 분석:** `logs/mixed_int_quant/per_layer_ops/`

### Quantization 방식

- **Weight**: block-wise symmetric INT quantization, INT32 scale
- **Activation**: per-block symmetric INT quantization (선택적)
- **Weight-only (WO)**: activation kurtosis/CV 통계 없는 레이어는 act quant 스킵

### 실험 Configuration (9개)

| # | Config key | target | LM min | DiT min | Act quant |
|---|-----------|--------|--------|---------|-----------|
| 1 | `lm8/dit8 INT16` | all | INT8 | INT8 | INT16 (강제) |
| 2 | `lm6/dit4` | all | INT6 | INT4 | natural (통계 기반) |
| 3 | `lm1/dit1` | all | INT1 | INT1 | natural |
| 4 | `lm8 INT16` | lm only | INT8 | - | INT16 (강제) |
| 5 | `lm6` | lm only | INT6 | - | natural |
| 6 | `lm1` | lm only | INT1 | - | natural |
| 7 | `dit8 INT16` | dit only | - | INT8 | INT16 (강제) |
| 8 | `dit4` | dit only | - | INT4 | natural |
| 9 | `dit1` | dit only | - | INT1 | natural |

### 결과 (50ep)

| Config | target | sp% | ob% | go% | l10% | avg% | avg_w | avg_a | qMB | x_BF16 | mem% | ops%* |
|--------|--------|-----|-----|-----|------|------|-------|-------|-----|--------|------|-------|
| INT8+INT16a all | all | 94 | 92 | 96 | 68 | **87.5** | 8.00 | 16.00 | 2667 | 1.73x | 42.1 | 87.5 |
| INT6/4 all | all | 94 | 82 | 94 | 50 | 80.0 | 5.16 | 6.64 | 2022 | 2.28x | 56.1 | 89.7 |
| INT1 all | all | 80 | 42 | 96 | 22 | 60.0 | 3.57 | 6.64 | 1356 | 3.39x | 70.5 | 93.5 |
| INT8+INT16a lm | lm | 98 | 94 | 94 | 68 | **88.5** | 8.00 | 16.00 | 2209 | 1.71x | 41.6 | 87.5 |
| INT6 lm | lm | 98 | 92 | 98 | 62 | **87.5** | 6.02 | 7.46 | 1737 | 2.18x | 54.1 | 89.2 |
| INT1 lm | lm | 90 | 72 | 82 | 70 | 78.5 | 3.46 | 7.46 | 1113 | 3.40x | 70.6 | 93.6 |
| INT8+INT16a dit | dit | 94 | 88 | 96 | 76 | **88.5** | 8.00 | 16.00 | 458 | 1.79x | 44.1 | 87.5 |
| INT4 dit | dit | 94 | 84 | 96 | 38 | 78.0 | 4.51 | 5.98 | 286 | 2.87x | 65.2 | 92.1 |
| INT1 dit | dit | 94 | 60 | 98 | 24 | 69.0 | 3.65 | 5.98 | 243 | 3.37x | 70.3 | 92.8 |

*ops% = `1 - max(w_bits, a_bits)/8` 기준 (INT8=1.0)

### 주요 관찰
- **LM >> DiT** 민감도: target=dit 단독 quant는 accuracy 거의 유지 (DiT=18% of total)
- **INT6 lm only** best tradeoff: 87.5% avg, 2.18x 압축, acc 손실 거의 없음
- **libero_10** (long-horizon, 10 task suite)이 가장 취약

---

## 4. 실험 2 — Mixed FP Quantization

**스크립트:** `eval_mixed_fp_quant.py`  
**SVD correction:** 없음  
**결과 경로:** `logs/mixed_fp_quant/`  
**전체 비교 CSV:** `logs/mixed_fp_quant/fp_vs_int_comparison_full.csv`

### FP Format Pool (9종)

| Bit group | Format | 표현 값 집합 | Scale 방식 |
|-----------|--------|------------|-----------|
| 8-bit | **MXFP8** E4M3 | non-uniform FP (OCP MX spec) | E8M0 per block (8-bit) |
| 8-bit | NVFP8 E4M3 | non-uniform FP | FP16 per block (16-bit) |
| 6-bit | **MXFP6_E2M3** | E2M3, max≈7.5 | E8M0 per block |
| 6-bit | MXFP6_E3M2 | E3M2, max≈28 | E8M0 per block |
| 4-bit | **NVFP4** | {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} | FP16 per block |
| 4-bit | MXFP4 E2M1 | uniform E2M1 | E8M0 per block |
| 3-bit | **FP3_E1M1** | {0, 1, 2, 3} | FP16 per block |
| 3-bit | FP3_E2M0 | {0, 1, 2, 4} | FP16 per block |
| 3-bit | FP3_E0M2 | {0, 0.25, 0.5, 0.75} | FP16 per block |

Bold = 각 bit-group의 default sub-format.

**INT vs FP 차이**: INT는 균등 간격(linear grid), FP는 비균등(log-like, 0 근처 정밀). 같은 bit수에서 FP가 bell-curve 분포 weight에 이론상 유리.

### Quantization 방식

- **Weight**: QUANT_FNS[fmt](W_fp32, block_size, scale_dtype) → dequant BF16 저장 (simulate, no packing)
- **Activation**: 동일 QUANT_FNS로 동적 quantize → 결과를 원본 dtype으로 복원
- **Weight-only (WO)**: activation 통계 없는 레이어 (NaN stats) → act quant 스킵 (기본)
- **force_act_quant**: NaN-stats 레이어도 min format으로 강제 act quant

### 실험 Configuration

| # | Config key | lm_min | dit_min | fmt_3bit | force_act | 설명 |
|---|-----------|--------|---------|----------|-----------|------|
| 1 | `mxfp8_all_no_fa` | 8 | 8 | - | ✗ | MXFP8 all, WO 레이어 유지 |
| 2 | `nvfp8_lm8_dit8_fa` | 8 | 8 | - | ✓ | NVFP8, 강제 act quant |
| 3 | `mxfp8_lm8_dit8_fa` | 8 | 8 | - | ✓ | MXFP8, 강제 act quant |
| 4 | `mxfp8_lm8_dit4_fa` | 8 | 4 | - | ✓ | LM=8bit, DiT=4bit mix |
| 5 | `mxfp8_lm6_dit4_fa` | 6 | 4 | - | ✓ | LM=6bit, DiT=4bit mix |
| 6 | `fp3e1m1_lm3_dit3_fa` | 3 | 3 | FP3_E1M1 | ✓ | 3-bit {0,1,2,3} |
| 7 | `fp3e2m0_lm3_dit3_fa` | 3 | 3 | FP3_E2M0 | ✓ | 3-bit {0,1,2,4} ★ |
| 8 | `fp3e0m2_lm3_dit3_fa` | 3 | 3 | FP3_E0M2 | ✓ | 3-bit {0,0.25,0.5,0.75} |

### CLI 사용법

```bash
# MXFP8 all, WO 레이어 유지 (no force_act)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_mixed_fp_quant.py \
    --task libero_10 --n_episodes 50 --batch_size 5 \
    --target all --lm_min_bits 8 --dit_min_bits 8 \
    --scale_dtype FP16 --output_dir logs/mixed_fp_quant/mxfp8_all

# FP3_E2M0, 강제 act quant
python eval_mixed_fp_quant.py \
    --task libero_10 --n_episodes 50 --batch_size 5 \
    --target all --lm_min_bits 3 --dit_min_bits 3 \
    --fmt_3bit FP3_E2M0 --force_act_quant \
    --scale_dtype FP16 --output_dir logs/mixed_fp_quant/fp3e2m0_all

# Sub-format 선택 가능 인자
# --fmt_8bit {MXFP8, NVFP8}
# --fmt_6bit {MXFP6_E2M3, MXFP6_E3M2}
# --fmt_4bit {NVFP4, MXFP4}
# --fmt_3bit {FP3_E1M1, FP3_E2M0, FP3_E0M2}
```

### 결과 (50ep, target=all)

| # | Config | sp% | ob% | go% | l10% | avg% | avg_w | avg_a | qMB | x_BF16 | mem% | ops%* |
|---|--------|-----|-----|-----|------|------|-------|-------|-----|--------|------|-------|
| 1 | MXFP8 all (no fa) | 86 | 66 | 88 | 54 | **73.5** | 8.00 | 7.71 | 2394 | 1.92x | 48.0 | 47.5 |
| 2 | NVFP8 lm8/dit8+fa | 90 | 66 | 82 | 52 | 72.5 | 8.00 | 7.07 | 2485 | 1.85x | 46.0 | 50.0 |
| 3 | MXFP8 lm8/dit8+fa | 88 | 68 | 76 | 44 | 69.0 | 8.00 | 7.07 | 2394 | 1.92x | 48.0 | 50.0 |
| 4 | MXFP8 lm8/dit4+fa | 86 | 60 | 74 | 42 | 65.5 | 7.40 | 7.07 | 2228 | 2.07x | 51.6 | 51.9 |
| 5 | MXFP8 lm6/dit4+fa | 86 | 54 | 84 | 36 | 65.0 | 5.76 | 7.07 | 1756 | 2.62x | 61.9 | 54.8 |
| 6 | FP3_E1M1 lm3/dit3+fa | 70 | 6 | 52 | 10 | 34.5 | 3.47 | 7.07 | 1176 | 3.92x | 74.5 | 54.8 |
| 7 | **FP3_E2M0 lm3/dit3+fa** | **90** | **62** | **76** | **62** | **72.5** | 3.47 | 7.07 | 1176 | **3.92x** | **74.5** | 54.8 |
| 8 | FP3_E0M2 lm3/dit3+fa | 66 | 6 | 50 | 4 | 31.5 | 3.47 | 7.07 | 1176 | 3.92x | 74.5 | 54.8 |

*ops% = `1 - max(w_bits, a_bits)/16` 기준 (BF16=1.0)

---

## 5. INT vs FP 통합 비교

| Config | fmt | sp% | ob% | go% | l10% | avg% | avg_w | avg_a | qMB | x_BF16 | mem% |
|--------|-----|-----|-----|-----|------|------|-------|-------|-----|--------|------|
| BF16 (baseline) | - | ~98 | ~96 | ~98 | ~76 | ~92 | 16.0 | 16.0 | 4605 | 1.00x | 0 |
| **INT8+INT16a lm** | INT | 98 | 94 | 94 | 68 | **88.5** | 8.00 | 16.00 | 2209 | 1.71x | 41.6 |
| **INT6 lm** | INT | 98 | 92 | 98 | 62 | **87.5** | 6.02 | 7.46 | 1737 | 2.18x | 54.1 |
| INT8+INT16a all | INT | 94 | 92 | 96 | 68 | 87.5 | 8.00 | 16.00 | 2667 | 1.73x | 42.1 |
| INT6/4 all | INT | 94 | 82 | 94 | 50 | 80.0 | 5.16 | 6.64 | 2022 | 2.28x | 56.1 |
| INT1 lm | INT | 90 | 72 | 82 | 70 | 78.5 | 3.46 | 7.46 | 1113 | 3.40x | 70.6 |
| INT1 all | INT | 80 | 42 | 96 | 22 | 60.0 | 3.57 | 6.64 | 1356 | 3.39x | 70.5 |
| MXFP8 all (no fa) | FP | 86 | 66 | 88 | 54 | 73.5 | 8.00 | 7.71 | 2394 | 1.92x | 48.0 |
| NVFP8 lm8/dit8+fa | FP | 90 | 66 | 82 | 52 | 72.5 | 8.00 | 7.07 | 2485 | 1.85x | 46.0 |
| **FP3_E2M0 lm3/dit3+fa** | FP | 90 | 62 | 76 | 62 | **72.5** | 3.47 | 7.07 | 1176 | **3.92x** | **74.5** |
| MXFP8 lm6/dit4+fa | FP | 86 | 54 | 84 | 36 | 65.0 | 5.76 | 7.07 | 1756 | 2.62x | 61.9 |

---

## 6. 주요 발견 및 메모

### Accuracy
- **INT 우위**: INT8+INT16act (87.5%) > MXFP8 (73.5%). activation을 16-bit로 유지하는 INT 방식이 FP8 act quant보다 크게 유리.
- **FP3_E2M0 이례적 성능**: avg 72.5%로 3-bit임에도 MXFP8(73.5%)에 근접. {0,1,2,4} 지수 표현이 range를 확보하면서 중요 weight를 살림.
- **FP3_E1M1/E0M2 발산**: E1M1={0,1,2,3}은 정수값만 → 작은 weight 소실. E0M2={0,0.25,0.5,0.75}는 max=0.75로 범위 부족 → libero_object/l10 0~6%.
- **LM 민감도 >> DiT**: LM만 quantize하면 accuracy 유지, DiT만 quantize하면 거의 영향 없음. LM이 전체 accuracy의 병목.

### Compression
- **best accuracy tradeoff**: INT6 lm only — 87.5% avg, 2.18x 압축 (LM만 quantize)
- **best compression**: FP3_E2M0 lm3/dit3 — 3.92x, 74.5% mem 절감, avg 72.5%

### force_act_quant 주의
- NaN-stats 레이어(특수 projection)를 강제 quantize 시 accuracy 하락
- MXFP8 lm8/dit8: force 없이 73.5% → force 있으면 69.0% (-4.5%p)
- WO 레이어는 그대로 두는 것이 안전

### libero_object 취약
- FP/INT activation quantize 시 특히 큰 drop (일부 config에서 -30%p 이상)
- object recognition 특성상 activation precision에 민감

### NVFP8 vs MXFP8
- libero_spatial: NVFP8(90%) > MXFP8(88%), but libero_10: NVFP8(52%) < MXFP8(54%)
- 전체 avg 거의 동일 (72.5% vs 73.5%), scale precision 차이(FP16 vs E8M0)의 영향 미미

---

## 7. 파일 구조

```
logs/
├── dist_analysis_v4/
│   └── dist_stats.json                    # per-layer kurtosis/CV 통계
├── mixed_int_quant/
│   ├── final_comparison_9configs_wide.csv # INT 9-config 비교표
│   ├── per_layer_ops/                     # config별 per-layer OPs CSV (9개)
│   └── component_sweep/                   # task별 JSON 결과
└── mixed_fp_quant/
    ├── fp_vs_int_comparison_full.csv      # INT+FP 통합 비교표 ★
    ├── mxfp8_all_50ep/                    # config #1 결과
    └── sweep_50ep/                        # config #2~8 결과
        ├── nvfp8_lm8_dit8/
        ├── mxfp8_lm8_dit8_fa/
        ├── mxfp8_lm8_dit4_fa/
        ├── mxfp8_lm6_dit4_fa/
        ├── fp3e1m1_lm3_dit3_fa/
        ├── fp3e2m0_lm3_dit3_fa/
        └── fp3e0m2_lm3_dit3_fa/
```
