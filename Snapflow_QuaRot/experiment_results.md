# Snapflow_QuaRot Experiment Results

**Model**: pi0.5 (PaliGemma VLM + Gemma-300M action expert)  
**Benchmark**: LIBERO-10 (10 tasks × 10 episodes, seed=1000)  
**Path A**: QuaRot baked (R1+R3 Hadamard), NFE=1, FP16 — 100%

---

## 0. 양자화 포맷 상세 스펙

모든 실험은 **weight-only** (W4A16) 또는 **W4A4** 기준. Path A 자체는 양자화 없음 (FP16).

### Weight Quantization 포맷별 스펙

| 포맷 | Weight bits | Weight 표현 | Activation bits | Scale 포맷 | Scale 레벨 수 |
|------|------------|------------|----------------|-----------|-------------|
| Path A (기준) | FP16 | 연속 | FP16 | — | — |
| INT4 (FP16 scale) | 4-bit INT | 16개 균일 | FP16 | **FP16** | ~65536 |
| INT4 (INT16 scale) | 4-bit INT | 16개 균일 | FP16 | **INT16** | 65536 (정수) |
| NVFP4 E2M1 | 4-bit FP | 9개 양수 | FP16 | **FP8 E4M3** | ~448 |
| W4A4 | 4-bit INT | 16개 균일 | **4-bit INT** | FP16 | ~65536 |

### NVFP4 Scale 상세 (FP8 E4M3)

NVFP4 block scale은 `scale_bits=(4,3)` → **FP8 E4M3** (4 exponent bits, 3 mantissa bits, 8-bit 총).

```python
# modelopt tensor_quant.py
(4, 3): "E4M3"   # FP8 E4M3, max=448
```

- NVFP4 weight: E2M1 (4-bit), 9개 양수 magnitude
- NVFP4 scale: FP8 E4M3 (8-bit), ~448개 표현값
- → **double quantization**: weight 오류 + scale 오류 중첩

### Arithmetic 비교

| 포맷 | Weight GEMM | Scale 적용 | Arithmetic 타입 |
|------|------------|-----------|---------------|
| INT4 FP16 scale | INT4 × scale | FP16 dequant | FP×FP |
| INT4 INT16 scale | INT4 × scale | INT16 dequant | INT×INT |
| NVFP4 | E2M1 × scale | FP8 dequant | FP×FP |
| W4A4 | INT4 × INT4 | FP16 scale | INT×INT |

---

## 1. 전체 결과 요약

| 방법 | Weight | Activation | Scale | NFE | pc_success | Path A 대비 |
|------|--------|-----------|-------|-----|-----------|------------|
| FP16 (NFE=10) | FP16 | FP16 | — | 10 | 92.4% | — |
| **Path A** | FP16 | FP16 | — | 1 | **100.0%** | 기준 |
| INT4 PTQ g=16 | INT4 | FP16 | FP16 | 1 | 44.0% | -56%p |
| INT4 PTQ g=8 | INT4 | FP16 | FP16 | 1 | 71.0% | -29%p |
| INT4 PTQ g=4 | INT4 | FP16 | FP16 | 1 | 82.0% | -18%p |
| NVFP4 PTQ b=4 | E2M1 | FP16 | FP8 E4M3 | 1 | 21.0% | -79%p |
| NVFP4 PTQ b=8 | E2M1 | FP16 | FP8 E4M3 | 1 | 19.0% | -81%p |
| NVFP4 PTQ b=16 | E2M1 | FP16 | FP8 E4M3 | 1 | 21.0% | -79%p |
| INT4 QAD g=16, FP16 scale | INT4 | FP16 | FP16 | 1 | 86.0% | -14%p |
| INT4 QAD g=16, INT16 scale | INT4 | FP16 | INT16 | 1 | 85.0% | -15%p |
| INT4 QAD g=8, INT16 scale | INT4 | FP16 | INT16 | 1 | 92.0% | -8%p |
| **INT4 QAD g=8, FP16 scale** | INT4 | FP16 | FP16 | 1 | **97.0%** | **-3%p** |
| NVFP4 QAD b=8 | E2M1 | FP16 | FP8 E4M3 | 1 | 79.0% | -21%p |
| W4A4 QAD g=8 | INT4 | **INT4** | FP16 | 1 | **93.0%** | -7%p |

---

## 2. INT4 PTQ — Group Size별 비교 (W4A16, FP16 scale)

| Group Size | Weight | Activation | pc_success |
|-----------|--------|-----------|-----------|
| g=16 | INT4 | FP16 | 44.0% |
| g=8 | INT4 | FP16 | 71.0% |
| g=4 | INT4 | FP16 | 82.0% |

Group size가 작을수록 정밀도 향상 (scale당 커버하는 weight 수 감소). g=4가 PTQ 최고 성능.

---

## 3. NVFP4 PTQ — Block Size별 비교 (W4A16, FP8 E4M3 scale)

| Block Size | Weight | Activation | Scale | pc_success |
|-----------|--------|-----------|-------|-----------|
| b=4 | E2M1 | FP16 | FP8 E4M3 | 21.0% |
| b=8 | E2M1 | FP16 | FP8 E4M3 | 19.0% |
| b=16 | E2M1 | FP16 | FP8 E4M3 | 21.0% |

Block size 무관하게 19-21%. INT4 대비 크게 열위한 원인:
1. E2M1 유효 레벨 9개 (INT4 16개 대비 절반)
2. FP8 scale (~448 레벨) → INT4의 FP16 scale(65536 레벨) 대비 double quant error 추가
3. Log-spaced 간격이 Gaussian weight 분포에 비효율

---

## 4. Ablation Study — INT4 g=4, 오류 원인 분석

Path A에서 INT4 g=4 (W4A16) 적용 시 100% → 82% (-18%p).

| Ablation Mode | 설명 | pc_success | 감소 |
|--------------|------|-----------|------|
| fp16 (Path A) | 전체 FP16 기준 | 100.0% | — |
| expert_only | Gemma Expert만 INT4 | 89.0% | -11%p |
| llm_only | PaliGemma LLM만 INT4 | 78.0% | -22%p |
| expert_attn | Expert attention만 INT4 | 86.0% | -14%p |
| expert_mlp | Expert MLP만 INT4 | 86.0% | -14%p |

**결론**:
- LLM이 primary error source (-22%p)
- Expert도 유의미 (-11%p): attention ≈ MLP (-14%p each)

---

## 5. QAD (Quantization-Aware Distillation)

### 개념

- **Teacher**: FP16 Path A frozen, 학습 시 2-step Euler forward로 velocity target 계산
  - `v_target = 0.5*(v(t=1) + v(t=0.5))` — trapezoidal 평균 (inference 아님, training-only)
  - SnapFlow NFE=2가 아님: 같은 Path A 모델을 2번 호출해 더 정확한 supervision signal 생성
- **Student**: Path A + fake-quant (INT4/NVFP4/W4A4), NFE=1 (s=0)
- **Loss**: `MSE(v_pred[:,:,:7], v_target[:,:,:7])` — action 7-dim velocity만
- **Freeze**: LLM (PaliGemma) + `layernorm.dense` (AdaLN) + lm_head + action_proj
- **학습 대상**: Gemma Expert (attention + MLP) — 약 317M params
- **설정**: 500 steps, lr=1e-4, CosineAnnealingLR, batch_size=4

### QAD vs PTQ 비교 — Scale Precision 포함

| Format | Weight | Activation | Scale | PTQ | QAD | 개선 |
|--------|--------|-----------|-------|-----|-----|------|
| INT4 g=8, FP16 scale | INT4 | FP16 | FP16 | 71% | **97%** | +26%p |
| INT4 g=8, INT16 scale | INT4 | FP16 | INT16 | 71% | **92%** | +21%p |
| INT4 g=16, FP16 scale | INT4 | FP16 | FP16 | 44% | **86%** | +42%p |
| INT4 g=16, INT16 scale | INT4 | FP16 | INT16 | 44% | **85%** | +41%p |
| NVFP4 b=8 | E2M1 | FP16 | FP8 E4M3 | 19% | **79%** | +60%p |
| W4A4 g=8 | INT4 | **INT4** | FP16 | — | **93%** | — |

### Scale Precision이 QAD 성능에 미치는 영향

| Group Size | FP16 scale QAD | INT16 scale QAD | 차이 |
|-----------|---------------|----------------|------|
| g=8 | 97% | 92% | **-5%p** |
| g=16 | 86% | 85% | -1%p |

- g=8에서 INT16 scale penalty가 더 큰 이유: block이 작을수록 scale 수가 많아 scale 정밀도 영향 증대
- g=16에서는 scale 수 자체가 적어 INT16 vs FP16 차이가 미미

### QAD Training Loss 비교

| 설정 | Weight | Activation | Scale | best_loss | last_loss |
|------|--------|-----------|-------|-----------|-----------|
| INT4 g=8, FP16 | INT4 | FP16 | FP16 | 0.005241 | 0.013547 |
| INT4 g=8, INT16 | INT4 | FP16 | INT16 | 0.003626 | 0.011416 |
| INT4 g=16, FP16 | INT4 | FP16 | FP16 | 0.009405 | 0.013727 |
| INT4 g=16, INT16 | INT4 | FP16 | INT16 | 0.009319 | 0.016335 |
| NVFP4 b=8 | E2M1 | FP16 | FP8 E4M3 | 0.009139 | 0.021622 |

### QAD Per-task 결과

#### INT4 g=8, FP16 scale (97%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 100% | 100% | 100% | 90% | 90% | 100% | 100% | 100% | 100% | 90% |

#### INT4 g=8, INT16 scale (92%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 90% | 100% | 100% | 60% | 90% | 100% | 100% | 100% | 90% | 90% |

#### INT4 g=16, FP16 scale (86%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 90% | 100% | 90% | 70% | 60% | 100% | 100% | 100% | 80% | 70% |

#### INT4 g=16, INT16 scale (85%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 90% | 100% | 70% | 80% | 80% | 100% | 90% | 100% | 80% | 60% |

#### NVFP4 b=8, FP8 E4M3 scale (79%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 100% | 100% | 100% | 40% | 70% | 100% | 100% | 100% | 10% | 70% |

---

## 6. 포맷별 성능 차이 원인 분석

### INT4 > NVFP4인 이유 (QAD 후에도)

| 원인 | INT4 | NVFP4 E2M1 |
|------|------|-----------|
| 유효 레벨 수 | 16개 (균일) | 9개 양수 (로그) |
| Scale 포맷 | FP16 (65536 레벨) | **FP8 E4M3 (~448 레벨)** |
| Double quant | 없음 | weight error + scale error 중첩 |
| Weight 분포 적합성 | Gaussian에 최적 (선형) | Gaussian에 비효율 (로그) |

### FP16 scale > INT16 scale인 이유

- FP16: 연속적 표현, ~65536 레벨, 지수부 덕분에 소수점 정밀도 높음
- INT16: 65536 정수 레벨, 고정 간격 → 매우 작거나 큰 scale 값 표현 시 정밀도 저하
- g=8에서 -5%p, g=16에서 -1%p → block 수가 많을수록 scale error 누적

### W4A4 vs W4A16

- W4A4 QAD 93%: activation INT4 추가로 W4A16(97%) 대비 4%p 손실 — 예상 대비 선방
- Task 8이 60%로 특히 약함 (activation outlier 민감 구간)
- activation per-tensor quantization 사용 (axis 없음) — calib/eval 간 sequence length 불일치 회피

---

## 7. 핵심 설계 결정 사항

- **`layernorm.dense` 동결**: AdaLN timestep conditioning. fine-tuning 시 이 레이어가 학습되면 0% 성능 붕괴 (stage8b 실패 원인). 반드시 frozen.
- **NVFP4 `type="dynamic"`**: `type="static"` 시 `scaled_e4m3()` 경로로 라우팅되어 E2M1 미지원 에러 발생.
- **MXFP4 PTQ 실패**: `scale_bits=(8,0)` E8M0 scale은 PTQ에서 0-15% 성능. Path A Hadamard rotated weight에 MXFP4 PTQ 부적합.
- **QAD teacher 2-step**: inference NFE=2가 아님. 학습 시 FP16 Path A를 2번 forward해 trapezoidal velocity target 계산 (더 정확한 supervision). Inference는 여전히 NFE=1.
- **W4A4 activation clip_ratio=0.9**: per-token symmetric quantization, outlier 제거용 clipping.
