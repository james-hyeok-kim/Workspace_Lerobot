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
| W4A4 QAD g=8 (DiT only) | INT4 | **INT4** | FP16 | 1 | **93.0%** | -7%p |
| **W4A8 LLM + W4A4 DiT QAD** | INT4(W4A8) | **FP8(LLM)+INT4(DiT)** | FP16 | 1 | **TBD** | TBD |

> Sequential W4A4 LLM+DiT QAD (§8): Phase 1 LLM QAD → 0%, Phase 2 DiT QAD → 0%. LLM activation INT4는 현재 방법으로 복구 불가.  
> QuantVLA-style W4A8 LLM PTQ + W4A4 DiT QAD (§9): 진행 중. LLM은 W4A8 (FP8 activation), DiT는 stage8f 93% checkpoint 재사용.

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

## 4. Ablation Study — 오류 원인 분석

### 4-1. W4A16 (Weight-only) g=4 Ablation

Path A에서 INT4 g=4 (W4A16) 적용 시 100% → 82% (-18%p).

| Ablation Mode | 설명 | pc_success | 감소 |
|--------------|------|-----------|------|
| fp16 (Path A) | 전체 FP16 기준 | 100.0% | — |
| expert_only | Gemma Expert만 INT4 | 89.0% | -11%p |
| llm_only | PaliGemma LLM만 INT4 | 78.0% | -22%p |
| expert_attn | Expert attention만 INT4 | 86.0% | -14%p |
| expert_mlp | Expert MLP만 INT4 | 86.0% | -14%p |

**결론**: LLM이 primary error source (-22%p), Expert도 유의미 (-11%p): attention ≈ MLP (-14%p each)

---

### 4-2. W4A4 (Weight+Activation) g=8 PTQ Ablation

W4A4 activation 추가 시 W4A16 대비 추가 손실 분석.

| Ablation Mode | W4A16 PTQ (g=4) | W4A4 PTQ (g=8) | Activation 추가 비용 |
|--------------|----------------|----------------|-------------------|
| fp16 (Path A) | 100.0% | 100.0% | — |
| expert_attn only | 86.0% | **85.0%** | -1%p |
| expert_mlp only | 86.0% | **86.0%** | 0%p |
| llm_only | 78.0% | **0.0%** | **-78%p** |

Per-task (expert_mlp W4A4 PTQ, 86%):
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 100% | 100% | 100% | 80% | 60% | 100% | 100% | 90% | 50% | 80% |

Per-task (expert_attn W4A4 PTQ, 85%):
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 100% | 100% | 100% | 80% | 70% | 100% | 90% | 90% | 60% | 60% |

Per-task (llm_only W4A4 PTQ, **0%**):
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

**결론 (W4A4 vs W4A16)**:
- expert_attn: -1%p (activation INT4 거의 무영향)
- expert_mlp: 0%p (activation INT4 무영향)
- **llm_only: -78%p → LLM activation 양자화(INT4)가 치명적** (W4A16 78% → W4A4 0%)
  - LLM attention의 softmax 입력, LayerNorm 입력 등 민감한 activation에 INT4 적용 → 언어/비전 이해 완전 붕괴
  - W4A16은 weight만 양자화 → 정보 유지, W4A4는 activation도 양자화 → 정보 소실
- Task 4, 8이 공통 취약 구간 (expert_attn/mlp에서)

### 4-3. W4A4 LLM QAD (Phase 1 Sequential) — 실패 분석

PTQ가 0%인 W4A4 LLM을 QAD로 복구 시도. 500 steps, velocity MSE loss, LLM만 학습 (DiT FP16 frozen).

| 방법 | best_loss | pc_success | 비고 |
|------|-----------|-----------|------|
| W4A4 LLM PTQ | — | 0.0% | activation INT4 치명 |
| W4A4 LLM QAD (500 steps) | 0.083677 | **0.0%** | loss 개선 없었음 |

Per-task (W4A4 LLM QAD, 0%):
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

**실패 원인 분석**:
- best_loss=0.083677 — DiT-only QAD 성공 케이스(0.005~0.013) 대비 10~17배 높음
- Gradient 경로: loss → DiT → cross-attention → LLM embedding → LLM weights. 이 긴 경로가 LLM 학습 신호를 희석시킴
- 2923M params(LLM) vs 317M params(DiT) — 500 steps는 LLM 수렴에 불충분
- **결론**: velocity MSE loss로는 W4A4 LLM 복구 불가. LLM activation INT4는 QAD로 해결 어려움

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
| **W4A4 g=8 (DiT only)** | INT4 | **INT4** | FP16 | **0.013284** | 0.020449 |

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

#### W4A4 g=8, FP16 scale — DiT only (93%)
| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| 성공률 | 100% | 100% | 100% | 100% | 90% | 100% | 90% | 100% | 60% | 90% |

- Task 8이 60%로 가장 취약 (activation outlier 민감 구간)
- Task 3이 W4A16 QAD(90%)에서 W4A4(100%)로 오히려 향상 — 노이즈 범위 내

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
- **범위 제한**: W4A4 QAD 93%는 DiT만 INT4 (LLM은 FP16). LLM까지 W4A4로 확장 시 0% (§8)
- best_loss 비교: W4A4 DiT QAD 0.013 → 93%, LLM QAD 0.084 → 0%. loss 값이 성능의 강력한 예측 지표

---

## 7. 핵심 설계 결정 사항

- **`layernorm.dense` 동결**: AdaLN timestep conditioning. fine-tuning 시 이 레이어가 학습되면 0% 성능 붕괴 (stage8b 실패 원인). 반드시 frozen.
- **NVFP4 `type="dynamic"`**: `type="static"` 시 `scaled_e4m3()` 경로로 라우팅되어 E2M1 미지원 에러 발생.
- **MXFP4 PTQ 실패**: `scale_bits=(8,0)` E8M0 scale은 PTQ에서 0-15% 성능. Path A Hadamard rotated weight에 MXFP4 PTQ 부적합.
- **QAD teacher 2-step**: inference NFE=2가 아님. 학습 시 FP16 Path A를 2번 forward해 trapezoidal velocity target 계산 (더 정확한 supervision). Inference는 여전히 NFE=1.
- **W4A4 activation clip_ratio=0.9**: per-token symmetric quantization, outlier 제거용 clipping.
- **W4A4 LLM QAD 실패**: velocity MSE loss로 LLM activation INT4 복구 불가. gradient 경로가 너무 길어 LLM 학습 신호 희석. W4A4 LLM은 PTQ든 QAD든 0% (Section 4-3).

---

## 8. Sequential W4A4 LLM→DiT QAD 실험

### 배경

DiT-only W4A4 QAD (93%)는 DiT activation INT4가 benign(-0~1%p)이어서 성공.
그러나 LLM activation INT4는 catastrophic (-78%p). LLM도 W4A4로 만들기 위한 sequential QAD 시도:
- **Phase 1**: W4A4 LLM 학습 (DiT FP16 frozen), velocity MSE loss
- **Phase 2**: Phase 1 체크포인트 + W4A4 DiT 학습

### Phase 1 결과 (W4A4 LLM QAD, 500 steps)

| 항목 | 값 |
|------|-----|
| Trainable params | 2923.3M (LLM) |
| Frozen params | 1220.1M (DiT + lm_head) |
| best_loss | 0.083677 |
| pc_success | **0.0%** |
| 참고 DiT-only QAD best_loss | 0.005–0.013 (→ 93–97%) |

Phase 1이 0%이므로 Phase 2는 망가진 LLM 위에서 DiT를 학습하는 상황.

### Phase 2 결과 (W4A4 DiT QAD on top of Phase 1 LLM, 500 steps)

| 항목 | 값 |
|------|-----|
| Trainable params | ~317M (DiT) |
| Frozen params | W4A4 LLM (Phase 1 ckpt) |
| best_loss | 0.075764 |
| pc_success | **0.0%** |

Per-task: 모든 task 0% (10/10 전부 실패)

### 전체 결과 요약

| Phase | 설정 | best_loss | pc_success |
|-------|------|-----------|-----------|
| Phase 1 | W4A4 LLM QAD (DiT FP16 frozen) | 0.083677 | **0.0%** |
| Phase 2 | W4A4 DiT QAD (Phase 1 LLM frozen) | 0.075764 | **0.0%** |
| 비교: DiT-only W4A4 QAD | W4A4 DiT (LLM FP16 frozen) | 0.013284 | 93.0% |

### 결론

W4A4 LLM + W4A4 DiT 전체 INT4는 현재 velocity MSE 접근법으로 불가.
- Phase 1 LLM이 망가진 상태(0%)에서 Phase 2 DiT 학습 → DiT도 의미있는 신호 없음
- best_loss 0.075~0.084는 성공한 DiT-only QAD(0.013)보다 6~17배 높음
- **핵심 제약**: LLM activation INT4는 PTQ든 QAD든 현재 방식으로는 복구 불가

**실현 가능한 최선**: W4A16 LLM (PTQ 71% → QAD 97%) + W4A4 DiT (QAD 93%)

---

## 9. LLM W4A8 PTQ + DiT W4A4 QAD 결합 실험 (QuantVLA-style)

### 배경

§8 의 Sequential W4A4 QAD 실패 원인: **LLM activation INT4가 치명적** (§4-2: llm_only W4A4 PTQ 0%). velocity MSE gradient 가 DiT → cross-attention → LLM 까지 닿지 않아 학습 자체가 불가.

QuantVLA 논문 (arXiv 2602.20309) 의 핵심 통찰: LLM 은 **W4A8** (activation = FP8 E4M3, weight = INT4 blockwise-128) + **AWQ-lite** 로 복구 가능. DiT 는 FP16 유지 또는 별도 경량 처리. 논문 보고치: LIBERO-10 97.6% (DuQuant + ATM + OHB 포함).

이번 실험은 QuantVLA 의 LLM 부분만 채택 (DuQuant / ATM / OHB 미적용) + 기존 W4A4 DiT QAD 체크포인트 결합.

### 실험 설계

| 컴포넌트 | 설정 | 출처 |
|----------|------|------|
| LLM (PaliGemma + vision tower) | **W4A8** — weight INT4 blockwise-128, activation FP8 E4M3, awq_lite calib 32 batch | modelopt `W4A8_AWQ_BETA_CFG` 기반 |
| DiT (gemma_expert) | **W4A4** — weight INT4 g=8, activation INT4, max calib | `results/stage8f_w4a4_g8/best_student.pt` (93%) |
| NFE | 1 | 모든 QAD 실험과 동일 |
| n_action_steps | 10 | 전체 고정 |
| 추가 학습 | **없음** — PTQ only | DiT 는 checkpoint 재사용, LLM 은 awq_lite calibration only |

### 2-pass mtq.quantize 프로토콜

```
Pass 1: DiT W4A4 (scope=expert) 등록 + max calibration (32 batch)
Load:   results/stage8f_w4a4_g8/best_student.pt → DiT trained weights + amax 복원
Pass 2: LLM/vision/projector W4A8 등록 + awq_lite calibration (32 batch)
Eval:   LIBERO-10 10 task × 10 ep
```

Pass 2 config 에 `"default"` 키 없음 → gemma_expert quantizer state 보존. DiT amax before/after 비교로 검증.

### Script

`scripts/stage9_quantvla_llm_w4a8.py`

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/stage9_quantvla_llm_w4a8.py \
  --student_ckpt results/stage8f_w4a4_g8/best_student.pt \
  --output_dir   results/stage9_quantvla_w4a8_w4a4dit \
  --calib_batches 32 --eval_episodes 10 --num_inference_steps 1
```

### 결과 (실행 후 기록)

| 항목 | 값 |
|------|-----|
| LLM 양자화 | W4A8 (INT4-blockwise128 + FP8 E4M3 activation, awq_lite) |
| DiT 양자화 | W4A4 g=8 (stage8f checkpoint, 93% 훈련됨) |
| pc_success | **TBD** |
| Path A 대비 | TBD |

Per-task: TBD

### 예상 / 목표

- **성공 기준**: pc_success ≥ 80%
- **기대 범위**: 80~95% — LLM FP8 activation 은 INT4 대비 훨씬 안전, DiT 는 이미 93% 검증됨
- **QuantVLA 논문과의 gap**: DuQuant (rotation) + ATM + OHB 미적용 → 97.6% 대비 낮을 수 있음
- **다음 단계**: 결과 < 80% 이면 ATM + OHB 추가 (2~3일 추가), 결과 ≥ 80% 이면 종료 or DuQuant 검토
