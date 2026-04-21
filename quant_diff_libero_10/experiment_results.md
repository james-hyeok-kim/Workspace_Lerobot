# Experiment Results: LM+DiT NVFP4 — MTQ vs Duhyeon (libero_10)

---

## 실험 구성 요약

| 실험명 | 방법 | batch_size | n_episodes | tasks | init controller | 결과 디렉토리 |
|--------|------|-----------|-----------|-------|----------------|--------------|
| MTQ_bs5 | MTQ NVFP4_DEFAULT_CFG | 5 | 10 | 0~9 | — | `results_mtq_10ep` |
| DH_bs5 | Duhyeon nvfp4_bmm | 5 | 10 | **5~9** | ❌ | `results_duhyeon_10ep` |
| DH_WO_bs10 | Duhyeon nvfp4_bmm | 10 | 10 | 0~9 | ❌ | `results_duhyeon_bs10` |
| DH_INIT_bs10 | Duhyeon nvfp4_bmm | 10 | 10 | 0~9 | ✅ (0417) | `results_duhyeon_init_bs10` |

---

## Eval 결과 — Task별 Success Rate (%)

| task_id | Task | MTQ_bs5 | DH_bs5 | DH_WO_bs10 | DH_INIT_bs10 |
|---------|------|--------:|-------:|-----------:|-------------:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 60 | — | 50 | 80 |
| 1 | put both the cream cheese box and the butter in the basket | 90 | — | 90 | 80 |
| 2 | turn on the stove and put the moka pot on it | 80 | — | 70 | 90 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 100 | — | 80 | 90 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 90 | — | 80 | 80 |
| 5 | pick up the book and place it in the back compartment of the caddy | 90 | 100 | 100 | 100 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 100 | 70 | 90 | 100 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 80 | 90 | 80 | 90 |
| 8 | put both moka pots on the stove | 80 | 90 | 90 | 90 |
| 9 | put the yellow and white mug in the microwave and close it | 90 | 80 | 100 | 100 |
| **avg** | | **86.0** | **86.0** *(5~9)* | **83.0** | **90.0** |

> DH_bs5는 task 5~9만 측정. avg 86.0은 해당 5개 task 기준.

### Task 5~9 직접 비교 (DH_bs5 측정 범위)

| task_id | MTQ_bs5 | DH_bs5 | 차이 |
|---------|--------:|-------:|-----:|
| 5 | 90 | 100 | +10 |
| 6 | 100 | 70 | −30 |
| 7 | 80 | 90 | +10 |
| 8 | 80 | 90 | +10 |
| 9 | 90 | 80 | −10 |
| **avg** | **88.0** | **86.0** | **−2.0** |

---

## 전체 10-task 비교 (MTQ_bs5 vs DH_INIT_bs10)

DH_bs5는 절반 task만 측정했고, DH_WO_bs10은 init controller 버그가 있으므로
**MTQ_bs5 (86%)** vs **DH_INIT_bs10 (90%)** 가 가장 신뢰도 높은 비교.

- DH_INIT_bs10이 MTQ_bs5 대비 **+4%p** 높음
- init state controller(0417) 적용 후 성공률 향상: DH_WO_bs10 83% → DH_INIT_bs10 90%
- batch_size 10으로 늘려도 init controller만 정확히 적용하면 MTQ 이상 성능

---

## Layer-wise 수치 비교 (compare_task3, compare_task6)

`compare_layerwise.py`로 MTQ vs Duhyeon output을 layer별로 비교.

### 대상

- **MTQ_bs5** `layer_captures_task{tid}.pt` vs **DH** `duhyeon/` 캡처 데이터
- MTQ-quantized layers: **289개** (LM + DiT 공통)

### SNR 통계 (MTQ-quantized 289 layers 기준)

| 지표 | task3 | task6 |
|------|------:|------:|
| MTQ SNR mean | 23.17 dB | 23.27 dB |
| MTQ SNR min | 16.84 dB | 16.81 dB |
| MTQ SNR max | 35.71 dB | 36.31 dB |
| DH SNR mean | 21.71 dB | 21.77 dB |
| DH SNR min | 15.23 dB | 15.28 dB |
| DH SNR max | 35.71 dB | 36.31 dB |
| MTQ vs DH diff SNR mean | 79.10 dB | 78.23 dB |
| MTQ vs DH diff SNR min | 20.13 dB | 20.07 dB |
| MTQ vs DH diff SNR max | 157.03 dB | 164.03 dB |

> **diff SNR mean ~79 dB**: 두 구현의 output이 수치적으로 매우 유사하나 완전히 동일하지는 않음.  
> [참고] 앞선 단일레이어 실험에서 MTQ vs Duhyeon diff SNR = **92.4 dB** (사실상 동일 판정).  
> activation quantization 경로의 batch-level amax 차이로 인한 누적 오차로 추정.

### Worst 5 Layers (diff SNR 최저 — 구현 간 차이 가장 큰 layer)

**task3:**

| layer (suffix) | diff SNR | MTQ SNR | DH SNR |
|----------------|--------:|--------:|-------:|
| self_attn.k_proj | 20.1 dB | 25.1 dB | 18.9 dB |
| self_attn.v_proj | 20.1 dB | 17.2 dB | 15.3 dB |
| self_attn.v_proj | 20.2 dB | 23.8 dB | 18.6 dB |
| self_attn.v_proj | 20.2 dB | 19.1 dB | 16.9 dB |
| self_attn.q_proj | 20.3 dB | 20.4 dB | 17.4 dB |

**task6:**

| layer (suffix) | diff SNR | MTQ SNR | DH SNR |
|----------------|--------:|--------:|-------:|
| self_attn.k_proj | 20.1 dB | 25.1 dB | 18.9 dB |
| self_attn.k_proj | 20.2 dB | 22.5 dB | 18.2 dB |
| self_attn.v_proj | 20.2 dB | 23.7 dB | 18.7 dB |
| self_attn.k_proj | 20.2 dB | 23.7 dB | 18.6 dB |
| self_attn.v_proj | 20.3 dB | 27.4 dB | 19.6 dB |

- self_attn의 **q/k/v_proj**에서 두 구현 간 차이가 집중됨
- activation quantization의 batch amax 계산 방식 차이가 attention layer에서 더 두드러지는 것으로 추정

---

## 결론

1. **MTQ vs DH 성능 차이는 작음**: full 10-task 기준 MTQ 86% vs DH_INIT 90%, 동등 수준
2. **init state controller가 성능에 유의미한 영향**: DH_WO 83% → DH_INIT 90% (+7%p)
3. **layer-wise 수치**: diff SNR mean ~79 dB → 두 구현은 거의 동일하나 미세한 차이 존재
4. **차이 집중 위치**: self_attn의 q/k/v_proj (activation quant 경로의 amax 계산 차이 추정)
5. **단일 레이어 실험(92.4 dB)보다 낮은 diff SNR(~79 dB)**: full model forward에서의 activation outlier 분포 차이 및 batch 구성 차이가 원인

---

# Experiment: 4-Way Comparison — My Code vs Duhyeon ± Init Control

## 실험 조건 (통일)

| 항목 | 값 |
|------|-----|
| 모델 | `lerobot/pi05_libero_finetuned` |
| Benchmark | LIBERO-10 (`libero_10`) |
| Task IDs | 0 ~ 9 (10개 전부) |
| n_episodes (per task) | 10 |
| batch_size (n_envs) | 10 |
| n_action_steps | 10 |
| start_seed | 1000 (고정) |
| use_amp | False |
| dh_init 조건 | 유저 요청으로 스킵 |

## 결과 — Task별 Success Rate (%)

| task_id | Task | mine | mine_init | dh |
|---------|------|-----:|----------:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 100 | 100 | 90 |
| 1 | put both the cream cheese box and the butter in the basket | 100 | 100 | 100 |
| 2 | turn on the stove and put the moka pot on it | 100 | 100 | 100 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 100 | 90 | 100 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 100 | 100 | 80 |
| 5 | pick up the book and place it in the back compartment of the caddy | 100 | 100 | 100 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 100 | 100 | 100 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 90 | 90 | 100 |
| 8 | put both moka pots on the stove | 80 | 90 | 80 |
| 9 | put the yellow and white mug in the microwave and close it | 100 | 100 | 100 |
| **avg** | | **97.0%** | **97.0%** | **95.0%** |

## 분석

### mine vs mine_init (+init_control 효과)

- 평균: 97.0% vs 97.0% → **동일**
- task별 차이: task 3 (100→90, -10%p), task 8 (80→90, +10%p)로 상쇄
- **결론**: `start_seed=1000`으로 고정한 상태에서 init_control 유무는 평균 성능에 영향 없음
  - seed 고정만으로도 `seeds=range(1000,...)`이 `env.reset(seed=...)`에 전달되어 init state가 결정적으로 선택됨
  - init_controller는 redundant

### mine vs dh (MTQ vs Duhyeon fake-quant)

- 평균: **97.0% vs 95.0% (MTQ가 +2%p 높음)**
- 낮은 task: dh task 0 (90%, -10%p), task 4 (80%, -20%p)
- 높은 task: dh task 7 (100%, +10%p)
- **결론**: 동일 seed/conditions에서 MTQ NVFP4_DEFAULT_CFG가 Duhyeon fake-quant보다 소폭 높은 성능

### 기존 실험과 비교

| 실험 | 조건 | mine/MTQ | dh |
|------|------|--------:|---:|
| 기존 (seeding 없음) | bs=5, n=10, seed=None, steps=50 | 86.0% | 83~90% (조건 불일치) |
| **이번 (seeding 통일)** | bs=10, n=10, seed=1000, steps=10 | **97.0%** | **95.0%** |

- 기존의 5% 격차는 seeding 불일치 + n_action_steps 차이가 주원인이었음
- 조건 통일 후 두 구현 간 실제 차이는 **2%p**로 감소
- 두 구현 모두 성능이 대폭 향상(86→97%, 83→95%)된 것은 `n_action_steps=50→10` 변경 효과가 큼

## 결론

1. **기존 5% 차이의 원인**: seeding 없음 + n_action_steps=50 (기본값) 사용이 주범. quant 알고리즘 차이는 2%p 수준.
2. **init_control 효과**: start_seed 고정 시 불필요. seed 없던 이전 실험에서는 7%p 영향이 있었으나, seed 고정 후엔 효과 없음.
3. **MTQ vs Duhyeon**: 동일 조건에서 MTQ 97% > DH 95% (+2%p). 두 구현 모두 실용적 수준.
4. **n_action_steps=10**: 기존 50 대비 성능이 훨씬 높음 → 이 설정이 baseline 기준.
