# Quantization Experiments — Handoff Notes

**Date**: 2026-04-02  
**Branch**: main  
**Working dir**: `/home/jameskimh/workspace/Workspace_Lerobot/lerobot`

---

## 완료된 작업

### 1. NVFP4 MTQ 실험 (`eval_nvfp4_mtq.py` + `run_nvfp4.sh`)

NVIDIA ModelOpt(MTQ) `NVFP4_DEFAULT_CFG`를 pi05 정책에 적용.

**주요 수정 사항:**
- `eval_nvfp4_mtq.py`: MTQ quantize 후 `torch.compile` 비활성화 블록 추가
  - `mtq.quantize()` 이후 `_dynamo.reset()` 호출
  - `inner_model.sample_actions` / `.forward`의 `_torchdynamo_orig_callable` 복원
- `run_nvfp4.sh`:
  - `TEST_MODE=1` → libero_10 1 episode
  - `FULL_MODE` → 4개 suite (libero_10/spatial/object/goal), 50 episodes, batch=10
  - 결과 JSON: `logs/nvfp4_full/<suite>.json`

### 2. RPCA SVD 실험 (`eval_rpca_svd.py` + `run_rpca.sh`)

Robust PCA (Inexact ALM) 기반 가중치 분해 후 양자화.

**알고리즘**: W = L (low-rank) + S (sparse outlier)
- L은 양자화, S는 fp16 residual로 유지
- `max_rpca_dim=1024` 초과 레이어는 RPCA 스킵 → Direct 양자화

**지원 Scheme**:
| Scheme | 설명 |
|--------|------|
| `int8_w` | INT8 weight-only, per-channel symmetric |
| `int8_wa` | INT8 weight+activation |
| `int4_w` | INT4 weight-only, block-wise (block=16) |
| `int4_wa` | INT4 weight+activation |
| `int2_w` | INT2 weight-only, per-channel symmetric |
| `int2_wa` | INT2 weight+activation |
| `ternary_w` | Ternary weight {-α, 0, +α} (TWN) |
| `nvfp4_wa` | NVFP4 via ModelOpt MTQ, weight+activation |

**Custom Module:**
- `RPCAQuantizedLinear`: L_dequant + S를 fp16 버퍼로 저장; `.weight`, `.out_features`, `.in_features` property 제공
- `RPCANvfp4Wrapper`: 내부에 `nn.Linear(device=L.device, dtype=L.dtype)` 생성 후 MTQ 적용; 동일 property 제공

**Coverage 리포트** (`build_quant_coverage` / `print_coverage_table`):
- Component별: vision / llm / expert / action_head
- Role별: attn_Q/K/V/O, mlp, proj 등
- Direct vs RPCA 구분

**run_rpca.sh:**
- `TEST_MODE=1` → schemes="int8_w nvfp4_wa", 1 episode
- `FULL_MODE` → 8개 scheme, 50 episodes
- `SCHEMES="int4_w"` 환경변수로 직접 지정 가능

### 3. LeRobot upstream 버그 수정

**파일**: `src/lerobot/scripts/lerobot_eval.py:371`

```python
# Before (버그)
else:
    all_seeds.append(None)

# After (수정)
else:
    all_seeds.extend([None] * env.num_envs)
```

**원인**: `all_seeds`가 배치당 1개씩만 추가되어 `sum_rewards`, `max_rewards`, `all_successes`(배치당 `num_envs`개)와 길이 불일치 → `zip(..., strict=True)` 에서 `ValueError` 발생.  
**증상**: `ValueError: zip() argument 4 is shorter than arguments 1-3`

---

## 알려진 이슈 / 주의사항

1. **torch.compile 충돌**: MTQ `quantize()` 후 반드시 `_dynamo.reset()` + 원본 callable 복원 필요. pi05의 `model.sample_actions` / `model.forward`가 `torch.compile`로 래핑되어 있어 `_FoldedCallback`과 충돌.

2. **RPCA 적용 범위**: `max_rpca_dim=1024` 기준으로 pi05 구조상:
   - Vision encoder(SigLIP, 1152-dim), LLM(Gemma, 2048-dim) → Direct 양자화
   - Expert k/v proj, action_head MLP → RPCA 적용

3. **CUDA device 지정**: `RPCANvfp4Wrapper` 내부 `nn.Linear` 생성 시 반드시 `device=L.device, dtype=L.dtype` 전달 필요. 미전달 시 CPU tensor로 생성되어 CUDA assert 발생.

4. **Coverage 집계**: `RPCANvfp4Wrapper`는 children을 가지는 non-leaf 모듈이므로 `build_quant_coverage`에서 `isinstance` 체크로 별도 처리.

---

## 실행 방법

```bash
# NVFP4 TEST (1 episode)
TEST_MODE=1 bash run_nvfp4.sh

# NVFP4 FULL (4 suites × 50 episodes)
bash run_nvfp4.sh

# RPCA TEST (int8_w + nvfp4_wa, 1 episode)
TEST_MODE=1 bash run_rpca.sh

# RPCA FULL (8 schemes × 50 episodes)
bash run_rpca.sh

# RPCA 특정 scheme만
SCHEMES="int4_w int4_wa" bash run_rpca.sh
```

---

## 파일 목록

| 파일 | 설명 |
|------|------|
| `eval_nvfp4_mtq.py` | NVFP4 MTQ 양자화 + LIBERO eval |
| `run_nvfp4.sh` | NVFP4 실험 runner (TEST/FULL 모드) |
| `eval_rpca_svd.py` | RPCA 기반 다중 scheme 양자화 + LIBERO eval |
| `run_rpca.sh` | RPCA 실험 runner (TEST/FULL 모드) |
| `check_pi05_precision.py` | HF Hub 모델 weight dtype 분포 확인 |
| `run_check_pi05_precision.sh` | precision 확인 runner |
| `src/lerobot/scripts/lerobot_eval.py` | upstream 버그 수정 포함 |

---

## 다음 실험 방향 (제안)

- FULL MODE로 4 suite × 8 scheme 전체 결과 수집
- Baseline (fp32/bf16) 대비 accuracy drop 정량화
- RPCA rank / lambda 하이퍼파라미터 탐색 (rank=32, lam_scale=1.0 기본값)
- GROOT N1.5 모델로 동일 실험 확장
