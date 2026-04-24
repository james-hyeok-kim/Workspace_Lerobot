# Snapflow_QuaRot 실험 설계

## 목표

pi0.5 (PaliGemma-2B + Gemma-300M action expert) 모델에 대해:
1. **SnapFlow** (1-NFE distillation) 으로 추론 속도 향상
2. **QuaRot** (rotation-aware INT4 quantization) 으로 메모리/연산 압축
3. 두 기법 결합 시 LIBERO-10 성능 유지 여부 측정

## 모델

| 항목 | 값 |
|------|-----|
| 기반 모델 | `lerobot/pi05_libero_finetuned` (로컬: `/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned`) |
| 아키텍처 | PaliGemma-2B (VLM backbone) + Gemma-300M (action expert DiT) |
| 학습 config | `chunk_size=50`, `n_action_steps=50`, `num_inference_steps=10` |
| 이미지 해상도 | 256×256 (agentview), 256×256 (wrist), 224×224 (empty_camera_0) |
| State 차원 | 8D (eef_pos×3 + axis_angle×3 + gripper×2) |
| Action 차원 | 7D (relative control) |

## 벤치마크

| Suite | 설명 | Max Steps | 예상 성능 |
|-------|------|-----------|-----------|
| `libero_spatial` | 10 tasks, spatial relations | 280 | ~97% (vanilla eval 확인) |
| `libero_10` | 10 tasks, Long Horizon | 520 | 조사 중 |

## 7 Stage 파이프라인

| Stage | 이름 | NFE | SnapFlow | QuaRot | OHB | W4A4 | 목표 |
|-------|------|-----|----------|--------|-----|------|------|
| 0 | Baseline FP16 | 10 | — | — | — | — | baseline pc_success 확립 |
| 1 | SnapFlow 1-NFE | 1 | ✓ | — | — | — | baseline × 0.8 이상 |
| 2 | QuaRot LLM (R1,R2) | 10 | — | LLM | — | — | ≈ Stage 0 (FP 무손실) |
| 3 | QuaRot LLM+DiT (R1,R2,R3) | 10 | — | LLM+DiT | — | — | ≈ Stage 0 |
| 4 | OHB + AdaLN | 10 | — | LLM+DiT | ✓ | — | ≈ Stage 0 |
| 5 | W4A4 | 10 | — | LLM+DiT+R4 | ✓ | ✓ | baseline × 0.7 이상 |
| 6 | E2E 통합 | 1 | ✓ | LLM+DiT+R4 | ✓ | ✓ | baseline × 0.8 이상, 8× speedup |

## 현재 실험 상태 (2026-04-22)

### 완료
- 모델 로컬 복사: `/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned`
- 7-stage scaffold 구축 (configs/, scripts/, common/, quarot/, quant/, analysis/)
- QuaRot R2 rotation 버그 수정 (`down_proj` output-side rotation)
- Vanilla eval 재현: libero_spatial 97% (standard `lerobot-eval` CLI)

### 진행 중
- **Stage 0 0% 원인 분석**: libero_10에서 policy는 정상 동작하나 success=0 지속
  - 디버그로 확인: 액션 출력 정상, task_description 정상, 환경 구성 정상
  - 가설 1: libero_10 자체 난이도 문제 (Long Horizon)
  - 가설 2: `n_action_steps=50`이 libero_10에서 너무 낮은 재쿼리 빈도
  - 현재 테스트: libero_spatial vs libero_10 동일 파이프라인 3회 비교 (PID 235895)

### 미완료
- Stage 0 baseline 완성 (유효한 libero_10 성능)
- Stage 1 SnapFlow distillation (사용자 직접 실행 필요)
- Stage 2~6

## 설계 원칙

### lerobot 코드 비수정
- PI05Policy를 Hub에서 로드 후 **in-memory**로 변환
- `RMSNorm` 교체, `nn.Linear` 래핑, `TensorQuantizer` 등록 모두 in-place
- `modeling_pi05.py`, `eval_policy.py`, `libero.py` 수정 금지

### 관측 파이프라인
```
LIBERO env (256×256 images, 8D state)
  → preprocess_observation() [numpy→tensor, channel first]
  → add_envs_task() [task description 추가]
  → LiberoProcessorStep() [image 180° flip, robot_state→observation.state 변환]
  → policy preprocessor [MEAN_STD 정규화, tokenization, device 배치]
  → PI05Policy.select_action()
  → postprocessor [역정규화]
  → env.step() [7D relative action]
```

### 정규화 매핑 (policy_preprocessor.json 기준)
| Feature | 방식 |
|---------|------|
| VISUAL | IDENTITY (정규화 없음, resize 없음) |
| STATE | MEAN_STD |
| ACTION | MEAN_STD |

## 성능 측정 지표

| 지표 | 정의 |
|------|------|
| `pc_success` | 에피소드 성공률 (%) — LIBERO `check_success()` 기반 |
| `avg_sum_reward` | 에피소드 누적 보상 평균 |
| `latency_p50_ms` | 정책 forward pass 지연 P50 |
| NFE | 에피소드당 denoising step 수 |

## 주요 의존성

| 컴포넌트 | 경로 |
|---------|------|
| pi0.5 정책 | `/home/jovyan/workspace/Workspace_Lerobot/lerobot/src/lerobot/policies/pi05/` |
| LIBERO 환경 | `/home/jovyan/workspace/Workspace_Lerobot/lerobot/src/lerobot/envs/libero.py` |
| ModelOpt | `/home/jovyan/workspace/Workspace_Lerobot/TensorRT-Model-Optimizer/` |
| 데이터셋 | `/data/jameskimh/james_libero_datasets/` |
| 사전학습 모델 | `/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned/` |

## 환경 변수

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=Snapflow_QuaRot:lerobot/src:TensorRT-Model-Optimizer:$PYTHONPATH
```

## 알려진 문제점 및 해결 방안

| 문제 | 해결 |
|------|------|
| R2 rotation dim mismatch (4096 vs 1024) | `down_proj`를 input-side가 아닌 output-side rotation으로 수정 |
| n_action_steps=10 초기 설정 (계획 기본값) | n_action_steps=50으로 변경 (모델 학습 설정 일치) |
| libero_10 0% success (원인 조사 중) | 비교 실험 진행 중 |
| 에피소드당 7분 소요 (libero_10) | 물리 시뮬레이션 병목 (libero_10 객체 수 많음) |

## 실행 방법

```bash
cd /home/jovyan/workspace/Workspace_Lerobot/Snapflow_QuaRot
bash scripts/run_stage.sh 0   # Baseline
bash scripts/run_stage.sh 2   # QuaRot LLM
bash scripts/run_stage.sh 3   # QuaRot LLM+DiT
bash scripts/run_stage.sh 4   # OHB + AdaLN
bash scripts/run_stage.sh 5   # W4A4
bash scripts/run_stage.sh 6   # E2E (Stage 1 완료 후)
```
