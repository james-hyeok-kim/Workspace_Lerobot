# Snapflow_QuaRot — pi0.5 실험 프레임워크

pi0.5 모델을 대상으로 LIBERO-10 벤치마크 위에서 SnapFlow distillation 과 QuaRot
rotation-aware quantization 을 단계별로 적용하는 7-stage 실험 프레임워크.

## Stage 구조

| Stage | 이름 | NFE | n_action_steps |
|---|---|---|---|
| 0 | Baseline FP16 | 10 | 10 |
| 1 | SnapFlow distill (1-NFE) | 1 | 10 |
| 2 | QuaRot — LLM 만 | 10 | 10 |
| 3 | QuaRot — LLM+DiT | 10 | 10 |
| 4 | OHB + AdaLN 처리 | 10 | 10 |
| 5 | W4A4 공격적 quant | 10 | 10 |
| 6 | E2E 통합 | 1 | 10 |

## 환경 변수

```bash
# ModelOpt 경로 (Hadamard transform 등 QuaRot 빌딩블록)
export MODEL_OPT_DIR=/home/jovyan/workspace/Workspace_Lerobot/TensorRT-Model-Optimizer
export PYTHONPATH=$MODEL_OPT_DIR:$PYTHONPATH

# LeRobot src
export LEROBOT_SRC=/home/jovyan/workspace/Workspace_Lerobot/lerobot/src
export PYTHONPATH=$LEROBOT_SRC:$PYTHONPATH

# LIBERO 데이터셋
export LIBERO_DATASET_PATH=/data/james_libero_datasets

# GPU headless 렌더링
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# HuggingFace 토큰 (private Hub 접근 시)
# export HF_TOKEN=...
```

## 실행 순서

```bash
cd /home/jovyan/workspace/Workspace_Lerobot/Snapflow_QuaRot

# Stage 0 — FP16 baseline 기준선 확정 (eval + calib stat 수집)
bash scripts/run_stage.sh 0

# Stage 1 — SnapFlow distillation (학습은 사용자 직접)
#   1) 학습 실행 (10k+ step, GPU 집약적):
python scripts/stage1_snapflow_distill.py --config configs/stage1_snapflow.yaml
#   2) 학습 완료 후 artifacts/stage1_student.safetensors 배치 확인
#   3) eval:
bash scripts/run_stage.sh 1

# Stage 2~6 순서대로 실행
for s in 2 3 4 5 6; do bash scripts/run_stage.sh $s; done

# 최종 leaderboard
cat results/leaderboard.md
```

## 디렉토리 구조

```
common/         — 공통 모듈 (recipe, policy_loader, eval_driver, smoke, results_db)
quarot/         — QuaRot rotation 구현 (rotations, fuse_rmsnorm, offline_rotate, online_hadamard, ohb, adaln_handler)
snapflow/       — SnapFlow distillation (teacher_student, distill_loss, trainer, data)
quant/          — W4A4 quantization (w4a4_recipe, modelopt_bridge, calib_quantize)
analysis/       — activation stats, delta plots
configs/        — stage별 YAML config
scripts/        — stage별 entry point + run_stage.sh
artifacts/      — stage 간 전달 산출물 (.pt, .safetensors, .json)
results/        — 평가 결과 (leaderboard.md, stage{N}/)
```

## 검증 프로토콜 (모든 stage 공통)

1. **Smoke**: `common/smoke.py` — 1 episode, task 0, seed 0. NaN 없음, 실행 완료 확인.
2. **Numerical equiv** (rotation only, Stage 2–3): FP16 출력 대비 max-rel-err < 1e-3.
3. **Activation stat plot**: `analysis/activation_stats.py` — max-abs / kurtosis heatmap.
4. **Full eval**: LIBERO-10, 10 task × 10 seed (n_episodes=100).
5. **Leaderboard row**: `results_db.append({stage, pc_success, latency_ms, nfe, w_bits, a_bits})`.

## 의존성

```bash
# fast_hadamard_transform (QuaRot R4 online Hadamard 에 필요)
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```
