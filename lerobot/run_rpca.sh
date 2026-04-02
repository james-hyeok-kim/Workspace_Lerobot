#!/bin/bash
# RPCA + Quantization Evaluation Runner
#
# Usage:
#   TEST_MODE=1 bash run_rpca.sh              → TEST: int8_w + nvfp4_wa, 1 episode
#   bash run_rpca.sh                          → FULL: 전체 8개 스킴, 50 episodes
#   SCHEMES="int4_w int4_wa" bash run_rpca.sh → 스킴 직접 지정 (FULL 모드 기반)

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR="$(pwd)"
LEROBOT_DIR="${BASE_DIR}"
MODEL_OPT_DIR="${BASE_DIR}/../TensorRT-Model-Optimizer"

# ── 환경 변수 ─────────────────────────────────────────────────────────────────
export PYTHONPATH="${LEROBOT_DIR}/src:${LEROBOT_DIR}:${MODEL_OPT_DIR}:${PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all

# HF 인증 (~/.env 에 HF_TOKEN=... 형태로 저장)
if [ -f ~/.env ]; then
    export $(grep -v '^#' ~/.env | xargs)
fi

# ── 모드 설정 ─────────────────────────────────────────────────────────────────
# TEST_MODE: 1 또는 true → TEST MODE, 그 외 → FULL MODE
TEST_MODE=0

MODEL_ID="lerobot/pi05_libero_finetuned"
TASK="libero_10"
DEVICE="cuda"

# RPCA 하이퍼파라미터
RPCA_RANK=32           # 저랭크 성분 최대 rank
RPCA_LAM_SCALE=1.0     # lambda 스케일 (lambda = scale/sqrt(max(m,n)))
MAX_RPCA_DIM=1024      # 이 차원 초과 레이어는 직접 양자화 (속도 확보)

if [[ "${TEST_MODE}" == "1" || "${TEST_MODE}" == "true" ]]; then
    # ── TEST MODE ─────────────────────────────────────────────────────────────
    SCHEMES="${SCHEMES:-int8_w nvfp4_wa}"
    N_EPISODES=1
    BATCH_SIZE=1
    LOG_DIR="${BASE_DIR}/logs/rpca_test"
    OUTPUT_DIR="${LOG_DIR}"

    echo "=================================================="
    echo " RPCA TEST MODE"
    echo " Model   : ${MODEL_ID}"
    echo " Task    : ${TASK}"
    echo " Schemes : ${SCHEMES}"
    echo " Episodes: ${N_EPISODES}"
    echo " Output  : ${OUTPUT_DIR}"
    echo "=================================================="

    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run.log"

    CUDA_VISIBLE_DEVICES=0 python -u "${LEROBOT_DIR}/eval_rpca_svd.py" \
        --pretrained_path "${MODEL_ID}" \
        --task "${TASK}" \
        --n_episodes "${N_EPISODES}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --schemes ${SCHEMES} \
        --rpca_rank "${RPCA_RANK}" \
        --rpca_lam_scale "${RPCA_LAM_SCALE}" \
        --max_rpca_dim "${MAX_RPCA_DIM}" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

else
    # ── FULL MODE: 전체 스킴 × libero_10 ─────────────────────────────────────
    SCHEMES="${SCHEMES:-int8_w int8_wa int4_w int4_wa int2_w int2_wa ternary_w nvfp4_wa}"
    N_EPISODES=50
    BATCH_SIZE=10
    LOG_DIR="${BASE_DIR}/logs/rpca_full"
    OUTPUT_DIR="${LOG_DIR}"

    echo "=================================================="
    echo " RPCA FULL MODE"
    echo " Model   : ${MODEL_ID}"
    echo " Task    : ${TASK}"
    echo " Schemes : ${SCHEMES}"
    echo " Episodes: ${N_EPISODES} per scheme"
    echo " Batch   : ${BATCH_SIZE} envs"
    echo " Output  : ${OUTPUT_DIR}"
    echo "=================================================="

    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/run.log"

    CUDA_VISIBLE_DEVICES=0 python -u "${LEROBOT_DIR}/eval_rpca_svd.py" \
        --pretrained_path "${MODEL_ID}" \
        --task "${TASK}" \
        --n_episodes "${N_EPISODES}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --schemes ${SCHEMES} \
        --rpca_rank "${RPCA_RANK}" \
        --rpca_lam_scale "${RPCA_LAM_SCALE}" \
        --max_rpca_dim "${MAX_RPCA_DIM}" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}
fi

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[OK] 완료. 결과: ${OUTPUT_DIR}"
else
    echo "[ERROR] 실패 (exit ${EXIT_CODE}). 로그: ${LOG_FILE}"
fi
exit ${EXIT_CODE}
