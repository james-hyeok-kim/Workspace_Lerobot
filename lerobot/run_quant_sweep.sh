#!/usr/bin/env bash
# run_quant_sweep.sh
# LM vs DiT Quantization Sweep (libero_spatial, 10 episodes)
#
# TEST MODE:
#   TEST_MODE=1 bash run_quant_sweep.sh
#
# FULL MODE:
#   bash run_quant_sweep.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${SCRIPT_DIR}"
MODEL_OPT_DIR="${SCRIPT_DIR}/../TensorRT-Model-Optimizer"

export PYTHONPATH="${LEROBOT_DIR}/src:${LEROBOT_DIR}:${MODEL_OPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ── 파라미터 ───────────────────────────────────────────────────────────────────
PRETRAINED_PATH="lerobot/pi05_libero_finetuned"
TASK="libero_spatial"
DEVICE="cuda"

if [[ "${TEST_MODE:-0}" == "1" ]]; then
    echo "[MODE] TEST — 2 episodes, fp32 + int8wa, all target only"
    N_EPISODES=2
    BATCH_SIZE=1
    SCHEMES="fp32 int8wa"
    TARGETS="all"
    STEP_SWEEP=""
    OUTPUT_DIR="${LEROBOT_DIR}/logs/quant_test"
else
    echo "[MODE] FULL — 10 episodes, all schemes & targets + step sweep"
    N_EPISODES=10
    BATCH_SIZE=5
    SCHEMES="fp32 int8wa nvfp4wa int4wa int3wa"
    TARGETS="lm_only dit_only all"
    STEP_SWEEP="--step_sweep --step_values 1 3 5 10"
    OUTPUT_DIR="${LEROBOT_DIR}/logs/quant_sweep"
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Output dir : ${OUTPUT_DIR}"
echo "[INFO] Log file   : ${LOG_FILE}"
echo "[INFO] Schemes    : ${SCHEMES}"
echo "[INFO] Targets    : ${TARGETS}"
echo ""

# ── 실행 ───────────────────────────────────────────────────────────────────────
python "${LEROBOT_DIR}/eval_quant_sweep.py" \
    --pretrained_path "${PRETRAINED_PATH}" \
    --task            "${TASK}" \
    --n_episodes      "${N_EPISODES}" \
    --batch_size      "${BATCH_SIZE}" \
    --device          "${DEVICE}" \
    --output_dir      "${OUTPUT_DIR}" \
    --schemes         ${SCHEMES} \
    --targets         ${TARGETS} \
    ${STEP_SWEEP} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "[DONE] Summary: ${OUTPUT_DIR}/quant_sweep_summary.json"
