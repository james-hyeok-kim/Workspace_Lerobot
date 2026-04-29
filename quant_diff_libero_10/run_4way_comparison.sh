#!/bin/bash
# 4-way comparison: my code vs Duhyeon code, with/without init_state_controller
# Conditions:
#   mine        : MTQ NVFP4_DEFAULT_CFG (LM+DiT), no init control
#   mine_init   : MTQ NVFP4_DEFAULT_CFG (LM+DiT), with init control
#   dh          : Duhyeon fake-quant hook, no init control
#   dh_init     : Duhyeon fake-quant hook, with init control
#
# All conditions: n_action_steps=10, batch_size=10, n_episodes=10, start_seed=1000

set -euo pipefail
cd "$(dirname "$0")"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0,1

PYTHON=/home/jameskimh/.libero/bin/python

COMMON=(
    --pretrained_path lerobot/pi05_libero_finetuned
    --n_episodes 10
    --batch_size 10
    --start_seed 1000
    --n_action_steps 10
)

OUT_BASE="/data/jameskimh/james_lerobot_results/quant_diff_libero_10/results_4way"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================"
echo "[4-WAY] Start: ${TIMESTAMP}"
echo "[4-WAY] n_episodes=10  batch_size=10  seed=1000  n_action_steps=10"
echo "======================================================"

# ── (1) 내 코드: MTQ NVFP4 ───────────────────────────────────────────────────
echo -e "\n[1/4] mine — MTQ NVFP4, no init_control"
${PYTHON} eval_nvfp4_lm_dit.py "${COMMON[@]}" \
    --output_dir "${OUT_BASE}/mine" \
    2>&1 | tee "${LOG_DIR}/mine_${TIMESTAMP}.log"

# ── (2) 내 코드 + init control ────────────────────────────────────────────────
echo -e "\n[2/4] mine_init — MTQ NVFP4, with init_control"
${PYTHON} eval_nvfp4_lm_dit.py "${COMMON[@]}" --init_control \
    --output_dir "${OUT_BASE}/mine_init" \
    2>&1 | tee "${LOG_DIR}/mine_init_${TIMESTAMP}.log"

# ── (3) Duhyeon 코드 ──────────────────────────────────────────────────────────
echo -e "\n[3/4] dh — Duhyeon fake-quant, no init_control"
${PYTHON} eval_duhyeon_nocapture.py "${COMMON[@]}" \
    --output_dir "${OUT_BASE}/dh" \
    2>&1 | tee "${LOG_DIR}/dh_${TIMESTAMP}.log"

# ── (4) Duhyeon 코드 + init control ──────────────────────────────────────────
echo -e "\n[4/4] dh_init — Duhyeon fake-quant, with init_control"
${PYTHON} eval_duhyeon_nocapture.py "${COMMON[@]}" --init_control \
    --output_dir "${OUT_BASE}/dh_init" \
    2>&1 | tee "${LOG_DIR}/dh_init_${TIMESTAMP}.log"

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "\n======================================================"
echo "[4-WAY] RESULTS SUMMARY"
echo "======================================================"
for cond in mine mine_init dh dh_init; do
    SUMMARY="${OUT_BASE}/${cond}/eval_summary.json"
    if [ ! -f "${SUMMARY}" ]; then
        SUMMARY="${OUT_BASE}/${cond}/eval_summary_nocarture.json"
    fi
    if [ -f "${SUMMARY}" ]; then
        AVG=$(${PYTHON} -c "import json; d=json.load(open('${SUMMARY}')); print(f\"{d.get('avg_success', 'N/A'):.1f}%\")" 2>/dev/null || echo "N/A")
        echo "  ${cond:<12}: ${AVG}"
    else
        echo "  ${cond:<12}: (result not found)"
    fi
done
echo "======================================================"
echo "[4-WAY] Done. Logs: ${LOG_DIR}/"
