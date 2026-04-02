#!/bin/bash
# NVFP4 MTQ Quantization Evaluation Runner
# Usage:
#   TEST_MODE=1 bash run_nvfp4.sh   → 1 task, 1 episode (빠른 검증)
#   bash run_nvfp4.sh               → 전체 suite, 50 episodes (full eval)

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
# TEST_MODE: 1 또는 true 로 설정하면 TEST MODE, 그 외는 FULL MODE
# TEST_MODE="${TEST_MODE:-0}"
TEST_MODE=0

MODEL_ID="lerobot/pi05_libero_finetuned"
DEVICE="cuda"

if [[ "${TEST_MODE}" == "1" || "${TEST_MODE}" == "true" ]]; then
    # ── TEST MODE: 빠른 검증 ──────────────────────────────────────────────────
    TASK="libero_10"
    N_EPISODES=1
    BATCH_SIZE=1
    LOG_DIR="${BASE_DIR}/logs/nvfp4_test"
    OUTPUT_JSON="${LOG_DIR}/nvfp4_result.json"
    LOG_FILE="${LOG_DIR}/run.log"

    echo "=================================================="
    echo " NVFP4 TEST MODE"
    echo " Model   : ${MODEL_ID}"
    echo " Task    : ${TASK}"
    echo " Episodes: ${N_EPISODES}"
    echo " Output  : ${OUTPUT_JSON}"
    echo "=================================================="

    mkdir -p "${LOG_DIR}"
    CUDA_VISIBLE_DEVICES=0 python -u "${LEROBOT_DIR}/eval_nvfp4_mtq.py" \
        --pretrained_path "${MODEL_ID}" \
        --task "${TASK}" \
        --n_episodes "${N_EPISODES}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --output_json "${OUTPUT_JSON}" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}
else
    # ── FULL MODE: 전체 suite × 전체 task 평가 ───────────────────────────────
    N_EPISODES=50
    BATCH_SIZE=10       # 10개 env 병렬 → 5 batches per task
    LOG_DIR="${BASE_DIR}/logs/nvfp4_full"
    # 평가할 LIBERO suite 목록
    SUITES=("libero_10" "libero_spatial" "libero_object" "libero_goal")
    echo "=================================================="
    echo " NVFP4 FULL MODE"
    echo " Model   : ${MODEL_ID}"
    echo " Suites  : ${SUITES[*]}"
    echo " Episodes: ${N_EPISODES} per task"
    echo " Batch   : ${BATCH_SIZE} envs"
    echo " Log dir : ${LOG_DIR}"
    echo "=================================================="
    mkdir -p "${LOG_DIR}"

    EXIT_CODE=0
    for TASK in "${SUITES[@]}"; do
        OUTPUT_JSON="${LOG_DIR}/nvfp4_${TASK}.json"
        LOG_FILE="${LOG_DIR}/${TASK}.log"
        echo ""
        echo "[INFO] === Suite: ${TASK} ==="
        CUDA_VISIBLE_DEVICES=0 python -u "${LEROBOT_DIR}/eval_nvfp4_mtq.py" \
            --pretrained_path "${MODEL_ID}" \
            --task "${TASK}" \
            --n_episodes "${N_EPISODES}" \
            --batch_size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --output_json "${OUTPUT_JSON}" \
            2>&1 | tee -a "${LOG_FILE}"
        PIPE_EXIT=${PIPESTATUS[0]}
        if [ ${PIPE_EXIT} -ne 0 ]; then
            echo "[ERROR] Suite ${TASK} 실패 (exit ${PIPE_EXIT})" | tee -a "${LOG_FILE}"
            EXIT_CODE=${PIPE_EXIT}
        else
            echo "[OK] Suite ${TASK} 완료 → ${OUTPUT_JSON}" | tee -a "${LOG_FILE}"
        fi
    done

    # ── 전체 suite summary JSON 생성 ─────────────────────────────────────────
    export _NVFP4_LOG_DIR="${LOG_DIR}"
    python -u - <<'PYEOF'
import json, glob, os
log_dir = os.environ["_NVFP4_LOG_DIR"]
files = sorted(f for f in glob.glob(f"{log_dir}/nvfp4_*.json") if "summary" not in os.path.basename(f))
results = {}
for fpath in files:
    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        cfg  = data.get("config", {})
        agg  = data.get("eval_results", {}).get("aggregated", {})
        qc   = data.get("quantization", {})
        task = cfg.get("task", os.path.basename(fpath))
        results[task] = {
            "pc_success":         agg.get("pc_success",        float("nan")),
            "avg_sum_reward":     agg.get("avg_sum_reward",     float("nan")),
            "total_leaf_modules": qc.get("total_leaf_modules",  0),
            "quantized_count":    qc.get("quantized_count",     0),
            "n_episodes":         cfg.get("n_episodes",         0),
        }
    except Exception as e:
        results[os.path.basename(fpath)] = {"error": str(e)}
ranking = sorted(
    [{"task": t, "pc_success": v.get("pc_success", float("nan"))} for t, v in results.items()],
    key=lambda x: (float("nan") if x["pc_success"] != x["pc_success"] else -x["pc_success"]),
)
valid = [v["pc_success"] for v in results.values() if v.get("pc_success") == v.get("pc_success")]
avg_success = sum(valid) / len(valid) if valid else float("nan")
summary = {
    "quant_config": "NVFP4_DEFAULT_CFG",
    "avg_success_rate": round(avg_success, 2),
    "ranking": ranking,
    "results": results,
}
out = os.path.join(log_dir, "nvfp4_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"[INFO] Summary → {out}  |  avg_success={avg_success:.1f}%")
for r in ranking:
    print(f"  {r['task']:<20} {r['pc_success']:.1f}%")
PYEOF

fi

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[OK] 완료. 결과: ${LOG_DIR}"
else
    echo "[ERROR] 실패 (exit ${EXIT_CODE}). 로그: ${LOG_DIR}/*.log"
fi
exit ${EXIT_CODE}