#!/usr/bin/env bash
# Usage: bash scripts/run_stage.sh <stage_id>
# Example: bash scripts/run_stage.sh 0

set -euo pipefail

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
    echo "Usage: bash scripts/run_stage.sh <0|1|2|3|4|5|6>"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORKSPACE"

export MODEL_OPT_DIR="${MODEL_OPT_DIR:-$(cd "$WORKSPACE/../TensorRT-Model-Optimizer" && pwd)}"
export LEROBOT_SRC="${LEROBOT_SRC:-$(cd "$WORKSPACE/../lerobot/src" && pwd)}"
export PYTHONPATH="$WORKSPACE:$LEROBOT_SRC:$MODEL_OPT_DIR:${PYTHONPATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "=========================================="
echo "  Snapflow_QuaRot — Stage $STAGE"
echo "  WORKSPACE: $WORKSPACE"
echo "  MODEL_OPT_DIR: $MODEL_OPT_DIR"
echo "=========================================="

case "$STAGE" in
    0) python scripts/stage0_baseline.py --config configs/stage0_baseline.yaml ;;
    1) python scripts/stage1_eval_student.py --config configs/stage1_snapflow.yaml ;;
    2) python scripts/stage2_quarot_llm.py --config configs/stage2_quarot_llm.yaml ;;
    3) python scripts/stage3_quarot_dit.py --config configs/stage3_quarot_llm_dit.yaml ;;
    4) python scripts/stage4_ohb_adaln.py --config configs/stage4_ohb_adaln.yaml ;;
    5) python scripts/stage5_w4a4.py --config configs/stage5_w4a4.yaml ;;
    6) python scripts/stage6_e2e.py --config configs/stage6_e2e.yaml ;;
    *)
        echo "Unknown stage: $STAGE (must be 0–6)"
        exit 1 ;;
esac
