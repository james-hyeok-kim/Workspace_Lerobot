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

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
# EGL은 GPU 0 사용 (GPU 2/3은 EGL init 실패). CUDA 추론은 GPU 2,3으로 제한.
export MUJOCO_EGL_DEVICE_ID="0"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3}"
export LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-/data/jameskimh/james_libero_datasets}"

# EGL 라이브러리 경로: libEGL.so (egl_libs) + libEGL_nvidia.so.0 (x86_64-linux-gnu)
export LD_LIBRARY_PATH="/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "  Snapflow_QuaRot — Stage $STAGE"
echo "  WORKSPACE: $WORKSPACE"
echo "  MODEL_OPT_DIR: $MODEL_OPT_DIR"
echo "=========================================="

case "$STAGE" in
    0) python scripts/stage0_baseline_simple.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage0 ;;
    1) python scripts/stage1_eval_student.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage1 ;;
    2) python scripts/stage2_quarot_llm.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage2 ;;
    3) python scripts/stage3_quarot_dit.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage3 ;;
    4) python scripts/stage4_ohb_adaln.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage4 ;;
    5) python scripts/stage5_w4a4.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage5 ;;
    6) python scripts/stage6_e2e.py --config configs/stage6_e2e.yaml ;;
    6r) python scripts/stage6_snapflow_rotation_only.py \
           --device cuda:1 \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage6_rotation_only ;;
    5b) python scripts/stage5b_snapflow_llm_w4a4.py \
           --task_ids 0 1 2 3 4 5 6 7 8 9 \
           --n_episodes 10 --batch_size 10 \
           --start_seed 1000 \
           --output_dir results/stage5b ;;
    *)
        echo "Unknown stage: $STAGE (must be 0–6, 6r, 5b)"
        exit 1 ;;
esac
