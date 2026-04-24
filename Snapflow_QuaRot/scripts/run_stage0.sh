#!/bin/bash
# Stage 0 — FP16 baseline eval (transformers 5.3.0 required)
# Usage: bash scripts/run_stage0.sh [--n_episodes N] [--task_ids 0 1 2 ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_DATASET_PATH=/data/jameskimh/james_libero_datasets
export LD_LIBRARY_PATH=/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${ROOT}:${ROOT}/../lerobot/src:${ROOT}/../TensorRT-Model-Optimizer:${PYTHONPATH:-}"

echo "[START] $(date)"
echo "[INFO] transformers version: $(python3 -c 'import transformers; print(transformers.__version__)')"

python3 "${SCRIPT_DIR}/stage0_baseline_simple.py" "$@"

echo "[END] $(date)"
