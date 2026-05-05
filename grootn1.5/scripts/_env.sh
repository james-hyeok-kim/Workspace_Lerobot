#!/usr/bin/env bash
# Common environment variables for GR00T-N1.5 LIBERO evaluation.
# Source this at the top of every script: source "$(dirname "$0")/_env.sh"

set -a

# EGL rendering requires GPU 0; CUDA inference uses GPU 2
# Include GPU 0 in CUDA_VISIBLE_DEVICES so MUJOCO_EGL_DEVICE_ID=0 passes robosuite assertion
CUDA_VISIBLE_DEVICES=0,2
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=0
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
LD_LIBRARY_PATH="/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# All HuggingFace downloads go to /data/jameskimh/groot_n1p5 (never workspace)
HF_HOME=/data/jameskimh/groot_n1p5/hf_cache
HF_HUB_CACHE=/data/jameskimh/groot_n1p5/hf_cache/hub
GROOT_OUTPUT_ROOT=/data/jameskimh/groot_n1p5

# LeRobot src (workspace editable, no install needed)
_WS_ROOT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
PYTHONPATH="${_WS_ROOT}/lerobot/src:${_WS_ROOT}/grootn1.5:${PYTHONPATH:-}"

set +a

# Ensure data directories exist
mkdir -p "${HF_HUB_CACHE}" "${GROOT_OUTPUT_ROOT}/results" "${GROOT_OUTPUT_ROOT}/logs"
