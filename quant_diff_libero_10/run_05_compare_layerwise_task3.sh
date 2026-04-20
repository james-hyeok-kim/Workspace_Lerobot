#!/bin/bash
# Layer-wise MTQ vs Duhyeon 비교 — task 3
# Output: compare_task3/

cd "$(dirname "$0")"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python compare_layerwise.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --task_id 3 \
    --output_dir compare_task3
