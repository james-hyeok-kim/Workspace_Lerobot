#!/bin/bash
# Duhyeon nvfp4_bmm — task 0~9, batch_size=10, 10 episodes (with init state controller 0417)
# Output: results_duhyeon_init_bs10/

cd "$(dirname "$0")"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_duhyeon_init.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --n_episodes 10 \
    --batch_size 10 \
    --output_dir results_duhyeon_init_bs10
