#!/bin/bash
# Duhyeon nvfp4_bmm — task 5~9 only, batch_size=5, 10 episodes (no init controller)
# Output: results_duhyeon_10ep/

cd "$(dirname "$0")"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_duhyeon_nocapture.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --task_ids 5 6 7 8 9 \
    --n_episodes 10 \
    --batch_size 5 \
    --output_dir results_duhyeon_10ep
