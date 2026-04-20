#!/bin/bash
# MTQ NVFP4_DEFAULT_CFG (LM+DiT) — task 0~9, batch_size=5, 10 episodes
# Output: results_mtq_10ep/

cd "$(dirname "$0")"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_nvfp4_lm_dit.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --n_episodes 10 \
    --batch_size 5 \
    --output_dir results_mtq_10ep
