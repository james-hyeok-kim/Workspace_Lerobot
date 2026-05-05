#!/bin/bash
# Auto-run multisuite eval after libero_10 QAD training completes
# Usage: bash run_multisuite_after_lib10.sh <config> <gpu0> <gpu1> <gpu2>
# config: w4a16 or w8a16

CONFIG=$1
GPU0=$2
GPU1=$3
GPU2=$4

export LD_LIBRARY_PATH=/tmp/libglvnd_extract/usr/lib/x86_64-linux-gnu:/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MODEL_OPT_DIR=/home/jovyan/workspace/Workspace_Lerobot/TensorRT-Model-Optimizer
export PYTHONPATH=$MODEL_OPT_DIR:$PYTHONPATH
export MUJOCO_GL=egl

SCRIPT=/home/jovyan/workspace/Workspace_Lerobot/Snapflow_QuaRot/scripts/stage10_w4a16llm_w4a4dit_qad.py
cd /home/jovyan/workspace/Workspace_Lerobot/Snapflow_QuaRot

if [ "$CONFIG" = "w4a16" ]; then
  LLM_BITS=4; LLM_GS=4; OUT_DIR=results/w4a16llm_w4a8dit_qad
else
  LLM_BITS=8; LLM_GS=16; OUT_DIR=results/w8a16llm_w4a8dit_qad
fi

JSON_PATTERN="${OUT_DIR}/libero_10_w${LLM_BITS}a16llm_g${LLM_GS}_w4a8dit_qad.json"
echo "[watcher:$CONFIG] Polling for: $JSON_PATTERN"

while [ ! -f "$JSON_PATTERN" ]; do sleep 60; done
echo "[watcher:$CONFIG] libero_10 done — launching 3-suite parallel eval on GPUs $GPU0 $GPU1 $GPU2"

CUDA_VISIBLE_DEVICES=$GPU0 python $SCRIPT \
  --dit_act_bits 8 --llm_weight_bits $LLM_BITS --llm_group_size $LLM_GS --dit_group_size 8 \
  --skip_train --output_dir $OUT_DIR \
  --eval_episodes 50 --n_envs 10 --suite libero_spatial \
  > ${OUT_DIR}/libero_spatial.log 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU1 python $SCRIPT \
  --dit_act_bits 8 --llm_weight_bits $LLM_BITS --llm_group_size $LLM_GS --dit_group_size 8 \
  --skip_train --output_dir $OUT_DIR \
  --eval_episodes 50 --n_envs 10 --suite libero_object \
  > ${OUT_DIR}/libero_object.log 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU2 python $SCRIPT \
  --dit_act_bits 8 --llm_weight_bits $LLM_BITS --llm_group_size $LLM_GS --dit_group_size 8 \
  --skip_train --output_dir $OUT_DIR \
  --eval_episodes 50 --n_envs 10 --suite libero_goal \
  > ${OUT_DIR}/libero_goal.log 2>&1 &

wait
echo "[watcher:$CONFIG] All 3 suites complete"
