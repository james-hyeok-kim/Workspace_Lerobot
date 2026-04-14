#!/bin/bash
# Component sweep: target=lm / target=dit for 3 configs × 4 tasks × 50 episodes
# Total: 24 experiments, ~8 hours sequential

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASKS="libero_spatial libero_object libero_goal libero_10"
OUT_BASE="logs/mixed_int_quant/component_sweep"
N=50
BS=5

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local task="$1"; shift
    log "START: $* --task $task"
    python eval_mixed_int_quant.py \
        --task "$task" --n_episodes $N --batch_size $BS \
        --output_dir "$OUT_BASE/$task" \
        "$@"
    log "DONE:  $* --task $task  exit=$?"
}

# ─────────────────────────────────────────────────────────────────────────────
# Config 1: lm8/dit8 INT16, target=lm  (LM only, INT8 weight, INT16 act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 1/6: lm8 INT16 act, target=lm ==="
for task in $TASKS; do
    run_exp "$task" --target lm \
        --lm_min_bits 8 \
        --lm_a_min_bits 16 --dit_a_min_bits 16 \
        --force_act_quant --act_quant_mode per_block
done

# ─────────────────────────────────────────────────────────────────────────────
# Config 2: lm8/dit8 INT16, target=dit  (DiT only, INT8 weight, INT16 act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 2/6: dit8 INT16 act, target=dit ==="
for task in $TASKS; do
    run_exp "$task" --target dit \
        --dit_min_bits 8 \
        --lm_a_min_bits 16 --dit_a_min_bits 16 \
        --force_act_quant --act_quant_mode per_block
done

# ─────────────────────────────────────────────────────────────────────────────
# Config 3: lm6, target=lm  (LM only, natural INT4-8 weight, natural act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 3/6: lm6 per_block, target=lm ==="
for task in $TASKS; do
    run_exp "$task" --target lm \
        --lm_min_bits 6 --act_quant_mode per_block
done

# ─────────────────────────────────────────────────────────────────────────────
# Config 4: dit4, target=dit  (DiT only, natural INT4-8 weight, natural act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 4/6: dit4 per_block, target=dit ==="
for task in $TASKS; do
    run_exp "$task" --target dit \
        --dit_min_bits 4 --act_quant_mode per_block
done

# ─────────────────────────────────────────────────────────────────────────────
# Config 5: lm1, target=lm  (LM only, aggressive INT1-8 weight, natural act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 5/6: lm1 per_block, target=lm ==="
for task in $TASKS; do
    run_exp "$task" --target lm \
        --lm_min_bits 1 --act_quant_mode per_block
done

# ─────────────────────────────────────────────────────────────────────────────
# Config 6: dit1, target=dit  (DiT only, aggressive INT1-8 weight, natural act)
# ─────────────────────────────────────────────────────────────────────────────
log "=== Config 6/6: dit1 per_block, target=dit ==="
for task in $TASKS; do
    run_exp "$task" --target dit \
        --dit_min_bits 1 --act_quant_mode per_block
done

log "=== ALL 24 EXPERIMENTS COMPLETE ==="
