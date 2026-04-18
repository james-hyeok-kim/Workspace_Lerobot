#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

LOG_NAME="e2e_quant.log"
PASS_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == --log=* ]]; then
        LOG_NAME="${arg#--log=}"
    else
        PASS_ARGS+=("$arg")
    fi
done
LOG_FILE="duhyeon/thanos/results/${LOG_NAME}"

python duhyeon/thanos/e2e_quant.py \
    --suites=all \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero --env.task=libero_10 \
    --eval.batch_size=10 --eval.n_episodes=50 \
    --policy.n_action_steps=10 --policy.device=cuda \
    --policy.compile_model=false \
    "${PASS_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"