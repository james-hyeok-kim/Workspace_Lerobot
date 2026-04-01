#!/bin/bash
# RPCA/SVD 실험 실행 스크립트
python eval_rpca_svd.py \
    --policy_paths "google/gemma-vla" "physical-intelligence/pi0" \
    --rank 32 \
    --alpha 0.5 \
    --outlier_ratio 0.01 \
    --env_name "lerobot/pusht_image"