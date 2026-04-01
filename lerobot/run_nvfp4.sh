#!/bin/bash
# MTQ NVFP4 실험 실행 스크립트

# mtq 라이브러리가 없다면 설치 시도 (필요시 주석 해제)
# pip install microscaling-toolkit

python eval_nvfp4_mtq.py \
    --policy_paths "google/gemma-vla" "physical-intelligence/pi0" \
    --env_name "lerobot/pusht_image"