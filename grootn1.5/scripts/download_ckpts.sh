#!/usr/bin/env bash
# Download GR00T-N1.5 model weights and Tacoin LIBERO finetune checkpoints.
# All files go to $HF_HOME (= /data/jameskimh/groot_n1p5/hf_cache).
# Already-cached repos are skipped automatically by huggingface-cli.

set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "[download] HF_HOME = $HF_HOME"

echo ""
echo "[1/6] nvidia/GR00T-N1.5-3B (base model) ..."
huggingface-cli download nvidia/GR00T-N1.5-3B

echo ""
echo "[2/6] lerobot/eagle2hg-processor-groot-n1p5 (Eagle tokenizer assets) ..."
huggingface-cli download lerobot/eagle2hg-processor-groot-n1p5

echo ""
echo "[3/6] Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL ..."
huggingface-cli download Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL

echo ""
echo "[4/6] Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT ..."
huggingface-cli download Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT

echo ""
echo "[5/6] Tacoin/GR00T-N1.5-3B-LIBERO-GOAL ..."
huggingface-cli download Tacoin/GR00T-N1.5-3B-LIBERO-GOAL

echo ""
echo "[6/6] Tacoin/GR00T-N1.5-3B-LIBERO-LONG ..."
huggingface-cli download Tacoin/GR00T-N1.5-3B-LIBERO-LONG

echo ""
echo "=== All downloads complete. Cached at: $HF_HUB_CACHE ==="
ls "$HF_HUB_CACHE" | grep -E "GR00T|eagle"
