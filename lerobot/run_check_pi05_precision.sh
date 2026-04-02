#!/bin/bash

# 📂 1. 경로 설정
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs/inspection"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/precision_check.log"

# 🔑 2. 인증 설정 (반드시 ~/.env 에 HF_TOKEN=... 이 있어야 합니다)
if [ -f ~/.env ]; then
    export $(grep -v '^#' ~/.env | xargs)
    echo "🔐 HF_TOKEN 주입 완료"
else
    echo "⚠️  ~/.env 파일을 찾을 수 없습니다. (인증 에러 위험)"
fi

# 🌍 3. 환경 변수
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# 🤖 4. 설정
MODEL_ID="lerobot/pi05_libero_finetuned"
OUTPUT_JSON="pi05_precision_report.json"

echo "🔍 정밀도 조사 시작..."

# 🚀 5. 실행 (환경 로드 없이 모델만 분석)
python -u check_pi05_precision.py \
    --model_id "$MODEL_ID" \
    --output "$OUTPUT_JSON" 2>&1 | tee "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 성공! 결과: $OUTPUT_JSON"
else
    echo "❌ 실패! 로그를 확인하세요: $LOG_FILE"
fi