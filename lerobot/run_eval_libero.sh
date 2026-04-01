#!/bin/bash
# ⚙️ 1. 사용자 설정 (James 전용)
TEST_MODE=1  # 1: 테스트용(1개 태스크), 0: 전체 태스크 실행

# 📂 2. 경로 및 환경 설정
BASE_DIR="$(pwd)"
LIBERO_DIR="${BASE_DIR}/LIBERO"
LEROBOT_DIR="${BASE_DIR}/lerobot"

# 📊 3. 로그 및 모드 설정
MODE_NAME=$([ "$TEST_MODE" -eq 1 ] && echo "TEST" || echo "FULL")
LOG_DIR="${BASE_DIR}/logs/${MODE_NAME}"
mkdir -p "$LOG_DIR"

# 🌍 4. 환경 변수 주입 (LIBERO 데이터셋 및 경로 고정)
export LIBERO_DATASET_PATH="/data/james_libero_datasets"
# 대화형 프롬프트 방지를 위해 LEROBOT_HOME 자동 지정
export HF_LEROBOT_HOME="${BASE_DIR}/.lerobot_home"

# .env 파일이 존재하면 읽어와서 export 합니다.
if [ -f ~/.env ]; then
  export $(echo $(cat .env | sed 's/#.*//g' | xargs) | envsubst)
fi

export PYTHONPATH="${LEROBOT_DIR}/src:${LIBERO_DIR}:${PYTHONPATH}"
# OSMesa 대신 EGL 사용 (속도 향상!)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NVIDIA_DRIVER_CAPABILITIES=all

# 🤖 5. 모델 정보 (Core 설정 기반)
MODEL_ID="lerobot/pi05_libero_finetuned"
POLICY_TYPE="pi05"

# 🚀 6. 실행 태스크 및 에피소드 설정
if [ "$TEST_MODE" -eq 1 ]; then
    echo "⚠️ [TEST MODE] 활성화 (로그: $LOG_DIR)"
    CATEGORIES=("libero_spatial")
    EPISODES=20      # Core 커맨드 기준
    ACTION_STEPS=10  # Core 커맨드 기준
else
    echo "🚀 [FULL MODE] 활성화 (로그: $LOG_DIR)"
    CATEGORIES=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")
    EPISODES=20      # 전체 모드에서도 Core의 안정적인 수치 유지
    ACTION_STEPS=10
fi

# 🔄 7. 메인 평가 루프
for CAT in "${CATEGORIES[@]}"
do
    CAT_LOG="${LOG_DIR}/${CAT}_experiments.log"
    echo -e "\n📊 카테고리 시작: $CAT"

    printf 'n\n' | CUDA_VISIBLE_DEVICES=0,1 HF_LEROBOT_HOME="${BASE_DIR}/.lerobot_home" lerobot-eval \
      --policy.type="${POLICY_TYPE}" \
      --policy.pretrained_path="${MODEL_ID}" \
      --env.type=libero \
      --env.task="${CAT}" \
      --eval.batch_size=1 \
      --eval.n_episodes="${EPISODES}" \
      --policy.n_action_steps="${ACTION_STEPS}" \
      --env.control_mode=relative \
      --policy.use_amp=true \
      --policy.dtype=bfloat16 \
      --job_name="james_pi05_eval" 2>&1 | tee "$CAT_LOG"

    echo -e "\n✅ $CAT 테스트 종료"
done
