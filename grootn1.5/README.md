# GR00T-N1.5 LIBERO 재현 실험

`nvidia/GR00T-N1.5-3B` 기반으로 LIBERO 4-suite success rate 를 측정하고
QuantVLA 논문 Table 2의 FP16 baseline 수치와 비교한다.

## 디렉토리 레이아웃

| 경로 | 내용 |
|------|------|
| `grootn1.5/scripts/` | 실행 스크립트 (이 repo) |
| `/data/jameskimh/groot_n1p5/hf_cache/` | HuggingFace 모델 캐시 |
| `/data/jameskimh/groot_n1p5/results/` | 평가 결과 json |
| `/data/jameskimh/groot_n1p5/logs/` | 실행 로그 |

## 의존성 확인

```bash
# lerobot (workspace editable, PYTHONPATH 로 로드)
PYTHONPATH=/home/jovyan/workspace/Workspace_Lerobot/lerobot/src \
  python -c "from lerobot.policies.groot import GrootPolicy; print('OK')"

# flash-attn (GR00T 필수, 없으면 설치)
python -c "import flash_attn; print(flash_attn.__version__)" || \
  pip install --user "flash-attn>=2.5.9,<3" --no-build-isolation

# LIBERO 시뮬레이터 (시스템 site-packages 에 이미 설치됨)
python -c "from libero.libero.envs import OffScreenRenderEnv; print('LIBERO OK')"
```

## 체크포인트 다운로드

```bash
cd /home/jovyan/workspace/Workspace_Lerobot/grootn1.5
bash scripts/download_ckpts.sh
```

다운로드 대상 (모두 `/data/jameskimh/groot_n1p5/hf_cache/hub/` 로 저장):
- `nvidia/GR00T-N1.5-3B` — base model (~6 GB)
- `lerobot/eagle2hg-processor-groot-n1p5` — Eagle tokenizer assets
- `Tacoin/GR00T-N1.5-3B-LIBERO-{SPATIAL,OBJECT,GOAL,LONG}` — suite별 finetune

## 스모크 테스트 (1 task × 1 episode, ~5분)

```bash
source scripts/_env.sh
python scripts/smoke_groot.py \
    --suite libero_spatial \
    --task_ids 0 \
    --n_episodes 1
# 결과: /data/jameskimh/groot_n1p5/results/smoke_<timestamp>/libero_spatial_smoke.json
```

## 풀 4-suite 평가 (10 task × 10 ep × 4 suite = 400 ep, 수 시간)

```bash
source scripts/_env.sh
RUN_ID=$(date +%Y%m%d_%H%M%S)
RESULTS=$GROOT_OUTPUT_ROOT/results/$RUN_ID
mkdir -p $RESULTS

declare -A SUITE_CKPT=(
    [libero_spatial]="Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL"
    [libero_object]="Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT"
    [libero_goal]="Tacoin/GR00T-N1.5-3B-LIBERO-GOAL"
    [libero_10]="Tacoin/GR00T-N1.5-3B-LIBERO-LONG"
)

for suite in libero_spatial libero_object libero_goal libero_10; do
    ckpt=${SUITE_CKPT[$suite]}
    python scripts/smoke_groot.py \
        --policy_path "$ckpt" \
        --suite "$suite" \
        --n_episodes 10 \
        --output_dir "$RESULTS/$suite"
done
```

결과 json 의 `overall.pc_success` 를 4 suite 평균해서 QuantVLA 표와 비교.

## QuantVLA FP16 baseline 참조값

| Suite | QuantVLA FP16 ref | 출처 |
|-------|-------------------|------|
| LIBERO-Spatial | 92.0 % | QuantVLA Tab. 2, 8 denoise steps |
| LIBERO-Object | 92.0 % | |
| LIBERO-Goal | 86.0 % | |
| LIBERO-Long | 76.0 % | |
| **Avg** | **86.5 %** | |

NVIDIA GR00T reference (같은 숫자): Spatial 92, Object 92, Long 76, Avg ~87%.
Tacoin 커뮤니티 finetune 으로 얻을 수 있는 결과: 평균 ≈ 87%, suite별 ±10%p 차이 예상.

## 알려진 제약

- QuantVLA 의 FP16 baseline 은 NVIDIA 내부 LIBERO finetune 체크포인트를 사용한 것으로 N1.5용
  공식 LIBERO 체크포인트는 비공개. 본 재현은 Tacoin 커뮤니티 체크포인트를 사용한다.
- 정확한 재현을 원한다면 `nvidia/GR00T-N1.5-3B` 에서 각 suite 별 30k step finetune 필요.
- GPU: `CUDA_VISIBLE_DEVICES=2` (lerobot 작업 GPU 할당 규칙).
