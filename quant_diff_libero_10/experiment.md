# Experiment: LM+DiT NVFP4_DEFAULT_CFG — Layer-wise Capture + libero_10 Eval

## 목적

`mtq.quantize(policy, config=NVFP4_DEFAULT_CFG)` 방식으로 LM + DiT를 quantize한 뒤
libero_10의 10개 task에 대해 eval을 수행하고, 모든 quantized Linear layer의
**input / weight / output**을 task별로 캡처한다.

나중에 Duhyeon의 NVFP4 구현 결과와 layer-wise로 비교하여 두 구현 간 차이의 원인을 규명한다.

---

## Quantization 설정

| 항목 | 값 |
|------|----|
| 방법 | `mtq.quantize(NVFP4_DEFAULT_CFG)` (ModelOpt MTQ) |
| 적용 대상 | LM (`language_model`) + DiT (`gemma_expert`) |
| 제외 대상 | vision_tower, embed_tokens, lm_head, action head projections |
| Format | NVFP4 E2M1 — `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` |
| Block size | 16 (last dim 기준) |
| Scale 구조 | two-level: global (float32) + per-block (FP8 E4M3fn) |
| Algorithm | max (amax calibration) |
| SVD | 없음 |
| Activation quant | dynamic (매 forward마다 amax 계산) |
| 실행 환경 | fake quant (B200이나 TRT-LLM 미설치 → dequant 후 BF16 matmul) |

### 적용 경로 (코드)

```python
# eval_quant_sweep.py:312-320 동일 경로
lm = policy.model.paligemma_with_expert.paligemma.model.language_model
mtq.quantize(lm, config=NVFP4_DEFAULT_CFG)

expert = policy.model.paligemma_with_expert.gemma_expert
mtq.quantize(expert, config=NVFP4_DEFAULT_CFG)
```

---

## 모델

| 항목 | 값 |
|------|----|
| 모델 | `lerobot/pi05_libero_finetuned` |
| 구조 | PI05Policy → PI05Pytorch → PaliGemmaWithExpertModel |
| LM | `paligemma_with_expert.paligemma.model.language_model` (Gemma 2B, 127 layers) |
| DiT | `paligemma_with_expert.gemma_expert` (Gemma 300M action expert, 167 layers) |

---

## Evaluation 설정

| 항목 | 값 |
|------|----|
| Benchmark | LIBERO-10 (`libero_10`) |
| Task IDs | 0 ~ 9 (10개 task 전부) |
| Episodes per task | 5 |
| Batch size | 5 |
| 평가 방법 | task_id별 순차 eval (`eval_policy`) |

### Task 목록 (libero_10)

| task_id | 명령어 |
|---------|--------|
| 0 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | put both the cream cheese box and the butter in the basket |
| 2 | turn on the stove and put the moka pot on it |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | pick up the book and place it in the back compartment of the caddy |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | put both moka pots on the stove |
| 9 | put the yellow and white mug in the microwave and close it |

---

## Layer Capture 방법

- quantized Linear layer 감지: `hasattr(module, "weight_quantizer")`
- **각 task_id의 첫 번째 forward pass**에서 capture
- `register_forward_hook` → capture 완료 후 즉시 hook 제거
- 캡처 데이터: `{layer_name: {"x": tensor, "W": tensor, "y": tensor}}`
- W는 quantize 전 원본 weight (dequant된 상태)

---

## 결과 파일 구조

```
results/
├── task_commands.json          # {task_id: task_description}
├── quant_report.json           # quantized layer 목록 및 커버리지
├── layer_captures_task0.pt     # task 0의 모든 quantized layer {x, W, y}
├── layer_captures_task1.pt
├── ...
├── layer_captures_task9.pt
├── eval_task0.json             # task 0 eval 결과 (success rate 등)
├── eval_task1.json
├── ...
├── eval_task9.json
└── eval_summary.json           # 10개 task 통합 결과
```

---

## 실행 방법

```bash
cd /home/jameskimh/workspace/Workspace_Lerobot/quant_diff_libero_10

# test mode (task 0, 1 episode)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_nvfp4_lm_dit.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --task_ids 0 \
    --n_episodes 1 \
    --batch_size 1 \
    --output_dir results_test

# full mode (task 0~9, 5 episodes each)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_nvfp4_lm_dit.py \
    --pretrained_path lerobot/pi05_libero_finetuned \
    --n_episodes 5 \
    --batch_size 5 \
    --output_dir results
```

---

## Batch Size 의미 및 영향

### batch_size란?

`batch_size = n_envs` — 동시에 실행하는 **병렬 환경(sub-env) 수**.

```
batch_size=5, n_episodes=10:
  env0: episode 0, 5        (2 rounds)
  env1: episode 1, 6
  ...
  env4: episode 4, 9

batch_size=10, n_episodes=10:
  env0: episode 0            (1 round)
  env1: episode 1
  ...
  env9: episode 9
```

- `n_episodes` = 총 시도 횟수 (success rate 분모)
- `batch_size` = 그 시도를 몇 개씩 묶어 병렬 처리하느냐

### MTQ vs Duhyeon에서 batch_size 영향

| 방법 | batch_size 영향 | 이유 |
|------|----------------|------|
| MTQ (NVFP4_DEFAULT_CFG) | 거의 없음 | weight-only fake-quant, per-block scale 고정 |
| Duhyeon (forward hook) | **있음** | activation scale이 batch 전체 global amax 기준 |

Duhyeon hook 내부:
```python
x_amax = x_2d.abs().max()   # ← batch 전체(n_envs개) observation의 global max
decode_scale = x_amax / (fp_max * FP8_MAX)
```

- `batch_size=5`: 5개 env observation의 amax → scale 결정
- `batch_size=10`: 10개 env observation의 amax → scale이 더 커질 가능성
- outlier가 한 env에 있으면 나머지 env 전체의 quantization precision이 낮아짐

---

## Init State Controller (0407 vs 0417)

### Init State란?

각 task별 `.init` 파일에 저장된 **사전 정의된 초기 배치** (로봇 joint position, 오브젝트 pose 등).
`init_state_id`로 인덱싱. **random seed와는 별개**.

```
init_state_id  →  init_states[id % len]  →  로봇/오브젝트 초기 배치
random seed    →  시뮬레이션 물리 랜덤성 (미세 흔들림 등)
```

### 기본 동작 (0407, DH_WO_INIT)

```python
def reset(self, seed=None, **kwargs):
    ...
    set_init_state(init_states[init_state_id])
    init_state_id += _reset_stride   # ← 항상, seed 무관하게 증가
```

**문제:** episode 중 성공 시 `step()`이 내부적으로 `reset(seed=None)` 자동 호출
→ `init_state_id`가 의도치 않게 증가 → 다음 episode가 예상과 다른 init state 사용

성공률이 높을수록 autoreset 빈도 증가 → init state 스킵 누적

### Controller 동작 (0417, DH_INIT)

```python
def _controlled_reset(seed=None, **kwargs):
    if seed is not None:     # eval 루프의 명시적 reset → 카운터 증가
        env.init_state_id = counter[0]
        counter[0] += stride
    else:                    # step() 중 autoreset (seed=None) → 무시
        pass
```

`seed is not None` = eval 루프가 의도적으로 새 episode 시작
`seed is None`     = episode 중간 성공 후 자동 reset → init_state_id 유지

### 실질적 차이

| | DH_WO_INIT (0407) | DH_INIT (0417) |
|---|---|---|
| autoreset 시 init_state_id | 증가 (스킵 발생) | 유지 |
| 실제 사용 init states | 불규칙 | 정확히 0~(n_episodes-1) |
| 재현성 | 낮음 | 높음 |
| 성공률 높을수록 영향 | 커짐 | 없음 |

### 실험 구성

| 실험명 | batch_size | n_episodes | init controller | 결과 디렉토리 |
|--------|-----------|-----------|----------------|--------------|
| MTQ_bs5 | 5 | 10 | — | `results_mtq_10ep` |
| DH_bs5 | 5 | 10 | ❌ | `results_duhyeon_10ep` |
| DH_WO_bs10 | 10 | 10 | ❌ | `results_duhyeon_bs10` |
| DH_INIT_bs10 | 10 | 10 | ✅ (0417) | `results_duhyeon_init_bs10` |

---

## 향후 비교 계획

Duhyeon 코드 결과 수령 후:
- `layer_captures_task{tid}.pt` (MTQ) vs Duhyeon의 동일 파일 비교
- per-layer MSE / SNR / abs_err 비교
- 차이가 큰 layer 식별 → eval 결과 차이 원인 규명

---

# Experiment: 4-Way Comparison — My Code vs Duhyeon ± Init Control

## 목적

기존 실험들은 `eval_policy(start_seed=None)`으로 seed가 고정되지 않아 재현성이 낮았고,
`eval_duhyeon_init.py`의 init_state_controller도 `start_seed`가 없어 사실상 no-op이었다.

이 실험에서 **4개 조건을 동일한 실험 조건**(`n_action_steps=10`, `batch_size=10`, `n_episodes=10`, `start_seed=1000`) 하에 비교하여:
1. MTQ vs Duhyeon 구현의 순수한 quant 알고리즘 차이를 측정
2. init_state_controller 유무의 영향을 정량화
3. 기존 5% 차이가 seeding 문제였는지 quant 차이였는지 규명

## 실험 조건

| 항목 | 값 |
|------|-----|
| 모델 | `lerobot/pi05_libero_finetuned` |
| Benchmark | LIBERO-10 (`libero_10`) |
| Task IDs | 0 ~ 9 (10개 전부) |
| n_episodes (per task) | 10 |
| batch_size (n_envs) | 10 |
| n_action_steps | **10** (기존 default 50에서 변경) |
| start_seed | **1000** (기존 None에서 변경) |
| use_amp | False |

## 4개 조건

| 조건 | Quant 방법 | init_control | 결과 디렉토리 |
|------|-----------|:------------:|--------------|
| mine | MTQ `NVFP4_DEFAULT_CFG` (LM+DiT) | ❌ | `/data/jameskimh/james_lerobot_results/quant_diff_libero_10/results_4way/mine` |
| mine_init | MTQ `NVFP4_DEFAULT_CFG` (LM+DiT) | ✅ | `/data/jameskimh/james_lerobot_results/quant_diff_libero_10/results_4way/mine_init` |
| dh | Duhyeon `nvfp4_bmm` forward hook | ❌ | `/data/jameskimh/james_lerobot_results/quant_diff_libero_10/results_4way/dh` |
| dh_init | Duhyeon `nvfp4_bmm` forward hook | ✅ | `/data/jameskimh/james_lerobot_results/quant_diff_libero_10/results_4way/dh_init` |

## init_state_controller 동작 원리

`start_seed=1000`을 `eval_policy`에 전달하면 각 episode가 `seed=1000, 1001, ...`으로 seeded reset을 받는다.
controller는 `seed is not None`(=외부 eval reset) 때만 `init_state_id`를 counter로 고정하고,
`seed is None`(=step() 내 autoreset) 때는 `init_state_id`를 건드리지 않는다.

- **controller 없을 때**: autoreset 때마다 `init_state_id += stride` → 성공률 높을수록 init state 스킵 누적
- **controller 있을 때**: 정확히 episode 0~9가 init_state 0~9에 대응

## 코드 변경 사항

`eval_nvfp4_lm_dit.py`, `eval_duhyeon_nocapture.py` 양쪽에:
- `--start_seed` (default=1000) 추가 → `eval_policy(start_seed=...)` 전달
- `--n_action_steps` (default=10) 추가 → `policy_cfg.n_action_steps` 덮어쓰기
- `--init_control` 플래그 추가 → `_install_init_state_controller()` 설치
- `eval_summary.json`에 위 설정값 기록

## 실행 방법

```bash
cd /home/jameskimh/workspace/Workspace_Lerobot/quant_diff_libero_10
bash run_4way_comparison.sh
```
