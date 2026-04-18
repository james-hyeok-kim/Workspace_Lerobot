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

## 향후 비교 계획

Duhyeon 코드 결과 수령 후:
- `layer_captures_task{tid}.pt` (MTQ) vs Duhyeon의 동일 파일 비교
- per-layer MSE / SNR / abs_err 비교
- 차이가 큰 layer 식별 → eval 결과 차이 원인 규명
