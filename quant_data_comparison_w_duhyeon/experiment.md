# NVFP4 Quantization Comparison — Experiment

## 목적

Duhyeon의 `mx_block_quant`(custom 구현)와 ModelOpt MTQ(`NVFP4QTensor`) 및
`eval_mixed_fp_quant._quantize_nvfp4`(직접 구현) 간의 수치적 동등성을 검증한다.

---

## 대상 레이어

```
model.paligemma_with_expert.gemma_expert.model.layers.0.mlp.gate_proj
```

pi0.5 모델의 Gemma Expert 첫 번째 레이어 MLP gate projection.

| 텐서 | Shape | dtype |
|------|-------|-------|
| Input (x) | `[1, 50, 1024]` | bfloat16 |
| Weight (W) | `[4096, 1024]` | bfloat16 |

---

## 비교 대상 (4 Methods)

| ID | 구현 출처 | 핵심 방식 |
|----|----------|----------|
| **M1** | `eval_nvfp4_mtq.py` | ModelOpt `NVFP4QTensor.quantize()` |
| **M2** | `nvfp4_single_layer.py` | 동일 (`NVFP4QTensor`) — single-layer standalone |
| **M3** | `eval_mixed_fp_quant.py` `_quantize_nvfp4` | 직접 구현 (single-level FP16 scale) |
| **M4** | `duhyeon/compare.py` `mx_block_quant` | Duhyeon 직접 구현 (two-level FP8 scale) |

---

## Quantization 설정

### M1/M2, M4 — Two-level Scale (NVFP4_DEFAULT_CFG 동일 구조)

| 항목 | 값 |
|------|----|
| Format | NVFP4 (E2M1): `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` |
| Block size | 16 (last dim 기준) |
| Global scale | `amax / (6.0 × 448.0)` → float32 |
| Per-block scale | `block_amax / (6.0 × global)` → **FP8 E4M3fn** |
| FP4 결정 방식 | M1/M2: `searchsorted` + odd-index tie-break / M4: log2 기반 `_fp_quant` |

### M3 — Single-level Scale (eval_mixed_fp_quant)

| 항목 | 값 |
|------|----|
| Format | NVFP4 (E2M1) |
| Block size | 16 |
| Global scale | **없음** |
| Per-block scale | `block_amax / 6.0` → **FP16** |
| FP4 결정 방식 | `_snap_to_grid` (nearest neighbor) |

---

## 실험 단계

### Step 1 — NVFP4 single-layer quantization

```bash
./run_01_nvfp4_single_layer.sh
```

- `nvfp4_single_layer.py` 실행
- `NVFP4QTensor.quantize()` 로 x, W를 quantize → dequantize
- 출력: `nvfp4_quant_result.pt`

### Step 2 — 단일 레이어 결과 시각화

```bash
./run_02_plot_nvfp4_comparison.sh
```

- `plot_nvfp4_comparison.py` 실행
- input / weight / output에 대해 3×4 레이아웃 그래프 생성
- 출력: `nvfp4_comparison.png`

### Step 3 — 4-way 비교 (핵심)

```bash
./run_03_compare_4methods.sh
```

- `compare_4methods.py` 실행
- `duhyeon/output_compare_dh.pt` 에서 x, W, y_fp, y_fq_dh 로드
- M1/M2(MTQ), M3(mixed), M4(Duhyeon) 결과를 FP baseline과 비교
- 출력: `nvfp4_4methods_comparison.png` + 콘솔 수치

### Step 4 — Duhyeon vs DEFAULT_CFG 2-way 비교

```bash
./run_04_plot_compare.sh
```

- `plot_compare.py` 실행
- `compare_result.pt` 에서 y_fp, y_fq_dh, y_mine 로드
- Duhyeon / DEFAULT_CFG / (Duhyeon − DEFAULT_CFG) 3-row 그래프
- 출력: `nvfp4_comparison_w_duhyeon.png`

---

## 파일 구조

```
quant_data_comparison_w_duhyeon/
├── run_01_nvfp4_single_layer.sh      # Step 1 실행 스크립트
├── run_02_plot_nvfp4_comparison.sh   # Step 2 실행 스크립트
├── run_03_compare_4methods.sh        # Step 3 실행 스크립트 (핵심)
├── run_04_plot_compare.sh            # Step 4 실행 스크립트
│
├── nvfp4_single_layer.py             # Step 1 스크립트
├── plot_nvfp4_comparison.py          # Step 2 스크립트
├── compare_4methods.py               # Step 3 스크립트
├── plot_compare.py                   # Step 4 스크립트
│
├── nvfp4_quant_result.pt             # Step 1 결과
├── compare_result.pt                 # Step 3 중간 결과 (y_fp, y_fq_dh, y_mine)
│
├── nvfp4_comparison.png              # Step 2 그래프
├── nvfp4_4methods_comparison.png     # Step 3 그래프
├── nvfp4_comparison_w_duhyeon.png    # Step 4 그래프
│
├── duhyeon/
│   ├── compare.py                    # Duhyeon 원본 비교 코드
│   ├── input_weight_captured.pt      # 원본 x, W 텐서
│   └── output_compare_dh.pt          # x, W, y_fp, y_fq_dh
│
├── experiment.md                     # 실험 설명 (이 파일)
└── experiment_result.md              # 실험 결과
```
