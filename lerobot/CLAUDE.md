# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeRobot is a PyTorch-based robotics library by HuggingFace that provides models, datasets, and tools for real-world robotics. It supports training and deploying policies on physical robots, simulated environments, and the HuggingFace Hub.

## Installation

```bash
# Install in dev mode (required for src/ layout)
pip install -e ".[dev,test]"

# Install with specific extras (e.g., for pi0/pi05 policies)
pip install -e ".[pi,libero]"

# Install pre-commit hooks
pre-commit install
```

## Common Commands

### Training

```bash
# Train a policy (uses draccus config system with dot-notation CLI overrides)
lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_insertion_scripted \
  --policy.type=act \
  --output_dir=outputs/train/my_run

# Train from a pretrained config on the Hub
lerobot-train --config_path=lerobot/act-aloha-sim-transfer-cube-vit-s
```

### Evaluation

```bash
# Evaluate a trained policy
lerobot-eval --config_path=outputs/train/my_run/checkpoints/last

# Run NVFP4 quantization evaluation (custom script in repo root)
bash run_nvfp4.sh

# Check model weight precision
bash run_check_pi05_precision.sh
```

### Data & Visualization

```bash
lerobot-dataset-viz --repo-id=lerobot/aloha_sim_insertion_scripted
lerobot-record --robot-type=so100 ...
lerobot-info
```

### Tests

```bash
# Full test suite (requires git-lfs artifacts)
git lfs install && git lfs pull
pytest -sv ./tests

# Run a single test file
pytest -sv tests/policies/pi0_pi05/test_pi05.py

# Run a specific test
pytest -sv tests/policies/test_policies.py::test_act_policy
```

### Linting & Formatting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Format only
ruff format src/ tests/

# Lint with auto-fix
ruff check --fix src/ tests/

# Type checking (partial — only some modules have mypy enabled)
mypy src/lerobot/configs/ src/lerobot/motors/ src/lerobot/cameras/
```

## Architecture

### Package Layout

Source lives under `src/lerobot/` (editable install via `pip install -e .`). The key modules:

- **`configs/`** — Dataclass-based config system using `draccus`. `PreTrainedConfig` is the base for all policy configs; `TrainPipelineConfig` is the training entrypoint config.
- **`policies/`** — Each policy is a subdirectory with `configuration_*.py`, `modeling_*.py`, and `processor_*.py`. All policies extend `PreTrainedPolicy` (in `policies/pretrained.py`). The factory (`policies/factory.py`) instantiates policies by type string.
- **`processor/`** — Input/output processing pipeline. `PolicyProcessorPipeline` applies a chain of transformations (normalization, renaming, tokenization, device placement) between dataset batches and policy inputs.
- **`datasets/`** — `LeRobotDataset` wraps Parquet + MP4 data. Supports HF Hub push/pull, streaming, multi-dataset merging.
- **`robots/`** — Hardware abstraction. Each robot subdir implements the `Robot` interface: `connect()`, `get_observation()`, `send_action()`.
- **`motors/`** — Low-level motor driver abstraction for Dynamixel, Feetech, Damiao, Robstride buses.
- **`cameras/`** — Camera drivers (OpenCV, RealSense, ZMQ, Reachy2).
- **`teleoperators/`** — Teleoperation devices (leaders, gamepads, keyboard, phone).
- **`envs/`** — Simulation environment wrappers (Aloha, PushT, LIBERO, MetaWorld).
- **`scripts/`** — Entry point scripts mapped to CLI commands in `pyproject.toml`.
- **`optim/`** — Optimizer and LR scheduler configs/factories.

### Config System (draccus)

Configs are Python dataclasses decorated with `draccus.ChoiceRegistry` for polymorphic dispatch. CLI overrides use dot-notation (e.g. `--policy.type=act --policy.n_obs_steps=2`). The `configs/parser.py:wrap` decorator handles:
- Loading configs from HF Hub via `--config_path=<hub_id_or_local_path>`
- Plugin discovery for external robot/env extensions via `--env.discover_packages_path=my_pkg`
- Filtering path args before passing to draccus

### Policy Convention

Each policy directory follows this pattern:
- `configuration_*.py`: dataclass extending `PreTrainedConfig`, registered via `@PreTrainedConfig.register_subclass("name")`
- `modeling_*.py`: `nn.Module` extending `PreTrainedPolicy`; must implement `select_action(batch)` and `forward(batch)` for training loss
- `processor_*.py`: input/output feature processing for that policy

### Custom Scripts (This Workspace)

This workspace adds NVFP4 quantization experiments on top of upstream LeRobot:

- `eval_nvfp4_mtq.py` — NVIDIA ModelOpt (MTQ) quantization evaluation for LeRobot policies
- `run_nvfp4.sh` — Shell runner for NVFP4 experiments; requires `TensorRT-Model-Optimizer` on `PYTHONPATH`
- `check_pi05_precision.py` + `run_check_pi05_precision.sh` — Inspect weight dtype distribution of a HF Hub model

These scripts require `modelopt` from `TensorRT-Model-Optimizer` (located at `../TensorRT-Model-Optimizer`). Set `MODEL_OPT_DIR` accordingly or source `run_nvfp4.sh` which sets it automatically.

### Environment Variables

- `LIBERO_DATASET_PATH` — Local path to LIBERO dataset files
- `HF_LEROBOT_HOME` — Override default LeRobot cache directory
- `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` — Required for headless GPU rendering
- `HF_TOKEN` — HuggingFace token (load from `~/.env`)


# Quantization Research — LeRobot (pi0.5 & GROOT N1.5)

## 프로젝트 목적
LeRobot의 pi0.5 및 GROOT N1.5 모델에 Post-Training Quantization(PTQ)을 적용하여
evaluation accuracy를 최대한 유지하면서 모델 데이터를 효과적으로 quantization하는 방법을 탐색한다.

---

## 모델
| 모델 | 설명 |
|------|------|
| pi0.5 | LeRobot diffusion-based policy |
| GROOT N1.5 | LeRobot transformer-based policy |

---

## 평가 기준 (LIBERO benchmark)

### Metric
LIBERO 모델의 5가지 task success rate:
| Metric | Dataset 경로 |
|--------|-------------|
| Spatial | `/data/james_libero_datasets/libero_spatial/` |
| Object | `/data/james_libero_datasets/libero_object/` |
| Goal | `/data/james_libero_datasets/libero_goal/` |
| Long | `/data/james_libero_datasets/libero_10/` |
| Avg | 위 4개 평균 |

> `libero_90/`은 대규모 pretraining용 데이터셋

### Baseline
- fp32 또는 bf16 체크포인트 기준
- 허용 accuracy drop: **탐색 중** (현 단계에서는 drop 최소화 방향으로 실험)

---

## 실험 범위

### Quantization 방식
- **PTQ only** (QAT 제외)

### 타겟 Bit-width / Format
| 타입 | 비고 |
|------|------|
| INT8 | 기본 baseline quant |
| INT4 | 적극적 압축 |
| INT2 | 극한 압축, accuracy 트레이드오프 탐색 |
| NVFP4 | NVIDIA Blackwell FP4 format |
| Custom format | 새로운 quant 방식 직접 설계 |

### 구현 방식
- **직접 구현 (custom)** 위주
- 필요 시 기존 라이브러리(bitsandbytes, GPTQ, AWQ 등) 참조

---

## 하드웨어 환경
- NVIDIA GPU 서버 (A100 / H100급)

---

## 주의사항
- LeRobot diffusion policy(pi0.5)는 action chunking 구조로 activation range가 불규칙할 수 있음
- Quantization 적용 시 모델 아키텍처별 특성(attention, MLP, conv 등) 고려 필요
- Custom format 실험 시 구현 내용과 근거를 실험 로그에 명확히 기록할 것