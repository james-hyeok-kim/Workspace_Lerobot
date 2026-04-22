"""Stage 1 — SnapFlow Distillation Training (사용자가 직접 실행).

This script trains a 1-NFE student policy from the 10-NFE teacher.
After training, the student checkpoint is saved to
artifacts/stage1_student.safetensors.

Then run:
  bash scripts/run_stage.sh 1
to evaluate the distilled student.

TODOs (see snapflow/distill_loss.py):
  - Implement _get_velocity_at() using PI05Pytorch.embed_prefix + .denoise_step
  - Verify dataset repo_id is accessible

Usage:
    python scripts/stage1_snapflow_distill.py --config configs/stage1_snapflow.yaml
"""

import argparse
import sys
from pathlib import Path

for _p in [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2] / "lerobot" / "src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snapflow.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_snapflow.yaml")
    args = parser.parse_args()
    train(args.config)
