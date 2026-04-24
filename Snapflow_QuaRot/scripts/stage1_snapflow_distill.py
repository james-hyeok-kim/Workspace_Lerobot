"""Stage 1 — SnapFlow distillation training entry point.

Run (single GPU, smoke test):
    CUDA_VISIBLE_DEVICES=2 python scripts/stage1_snapflow_distill.py \
        --max_steps 5 --batch_size 1 --num_workers 0

Run (2 GPUs, full training):
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
        scripts/stage1_snapflow_distill.py \
        --train_steps 10000 --batch_size 4

After training, evaluate:
    bash scripts/run_stage.sh 1
"""

import argparse
import sys
from pathlib import Path

for _p in [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2] / "lerobot" / "src"),
    str(Path(__file__).resolve().parents[2] / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snapflow.trainer import (
    PRETRAINED_PATH, NORMALIZER_STATS_PATH, DATASET_PATH, OUTPUT_PATH, train
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--normalizer_stats_path", default=NORMALIZER_STATS_PATH)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--train_steps", type=int, default=10_000)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override train_steps (quick smoke test)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    steps = args.max_steps if args.max_steps is not None else args.train_steps

    train(
        pretrained_path=args.pretrained_path,
        normalizer_stats_path=args.normalizer_stats_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        train_steps=steps,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_every=args.save_every,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
