"""SnapFlow distillation trainer — DDP on GPU 2,3.

Usage:
    torchrun --nproc_per_node=2 scripts/stage1_snapflow_distill.py \
        --dataset_path /data/jameskimh/james_libero_datasets/libero_10 \
        --pretrained_path /data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned \
        --output_path artifacts/stage1_student.safetensors \
        --train_steps 10000

Or with explicit CUDA_VISIBLE_DEVICES=2,3:
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 scripts/stage1_snapflow_distill.py ...

Smoke test (5 steps, batch 1):
    CUDA_VISIBLE_DEVICES=2 python scripts/stage1_snapflow_distill.py \
        --max_steps 5 --batch_size 1 --num_workers 0
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DATASET_PATH = "/data/jameskimh/james_libero_datasets/libero_10"
OUTPUT_PATH = "artifacts/stage1_student.safetensors"


def _setup_paths():
    root = Path(__file__).resolve().parents[2]
    for p in [
        str(root / "Snapflow_QuaRot"),
        str(root / "lerobot" / "src"),
        str(root / "TensorRT-Model-Optimizer"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_ddp() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_ddp() else 1


def _load_policy(pretrained_path: str, device: torch.device):
    """Load PI05Policy with n_action_steps=10 and num_inference_steps=10."""
    import torch._dynamo as _dynamo
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.envs.configs import LiberoEnv

    _dynamo.reset()
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = str(device)
    policy_cfg.use_amp = False
    policy_cfg.n_action_steps = 10
    policy_cfg.num_inference_steps = 10

    env_cfg = LiberoEnv(task="libero_10", task_ids=list(range(10)))
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    return policy.to(device)


def train(
    pretrained_path: str = PRETRAINED_PATH,
    normalizer_stats_path: str = NORMALIZER_STATS_PATH,
    dataset_path: str = DATASET_PATH,
    output_path: str = OUTPUT_PATH,
    train_steps: int = 10_000,
    batch_size: int = 4,
    lr: float = 1e-5,
    grad_clip: float = 1.0,
    save_every: int = 1000,
    num_workers: int = 4,
    seed: int = 42,
):
    _setup_paths()

    # DDP init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    use_ddp = "LOCAL_RANK" in os.environ and torch.cuda.device_count() > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        if _rank() == 0:
            log.info(f"DDP initialized: {_world_size()} processes")

    from snapflow.data import make_libero_dataloader, LiberoHDF5Dataset
    from snapflow.distill_loss import SnapFlowLoss
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # ── Load policies ────────────────────────────────────────────────────────
    if _rank() == 0:
        log.info("Loading teacher policy (frozen)...")
    teacher = _load_policy(pretrained_path, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if _rank() == 0:
        log.info("Loading student policy (trainable)...")
    import copy
    student = copy.deepcopy(teacher)
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)

    # Wrap student in DDP
    if use_ddp:
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=False)

    student_core = student.module if use_ddp else student

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = LiberoHDF5Dataset(
        dataset_path=dataset_path,
        normalizer_stats_path=normalizer_stats_path,
        seed=seed,
    )
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            generator=g,
            persistent_workers=(num_workers > 0),
        )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in student_core.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=lr * 0.1)

    loss_fn = SnapFlowLoss(action_dim=7)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _rank() == 0:
        log.info(f"Training: {train_steps} steps, batch={batch_size}, lr={lr}")
        log.info(f"Dataset: {len(dataset)} samples")

    loader_iter = iter(loader)
    step_times = []

    for step in range(train_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            if use_ddp:
                sampler.set_epoch(step)
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        t0 = time.perf_counter()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = loss_fn(teacher, student_core, batch)

        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student_core.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        scheduler.step()

        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        if _rank() == 0 and step % 50 == 0:
            avg_t = sum(step_times[-50:]) / len(step_times[-50:])
            remaining = (train_steps - step) * avg_t
            log.info(
                f"step {step:5d}/{train_steps}  "
                f"loss={loss.item():.5f}  "
                f"v_T_norm={loss_dict['v_teacher_norm'].item():.3f}  "
                f"v_S_norm={loss_dict['v_student_norm'].item():.3f}  "
                f"step_t={avg_t:.2f}s  "
                f"ETA={remaining/3600:.1f}h"
            )

        if _rank() == 0 and (step + 1) % save_every == 0:
            _save_checkpoint(student_core, out_path.parent / f"step{step+1:05d}.safetensors")
            log.info(f"Checkpoint saved at step {step+1}")

    # Final save
    if _rank() == 0:
        _save_checkpoint(student_core, out_path)
        log.info(f"Final student checkpoint: {out_path}")

    if use_ddp:
        dist.destroy_process_group()


def _save_checkpoint(policy, path: Path):
    from safetensors.torch import save_file
    state = {k: v.contiguous().cpu() for k, v in policy.state_dict().items()}
    save_file(state, str(path))
