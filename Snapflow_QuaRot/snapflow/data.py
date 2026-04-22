"""LIBERO dataset iterator for SnapFlow distillation training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def make_libero_dataloader(
    dataset_repo_id: str = "lerobot/libero_10_lerobot",
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader over LIBERO demonstrations for distillation.

    The dataloader yields dicts with:
    - "action": [B, T, D] action chunks
    - Observation keys expected by pi0.5 preprocessor
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    log.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(repo_id=dataset_repo_id)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    log.info(f"DataLoader ready: {len(dataset)} samples, batch_size={batch_size}")
    return loader
