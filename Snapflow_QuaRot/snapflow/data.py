"""LIBERO HDF5 dataset for SnapFlow distillation training.

Reads directly from local HDF5 demo files — no LeRobotDataset dependency.

Each sample:
  - observation.images.image       [3, H, W] float32, [0, 1]  (channels-first)
  - observation.images.image2      [3, H, W] float32, [0, 1]  (channels-first)
  - observation.language.tokens    [max_length] int64
  - observation.language.attention_mask [max_length] bool
  - action                         [chunk_size, action_dim] float32
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

CHUNK_SIZE = 50
ACTION_DIM = 7
STATE_DIM = 8
TOKENIZER_MAX_LENGTH = 200
IMAGE_KEY1 = "observation.images.image"    # agentview_rgb
IMAGE_KEY2 = "observation.images.image2"   # eye_in_hand_rgb
LANG_TOKENS = "observation.language.tokens"
LANG_MASK = "observation.language.attention_mask"
ACTION_KEY = "action"


class LiberoHDF5Dataset(Dataset):
    """Streams training samples from LIBERO-10 HDF5 demo files."""

    _TOKENIZER_CACHE = (
        "/home/jovyan/.cache/huggingface/hub"
        "/models--google--paligemma-3b-pt-224"
        "/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
    )

    def __init__(
        self,
        dataset_path: str,
        normalizer_stats_path: str,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        chunk_size: int = CHUNK_SIZE,
        seed: int = 42,
    ):
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size

        # Index: list of (hdf5_path, demo_key, timestep, language_instruction)
        self._index: list[tuple[str, str, int, str]] = []
        hdf5_files = sorted(self.dataset_path.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files in {dataset_path}")

        for fpath in hdf5_files:
            with h5py.File(fpath, "r") as f:
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang_instr = problem_info.get("language_instruction", fpath.stem)
                demo_keys = sorted(f["data"].keys())
                for dk in demo_keys:
                    T = f["data"][dk]["actions"].shape[0]
                    # Only include timesteps where we can form a full chunk
                    for t in range(max(1, T - chunk_size + 1)):
                        self._index.append((str(fpath), dk, t, lang_instr))

        rng = np.random.default_rng(seed)
        rng.shuffle(self._index)
        log.info(f"LiberoHDF5Dataset: {len(self._index)} samples from {len(hdf5_files)} files")

        # Load normalizer stats (q01, q99 for state)
        # Accepts .safetensors or .pt format
        stats_path = Path(normalizer_stats_path)
        if stats_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            stats = load_file(str(stats_path))
            self.state_q01 = stats["observation.state.q01"].numpy().astype(np.float32)
            self.state_q99 = stats["observation.state.q99"].numpy().astype(np.float32)
        else:
            stats = torch.load(stats_path, map_location="cpu", weights_only=True)
            self.state_q01 = stats["q01"].numpy().astype(np.float32)
            self.state_q99 = stats["q99"].numpy().astype(np.float32)

        # Load tokenizer (로컬 캐시 우선, 없으면 HF에서 다운로드)
        from transformers import AutoTokenizer
        local_cache = Path(self._TOKENIZER_CACHE)
        tok_path = str(local_cache) if local_cache.exists() else tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=local_cache.exists())

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantile normalize to [-1, 1]."""
        denom = self.state_q99 - self.state_q01
        denom = np.where(denom < 1e-6, 1.0, denom)
        return 2.0 * (state - self.state_q01) / denom - 1.0

    def _tokenize(self, lang_instr: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build and tokenize the task+state prompt."""
        norm_state = self._normalize_state(state)
        discretized = np.digitize(norm_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str = " ".join(map(str, discretized))
        cleaned = lang_instr.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        enc = self.tokenizer(
            prompt,
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return enc["input_ids"][0].astype(np.int64), enc["attention_mask"][0].astype(bool)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        fpath, demo_key, t0, lang_instr = self._index[idx]

        with h5py.File(fpath, "r", swmr=False) as f:
            demo = f["data"][demo_key]
            T = demo["actions"].shape[0]

            # Images at t0: uint8 [H, W, 3] → float32 [0, 1] → [3, H, W] channels-first
            agentview = demo["obs"]["agentview_rgb"][t0].astype(np.float32) / 255.0
            agentview = agentview.transpose(2, 0, 1)         # [3, H, W]
            eye_in_hand = demo["obs"]["eye_in_hand_rgb"][t0].astype(np.float32) / 255.0
            eye_in_hand = eye_in_hand.transpose(2, 0, 1)    # [3, H, W]

            # State at t0
            ee_states = demo["obs"]["ee_states"][t0]       # [6]
            gripper = demo["obs"]["gripper_states"][t0]    # [2]
            state = np.concatenate([ee_states, gripper], axis=0).astype(np.float32)

            # Action chunk [t0 : t0+chunk_size] padded to chunk_size
            t_end = min(t0 + self.chunk_size, T)
            action_raw = demo["actions"][t0:t_end].astype(np.float32)  # [<=chunk_size, 7]

        # Pad action to chunk_size if needed
        pad_len = self.chunk_size - action_raw.shape[0]
        if pad_len > 0:
            action_raw = np.pad(action_raw, ((0, pad_len), (0, 0)), mode="edge")

        # Tokenize
        tokens, attn_mask = self._tokenize(lang_instr, state)

        return {
            IMAGE_KEY1: torch.from_numpy(agentview),         # [H, W, 3]
            IMAGE_KEY2: torch.from_numpy(eye_in_hand),       # [H, W, 3]
            LANG_TOKENS: torch.from_numpy(tokens),           # [200]
            LANG_MASK: torch.from_numpy(attn_mask),          # [200]
            ACTION_KEY: torch.from_numpy(action_raw),        # [chunk_size, 7]
        }


def make_libero_dataloader(
    dataset_path: str,
    normalizer_stats_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    dataset = LiberoHDF5Dataset(
        dataset_path=dataset_path,
        normalizer_stats_path=normalizer_stats_path,
        seed=seed,
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
