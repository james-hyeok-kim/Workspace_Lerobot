"""Collect per-Linear activation statistics from a LIBERO subset.

Usage:
    stats = capture_calib_stats(policy, env_cfg, n_chunks=64)
    torch.save(stats, "artifacts/stage0_calib_stats.pt")

stats structure:
    dict[layer_name -> {
        "max_abs": float,        # max absolute value across all captured activations
        "kurtosis": float,       # kurtosis of flattened activations
        "mean": float,
        "std": float,
        "n_samples": int,
    }]
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from common.hooks import ActivationStore

log = logging.getLogger(__name__)


def _kurtosis(x: torch.Tensor) -> float:
    """Sample kurtosis (Fisher, excess=True, i.e. normal=0)."""
    x = x.float().flatten()
    mu = x.mean()
    sigma = x.std() + 1e-8
    z = (x - mu) / sigma
    return (z.pow(4).mean() - 3.0).item()


def capture_calib_stats(
    policy: nn.Module,
    env_cfg,
    n_chunks: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, dict]:
    """Run n_chunks inference steps and collect activation stats for every Linear.

    Returns a dict keyed by module name with max_abs, kurtosis, mean, std.
    """
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    log.info(f"Capturing calibration stats over {n_chunks} forward passes...")

    # Attach activation hooks to all nn.Linear inside the inner model
    inner = getattr(policy, "model", policy)

    store = ActivationStore()
    store.register(inner, module_types=(nn.Linear,))

    policy.eval()
    n_collected = 0

    # Minimal rollout: just collect activations from select_action calls
    # We do this by running a short eval without tracking env success
    try:
        from lerobot.envs.factory import make_env
        envs_dict = make_env(env_cfg, n_envs=1)
        suite = next(iter(envs_dict))
        task_id = next(iter(envs_dict[suite]))
        env = envs_dict[suite][task_id]

        from lerobot.envs.factory import make_env_pre_post_processors
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy.config
        )

        obs_raw, _ = env.reset(seed=seed)

        with torch.no_grad():
            while n_collected < n_chunks:
                obs = env_preprocessor(obs_raw)
                _ = policy.select_action(obs)
                n_collected += 1
                obs_raw, _, terminated, truncated, _ = env.step(
                    torch.zeros(env.action_space.shape, device=device)
                )
                if terminated.any() or truncated.any():
                    obs_raw, _ = env.reset()

        env.close()
    except Exception as e:
        log.warning(f"Env-based calibration failed ({e}), using random-input fallback.")
        store.remove()
        return {}
    finally:
        store.remove()

    # Aggregate per-layer stats
    stats: dict[str, dict] = {}
    for name, tensors in store.data.items():
        if not tensors:
            continue
        all_acts = torch.cat([t.reshape(-1) for t in tensors])
        stats[name] = {
            "max_abs": all_acts.abs().max().item(),
            "kurtosis": _kurtosis(all_acts),
            "mean": all_acts.mean().item(),
            "std": all_acts.std().item(),
            "n_samples": all_acts.numel(),
        }

    log.info(f"Captured stats for {len(stats)} Linear layers.")
    return stats


def save_calib_stats(stats: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    log.info(f"Calib stats saved to {path}")


def load_calib_stats(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Calib stats not found: {path}. Run Stage 0 first.")
    return torch.load(path, weights_only=False)
