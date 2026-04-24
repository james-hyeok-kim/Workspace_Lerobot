"""Collect per-Linear activation statistics from a LIBERO subset.

Usage:
    stats = capture_calib_stats(policy, preprocessor, env_preprocessor, ..., n_chunks=64)
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
from pathlib import Path

import torch
from torch import nn

from common.hooks import OnlineStatsStore

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
    preprocessor=None,
    env_preprocessor=None,
    env_postprocessor=None,
    postprocessor=None,
) -> dict[str, dict]:
    """Run n_chunks inference steps and collect activation stats for every Linear.

    Uses eval_policy_all with full preprocessing pipeline to avoid preprocessor issues.

    Args:
        policy: The policy to capture stats from.
        env_cfg: LiberoEnv config (used to create calib envs).
        n_chunks: Approximate number of select_action calls to collect.
        preprocessor: VLM preprocessor from load_policy (optional but recommended).
        env_preprocessor: Env-specific preprocessor from load_policy (optional).
        env_postprocessor: Env-specific postprocessor from load_policy (optional).
        postprocessor: Policy output postprocessor from load_policy (optional).
    Returns:
        dict keyed by module name with max_abs, kurtosis, mean, std.
    """
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies.factory import make_pre_post_processors

    log.info(f"Capturing calibration stats over ~{n_chunks} forward passes...")

    # Restrict to task 0 for speed
    calib_env_cfg = LiberoEnv(
        task=env_cfg.task if hasattr(env_cfg, "task") else "libero_10",
        task_ids=[0],
    )

    # Build missing preprocessors from policy config
    if env_preprocessor is None or env_postprocessor is None:
        _ep, _epst = make_env_pre_post_processors(
            env_cfg=calib_env_cfg, policy_cfg=policy.config
        )
        env_preprocessor = env_preprocessor or _ep
        env_postprocessor = env_postprocessor or _epst

    if preprocessor is None or postprocessor is None:
        pretrained_path = getattr(policy.config, "pretrained_path", "") or ""
        try:
            _pre, _post = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=pretrained_path,
            )
        except Exception as e:
            log.warning(f"Could not build preprocessor ({e}); calib may be inaccurate.")
            _pre, _post = (lambda x: x), (lambda x: x)
        preprocessor = preprocessor or _pre
        postprocessor = postprocessor or _post

    # Attach activation hooks (online stats — no tensor accumulation)
    inner = getattr(policy, "model", policy)
    store = OnlineStatsStore()
    store.register(inner, module_types=(nn.Linear,))

    # Count actual select_action calls via a counter hook
    n_collected = [0]
    _orig_select = policy.select_action

    def _counting_select(obs):
        result = _orig_select(obs)
        n_collected[0] += 1
        return result

    policy.select_action = _counting_select
    policy.eval()

    # n_episodes: enough to accumulate n_chunks calls
    # Each episode is at most 520 steps with n_action_steps=10 → up to 52 select_action calls
    # So 2 episodes should yield ~100 calls, which is > 64
    n_episodes = max(2, (n_chunks // 50) + 1)

    try:
        envs_dict = make_env(calib_env_cfg, n_envs=1)

        with torch.no_grad():
            eval_policy_all(
                envs=envs_dict,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=n_episodes,
                start_seed=seed,
            )

        for suite_envs in envs_dict.values():
            for env in suite_envs.values():
                env.close()

    except Exception as e:
        log.warning(f"Env-based calibration failed ({e}), calib stats will be empty.")
        store.remove()
        policy.select_action = _orig_select
        return {}
    finally:
        store.remove()
        policy.select_action = _orig_select

    log.info(f"Calibration: {n_collected[0]} forward passes completed.")

    # Retrieve finalized online stats (no tensor concatenation needed)
    stats: dict[str, dict] = {
        name: s for name, s in store.data.items() if s["n_samples"] > 0
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
