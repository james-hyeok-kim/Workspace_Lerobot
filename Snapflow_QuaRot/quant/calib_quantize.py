"""Calibration loop for ModelOpt quantizers.

Runs N forward passes through the policy using LIBERO data
to calibrate scale factors for the TensorQuantizer instances
inserted by mtq.quantize().
"""

from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


def make_calibration_loop(
    policy: nn.Module,
    env_cfg,
    n_chunks: int = 64,
    device: str = "cuda",
) -> callable:
    """Return a forward_loop callable for mtq.quantize calibration.

    mtq.quantize(policy, config=cfg, forward_loop=forward_loop)

    The callable receives the (possibly quantized) model and runs
    N forward passes to collect calibration statistics.
    """
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    envs_dict = make_env(env_cfg, n_envs=1)
    suite = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite]))
    env = envs_dict[suite][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    obs_raw, _ = env.reset(seed=0)
    n_collected = [0]

    def forward_loop(model):
        nonlocal obs_raw
        model.eval()
        with torch.no_grad():
            while n_collected[0] < n_chunks:
                obs = env_preprocessor(obs_raw)
                _ = model.select_action(obs)
                n_collected[0] += 1

                obs_raw, _, terminated, truncated, _ = env.step(
                    torch.zeros(env.action_space.shape, device=device)
                )
                if terminated.any() or truncated.any():
                    obs_raw, _ = env.reset()

                if n_collected[0] % 16 == 0:
                    log.info(f"  Calibration: {n_collected[0]}/{n_chunks} chunks")

        env.close()

    return forward_loop
