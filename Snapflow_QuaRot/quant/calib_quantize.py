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

    Uses eval_policy_all for robust preprocessing (same as calib_capture.py).
    """
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = getattr(policy.config, "pretrained_path", "") or ""
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=pretrained_path,
        )
    except Exception as e:
        log.warning(f"Could not build preprocessor ({e}); using identity.")
        preprocessor = lambda x: x
        postprocessor = lambda x: x

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    n_episodes = max(2, (n_chunks // 50) + 1)

    def forward_loop(model):
        model.eval()
        envs_dict = make_env(env_cfg, n_envs=1)
        with torch.no_grad():
            try:
                eval_policy_all(
                    envs=envs_dict,
                    policy=model,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=n_episodes,
                    start_seed=0,
                )
            except Exception as e:
                log.warning(f"MTQ calibration loop error: {e} — quantizers may use fallback amax.")
        for suite_envs in envs_dict.values():
            for env in suite_envs.values():
                try:
                    env.close()
                except Exception:
                    pass
        log.info(f"MTQ calibration: {n_episodes} episodes completed.")

    return forward_loop
