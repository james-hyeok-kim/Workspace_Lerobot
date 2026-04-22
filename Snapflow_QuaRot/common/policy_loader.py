"""Load pi0.5 from Hub and apply a Recipe in-memory.

lerobot repo is never modified — all mutations happen on the returned nn.Module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

# Ensure lerobot src is on path
_root = Path(__file__).resolve().parents[2]
_lerobot_src = _root / "lerobot" / "src"
for _p in [str(_root / "Snapflow_QuaRot"), str(_lerobot_src)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env_pre_post_processors

from common.recipe import Recipe

log = logging.getLogger(__name__)


def load_policy(recipe: Recipe, device: str | None = None):
    """Load pi0.5 and apply recipe transforms.

    Returns:
        (policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor)
    """
    device = device or recipe.eval.device

    log.info(f"Loading policy from: {recipe.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(recipe.pretrained_path)
    policy_cfg.pretrained_path = recipe.pretrained_path
    policy_cfg.device = device

    # Override policy parameters from recipe
    policy_cfg.n_action_steps = recipe.policy.n_action_steps
    policy_cfg.num_inference_steps = recipe.policy.num_inference_steps

    env_cfg = LiberoEnv(task=recipe.eval.task)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=recipe.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # Apply recipe transforms (order matters)
    if recipe.snapflow.enabled and recipe.snapflow.student_ckpt:
        _apply_snapflow_weights(policy, recipe.snapflow.student_ckpt)

    if recipe.quarot.enabled:
        from quarot.rotate_pi05 import apply_quarot
        apply_quarot(policy, recipe.quarot)
        log.info("QuaRot applied.")

    if recipe.ohb.enabled:
        from quarot.ohb import apply_ohb
        apply_ohb(policy, recipe.ohb, recipe.calib_stats_path)
        log.info("OHB applied.")

    if recipe.w4a4.enabled:
        from quant.modelopt_bridge import apply_w4a4
        apply_w4a4(policy, recipe.w4a4)
        log.info("W4A4 quantization applied.")

    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _apply_snapflow_weights(policy, ckpt_path: str):
    """Load distilled student weights into policy."""
    from safetensors.torch import load_file

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SnapFlow student checkpoint not found: {ckpt_path}\n"
            "Run stage1_snapflow_distill.py first and place the checkpoint at the above path."
        )
    state = load_file(str(path))
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"Missing keys when loading student ckpt: {missing[:5]}...")
    if unexpected:
        log.warning(f"Unexpected keys in student ckpt: {unexpected[:5]}...")
    log.info(f"SnapFlow student weights loaded from {ckpt_path}")
