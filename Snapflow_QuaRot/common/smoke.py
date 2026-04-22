"""Smoke test — 1 episode, task 0, seed 0.

Verifies: no NaN in outputs, episode completes without exception.
Used as a fast gate before running a full eval.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def run_smoke(
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    recipe,
) -> bool:
    """Run 1-episode smoke test. Returns True on success."""
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv
    from contextlib import nullcontext

    env_cfg = LiberoEnv(task=recipe.eval.task)
    envs_dict = make_env(env_cfg, n_envs=1)
    suite = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite]))
    env = envs_dict[suite][task_id]

    log.info("Running smoke test (1 episode, seed=0)...")

    try:
        with torch.no_grad():
            eval_info = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=1,
                start_seed=0,
            )
        env.close()
    except Exception as e:
        log.error(f"Smoke test FAILED with exception: {e}")
        return False

    agg = eval_info.get("aggregated", {})
    pc = agg.get("pc_success", None)

    # Check for NaN
    if pc is not None and pc != pc:  # NaN check
        log.error("Smoke test FAILED: pc_success is NaN")
        return False

    log.info(f"Smoke test PASSED: pc_success={pc}")
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.recipe import Recipe
    from common.policy_loader import load_policy

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/stage0_baseline.yaml"
    recipe = Recipe.from_yaml(config_path)
    policy, pre, post, env_pre, env_post = load_policy(recipe)
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    sys.exit(0 if ok else 1)
