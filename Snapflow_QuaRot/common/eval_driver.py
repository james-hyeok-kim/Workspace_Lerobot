"""Eval driver — wraps lerobot.scripts.lerobot_eval.eval_policy with in-memory policy.

Avoids spawning a subprocess; the modified policy (QuaRot / W4A4 / SnapFlow) is
passed directly to the lerobot eval loop.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from common.recipe import Recipe
from common.metrics import LatencyTimer

log = logging.getLogger(__name__)


def run_eval(
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    recipe: Recipe,
) -> dict:
    """Run LIBERO-10 evaluation and return results dict.

    Returns:
        {
          "aggregated": {"pc_success", "avg_sum_reward", "avg_max_reward"},
          "latency":    {"p50_ms", "p95_ms"},
          "config":     {stage name, nfe, n_action_steps, ...},
        }
    """
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv

    device = recipe.eval.device
    env_cfg = LiberoEnv(task=recipe.eval.task)

    log.info(f"Creating LIBERO env: task={recipe.eval.task}, n_envs={recipe.eval.batch_size}")
    envs_dict = make_env(env_cfg, n_envs=recipe.eval.batch_size)
    suite = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite]))
    env = envs_dict[suite][task_id]

    log.info(f"Evaluating {recipe.eval.n_episodes} episodes (suite={suite}, task_id={task_id})")

    timer = LatencyTimer(device=device)
    timer.attach(policy)

    ctx = torch.autocast(device_type="cuda") if recipe.eval.use_amp else nullcontext()
    with torch.no_grad(), ctx:
        eval_info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=recipe.eval.n_episodes,
            start_seed=recipe.eval.start_seed,
        )

    timer.detach()
    env.close()

    latency = timer.summary()
    result = {
        "config": {
            "stage": recipe.name,
            "pretrained_path": recipe.pretrained_path,
            "task": recipe.eval.task,
            "n_episodes": recipe.eval.n_episodes,
            "nfe": recipe.policy.num_inference_steps,
            "n_action_steps": recipe.policy.n_action_steps,
            "snapflow": recipe.snapflow.enabled,
            "quarot": recipe.quarot.enabled,
            "quarot_scope": recipe.quarot.scope if recipe.quarot.enabled else "none",
            "ohb": recipe.ohb.enabled,
            "w4a4": recipe.w4a4.enabled,
        },
        "aggregated": eval_info.get("aggregated", {}),
        "latency": latency,
    }

    # Save result
    out_dir = Path(recipe.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Results saved to {out_path}")

    agg = result["aggregated"]
    log.info(
        f"[{recipe.name}] pc_success={agg.get('pc_success', 'N/A'):.1f}%  "
        f"avg_sum_reward={agg.get('avg_sum_reward', 'N/A'):.4f}  "
        f"latency_p50={latency.get('p50_ms', 'N/A'):.1f}ms"
    )

    return result
