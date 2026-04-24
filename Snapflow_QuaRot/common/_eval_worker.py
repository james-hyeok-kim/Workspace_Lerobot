"""Worker subprocess for parallel eval.

Called by parallel_eval.py with:
    python _eval_worker.py <config_yaml_path> <result_json_path>

Runs eval_policy_all on assigned task_ids (set in recipe.eval.task_ids).
CUDA_VISIBLE_DEVICES already set by parent.
"""

import json
import logging
import sys
from pathlib import Path
from contextlib import nullcontext

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    cfg_path = sys.argv[1]
    result_path = sys.argv[2]

    from common.recipe import Recipe
    from common.policy_loader import load_policy
    from common.metrics import LatencyTimer
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv
    import torch

    recipe = Recipe.from_yaml(cfg_path)

    task_ids = recipe.eval.task_ids
    n_ep = recipe.eval.n_episodes_per_task

    log.info(
        f"[Worker GPU={torch.cuda.current_device()}] "
        f"task_ids={task_ids}, n_ep_per_task={n_ep} (seed={recipe.eval.start_seed})"
    )

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    env_cfg = LiberoEnv(task=recipe.eval.task, task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=recipe.eval.batch_size)

    timer = LatencyTimer(device=recipe.eval.device)
    timer.attach(policy)

    ctx = torch.autocast(device_type="cuda") if recipe.eval.use_amp else nullcontext()
    with torch.no_grad(), ctx:
        eval_info = eval_policy_all(
            envs=envs_dict,
            policy=policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=pre,
            postprocessor=post,
            n_episodes=n_ep,
            start_seed=recipe.eval.start_seed,
        )

    timer.detach()
    # Close all envs
    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            env.close()

    latency = timer.summary()
    overall = eval_info.get("overall", {})
    result = {
        "aggregated": {
            "pc_success": overall.get("pc_success"),
            "avg_sum_reward": overall.get("avg_sum_reward"),
            "avg_max_reward": overall.get("avg_max_reward"),
        },
        "latency": latency,
        "config": {
            "stage": recipe.name,
            "task_ids": task_ids,
            "n_episodes_per_task": n_ep,
            "nfe": recipe.policy.num_inference_steps,
        },
    }

    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    pc = overall.get("pc_success")
    pc_str = f"{pc:.1f}%" if pc is not None else "N/A"
    log.info(
        f"[Worker] Done: pc_success={pc_str}  "
        f"latency_p50={latency.get('p50_ms', 'N/A')}ms → {result_path}"
    )


if __name__ == "__main__":
    main()
