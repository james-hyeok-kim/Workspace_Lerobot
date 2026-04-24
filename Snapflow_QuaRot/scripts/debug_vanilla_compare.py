"""Quick comparison: run 3 episodes each on libero_spatial and libero_10
using the EXACT same pipeline as the working vanilla eval (lerobot-eval CLI).

This isolates whether 0% on libero_10 is a pipeline issue or task difficulty.
"""

import sys, os
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
for _p in [str(_root / "Snapflow_QuaRot"), str(_root / "lerobot" / "src"), str(_root / "TensorRT-Model-Optimizer")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
# EGL은 GPU 0이 필요함 (GPU 2/3는 EGL init 실패). CUDA 추론은 GPU 2로.
# CUDA_VISIBLE_DEVICES에 GPU 0을 포함하여 EGL 요구 충족;
# policy device="cuda:1"로 GPU 2에서만 CUDA 추론 실행.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["LD_LIBRARY_PATH"] = (
    "/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
)

import torch
import time
from contextlib import nullcontext

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
N_EPISODES = 3   # small number for quick check
N_ENVS = 1
N_ACTION_STEPS = 10  # same as vanilla eval

def run_eval(task: str, task_id: int = 0):
    print(f"\n{'='*60}")
    print(f"  Task: {task}, task_id={task_id}, n_action_steps={N_ACTION_STEPS}")
    print(f"{'='*60}")

    # Load config (same as lerobot-eval)
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = PRETRAINED_PATH
    policy_cfg.device = "cuda:1"  # cuda:1 within CUDA_VISIBLE_DEVICES=0,2 → physical GPU 2
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.use_amp = False   # match vanilla: use_amp=true actually; test both
    policy_cfg.dtype = "bfloat16"

    # Env
    env_cfg = LiberoEnv(task=task, task_ids=[task_id])
    envs_dict = make_env(env_cfg, n_envs=N_ENVS)
    suite_name = next(iter(envs_dict))
    env = envs_dict[suite_name][task_id]
    print(f"  Task description: {env.envs[0].task_description}")

    # Policy
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # Preprocessors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={"device_processor": {"device": "cuda:1"}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # Eval (same as vanilla)
    t0 = time.time()
    with torch.no_grad():
        result = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=N_EPISODES,
            start_seed=0,
        )
    elapsed = time.time() - t0

    agg = result.get("aggregated", {})
    pc = agg.get("pc_success", "N/A")
    per_ep = result.get("per_episode", [])

    print(f"\n  Results after {elapsed:.1f}s:")
    print(f"  pc_success: {pc}")
    print(f"  avg_sum_reward: {agg.get('avg_sum_reward', 'N/A')}")
    for ep in per_ep:
        print(f"    episode {ep['episode_ix']}: success={ep['success']}, reward={ep['sum_reward']:.3f}")

    env.close()
    del policy
    torch.cuda.empty_cache()
    return pc


# Test libero_spatial first (should be ~97%)
pc_spatial = run_eval("libero_spatial", task_id=0)

# Then libero_10 (unknown)
pc_10 = run_eval("libero_10", task_id=0)

print(f"\n{'='*60}")
print(f"  SUMMARY:")
print(f"  libero_spatial task_0: {pc_spatial}%")
print(f"  libero_10 task_0:      {pc_10}%")
print(f"{'='*60}")
