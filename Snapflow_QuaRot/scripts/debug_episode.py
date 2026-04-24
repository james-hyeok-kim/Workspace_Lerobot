"""Debug: run 1 episode on LIBERO task 0, print actions/rewards/success at each step.

Identifies why running_success_rate is always 0%.
"""

import sys
import os
from pathlib import Path

# Path setup
_root = Path(__file__).resolve().parents[2]
for _p in [
    str(_root / "Snapflow_QuaRot"),
    str(_root / "lerobot" / "src"),
    str(_root / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
TASK = "libero_10"
TASK_ID = 0
N_ACTION_STEPS = 50
MAX_STEPS = 100  # only run 100 steps for debug

print(f"=== Debug Episode ===")
print(f"  model: {PRETRAINED_PATH}")
print(f"  task: {TASK}, task_id={TASK_ID}, n_action_steps={N_ACTION_STEPS}")
print(f"  max_steps: {MAX_STEPS}")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task

# 1. Load config
print("\n[1] Loading policy config...")
policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
policy_cfg.pretrained_path = PRETRAINED_PATH
policy_cfg.device = "cuda"
policy_cfg.n_action_steps = N_ACTION_STEPS
policy_cfg.num_inference_steps = 10

print(f"  type: {policy_cfg.type}")
print(f"  n_action_steps: {policy_cfg.n_action_steps}")
print(f"  chunk_size: {getattr(policy_cfg, 'chunk_size', 'N/A')}")
print(f"  input_features: {list(policy_cfg.input_features.keys()) if policy_cfg.input_features else 'None'}")

# 2. Create env
print("\n[2] Creating LIBERO env...")
env_cfg = LiberoEnv(task=TASK, task_ids=[TASK_ID])
envs_dict = make_env(env_cfg, n_envs=1)
suite_name = next(iter(envs_dict))
env = envs_dict[suite_name][TASK_ID]
print(f"  suite: {suite_name}, task_id: {TASK_ID}")
print(f"  env.num_envs: {env.num_envs}")
print(f"  env type: {type(env)}")
print(f"  task_description: {env.envs[0].task_description}")

# 3. Load policy
print("\n[3] Loading policy...")
policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
policy.eval()
print(f"  policy type: {type(policy)}")
print(f"  device: {next(policy.parameters()).device}")

# 4. Build preprocessors
print("\n[4] Building preprocessors...")
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=PRETRAINED_PATH,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
env_preprocessor, env_postprocessor = make_env_pre_post_processors(
    env_cfg=env_cfg, policy_cfg=policy_cfg
)

# 5. Run episode
print("\n[5] Running episode...")
policy.reset()
obs, info = env.reset(seed=0)
print(f"  obs keys: {list(obs.keys())}")
if "pixels" in obs:
    print(f"  pixel keys: {list(obs['pixels'].keys())}")
    for k, v in obs["pixels"].items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
if "robot_state" in obs:
    eef_pos = obs["robot_state"]["eef"]["pos"]
    print(f"  eef_pos: {eef_pos}")

step = 0
total_reward = 0.0
success = False
done = False

with torch.no_grad():
    while not done and step < MAX_STEPS:
        # preprocess
        tensor_obs = preprocess_observation(obs)
        tensor_obs = add_envs_task(env, tensor_obs)
        tensor_obs = env_preprocessor(tensor_obs)
        tensor_obs = preprocessor(tensor_obs)

        if step == 0:
            print(f"\n  Processed obs keys: {list(tensor_obs.keys())}")
            for k, v in tensor_obs.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"    {k}: list len={len(v)}, first={v[0][:50] if v else ''}")

        action = policy.select_action(tensor_obs)
        action_dict = postprocessor(action)
        action_env = env_postprocessor({"action": action_dict})["action"]
        action_np = action_env.cpu().numpy()

        if action_np.ndim == 2:
            action_np_step = action_np[0]  # take first env's action
        else:
            action_np_step = action_np

        if step == 0:
            print(f"\n  First action: {action_np_step}")
            print(f"  Action min: {action_np_step.min():.4f}, max: {action_np_step.max():.4f}")

        obs_new, reward, terminated, truncated, info_step = env.step(action_np)

        if step % 20 == 0 or terminated or truncated:
            is_success_raw = info_step.get("is_success", False)
            print(f"  step={step:4d}: reward={reward[0]:.4f} terminated={terminated[0]} is_success={is_success_raw[0] if hasattr(is_success_raw, '__len__') else is_success_raw} action={action_np_step[:3]}")

        total_reward += float(reward[0]) if hasattr(reward, '__len__') else float(reward)

        # Check final_info for success
        if "final_info" in info_step:
            fi = info_step["final_info"]
            print(f"  *** final_info at step {step}: {fi}")
            success = bool(fi.get("is_success", False))

        done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated)
        done = done or bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
        obs = obs_new
        step += 1

print(f"\n=== Episode done ===")
print(f"  steps: {step}")
print(f"  total_reward: {total_reward:.4f}")
print(f"  success: {success}")
env.close()
