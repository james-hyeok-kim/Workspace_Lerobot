"""Stage 0 — FP16 baseline. Single-GPU, all LIBERO-10 tasks.

Requires: transformers==5.3.0  (5.5.x breaks pi0.5 attention masking)

Usage:
    MUJOCO_GL=egl python scripts/stage0_baseline_simple.py
    MUJOCO_GL=egl python scripts/stage0_baseline_simple.py --n_episodes 3 --task_ids 0
"""

import argparse
import json
import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in [
    str(_ROOT / "Snapflow_QuaRot"),
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
DEVICE = "cuda"
N_ACTION_STEPS = 10
NUM_INFERENCE_STEPS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Task IDs (default: 0-9 all)")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results/stage0")
    args = parser.parse_args()

    import transformers
    print(f"[INFO] transformers: {transformers.__version__}")
    print(f"[INFO] torch: {torch.__version__}")

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading policy from {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)

    print(f"[INFO] Creating {len(task_ids)} LIBERO-10 envs (batch={args.batch_size} each)...")
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # Disable torch.compile for inference stability
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is not None:
        for attr in ("sample_actions", "forward"):
            fn = getattr(inner, attr, None)
            if fn is not None:
                orig = (getattr(fn, "_torchdynamo_orig_callable", None)
                        or getattr(fn, "_orig_mod", None))
                if orig is not None:
                    setattr(inner, attr, orig)
                    print(f"[INFO] torch.compile disabled for model.{attr}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} episodes each...")
    with torch.no_grad():
        eval_info = eval_policy_all(
            envs=envs_dict,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            start_seed=args.start_seed,
        )

    # Close envs
    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    # Extract results
    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))
    rw = overall.get("avg_sum_reward", float("nan"))

    print(f"\n{'='*60}")
    print(f"[STAGE 0 RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")

    # Per-task breakdown
    per_task = eval_info.get("per_task", [])
    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task (per-group) breakdown:")
        for tg, agg in sorted(per_group.items()):
            t_pc = agg.get("pc_success", float("nan"))
            n_ep = agg.get("n_episodes", 0)
            print(f"  {tg}: {t_pc:.1f}%  (n={n_ep})")

    # Save result
    result = {
        "stage": "stage0_baseline",
        "pretrained_path": args.pretrained_path,
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "n_action_steps": N_ACTION_STEPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "transformers_version": transformers.__version__,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")

    # Update leaderboard
    leaderboard_path = Path("results/leaderboard.md")
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Stage | pc_success | avg_reward | NFE | n_action_steps | Notes |\n"
    header += "|---|---|---|---|---|---|\n"
    row = (f"| stage0_baseline | {pc:.1f}% | {rw:.4f} | {NUM_INFERENCE_STEPS} "
           f"| {N_ACTION_STEPS} | FP16, transformers 5.3.0 |\n")
    if leaderboard_path.exists():
        existing = leaderboard_path.read_text()
        if "stage0_baseline" not in existing:
            leaderboard_path.write_text(existing.rstrip() + "\n" + row)
        else:
            # Update existing row
            lines = existing.split("\n")
            lines = [row.rstrip() if "stage0_baseline" in l else l for l in lines]
            leaderboard_path.write_text("\n".join(lines))
    else:
        leaderboard_path.write_text(header + row)
    print(f"[LEADERBOARD] Updated → {leaderboard_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
