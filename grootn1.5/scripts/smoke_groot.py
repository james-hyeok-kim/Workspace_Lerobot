"""GR00T-N1.5 LIBERO smoke test — 1 task × N episodes.

Verifies the full evaluation pipeline (model load → env rollout → success output)
without running a time-consuming full 4-suite benchmark.

Usage:
    # Smoke: 1 task, 1 episode (default)
    source scripts/_env.sh
    python scripts/smoke_groot.py

    # Custom checkpoint / suite
    python scripts/smoke_groot.py \\
        --policy_path Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL \\
        --suite libero_spatial \\
        --task_ids 0 \\
        --n_episodes 1

Checkpoint ↔ suite mapping (Tacoin):
    libero_spatial → Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL
    libero_object  → Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT
    libero_goal    → Tacoin/GR00T-N1.5-3B-LIBERO-GOAL
    libero_10      → Tacoin/GR00T-N1.5-3B-LIBERO-LONG
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── resolve workspace root regardless of cwd ─────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_WS_ROOT = _SCRIPT_DIR.parents[2]  # workspace/Workspace_Lerobot

for _p in [
    str(_WS_ROOT / "lerobot" / "src"),
    str(_WS_ROOT / "grootn1.5"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

import torch

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.groot import GrootPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

SUITE_TO_CKPT = {
    "libero_spatial": "Tacoin/GR00T-N1.5-3B-LIBERO-SPATIAL",
    "libero_object":  "Tacoin/GR00T-N1.5-3B-LIBERO-OBJECT",
    "libero_goal":    "Tacoin/GR00T-N1.5-3B-LIBERO-GOAL",
    "libero_10":      "Tacoin/GR00T-N1.5-3B-LIBERO-LONG",
}

# Tacoin finetune was trained with n_action_steps=16, chunk_size=16
N_ACTION_STEPS = 16
CHUNK_SIZE = 16
# QuantVLA paper Table 4 default
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser(description="GR00T-N1.5 LIBERO smoke test")
    parser.add_argument(
        "--policy_path",
        default=None,
        help="HF repo-id or local path for GrootPolicy. Default: inferred from --suite.",
    )
    parser.add_argument(
        "--suite",
        default="libero_spatial",
        choices=list(SUITE_TO_CKPT.keys()),
        help="LIBERO suite to run (default: libero_spatial).",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=[0],
        help="Task indices within the suite (default: [0]).",
    )
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Episodes per task (default: 1).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Parallel env batch size (default: 1).")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: $GROOT_OUTPUT_ROOT/results/smoke_<timestamp>",
    )
    args = parser.parse_args()

    policy_path = args.policy_path or SUITE_TO_CKPT[args.suite]

    out_dir_default = os.path.join(
        os.environ.get("GROOT_OUTPUT_ROOT", "/data/jameskimh/groot_n1p5"),
        "results",
        f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    out_dir = Path(args.output_dir or out_dir_default)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] torch:  {torch.__version__}")
    print(f"[INFO] CUDA:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[INFO] policy: {policy_path}")
    print(f"[INFO] suite:  {args.suite}  task_ids={args.task_ids}  n_episodes={args.n_episodes}")
    print(f"[INFO] output: {out_dir}")

    # ── Load policy ───────────────────────────────────────────────────────────
    # Tacoin checkpoints have no model.safetensors (sharded format), so
    # GrootPolicy.from_pretrained detects them as base GR00T models and calls
    # GR00TN15.from_pretrained internally, which calls snapshot_download → HF cache.
    print("\n[STEP 1] Loading GR00T policy ...")
    policy = GrootPolicy.from_pretrained(
        policy_path,
        n_action_steps=N_ACTION_STEPS,
        chunk_size=CHUNK_SIZE,
        # embodiment_tag="new_embodiment" is the Tacoin default (no override needed)
    )
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()
    print("[INFO] Policy loaded.")

    # ── Build LIBERO env ──────────────────────────────────────────────────────
    print("\n[STEP 2] Creating LIBERO env ...")
    env_cfg = LiberoEnv(task=args.suite, task_ids=args.task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    # ── Build preprocessors ───────────────────────────────────────────────────
    # No pretrained_path → creates fresh GrootPackInputsStep + GrootEagleEncodeStep
    # dataset_stats=None → GR00T model handles normalization internally via metadata.json
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy.config)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    # ── Rollout ───────────────────────────────────────────────────────────────
    print(f"\n[STEP 3] Running {len(args.task_ids)} task(s) × {args.n_episodes} episode(s) ...")
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

    # ── Close envs ────────────────────────────────────────────────────────────
    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    # ── Results ───────────────────────────────────────────────────────────────
    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))
    rw = overall.get("avg_sum_reward", float("nan"))

    print(f"\n{'='*60}")
    print(f"[SMOKE RESULT]  suite={args.suite}  pc_success={pc:.1f}%  avg_reward={rw:.4f}")
    per_group = eval_info.get("per_group", {})
    if per_group:
        for tg, agg in sorted(per_group.items()):
            print(f"  task {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes',0)})")
    print(f"{'='*60}")

    result = {
        "mode": "smoke",
        "policy_path": policy_path,
        "suite": args.suite,
        "task_ids": args.task_ids,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "n_action_steps": N_ACTION_STEPS,
        "chunk_size": CHUNK_SIZE,
        "overall": overall,
        "per_group": {k: dict(v) for k, v in per_group.items()},
    }
    out_path = out_dir / f"{args.suite}_smoke.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[SAVED] {out_path}")

    # Quick sanity: success must be a number (not NaN), no crash = smoke pass
    if pc != pc:  # NaN check
        print("[WARN] pc_success is NaN — possible rollout issue.")
        sys.exit(1)
    print("[SMOKE PASS]")


if __name__ == "__main__":
    main()
