"""Stage 1 — evaluate SnapFlow 1-NFE student checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=0,2 python scripts/stage1_eval_student.py \
        --student_ckpt artifacts/stage1_student.safetensors \
        --task_ids 0 1 2 3 4 5 6 7 8 9 \
        --n_episodes 10 --batch_size 10 \
        --start_seed 1000 \
        --output_dir results/stage1 \
        --device cuda:1

    # or via run_stage.sh (uses default config)
    bash scripts/run_stage.sh 1
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
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
STUDENT_CKPT = "/data/jameskimh/james_lerobot_results/artifacts/stage1_student.safetensors"
DEVICE = "cuda"
N_ACTION_STEPS = 10
NUM_INFERENCE_STEPS = 1  # SnapFlow: 1-NFE


def _patch_embed_image(policy):
    """Fix transformers 4.57.6 API: get_image_features() now returns plain Tensor."""
    pali_with_expert = policy.model.paligemma_with_expert

    def patched_embed_image(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        pali = pali_with_expert.paligemma
        image_outputs = pali.model.get_image_features(image)
        if hasattr(image_outputs, "pooler_output"):
            features = image_outputs.pooler_output * pali.config.text_config.hidden_size ** 0.5
        else:
            # New API (4.57.6): returns projected_features / sqrt(hidden_size)
            features = image_outputs * pali.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    pali_with_expert.embed_image = patched_embed_image
    print("[INFO] embed_image patched for transformers API compatibility")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--student_ckpt", default=STUDENT_CKPT)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage1")
    args = parser.parse_args()

    import transformers
    print(f"[INFO] transformers: {transformers.__version__}")
    print(f"[INFO] torch: {torch.__version__}")

    ckpt_path = Path(args.student_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Student checkpoint not found: {ckpt_path}\n"
            f"Run distillation first:\n"
            f"  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 "
            f"scripts/stage1_snapflow_distill.py"
        )

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading base policy from {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.compile_model = False   # eager mode: avoids torch.compile retrace on method patches
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)

    print(f"[INFO] Creating {len(task_ids)} LIBERO-10 envs (batch={args.batch_size} each)...")
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    # Fix transformers 4.57.6 API in eager mode (safe: no compiled graph to retrace)
    _patch_embed_image(policy)

    # Load student weights
    print(f"[INFO] Loading student weights from {ckpt_path}")
    student_state = load_file(str(ckpt_path))

    # Split TTE state from policy state
    tte_state = {k[len("target_time_emb."):]: v
                 for k, v in student_state.items()
                 if k.startswith("target_time_emb.")}
    policy_state = {k: v for k, v in student_state.items()
                    if not k.startswith("target_time_emb.")}

    missing, unexpected = policy.load_state_dict(policy_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (ok if just normalization stats)")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    # Inject phi_s(s=0) into bias in-place for 1-NFE mode.
    # V4: inject into action_out_proj.bias (32-dim, action space)
    # V3 legacy: inject into time_mlp_out.bias (2048-dim, conditioning space — corrupts AdaRMS)
    original_bias = None
    injected_bias_param = None  # track which param was modified for restore
    if tte_state:
        from snapflow.target_time_emb import TargetTimeEmbedding
        mlp_in_shape = tte_state["mlp_in.weight"].shape   # [hidden_dim, sinusoidal_dim]
        mlp_out_shape = tte_state["mlp_out.weight"].shape  # [out_dim, hidden_dim]
        sinusoidal_dim = mlp_in_shape[1]
        hidden_dim = mlp_in_shape[0]
        out_dim = mlp_out_shape[0]
        is_legacy_v3 = (out_dim == sinusoidal_dim == hidden_dim)

        if is_legacy_v3:
            target_time_emb = TargetTimeEmbedding(embed_dim=out_dim).to(args.device)
            target_time_emb.load_state_dict(tte_state)
            target_time_emb.eval()
            print(f"[WARN] Legacy V3 TTE (embed_dim={out_dim}): injecting into time_mlp_out.bias — AdaRMS corrupted!")
            with torch.no_grad():
                phi_s_zero = target_time_emb(torch.tensor([0.0], device=args.device)).squeeze(0)
            injected_bias_param = policy.model.time_mlp_out.bias
        else:
            target_time_emb = TargetTimeEmbedding(
                out_dim=out_dim, sinusoidal_dim=sinusoidal_dim, hidden_dim=hidden_dim
            ).to(args.device)
            target_time_emb.load_state_dict(tte_state)
            target_time_emb.eval()
            print(f"[INFO] V4 TTE loaded (out_dim={out_dim}, sinusoidal_dim={sinusoidal_dim})")
            with torch.no_grad():
                phi_s_zero = target_time_emb(torch.tensor([0.0], device=args.device)).squeeze(0)
            injected_bias_param = policy.model.action_out_proj.bias

        original_bias = injected_bias_param.data.clone()
        injected_bias_param.data.add_(phi_s_zero.to(injected_bias_param.dtype))
        print(f"[INFO] phi_s(s=0) norm={phi_s_zero.norm():.4f} injected into {injected_bias_param.shape}")
    else:
        print("[INFO] No target_time_emb in checkpoint — FM-only mode")

    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep (NFE={NUM_INFERENCE_STEPS})...")
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

    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try:
                env.close()
            except Exception:
                pass

    # Restore bias (clean up in-place modification)
    if original_bias is not None and injected_bias_param is not None:
        injected_bias_param.data.copy_(original_bias)

    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))
    rw = overall.get("avg_sum_reward", float("nan"))

    print(f"\n{'='*60}")
    print(f"[STAGE 1 RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}  NFE={NUM_INFERENCE_STEPS}")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    result = {
        "stage": "stage1_snapflow",
        "student_ckpt": str(ckpt_path),
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "n_action_steps": N_ACTION_STEPS,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")

    # Update leaderboard
    leaderboard_path = Path("results/leaderboard.md")
    row = (f"| stage1_snapflow | {pc:.1f} | {rw:.4f} | — | — "
           f"| {NUM_INFERENCE_STEPS} | {N_ACTION_STEPS} | 16 | 16 | ✓ | — | — | — |\n")
    if leaderboard_path.exists():
        existing = leaderboard_path.read_text()
        if "stage1_snapflow" not in existing:
            leaderboard_path.write_text(existing.rstrip() + "\n" + row)
        else:
            lines = existing.split("\n")
            lines = [row.rstrip() if "stage1_snapflow" in l else l for l in lines]
            leaderboard_path.write_text("\n".join(lines))
    print(f"[LEADERBOARD] Updated → {leaderboard_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
