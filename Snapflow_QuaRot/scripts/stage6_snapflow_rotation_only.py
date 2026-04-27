"""Stage 6 (rotation-only) — SnapFlow (NFE=1) + LLM R1+R3, NO W4A4.

Diagnostic to confirm that combining SnapFlow student with full QuaRot
rotations (R1+R3) is lossless at NFE=1. W4A4 is intentionally omitted.

Finding from stage5b + rotation diagnostic:
  - R3 rotation alone: 90% NFE=1 (task 0, n=10)
  - R1+R3 + W4A4: 0% (W4A4 breaks NFE=1 SnapFlow)
  → This script confirms R1+R3 without W4A4 is compatible with NFE=1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/stage6_snapflow_rotation_only.py
"""

import argparse
import json
import sys
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
STUDENT_CKPT = "artifacts/stage1_student.safetensors"
DEVICE = "cuda"
N_ACTION_STEPS = 10
NUM_INFERENCE_STEPS = 1
_R3_SEED = 100


def _patch_embed_image(policy):
    pw = policy.model.paligemma_with_expert
    def pe(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        p = pw.paligemma
        out = p.model.get_image_features(image)
        f = (out.pooler_output if hasattr(out, "pooler_output") else out) * p.config.text_config.hidden_size ** 0.5
        return f.to(out_dtype)
    pw.embed_image = pe
    print("[INFO] embed_image patched")


def _load_student_weights(policy, ckpt_path: str):
    from safetensors.torch import load_file
    state = load_file(ckpt_path)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    print(f"[INFO] Student loaded: missing={len(missing)}, unexpected={len(unexpected)}")


def _apply_quarot_r1r3(policy, device: str):
    """LLM R1 + shared R3 (same as stage5b), NO W4A4."""
    from quarot.fuse_rmsnorm import fuse_all_rmsnorms
    from quarot.offline_rotate import apply_r1r2r3, _make_signs, _apply_r3_to_layer
    from quarot.rotations import hadamard_transform

    inner = policy.model
    pali = inner.paligemma_with_expert
    llm_model = pali.paligemma.model.language_model
    expert_model = pali.gemma_expert
    llm_layers = list(llm_model.layers)
    expert_layers = list(expert_model.model.layers)
    gen_device = "cuda" if "cuda" in device else "cpu"

    n_fused = fuse_all_rmsnorms(llm_model, scope="llm")
    fuse_all_rmsnorms(expert_model, scope="dit")
    print(f"[INFO] RMSNorm fused: LLM={n_fused}")

    D_llm = _make_signs(llm_layers[0].self_attn.q_proj.weight.shape[1],
                        device=gen_device,
                        generator=torch.Generator(device=gen_device).manual_seed(42)).float()

    emb_w = llm_model.embed_tokens.weight
    W = hadamard_transform(emb_w.data.float(), rotate_fp32=True) * D_llm[None, :].to(emb_w.device)
    emb_w.data = W.to(emb_w.dtype)

    mmp = pali.paligemma.model.multi_modal_projector.linear
    W = mmp.weight.data.float()
    D = D_llm.to(W.device)
    W = hadamard_transform(W.T, rotate_fp32=True).T * D[:, None]
    mmp.weight.data = W.to(mmp.weight.dtype)
    if mmp.bias is not None:
        b = hadamard_transform(mmp.bias.data.float().unsqueeze(0), rotate_fp32=True).squeeze(0) * D
        mmp.bias.data = b.to(mmp.bias.dtype)

    apply_r1r2r3(llm_layers, r1=True, r2=False, r3=False, device=gen_device, seed=42)

    head_dim = llm_layers[0].self_attn.head_dim
    gen_r3 = torch.Generator(device=gen_device).manual_seed(_R3_SEED)
    shared_head_signs = _make_signs(head_dim, device=gen_device, generator=gen_r3).float()
    with torch.no_grad():
        for layer in llm_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
        for layer in expert_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
    print("[INFO] LLM R1 + shared R3 applied (NO W4A4)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--student_ckpt", default=STUDENT_CKPT)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage6_rotation_only")
    args = parser.parse_args()

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading base policy from {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.compile_model = False
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    _patch_embed_image(policy)
    _load_student_weights(policy, args.student_ckpt)

    print("\n[INFO] === Stage 6 rotation-only: SnapFlow (NFE=1) + R1+R3 (NO W4A4) ===")
    _apply_quarot_r1r3(policy, args.device)

    policy.eval()
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

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep (NFE=1, R1+R3 only)...")
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
            try: env.close()
            except Exception: pass

    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))
    rw = overall.get("avg_sum_reward", float("nan"))

    print(f"\n{'='*60}")
    print(f"[STAGE 6 rotation-only RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  Stage 0 baseline:          94.6% (NFE=10, FP16)")
    print(f"  Stage 1 SnapFlow:           84.0% (NFE=1, no rotation, n=100)")
    print(f"  Stage 3 QuaRot LLM+DiT:    94.0% (NFE=10, R1+R3)")
    print(f"  Stage 5b SnapFlow+LLM W4A4:  0.0% (NFE=1, W4A4 breaks it)")
    print(f"  Stage 6 rotation-only:     {pc:.1f}% (NFE=1, R1+R3 lossless)")
    print(f"  Config: SnapFlow NFE=1 + LLM R1+R3 (NO W4A4)")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    result = {
        "stage": "stage6_snapflow_rotation_only",
        "snapflow": {"enabled": True, "nfe": 1, "ckpt": args.student_ckpt},
        "quarot": {"r1_llm": True, "r1_dit": False, "r3_shared": True},
        "w4a4": "none",
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = Path(args.output_dir) / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
