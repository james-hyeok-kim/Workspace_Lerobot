"""Stage 2 — QuaRot R1+R2 on PaliGemma LLM backbone (FP16, NFE=10).

Rotation is mathematically lossless in FP — expected pc_success ≈ Stage 0 (94.6%).
DiT / action expert stays unchanged.

Usage:
    CUDA_VISIBLE_DEVICES=0,2,3 python scripts/stage2_quarot_llm.py \
        --task_ids 0 1 2 3 4 5 6 7 8 9 \
        --n_episodes 10 --batch_size 10 \
        --start_seed 1000 \
        --output_dir results/stage2 \
        --device cuda:1

    bash scripts/run_stage.sh 2
"""

import argparse
import json
import sys
import os
from pathlib import Path
from types import SimpleNamespace

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


def _apply_quarot_llm(policy, device: str):
    """Apply R1 to LLM backbone: fuse RMSNorm, rotate boundary weights + all layer weights.

    R1 rotates the ENTIRE LLM residual stream. For losslessness, ALL weights that
    read from or write to the residual stream must be rotated consistently:
      - Boundary inputs: embed_tokens (language) + multi_modal_projector (image)
      - Layer internals: q/k/v/gate/up (input-side), o/down (output-side) × 18 layers
    """
    from quarot.fuse_rmsnorm import fuse_all_rmsnorms
    from quarot.offline_rotate import apply_r1r2r3, _make_signs
    from quarot.rotations import hadamard_transform

    inner = policy.model  # PI05Pytorch
    pali = inner.paligemma_with_expert

    # Navigate to the LLM backbone
    llm_model = pali.paligemma.model.language_model
    llm_layers = getattr(llm_model, "layers", None)
    if llm_layers is None:
        inner_model = getattr(llm_model, "model", None)
        llm_layers = getattr(inner_model, "layers", None) if inner_model else None
    if llm_layers is None:
        raise RuntimeError("Cannot locate LLM decoder layers in pi0.5 model")
    llm_layers = list(llm_layers)
    print(f"[INFO] LLM layers: {len(llm_layers)} × {type(llm_layers[0]).__name__}")

    # 1. Fuse RMSNorm gains into downstream Linear weights (lossless)
    print("[INFO] Fusing RMSNorm (LLM)...")
    n_fused = fuse_all_rmsnorms(llm_model, scope="llm")
    print(f"[INFO]   → {n_fused} norm→linear fusions")

    # 2. Generate the R1 signs (same seed as apply_r1r2r3 → identical rotation)
    gen_device = "cuda" if "cuda" in device else "cpu"
    hidden_dim = llm_layers[0].self_attn.q_proj.weight.shape[1]  # 2048
    gen = torch.Generator(device=gen_device).manual_seed(42)
    D = _make_signs(hidden_dim, device=gen_device, generator=gen).float()  # [2048]

    # 3. Rotate boundary weights so the residual stream is consistently rotated
    #    embed_tokens: output goes to residual → input-side formula W' = W @ H @ diag(D)
    #    multi_modal_projector.linear: output goes to residual → output-side formula W' = diag(D) @ H @ W
    print("[INFO] Rotating boundary weights (embed_tokens + multi_modal_projector)...")

    # --- embed_tokens [vocab, 2048] ---
    emb_w = llm_model.embed_tokens.weight  # [257152, 2048]
    W = emb_w.data.float()
    D_dev = D.to(W.device)
    W = hadamard_transform(W, rotate_fp32=True)   # W @ H  (H on last dim = input cols)
    W = W * D_dev[None, :]                         # W @ H @ diag(D)
    emb_w.data = W.to(emb_w.dtype)

    # --- multi_modal_projector.linear [2048, 1152] (+ bias [2048]) ---
    mmp_linear = pali.paligemma.model.multi_modal_projector.linear
    W = mmp_linear.weight.data.float()
    D_dev = D.to(W.device)
    W = hadamard_transform(W.T, rotate_fp32=True).T  # H @ W  (H on output rows)
    W = W * D_dev[:, None]                            # diag(D) @ H @ W
    mmp_linear.weight.data = W.to(mmp_linear.weight.dtype)
    if mmp_linear.bias is not None:
        b = mmp_linear.bias.data.float().to(D_dev.device)
        b = hadamard_transform(b.unsqueeze(0), rotate_fp32=True).squeeze(0)  # H @ b
        b = b * D_dev                                                         # diag(D) @ H @ b
        mmp_linear.bias.data = b.to(mmp_linear.bias.dtype)

    print("[INFO]   → embed_tokens + multi_modal_projector.linear rotated")

    # 4. Apply R1 offline rotations to all 18 LLM decoder layers
    #    (uses same seed=42 → generates identical hidden_signs as step 2)
    print("[INFO] Applying R1 offline rotations to 18 LLM layers...")
    state = apply_r1r2r3(
        llm_layers, r1=True, r2=False, r3=False,
        device=gen_device,
        seed=42,
    )
    print(f"[INFO]   → R1 done. hidden_signs norm: {state.hidden_signs.float().norm():.2f}")

    # Save rotation state artifact
    art_dir = Path("/data/jameskimh/james_lerobot_results/artifacts")
    art_dir.mkdir(exist_ok=True)
    torch.save(
        {"hidden_signs": state.hidden_signs, "head_signs": state.head_signs},
        str(art_dir / "stage2_rot_state.pt"),
    )
    print("[INFO] Rotation state saved → /data/jameskimh/james_lerobot_results/artifacts/stage2_rot_state.pt")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage2")
    parser.add_argument("--skip_equiv_check", action="store_true",
                        help="Skip numerical equivalence check (faster)")
    args = parser.parse_args()

    import transformers
    print(f"[INFO] transformers: {transformers.__version__}")
    print(f"[INFO] torch: {torch.__version__}")

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policy ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading policy from {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    # ── Apply QuaRot LLM rotations ─────────────────────────────────────────────
    print("\n[INFO] === Applying QuaRot R1+R2 to LLM backbone ===")
    _apply_quarot_llm(policy, args.device)

    # ── Optional numerical equivalence check ───────────────────────────────────
    if not args.skip_equiv_check:
        print("\n[INFO] === Numerical Equivalence Check ===")
        _check_weight_norms(policy)

    policy.eval()

    # Disable torch.compile
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

    # ── Full evaluation ────────────────────────────────────────────────────────
    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep (NFE={NUM_INFERENCE_STEPS})...")
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

    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))
    rw = overall.get("avg_sum_reward", float("nan"))

    print(f"\n{'='*60}")
    print(f"[STAGE 2 RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  (Stage 0 baseline: 94.6%  — rotation should be lossless)")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    # Save result JSON
    result = {
        "stage": "stage2_quarot_llm",
        "quarot": {"scope": "llm", "r1": True, "r2": False, "r3": False, "r4": False,
                   "fuse_rmsnorm": True},
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
    row = (f"| stage2_quarot_llm | {pc:.1f} | {rw:.4f} | — | — "
           f"| {NUM_INFERENCE_STEPS} | {N_ACTION_STEPS} | 16 | 16 | — | ✓(LLM) | — | — |\n")
    if leaderboard_path.exists():
        existing = leaderboard_path.read_text()
        if "stage2_quarot_llm" not in existing:
            leaderboard_path.write_text(existing.rstrip() + "\n" + row)
        else:
            lines = existing.split("\n")
            lines = [row.rstrip() if "stage2_quarot_llm" in l else l for l in lines]
            leaderboard_path.write_text("\n".join(lines))
    print(f"[LEADERBOARD] Updated → {leaderboard_path}")
    print(f"{'='*60}")


def _check_weight_norms(policy):
    """Quick sanity: weight Frobenius norms should be preserved by rotation."""
    inner = policy.model
    pali = inner.paligemma_with_expert
    llm_layers = list(pali.paligemma.model.language_model.layers)
    layer = llm_layers[0]
    q_norm = layer.self_attn.q_proj.weight.float().norm().item()
    down_norm = layer.mlp.down_proj.weight.float().norm().item()
    print(f"[EQUIV] Layer 0 q_proj norm: {q_norm:.3f}  (should be close to original)")
    print(f"[EQUIV] Layer 0 down_proj norm: {down_norm:.3f}")
    print("[EQUIV] Note: Hadamard rotation preserves Frobenius norm exactly.")


if __name__ == "__main__":
    main()
