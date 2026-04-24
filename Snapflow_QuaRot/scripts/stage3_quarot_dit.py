"""Stage 3 — QuaRot: LLM R1 + shared R3 on both LLM and DiT.

Architecture constraint: the action expert (DiT, Gemma-300M) uses AdaLN
(Adaptive RMSNorm with per-channel scale+shift conditioned on timestep).
R1 rotation of the DiT's residual stream is NOT lossless because the AdaLN
shift term is added in the original space and does not commute with an
orthogonal rotation of the hidden dimension.

Instead, Stage 3 applies:
  LLM: R1 (residual stream rotation, same as Stage 2) — lossless
  DiT: NONE (AdaLN prevents R1)
  Both: R3 (V ↔ o_proj within head_dim=256 space) — lossless for joint attention

For joint attention losslessness, LLM and DiT must use the SAME R3 head_signs:
  V_all = cat([R3 @ V_llm, R3 @ V_expert], dim=token)
  O = Att @ V_all = R3 @ O_orig  (R3 applied uniformly to all token V values)
  o_proj'(O_chunk) = W_o @ R3^T @ R3 @ O_chunk = W_o @ O_chunk ✓

Usage:
    bash scripts/run_stage.sh 3
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
DEVICE = "cuda"
N_ACTION_STEPS = 10
NUM_INFERENCE_STEPS = 10

# Separate seed for R3 head_signs so it's independent from R1 hidden_signs
_R3_SEED = 100


def _apply_quarot_stage3(policy, device: str):
    """Stage 3 rotation: LLM R1 + shared R3 for both LLM and DiT.

    LLM: R1 (residual stream, same as Stage 2) — 18 layers + boundary weights.
    Both: R3 (V ↔ o_proj, within head_dim=256) — same signs for joint attention.
    DiT: NO R1 (AdaLN per-channel scale/shift breaks residual-stream rotation).
    """
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

    # ── 1. Fuse LLM RMSNorm (same as Stage 2) ────────────────────────────────
    print("[INFO] Fusing RMSNorm (LLM)...")
    n_fused_llm = fuse_all_rmsnorms(llm_model, scope="llm")
    print(f"[INFO]   → LLM: {n_fused_llm} fusions")

    # Expert norms are AdaLN-conditioned (use_adarms=True): no static weight → 0 fusions expected
    n_fused_dit = fuse_all_rmsnorms(expert_model, scope="dit")
    print(f"[INFO]   → DiT: {n_fused_dit} fusions (0 expected: AdaLN norms have no static weight)")

    # ── 2. LLM R1 boundary weights (same as Stage 2) ─────────────────────────
    llm_hidden = llm_layers[0].self_attn.q_proj.weight.shape[1]  # 2048
    gen_llm = torch.Generator(device=gen_device).manual_seed(42)
    D_llm = _make_signs(llm_hidden, device=gen_device, generator=gen_llm).float()

    print("[INFO] Rotating LLM boundary weights (embed_tokens + multi_modal_projector)...")
    emb_w = llm_model.embed_tokens.weight
    W = emb_w.data.float()
    D = D_llm.to(W.device)
    W = hadamard_transform(W, rotate_fp32=True)
    W = W * D[None, :]
    emb_w.data = W.to(emb_w.dtype)

    mmp_linear = pali.paligemma.model.multi_modal_projector.linear
    W = mmp_linear.weight.data.float()
    D = D_llm.to(W.device)
    W = hadamard_transform(W.T, rotate_fp32=True).T
    W = W * D[:, None]
    mmp_linear.weight.data = W.to(mmp_linear.weight.dtype)
    if mmp_linear.bias is not None:
        b = mmp_linear.bias.data.float().to(D.device)
        b = hadamard_transform(b.unsqueeze(0), rotate_fp32=True).squeeze(0)
        b = b * D
        mmp_linear.bias.data = b.to(mmp_linear.bias.dtype)
    print("[INFO]   → LLM boundary rotated")

    # ── 3. Apply LLM R1 to 18 LLM decoder layers ─────────────────────────────
    print("[INFO] Applying R1 to LLM layers (r2=False, r3=False — R3 applied separately below)...")
    llm_state = apply_r1r2r3(llm_layers, r1=True, r2=False, r3=False,
                              device=gen_device, seed=42)
    print(f"[INFO]   → LLM R1 done (hidden_signs norm: {llm_state.hidden_signs.float().norm():.2f})")

    # ── 4. Shared R3 for joint-attention losslessness ─────────────────────────
    # LLM and DiT must use IDENTICAL head_signs so that V_all = R3 @ V_all
    # uniformly, allowing both o_proj weights to consistently absorb R3^T.
    head_dim = llm_layers[0].self_attn.head_dim  # 256
    gen_r3 = torch.Generator(device=gen_device).manual_seed(_R3_SEED)
    shared_head_signs = _make_signs(head_dim, device=gen_device, generator=gen_r3).float()
    print(f"[INFO] Shared R3 head_signs (seed={_R3_SEED}): norm={shared_head_signs.norm():.2f}")

    print("[INFO] Applying R3 to LLM layers (V ↔ o_proj, head_dim=256)...")
    with torch.no_grad():
        for layer in llm_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
    print("[INFO]   → LLM R3 done")

    print("[INFO] Applying R3 to DiT/expert layers (V ↔ o_proj, head_dim=256)...")
    with torch.no_grad():
        for layer in expert_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
    print("[INFO]   → DiT R3 done")

    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "llm_hidden_signs": llm_state.hidden_signs,
            "llm_head_signs": llm_state.head_signs,
            "shared_head_signs_r3": shared_head_signs,
        },
        str(art_dir / "stage3_rot_state.pt"),
    )
    print("[INFO] Rotation state saved → artifacts/stage3_rot_state.pt")
    return llm_state, shared_head_signs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage3")
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
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    print("\n[INFO] === Applying QuaRot Stage 3: LLM R1 + shared R3 (LLM+DiT) ===")
    _apply_quarot_stage3(policy, args.device)

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
    print(f"[STAGE 3 RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  (Stage 0 baseline: 94.6%  — rotation should be lossless)")
    print(f"  (Stage 2 LLM R1: 96.0%)")
    print(f"  Rotations: LLM R1 + shared R3 on LLM+DiT (DiT R1 skipped: AdaLN prevents lossless R1)")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    result = {
        "stage": "stage3_quarot_llm_dit",
        "quarot": {"scope": "llm+shared_r3", "r1_llm": True, "r1_dit": False, "r2": False,
                   "r3_shared": True, "r4": False, "fuse_rmsnorm_llm": True,
                   "note": "DiT R1 skipped: AdaLN per-channel scale/shift prevents lossless residual rotation"},
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

    leaderboard_path = Path("results/leaderboard.md")
    row = (f"| stage3_quarot_llm_dit | {pc:.1f} | {rw:.4f} | — | — "
           f"| {NUM_INFERENCE_STEPS} | {N_ACTION_STEPS} | 16 | 16 | — "
           f"| ✓R1(LLM)+R3(both) | — | — |\n")
    if leaderboard_path.exists():
        existing = leaderboard_path.read_text()
        if "stage3_quarot_llm_dit" not in existing:
            leaderboard_path.write_text(existing.rstrip() + "\n" + row)
        else:
            lines = existing.split("\n")
            lines = [row.rstrip() if "stage3_quarot_llm_dit" in l else l for l in lines]
            leaderboard_path.write_text("\n".join(lines))
    print(f"[LEADERBOARD] Updated → {leaderboard_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
