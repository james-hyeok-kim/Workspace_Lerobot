"""Stage 4 — OHB (Outlier Head Bypass) + AdaLN detection.

Rotations: identical to Stage 3 (LLM R1 + shared R3 for LLM+DiT, no DiT R1).
Added vs Stage 3:
  - OHB manifest built from Stage 0 calib stats (top-K% kurtosis layers marked FP16).
    Saved to artifacts/stage4_ohb_manifest.json for Stage 5 to consume.
  - AdaLN detection logged (DiT uses use_adarms=True; its norm modules are protected).

OHB does NOT modify model weights — it only tags layers for Stage 5 quantization.
Expected result: ~94% (same as Stage 3, since no quantization yet).

Usage:
    bash scripts/run_stage.sh 4
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
_R3_SEED = 100
CALIB_STATS_PATH = "/data/jameskimh/james_lerobot_results/artifacts/stage0_calib_stats.pt"
OHB_MANIFEST_PATH = "/data/jameskimh/james_lerobot_results/artifacts/stage4_ohb_manifest.json"
OHB_TOP_K_PCT = 5.0
OHB_METRIC = "kurtosis"


def _apply_quarot_stage4(policy, device: str):
    """Stage 4 rotations: identical to Stage 3 (LLM R1 + shared R3)."""
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

    # 1. Fuse RMSNorm (LLM only — DiT AdaLN norms have no static weight)
    print("[INFO] Fusing RMSNorm (LLM)...")
    n_fused_llm = fuse_all_rmsnorms(llm_model, scope="llm")
    n_fused_dit = fuse_all_rmsnorms(expert_model, scope="dit")
    print(f"[INFO]   → LLM: {n_fused_llm} fusions, DiT: {n_fused_dit} fusions (0 expected)")

    # 2. LLM R1 boundary weights
    llm_hidden = llm_layers[0].self_attn.q_proj.weight.shape[1]
    gen_llm = torch.Generator(device=gen_device).manual_seed(42)
    D_llm = _make_signs(llm_hidden, device=gen_device, generator=gen_llm).float()

    print("[INFO] Rotating LLM boundary weights...")
    emb_w = llm_model.embed_tokens.weight
    W = hadamard_transform(emb_w.data.float(), rotate_fp32=True)
    W = W * D_llm[None, :].to(W.device)
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

    # 3. LLM R1 to all 18 LLM decoder layers
    print("[INFO] Applying R1 to LLM layers...")
    llm_state = apply_r1r2r3(llm_layers, r1=True, r2=False, r3=False,
                              device=gen_device, seed=42)
    print(f"[INFO]   → LLM R1 done (hidden_signs norm: {llm_state.hidden_signs.float().norm():.2f})")

    # 4. Shared R3 for both LLM and DiT (same head_signs → joint attention lossless)
    head_dim = llm_layers[0].self_attn.head_dim
    gen_r3 = torch.Generator(device=gen_device).manual_seed(_R3_SEED)
    shared_head_signs = _make_signs(head_dim, device=gen_device, generator=gen_r3).float()
    print(f"[INFO] Shared R3 head_signs (seed={_R3_SEED}): norm={shared_head_signs.norm():.2f}")

    with torch.no_grad():
        for layer in llm_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
        print("[INFO]   → LLM R3 done")
        for layer in expert_layers:
            _apply_r3_to_layer(layer, shared_head_signs)
        print("[INFO]   → DiT R3 done")

    art_dir = Path("/data/jameskimh/james_lerobot_results/artifacts")
    art_dir.mkdir(exist_ok=True)
    torch.save(
        {"llm_hidden_signs": llm_state.hidden_signs,
         "shared_head_signs_r3": shared_head_signs},
        str(art_dir / "stage4_rot_state.pt"),
    )
    return llm_state, shared_head_signs


def _build_ohb_manifest():
    """Build and save OHB manifest from Stage 0 calib stats."""
    from quarot.ohb import build_ohb_manifest

    if not Path(CALIB_STATS_PATH).exists():
        print(f"[WARN] Calib stats not found at {CALIB_STATS_PATH} — skipping OHB manifest build.")
        return {}

    print(f"[INFO] Building OHB manifest (top {OHB_TOP_K_PCT}% by {OHB_METRIC})...")
    manifest = build_ohb_manifest(
        calib_stats_path=CALIB_STATS_PATH,
        top_k_pct=OHB_TOP_K_PCT,
        metric=OHB_METRIC,
        output_path=OHB_MANIFEST_PATH,
    )
    print(f"[INFO] OHB manifest: {len(manifest)} layers marked FP16 → {OHB_MANIFEST_PATH}")
    return manifest


def _detect_adaln(policy):
    """Log AdaLN detection results for diagnostic purposes."""
    inner = policy.model
    pali = inner.paligemma_with_expert

    llm_cfg = getattr(pali.paligemma.model.language_model, "config", None)
    expert_cfg = getattr(pali.gemma_expert, "config", None)

    use_adarms_llm = getattr(llm_cfg, "use_adarms", False) if llm_cfg else False
    use_adarms_expert = getattr(expert_cfg, "use_adarms", False) if expert_cfg else False

    # Count expert norm modules that have .dense (AdaLN conditioned)
    n_adaln_norms = sum(
        1 for m in pali.gemma_expert.modules()
        if hasattr(m, "dense") and not hasattr(m, "weight")
    )
    print(f"[INFO] AdaLN detection:")
    print(f"[INFO]   LLM use_adarms={use_adarms_llm} (R1 safe)")
    print(f"[INFO]   DiT use_adarms={use_adarms_expert} → {n_adaln_norms} conditioned norms (R1 skipped)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage4")
    args = parser.parse_args()

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

    print("\n[INFO] === Stage 4: OHB + AdaLN (rotations: LLM R1 + shared R3) ===")
    _apply_quarot_stage4(policy, args.device)
    _detect_adaln(policy)
    ohb_manifest = _build_ohb_manifest()

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

    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep...")
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
    print(f"[STAGE 4 RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  (Stage 0 baseline: 94.6%  |  Stage 3 LLM R1+R3: 94.0%)")
    print(f"  OHB: {len(ohb_manifest)} layers protected (FP16 for Stage 5)")
    print(f"  Rotations: LLM R1 + shared R3 (DiT R1 skipped: AdaLN)")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    result = {
        "stage": "stage4_ohb_adaln",
        "quarot": {"r1_llm": True, "r1_dit": False, "r3_shared": True,
                   "note": "DiT R1 skipped: AdaLN"},
        "ohb": {"top_k_pct": OHB_TOP_K_PCT, "metric": OHB_METRIC,
                "n_protected": len(ohb_manifest),
                "manifest_path": OHB_MANIFEST_PATH},
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")

    leaderboard_path = Path("results/leaderboard.md")
    row = (f"| stage4_ohb_adaln | {pc:.1f} | {rw:.4f} | — | — "
           f"| {NUM_INFERENCE_STEPS} | {N_ACTION_STEPS} | 16 | 16 | — "
           f"| ✓R1(LLM)+R3(both)+OHB | — | — |\n")
    if leaderboard_path.exists():
        existing = leaderboard_path.read_text()
        if "stage4_ohb_adaln" not in existing:
            leaderboard_path.write_text(existing.rstrip() + "\n" + row)
        else:
            lines = existing.split("\n")
            lines = [row.rstrip() if "stage4_ohb_adaln" in l else l for l in lines]
            leaderboard_path.write_text("\n".join(lines))
    print(f"[LEADERBOARD] Updated → {leaderboard_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
