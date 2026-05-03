"""Stage 9 v5 — FP8 Block-wise Activation (block_size=32) + QAD DiT.

v4 와의 차이:
  - LLM activation: per-tensor FP8 → block-wise FP8 (block_size=32, last dim)
  - Weight: 동일 (INT4 block=128 + FP8 secondary)
  - DiT: W4A4 QAD (stage8f 동일)
  - Protocol: v4 pre-load 방식 동일 (stage8f 먼저 로드 → combined calib → QAD amax 복원)

가설: per-tensor FP8 activation 이 LLM outlier 처리 실패의 주원인.
     block_size=32 로 세분화하면 outlier 영향이 해당 block 에 국한 → 품질 회복 기대.
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
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from snapflow.data import LiberoHDF5Dataset
from snapflow.distill_loss import get_velocity

PRETRAINED_PATH = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
PATH_A          = "/tmp/path_a.pt"
CALIB_PATH      = "/data/jameskimh/james_libero_datasets/libero_10"
NORMALIZER_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
DEVICE = "cuda"

DEFAULT_STUDENT_CKPT = "results/stage8f_w4a4_g8/best_student.pt"


# ── quantization config ───────────────────────────────────────────────────────

def _build_v5_config(dit_group_size: int = 8, act_block_size: int = 32) -> dict:
    """DiT W4A4 + LLM W4 + FP8 block-wise activation (block_size=act_block_size).

    v4 대비 변경:
      a_fp8: per-tensor {"num_bits": (4,3)} →
             block-wise {"num_bits": (4,3), "block_sizes": {-1: act_block_size}}
    """
    w4_dit = {"num_bits": 4, "block_sizes": {-1: dit_group_size}, "enable": True}
    a4_dit = {"num_bits": 4, "enable": True}
    w4_llm = {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True}
    w_fp8  = {"num_bits": (4, 3), "enable": True}
    # block-wise FP8 activation: outlier 가 해당 block 에만 영향
    a_fp8_block = {"num_bits": (4, 3), "block_sizes": {-1: act_block_size}, "enable": True}
    fp16   = {"enable": False}
    return {
        "quant_cfg": {
            "*gemma_expert*weight_quantizer":            w4_dit,
            "*gemma_expert*input_quantizer":             a4_dit,
            "*language_model*weight_quantizer":          [w4_llm, w_fp8],
            "*vision_tower*weight_quantizer":            [w4_llm, w_fp8],
            "*multi_modal_projector*weight_quantizer":   [w4_llm, w_fp8],
            "*language_model*input_quantizer":           a_fp8_block,
            "*vision_tower*input_quantizer":             a_fp8_block,
            "*multi_modal_projector*input_quantizer":    a_fp8_block,
            "*lm_head*":             fp16,
            "*[kv]_bmm_quantizer":   fp16,
            "default":               fp16,
        },
        "algorithm": "max",
    }


# ── calibration loop ──────────────────────────────────────────────────────────

def _make_calib_loop(device: str, n_batches: int = 32):
    def calib_loop(model):
        model.eval()
        dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=0)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                try:
                    images, img_masks = model._preprocess_images(batch)
                    tokens = batch[OBS_LANGUAGE_TOKENS]
                    masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
                    actions = model.prepare_action(batch)
                    noise = model.model.sample_noise(actions.shape, device)
                    t = torch.ones(actions.shape[0], device=device)
                    get_velocity(model.model, images, img_masks, tokens, masks, noise, t, s=t)
                except Exception as e:
                    print(f"[WARN] calib batch skipped: {e}")
                n += 1
                if n >= n_batches:
                    break
        print(f"[INFO] Calibration done: {n} batches")
    return calib_loop


# ── quantizer amax snapshot / verify ─────────────────────────────────────────

def _snapshot_amax(policy, pattern: str) -> dict:
    snap = {}
    for name, mod in policy.named_modules():
        if pattern not in name:
            continue
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            amax = getattr(q, "_amax", None)
            if amax is not None:
                snap[f"{name}.{qname}"] = amax.detach().clone()
    return snap


def _compare_amax(before: dict, after: dict, label: str):
    changed = []
    for k, v_before in before.items():
        v_after = after.get(k)
        if v_after is None:
            changed.append(f"{k}: MISSING after")
        elif not torch.allclose(v_before.cpu(), v_after.cpu(), atol=1e-6):
            diff = (v_after.cpu() - v_before.cpu()).abs().max()
            changed.append(f"{k}: changed (max_diff={diff:.4f})")
    if changed:
        print(f"[WARN] {label} amax CHANGED ({len(changed)} quantizers):")
        for c in changed[:5]:
            print(f"  {c}")
        if len(changed) > 5:
            print(f"  ... and {len(changed)-5} more")
    else:
        print(f"[OK] {label} amax fully preserved ({len(before)} quantizers)")


def _verify_amax(policy, out_dir: Path):
    bad, ok = [], []
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            if not getattr(q, "is_enabled", False):
                continue
            amax = getattr(q, "_amax", None)
            if amax is None or (isinstance(amax, torch.Tensor) and (amax <= 0).any()):
                bad.append(f"{name}.{qname}")
            else:
                ok.append(f"{name}.{qname}")
    print(f"[INFO] Quantizer amax: {len(ok)} valid, {len(bad)} invalid")
    if bad:
        print(f"[WARN] Invalid amax: {bad[:5]}{'...' if len(bad)>5 else ''}")
        for name, mod in policy.named_modules():
            for qname in ("input_quantizer", "weight_quantizer"):
                q = getattr(mod, qname, None)
                if q is None:
                    continue
                if not getattr(q, "is_enabled", False):
                    continue
                amax = getattr(q, "_amax", None)
                if amax is None or (isinstance(amax, torch.Tensor) and (amax <= 0).any()):
                    q._amax = torch.ones(1, device=next(policy.parameters()).device)
        print("[INFO] Patched invalid amax → 1.0")
    summary = {"valid": len(ok), "invalid": len(bad), "invalid_names": bad[:20]}
    with open(out_dir / "quantizer_amax_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return len(ok), len(bad)


# ── misc utils ────────────────────────────────────────────────────────────────

def _patch_embed_image(policy):
    pw = policy.model.paligemma_with_expert
    def pe(image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        p = pw.paligemma
        out = p.model.get_image_features(image)
        f = (out.pooler_output if hasattr(out, "pooler_output") else out) \
            * p.config.text_config.hidden_size ** 0.5
        return f.to(out_dtype)
    pw.embed_image = pe


def _reset_dynamo(policy):
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner:
        for attr in ("sample_actions", "forward"):
            fn = getattr(inner, attr, None)
            if fn:
                orig = (getattr(fn, "_torchdynamo_orig_callable", None)
                        or getattr(fn, "_orig_mod", None))
                if orig:
                    setattr(inner, attr, orig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage9 v5: FP8 block-wise activation (block_size=32)")
    parser.add_argument("--student_ckpt", default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--output_dir", default="results/stage9_fp8block_v5")
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--dit_group_size", type=int, default=8)
    parser.add_argument("--act_block_size", type=int, default=32,
                        help="FP8 activation block size (per last dim). Default=32.")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--task_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import modelopt.torch.quantization as mtq

    print(f"[INFO] Stage9 v5 — FP8 block-wise activation (block_size={args.act_block_size})")
    print(f"  LLM weight: INT4 block=128 + FP8 secondary")
    print(f"  LLM activation: FP8 E4M3 block={args.act_block_size} (vs per-tensor in v4)")
    print(f"  DiT: W4A4 g={args.dit_group_size}, QAD amax from stage8f")

    # ── Load base policy ──────────────────────────────────────────────────────
    print("\n[INFO] Building policy from pretrained ...")
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path        = PRETRAINED_PATH
    policy_cfg.device                 = args.device
    policy_cfg.use_amp                = False
    policy_cfg.compile_model          = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps         = 10
    policy_cfg.num_inference_steps    = args.num_inference_steps

    env_cfg   = LiberoEnv(task="libero_10", task_ids=args.task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.eval_batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(policy)

    sd = torch.load(PATH_A, map_location="cpu")
    policy.load_state_dict(sd, strict=False)
    policy.to(args.device)
    print("[INFO] Base policy (Path A) loaded")

    calib_loop = _make_calib_loop(args.device, n_batches=args.calib_batches)

    # ── Step 2: stage8f 를 plain FP16 model 에 먼저 로드 ─────────────────────
    student_ckpt = Path(args.student_ckpt)
    print(f"\n[Step 2] Pre-loading stage8f weights into plain FP16 model: {student_ckpt}")
    ckpt_data = torch.load(student_ckpt, map_location="cpu")
    ckpt_state = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data

    missing_pre, unexpected_pre = policy.load_state_dict(ckpt_state, strict=False)
    print(f"[Step 2] load_state_dict: missing={len(missing_pre)} unexpected={len(unexpected_pre)}")
    if unexpected_pre[:3]:
        print(f"  unexpected[:3]: {unexpected_pre[:3]}")

    # ── Step 3: combined config + max calib ───────────────────────────────────
    print(f"\n[Step 3] Combined DiT-W4A4 + LLM-W4-FP8block{args.act_block_size} max calib "
          f"(g_dit={args.dit_group_size}, {args.calib_batches} batches) ...")
    v5_cfg = _build_v5_config(dit_group_size=args.dit_group_size, act_block_size=args.act_block_size)
    mtq.quantize(policy, v5_cfg, forward_loop=calib_loop)

    llm_amax_after_calib = _snapshot_amax(policy, "language_model")
    n_llm_valid = sum(1 for v in llm_amax_after_calib.values() if (v > 0).all())
    dit_amax_after_calib = _snapshot_amax(policy, "gemma_expert")
    n_dit_calib = sum(1 for v in dit_amax_after_calib.values() if (v > 0).all())
    print(f"[Step 3] LLM valid amax: {n_llm_valid}/{len(llm_amax_after_calib)}")
    print(f"[Step 3] DiT valid amax (max calib, 임시): {n_dit_calib}/{len(dit_amax_after_calib)}")

    # ── Step 4: stage8f 재로드 → DiT QAD amax 복원 ────────────────────────────
    print(f"\n[Step 4] Re-loading stage8f → DiT QAD amax 복원 ...")
    missing_post, unexpected_post = policy.load_state_dict(ckpt_state, strict=False)
    print(f"[Step 4] load_state_dict: missing={len(missing_post)} unexpected={len(unexpected_post)}")

    dit_amax_final = _snapshot_amax(policy, "gemma_expert")
    n_dit_final = sum(1 for v in dit_amax_final.values() if (v > 0).all())
    print(f"[Step 4] DiT valid amax after QAD restore: {n_dit_final}/{len(dit_amax_final)}")

    llm_amax_final = _snapshot_amax(policy, "language_model")
    _compare_amax(llm_amax_after_calib, llm_amax_final, "LLM language_model")

    dit_changed = sum(
        1 for k in dit_amax_after_calib
        if k in dit_amax_final and not torch.allclose(
            dit_amax_after_calib[k].cpu(), dit_amax_final[k].cpu(), atol=1e-6
        )
    )
    print(f"[Step 4] DiT amax changed by QAD restore: {dit_changed}/{len(dit_amax_after_calib)}")

    n_valid, n_bad = _verify_amax(policy, out_dir)

    # ── Eval ─────────────────────────────────────────────────────────────────
    policy.eval()
    _reset_dynamo(policy)
    for p in policy.parameters():
        p.requires_grad_(False)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "tokenizer_processor": {"tokenizer_name": "/tmp/paligemma_tok_fast"},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"\n[INFO] Evaluating {len(args.task_ids)} tasks × {args.eval_episodes} ep "
          f"(NFE={args.num_inference_steps}) ...")
    with torch.no_grad():
        eval_info = eval_policy_all(
            envs=envs_dict,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.eval_episodes,
            start_seed=args.start_seed,
        )

    for suite_envs in envs_dict.values():
        for env in suite_envs.values():
            try: env.close()
            except Exception: pass

    overall = eval_info.get("overall", {})
    pc = overall.get("pc_success", float("nan"))

    print(f"\n{'='*60}")
    print(f"[Stage9 v5 — FP8 block={args.act_block_size} activation] pc_success={pc:.1f}%")
    print(f"  Reference:")
    print(f"    FP16 (Path A):                  100.0%")
    print(f"    W4A4 DiT QAD (stage8f):          93.0%")
    print(f"    Stage9 v4 (per-tensor FP8):       7.0%")
    print(f"  This result: {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ = t["metrics"]["successes"]
            pct  = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    result = {
        "stage": "stage9_fp8block_v5",
        "protocol": "v5_fp8_blockwise_activation",
        "llm_quant": f"W4-FP8block{args.act_block_size} "
                     f"(INT4_block128 weight + FP8E4M3_block{args.act_block_size} activation)",
        "dit_quant": f"W4A4 (g={args.dit_group_size}, QAD amax restored from stage8f)",
        "student_ckpt": str(student_ckpt),
        "num_inference_steps": args.num_inference_steps,
        "n_action_steps": 10,
        "calib_batches": args.calib_batches,
        "act_block_size": args.act_block_size,
        "overall": overall,
        "eval_info": eval_info,
        "reference": {
            "fp16_path_a": 100.0,
            "w4a4_dit_qad_stage8f": 93.0,
            "stage9_v4_per_tensor_fp8": 7.0,
        },
    }
    out_path = out_dir / "libero_10_fp8block_v5.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
