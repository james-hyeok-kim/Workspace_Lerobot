"""Stage 9 — QuantVLA-style W4A8 LLM PTQ + W4A4 DiT QAD (기존 checkpoint).

[Protocol]
  Pass 1 : DiT W4A4 (scope=expert) 등록 + max calibration (32 batches)
  Load   : results/stage8f_w4a4_g8/best_student.pt → DiT weights + amax 복원
  Pass 2 : LLM/vision/projector W4A8 등록 + awq_lite calibration (32 batches)
  Eval   : LIBERO-10 (10 task × 10 ep, NFE=1)

Reference: QuantVLA arXiv 2602.20309 (LLM W4A8 + AWQ-lite 부분만 구현, DuQuant/ATM/OHB 미적용)
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


# ── quantization configs ──────────────────────────────────────────────────────

def _build_dit_w4a4_config(group_size: int = 8) -> dict:
    """Pass 1: gemma_expert (DiT) only, W4A4, algorithm=max."""
    w4 = {"num_bits": 4, "block_sizes": {-1: group_size}, "enable": True}
    a4 = {"num_bits": 4, "enable": True}
    fp16 = {"enable": False}
    return {
        "quant_cfg": {
            "*gemma_expert*weight_quantizer": w4,
            "*gemma_expert*input_quantizer":  a4,
            "*[kv]_bmm_quantizer": fp16,
            "default": fp16,
        },
        "algorithm": "max",
    }


def _build_llm_w4a8_config() -> dict:
    """Pass 2: LLM/vision/projector only, W4A8 (INT4 blockwise + FP8E4M3), algorithm=awq_lite.

    Default 키 생략 → gemma_expert 의 Pass 1 quantizer state 는 건드리지 않음.
    QuantVLA 논문 기준: weight INT4 block=128 + FP8 secondary, activation FP8 E4M3.
    """
    w4_blockwise = {
        "num_bits": 4,
        "block_sizes": {-1: 128, "type": "static"},
        "enable": True,
    }
    w_fp8 = {"num_bits": (4, 3), "enable": True}   # FP8 E4M3 secondary quantizer
    a_fp8 = {"num_bits": (4, 3), "enable": True}   # FP8 E4M3 activation
    fp16  = {"enable": False}
    return {
        "quant_cfg": {
            "*language_model*weight_quantizer":          [w4_blockwise, w_fp8],
            "*vision_tower*weight_quantizer":            [w4_blockwise, w_fp8],
            "*multi_modal_projector*weight_quantizer":   [w4_blockwise, w_fp8],
            "*language_model*input_quantizer":           a_fp8,
            "*vision_tower*input_quantizer":             a_fp8,
            "*multi_modal_projector*input_quantizer":    a_fp8,
            "*lm_head*":             fp16,
            "*[kv]_bmm_quantizer":   fp16,
            # gemma_expert 는 패턴 미매칭 → Pass 1 quantizer state 보존
            # default 키 없음 → 미매칭 quantizer 는 그대로 유지
        },
        "algorithm": "awq_lite",
    }


# ── calibration loop (stage8f_qad.py:204-225 동일 패턴) ─────────────────────

def _make_calib_loop(device: str, n_batches: int = 32):
    def calib_loop(model):
        model.eval()
        dataset = LiberoHDF5Dataset(CALIB_PATH, NORMALIZER_PATH, seed=0)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
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
    """Snapshot _amax tensors for modules matching pattern (for before/after comparison)."""
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
            changed.append(f"{k}: changed (max_diff={( v_after.cpu()-v_before.cpu()).abs().max():.4f})")
    if changed:
        print(f"[WARN] {label} amax CHANGED after Pass 2 ({len(changed)} quantizers):")
        for c in changed[:5]:
            print(f"  {c}")
        if len(changed) > 5:
            print(f"  ... and {len(changed)-5} more")
    else:
        print(f"[OK] {label} amax fully preserved across Pass 2 ({len(before)} quantizers)")


def _verify_amax(policy, out_dir: Path):
    """Report enabled quantizer amax validity. Write summary to file."""
    bad, ok = [], []
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None or not getattr(q, "_enabled", False):
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
                amax = getattr(q, "_amax", None)
                if amax is None or (isinstance(amax, torch.Tensor) and (amax <= 0).any()):
                    q._amax = torch.ones(1, device=next(policy.parameters()).device)
        print("[INFO] Patched invalid amax → 1.0")
    summary = {"valid": len(ok), "invalid": len(bad), "invalid_names": bad[:20]}
    with open(out_dir / "quantizer_amax_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


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
    parser = argparse.ArgumentParser(description="QuantVLA W4A8 LLM + W4A4 DiT QAD eval")
    parser.add_argument("--student_ckpt", default=DEFAULT_STUDENT_CKPT,
                        help="W4A4 DiT QAD trained student checkpoint (default: stage8f best)")
    parser.add_argument("--output_dir", default="results/stage9_quantvla_w4a8_w4a4dit")
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--dit_group_size", type=int, default=8,
                        help="group_size for DiT W4A4 Pass 1 (must match checkpoint)")
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

    # ── Load base policy (FP32) ───────────────────────────────────────────────
    print("[INFO] Building policy from pretrained ...")
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

    # ── Pass 1: DiT W4A4 등록 + max calibration ───────────────────────────────
    print(f"\n[Pass 1] DiT W4A4 (g={args.dit_group_size}) fake-quant + max calibration ...")
    dit_cfg = _build_dit_w4a4_config(group_size=args.dit_group_size)
    mtq.quantize(policy, dit_cfg, forward_loop=calib_loop)
    print("[Pass 1] Done")

    # ── Load W4A4 DiT QAD checkpoint → restore trained weights + amax ────────
    student_ckpt = Path(args.student_ckpt)
    print(f"\n[Load] Restoring DiT QAD checkpoint: {student_ckpt}")
    ckpt_data = torch.load(student_ckpt, map_location="cpu")
    # stage8f saves state_dict directly (not nested under "model" key)
    if isinstance(ckpt_data, dict) and "model" in ckpt_data:
        ckpt_state = ckpt_data["model"]
    else:
        ckpt_state = ckpt_data
    missing, unexpected = policy.load_state_dict(ckpt_state, strict=False)
    print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  missing[:3]: {missing[:3]}")
    if unexpected[:3]:
        print(f"  unexpected[:3]: {unexpected[:3]}")

    # Verify DiT amax loaded correctly
    dit_amax_before = _snapshot_amax(policy, "gemma_expert")
    n_valid_dit = sum(1 for v in dit_amax_before.values() if (v > 0).all())
    print(f"[Load] DiT quantizers with valid amax: {n_valid_dit}/{len(dit_amax_before)}")

    # ── Pass 2: LLM W4A8 + awq_lite calibration ──────────────────────────────
    print(f"\n[Pass 2] LLM/vision/projector W4A8 (awq_lite, {args.calib_batches} batches) ...")
    llm_cfg = _build_llm_w4a8_config()
    mtq.quantize(policy, llm_cfg, forward_loop=calib_loop)
    print("[Pass 2] Done")

    # Verify DiT amax preserved
    dit_amax_after = _snapshot_amax(policy, "gemma_expert")
    _compare_amax(dit_amax_before, dit_amax_after, "DiT gemma_expert")

    # Verify all enabled quantizers have valid amax
    _verify_amax(policy, out_dir)

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
    print(f"[QuantVLA W4A8 LLM + W4A4 DiT QAD] pc_success={pc:.1f}%")
    print(f"  Reference:")
    print(f"    FP16 (Path A):              100.0%")
    print(f"    W4A4 DiT QAD (stage8f):      93.0%")
    print(f"    W4A4 LLM QAD (sequential):    0.0%")
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
        "stage": "stage9_quantvla_w4a8_w4a4dit",
        "llm_quant": "W4A8_awq_lite (INT4_blockwise128 + FP8E4M3)",
        "dit_quant": f"W4A4_max (g={args.dit_group_size}, from stage8f checkpoint)",
        "student_ckpt": str(student_ckpt),
        "num_inference_steps": args.num_inference_steps,
        "n_action_steps": 10,
        "calib_batches": args.calib_batches,
        "overall": overall,
        "eval_info": eval_info,
        "reference": {
            "fp16_path_a": 100.0,
            "w4a4_dit_qad_stage8f": 93.0,
            "w4a4_llm_qad_sequential": 0.0,
        },
    }
    out_path = out_dir / "libero_10_quantvla_w4a8_w4a4dit.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
