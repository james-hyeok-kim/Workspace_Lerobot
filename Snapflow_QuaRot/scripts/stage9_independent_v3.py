"""Stage 9 v3 — Independent Calibration: LLM W4A8 (FP16 DiT 기준) + DiT QAD stage8f 결합.

[Protocol v3 — Independent calibration, inference-time combine]
  Step 1 : fresh policy 로드
  Step 2 : mtq.quantize(v3 config, max calib)
           - DiT quantizer: enable=False 로 등록 → calib forward 에서 no-op (FP16 처럼 동작)
           - LLM quantizer: enable=True → FP16 DiT 분포 기준으로 max calibration
  Step 3 : stage8f 로드 (strict=False)
           - DiT weights → QAD trained values
           - DiT _amax buffers → QAD amax (quantizer disabled 상태여도 buffer 채워짐)
           - LLM amax → Step 2 calib 값 유지
  Step 4 : DiT quantizer 수동 enable (q.enable() per quantizer)
  Step 5 : verify amax + LIBERO-10 eval

v2 와의 차이:
  - DiT quantizer 가 calibration 동안 disabled → LLM 이 FP16 DiT 분포로 calibration
  - v2 는 DiT W4A4 fake-quant 상태에서 LLM calibration (원본 FP16 DiT 기준)
  - 두 버전 모두 inference 시엔 W4A8 LLM + W4A4 DiT 조합 → distribution mismatch 존재

Reference: QuantVLA arXiv 2602.20309
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

def _build_v3_calib_config(dit_group_size: int = 8) -> dict:
    """v3 전용 config: DiT quantizer 는 enable=False 로 등록만, LLM 만 max calib.

    DiT enable=False 효과:
      - TensorQuantizer 모듈은 등록됨 (stage8f 로드 시 _amax buffer 매칭 가능)
      - forward 는 no-op (pass-through) → calib 동안 DiT 는 FP16 처럼 동작
      - LLM 이 깨끗한 FP16 DiT 분포 기준으로 calibration 됨
    Step 4 에서 q.enable() 로 수동 활성화 후 inference.
    """
    # DiT: 등록은 하되 disabled
    w4_dit_off = {"num_bits": 4, "block_sizes": {-1: dit_group_size}, "enable": False}
    a4_dit_off = {"num_bits": 4, "enable": False}
    # LLM/vision/projector: 정상 enable + max calib
    w4_llm = {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True}
    w_fp8  = {"num_bits": (4, 3), "enable": True}
    a_fp8  = {"num_bits": (4, 3), "enable": True}
    fp16   = {"enable": False}
    return {
        "quant_cfg": {
            "*gemma_expert*weight_quantizer":            w4_dit_off,
            "*gemma_expert*input_quantizer":             a4_dit_off,
            "*language_model*weight_quantizer":          [w4_llm, w_fp8],
            "*vision_tower*weight_quantizer":            [w4_llm, w_fp8],
            "*multi_modal_projector*weight_quantizer":   [w4_llm, w_fp8],
            "*language_model*input_quantizer":           a_fp8,
            "*vision_tower*input_quantizer":             a_fp8,
            "*multi_modal_projector*input_quantizer":    a_fp8,
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


def _verify_amax(policy, out_dir: Path):
    """Enabled quantizer 의 amax 유효성 검증. _disabled attribute 사용 (modelopt 실제 API)."""
    bad, ok = [], []
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            # modelopt TensorQuantizer: is_enabled = not self._disabled (tensor_quantizer.py:469-471)
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
    parser = argparse.ArgumentParser(description="Stage9 v3: Independent calibration")
    parser.add_argument("--student_ckpt", default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--output_dir", default="results/stage9_independent_v3")
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--dit_group_size", type=int, default=8)
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

    # ── Load base policy ──────────────────────────────────────────────────────
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

    # ── Step 2: v3 config (DiT disabled) + LLM max calib ─────────────────────
    print(f"\n[Step 2] v3: LLM W4A8 max calib (DiT disabled during calib, FP16 분포 기준) "
          f"({args.calib_batches} batches) ...")
    v3_cfg = _build_v3_calib_config(dit_group_size=args.dit_group_size)
    mtq.quantize(policy, v3_cfg, forward_loop=calib_loop)

    # 확인: LLM amax 정상, DiT amax 는 미설정 상태
    llm_amax = _snapshot_amax(policy, "language_model")
    n_llm_valid = sum(1 for v in llm_amax.values() if (v > 0).all())
    print(f"[Step 2] LLM valid amax: {n_llm_valid}/{len(llm_amax)}")

    dit_amax_pre = _snapshot_amax(policy, "gemma_expert")
    n_dit_pre = sum(1 for v in dit_amax_pre.values() if (v > 0).all())
    print(f"[Step 2] DiT valid amax (should be 0, disabled): {n_dit_pre}/{len(dit_amax_pre)}")

    # DiT quantizer 존재 확인 (enable=False 여도 모듈은 등록돼야 함)
    n_dit_mods = sum(
        1 for name, mod in policy.named_modules()
        if "gemma_expert" in name and getattr(mod, "weight_quantizer", None) is not None
    )
    print(f"[Step 2] DiT QuantLinear modules found: {n_dit_mods}")

    # ── Step 3: stage8f 로드 → DiT weights + _amax buffers 채우기 ────────────
    student_ckpt = Path(args.student_ckpt)
    print(f"\n[Step 3] Loading stage8f checkpoint: {student_ckpt}")
    ckpt_data = torch.load(student_ckpt, map_location="cpu")
    ckpt_state = ckpt_data["model"] if isinstance(ckpt_data, dict) and "model" in ckpt_data else ckpt_data

    missing, unexpected = policy.load_state_dict(ckpt_state, strict=False)
    print(f"[Step 3] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"  missing[:3]: {missing[:3]}")

    dit_amax_post = _snapshot_amax(policy, "gemma_expert")
    n_dit_post = sum(1 for v in dit_amax_post.values() if (v > 0).all())
    print(f"[Step 3] DiT _amax buffers filled: {n_dit_post}/{len(dit_amax_post)}")

    # ── Step 4: DiT quantizer 수동 enable ────────────────────────────────────
    print("\n[Step 4] Enabling DiT quantizers ...")
    n_enabled = 0
    n_skipped = 0
    for name, mod in policy.named_modules():
        if "gemma_expert" not in name:
            continue
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            if hasattr(q, "enable"):
                q.enable()
                n_enabled += 1
            else:
                n_skipped += 1
    print(f"[Step 4] DiT quantizers enabled: {n_enabled}, skipped (no enable method): {n_skipped}")

    # 최종 verify
    dit_amax_final = _snapshot_amax(policy, "gemma_expert")
    n_dit_final = sum(1 for v in dit_amax_final.values() if (v > 0).all())
    print(f"[Step 4] DiT quantizers with valid amax after enable: {n_dit_final}/{len(dit_amax_final)}")

    n_valid, n_bad = _verify_amax(policy, out_dir)
    print(f"[Summary] LLM valid={n_llm_valid}/{len(llm_amax)}, DiT valid={n_dit_final}/{len(dit_amax_final)}, "
          f"Overall enabled+valid={n_valid}")

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
    print(f"[Stage9 v3 — Independent Calib] pc_success={pc:.1f}%")
    print(f"  Reference:")
    print(f"    FP16 (Path A):              100.0%")
    print(f"    W4A4 DiT QAD (stage8f):      93.0%")
    print(f"    Stage9 v2 (combined max):     4.0%")
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
        "stage": "stage9_independent_v3",
        "protocol": "v3_independent_calib",
        "llm_quant": "W4A8_max (INT4_blockwise128 + FP8E4M3 activation, calib with FP16 DiT)",
        "dit_quant": f"W4A4 (g={args.dit_group_size}, from stage8f QAD checkpoint, enable=False during calib)",
        "student_ckpt": str(student_ckpt),
        "num_inference_steps": args.num_inference_steps,
        "n_action_steps": 10,
        "calib_batches": args.calib_batches,
        "overall": overall,
        "eval_info": eval_info,
        "reference": {
            "fp16_path_a": 100.0,
            "w4a4_dit_qad_stage8f": 93.0,
            "stage9_v2_combined_max": 4.0,
        },
    }
    out_path = out_dir / "libero_10_independent_v3.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
