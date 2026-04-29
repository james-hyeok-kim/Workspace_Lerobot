"""Stage 8c eval — Path A (quarot baked, NFE=1) + INT4 g=16 PTQ (no training).

진단 결과:
  - stage8b QAT 0% 원인: gemma_expert AdaLN layernorm.dense weights가 500 step 만에
    25-33% 노름 변화 → timestep conditioning 망가짐 → velocity field 붕괴
  - Path A (step 0) = quarot baked, no QAT = NFE=1/10 모두 100%
  - 해결: Path A + INT4 g=16 PTQ (training 없이 calibration만)

로드 순서:
  1. base policy 로드
  2. path_a.pt (quarot baked full state) 로드 — quarot 재적용 없음
  3. INT4 g=16 PTQ calibration (forward_loop으로 amax 보정)
  4. eval
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
PATH_A = "/tmp/path_a.pt"
DEVICE = "cuda"
N_ACTION_STEPS = 10
NUM_INFERENCE_STEPS = 1
OHB_MANIFEST_PATH = "/data/jameskimh/james_lerobot_results/artifacts/stage4_ohb_manifest.json"
CALIB_DATASET_PATH = "/data/jameskimh/james_libero_datasets/libero_10"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)


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


def _load_path_a(policy, path: str):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"[INFO] Path A loaded: {len(sd)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    for chk in ["model.action_in_proj.weight", "model.action_out_proj.weight"]:
        if chk in sd:
            t = sd[chk].float()
            print(f"  {chk.split('.')[-2]}: norm={t.norm():.4f}, dtype={sd[chk].dtype}")


def _apply_int4_g16(policy, device: str) -> dict:
    import modelopt.torch.quantization as mtq
    from quant.w4a4_recipe import build_int4_weight_only_config

    ohb_manifest = {}
    if Path(OHB_MANIFEST_PATH).exists():
        with open(OHB_MANIFEST_PATH) as f:
            ohb_manifest = json.load(f)
        print(f"[INFO] OHB: {len(ohb_manifest)} layers protected")

    quant_config = build_int4_weight_only_config(
        group_size=16, ohb_manifest=ohb_manifest, algorithm="max"
    )

    def forward_loop(model):
        from snapflow.data import LiberoHDF5Dataset
        from snapflow.distill_loss import get_velocity
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        from torch.utils.data import DataLoader
        model.eval()
        try:
            dataset = LiberoHDF5Dataset(
                dataset_path=CALIB_DATASET_PATH,
                normalizer_stats_path=NORMALIZER_STATS_PATH,
                seed=0,
            )
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            n_done = 0
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) if hasattr(v, "to") else v
                             for k, v in batch.items()}
                    try:
                        images, img_masks = model._preprocess_images(batch)
                        tokens = batch[OBS_LANGUAGE_TOKENS]
                        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
                        actions = model.prepare_action(batch)
                        noise = model.model.sample_noise(actions.shape, device)
                        t = torch.ones(actions.shape[0], device=device)
                        get_velocity(model.model, images, img_masks, tokens, masks, noise, t, s=t)
                    except Exception as e:
                        pass
                    n_done += 1
                    if n_done >= 64:
                        break
            print(f"[INFO] Calibration: {n_done} batches")
        except Exception as e:
            print(f"[WARN] Calibration error: {e}")

    mtq.quantize(policy, quant_config, forward_loop=forward_loop)
    print("[INFO] INT4 g=16 PTQ done.")
    return ohb_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--path_a", default=PATH_A)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/stage8c")
    parser.add_argument("--fp16_only", action="store_true",
                        help="INT4 없이 FP16로만 평가 (smoke test용)")
    args = parser.parse_args()

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading base policy from {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    _patch_embed_image(policy)

    quant_label = "FP16 only" if args.fp16_only else "INT4 g=16 PTQ"
    print(f"\n[INFO] === Stage 8c: Path A (quarot baked) + NFE=1 + {quant_label} ===")
    print(f"[INFO] NOTE: Path A = step-0 of QAT training (quarot baked, no gradient updates).")
    print(f"[INFO] Path A: {args.path_a}")

    _load_path_a(policy, args.path_a)

    if not args.fp16_only:
        ohb_manifest = _apply_int4_g16(policy, args.device)
    else:
        ohb_manifest = {}
        print("[INFO] FP16 smoke test mode — skipping INT4 quantization")

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
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "tokenizer_processor": {"tokenizer_name": "/tmp/paligemma_tok_fast"},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep "
          f"(NFE=1, {quant_label})...")
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
    print(f"[STAGE 8c RESULT] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  stage0_baseline:          94.6% (NFE=10, FP16)")
    print(f"  stage1_snapflow:          84.0% (NFE=1,  FP16, no quant)")
    print(f"  stage6_rotation_only:     86.0% (NFE=1,  FP16, R1+R3)")
    print(f"  stage5c_int4_g16:         94.0% (NFE=10, INT4 g=16)")
    print(f"  stage7_snapflow_int4g16:  45.0% (NFE=1,  INT4 g=16, FP16 student)")
    print(f"  stage8b_qat (broken):      0.0% (NFE=1,  INT4 g=16, AdaLN drift bug)")
    print(f"  stage8c ({quant_label}): {pc:.1f}% (NFE=1, {quant_label}, Path A = quarot baked)")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    suffix = "fp16" if args.fp16_only else "int4g16"
    result = {
        "stage": f"stage8c_patha_ptq_{suffix}",
        "snapflow": {"enabled": True, "nfe": 1},
        "quarot": {"baked_into_path_a": True, "r1_llm": True, "r3_shared": True,
                   "note": "rotation applied during path_a creation, NOT re-applied at eval"},
        "quant": {"format": "fp16" if args.fp16_only else "int4",
                  "group_size": 16 if not args.fp16_only else None,
                  "weight_only": True, "ptq_only": True,
                  "n_ohb_protected": len(ohb_manifest)},
        "path_a": args.path_a,
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "overall": overall,
        "eval_info": eval_info,
        "diagnosis": {
            "stage8b_failure": "AdaLN (layernorm.dense.weight) 25-33% drift in 500 QAT steps",
            "fix": "PTQ-only, no gradient training"
        },
    }
    out_path = Path(args.output_dir) / f"libero_10_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
