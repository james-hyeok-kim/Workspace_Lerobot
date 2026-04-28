"""Stage 8e — Path A + MXFP4/NVFP4 weight-only PTQ (B200 네이티브 FP4 포맷).

MXFP4: E2M1 (2 exponent, 1 mantissa), per-block scale in INT8 (MX standard).
NVFP4: E2M1, per-block scale in FP4 (NVIDIA variant, dynamic 경로).
INT4 대비 floating-point 표현으로 동적 범위가 넓어 정확도 유리.
block_size 4 / 8 / 16 비교.

로드 순서:
  1. base policy
  2. path_a.pt (quarot baked: R1+R3)
  3. MXFP4 weight-only PTQ (configurable block_size)
  4. eval (NFE=1, all 10 LIBERO-10 tasks)
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


def _build_nvfp4_weight_only_config(block_size: int) -> dict:
    """NVFP4 E2M1 weight-only config (NVIDIA FP4, dynamic_block_quant 경로).

    NVFP4_DEFAULT_CFG 구조와 동일:
      num_bits=(2,1), type="dynamic", scale_bits=(4,3) → dynamic_block_quant 경로.
    "type": "static"을 사용하면 scaled_e4m3() 경로로 빠져 E2M1 미지원 에러 발생.
    algorithm="max": amax 기반 보정 (forward_loop 없이도 동작).
    """
    fp4_w = {
        "num_bits": (2, 1),
        "block_sizes": {-1: block_size, "type": "dynamic", "scale_bits": (4, 3)},
        "enable": True,
    }
    fp16 = {"enable": False}

    return {
        "quant_cfg": {
            "*weight_quantizer": fp4_w,
            "*input_quantizer": fp16,
            "*lm_head*weight_quantizer": fp16,
            "*action_in_proj*weight_quantizer": fp16,
            "*action_out_proj*weight_quantizer": fp16,
            "*[kv]_bmm_quantizer": fp16,
            "default": fp16,
        },
        "algorithm": "max",
    }


def _apply_nvfp4_quant(policy, device: str, block_size: int):
    import modelopt.torch.quantization as mtq

    cfg = _build_nvfp4_weight_only_config(block_size)
    print(f"[INFO] NVFP4 weight-only: block_size={block_size}, algorithm=max (dynamic)")

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
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                    try:
                        images, img_masks = model._preprocess_images(batch)
                        tokens = batch[OBS_LANGUAGE_TOKENS]
                        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
                        actions = model.prepare_action(batch)
                        noise = model.model.sample_noise(actions.shape, device)
                        t = torch.ones(actions.shape[0], device=device)
                        get_velocity(model.model, images, img_masks, tokens, masks, noise, t, s=t)
                    except Exception:
                        pass
                    n_done += 1
                    if n_done >= 64:
                        break
            print(f"[INFO] Calibration: {n_done} batches")
        except Exception as e:
            print(f"[WARN] Calibration: {e}")

    mtq.quantize(policy, cfg, forward_loop=forward_loop)
    print(f"[INFO] NVFP4 PTQ done: block_size={block_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=16,
                        help="NVFP4 block size (4, 8, or 16)")
    # MXFP4는 algorithm=None (dynamic), argparse에서는 사용 안 함
    parser.add_argument("--task_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--path_a", default=PATH_A)
    args = parser.parse_args()

    out_dir = args.output_dir or f"results/stage8e_mxfp4_b{args.block_size}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = N_ACTION_STEPS
    policy_cfg.num_inference_steps = NUM_INFERENCE_STEPS

    env_cfg = LiberoEnv(task="libero_10", task_ids=args.task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(policy)

    sd = torch.load(args.path_a, map_location="cpu")
    policy.load_state_dict(sd, strict=False)
    print(f"[INFO] Path A loaded from {args.path_a}")

    _apply_nvfp4_quant(policy, args.device, args.block_size)

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

    print(f"\n[INFO] MXFP4 b={args.block_size} | {len(args.task_ids)} tasks × {args.n_episodes} ep | NFE=1")
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

    print(f"\n{'='*60}")
    print(f"[STAGE 8e NVFP4 b={args.block_size}] pc_success={pc:.1f}%")
    print(f"  Reference:")
    print(f"    FP16 (Path A):          100.0%")
    print(f"    INT4 g=16 max:           50.0%")
    print(f"    INT4 g=8  max:           71.0%")
    print(f"    INT4 g=4  max:           82.0%")
    print(f"  NVFP4 b={args.block_size:2d} (dynamic):           {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ = t["metrics"]["successes"]
            pct = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    result = {
        "stage": "stage8e_nvfp4",
        "quant": {
            "format": "nvfp4",
            "num_bits": "(2,1) E2M1",
            "block_size": args.block_size,
            "scale_bits": "(4,3) FP4",
            "algorithm": "max+dynamic",
            "weight_only": True,
        },
        "snapflow": {"nfe": 1},
        "quarot": {"baked_into_path_a": True},
        "overall": overall,
        "eval_info": eval_info,
        "reference": {
            "fp16": 100.0,
            "int4_g16": 50.0,
            "int4_g8": 71.0,
            "int4_g4": 82.0,
        },
    }
    out_path = Path(out_dir) / f"libero_10_nvfp4_b{args.block_size}_max.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
