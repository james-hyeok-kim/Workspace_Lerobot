"""Stage 5c — Quantization format sweep.

Tests INT4 with smaller group sizes (16/32) and FP4 formats (MXFP4/NVFP4).

Formats:
  int4_g128  : INT4 weight-only, group=128 (stage5 baseline)
  int4_g32   : INT4 weight-only, group=32
  int4_g16   : INT4 weight-only, group=16
  mxfp4      : MXFP4 weight-only, block=32, E8M0 scale
  nvfp4      : NVFP4 weight-only, block=16, E4M3 scale

All use: LLM R1+R3 rotations + OHB manifest + NFE=10.

Usage (single format):
    bash scripts/run_stage.sh 5c --quant_format int4_g32

Usage (manual):
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \\
    LD_LIBRARY_PATH=/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu \\
    CUDA_VISIBLE_DEVICES=0,2 HF_HUB_OFFLINE=1 \\
    python scripts/stage5c_quant_formats.py --quant_format nvfp4 --device cuda:1
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
OHB_MANIFEST_PATH = "artifacts/stage4_ohb_manifest.json"
CALIB_DATASET_PATH = "/data/jameskimh/james_libero_datasets/libero_10"
NORMALIZER_STATS_PATH = (
    "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
    "/policy_preprocessor_step_2_normalizer_processor.safetensors"
)

FORMATS = ["int4_g128", "int4_g32", "int4_g16", "mxfp4", "nvfp4"]


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


def _apply_quarot(policy, device: str):
    """LLM R1 + shared R3 (identical to stage3/4/5)."""
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
    print(f"[INFO] RMSNorm fused: {n_fused} LLM")

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
    print("[INFO] R1+R3 applied")


def _build_quant_config(quant_format: str, ohb_manifest: dict) -> dict:
    from quant.w4a4_recipe import (
        build_int4_weight_only_config,
        build_mxfp4_weight_only_config,
        build_nvfp4_weight_only_config,
    )
    if quant_format == "int4_g128":
        return build_int4_weight_only_config(group_size=128, ohb_manifest=ohb_manifest, algorithm="max")
    elif quant_format == "int4_g32":
        return build_int4_weight_only_config(group_size=32, ohb_manifest=ohb_manifest, algorithm="max")
    elif quant_format == "int4_g16":
        return build_int4_weight_only_config(group_size=16, ohb_manifest=ohb_manifest, algorithm="max")
    elif quant_format == "mxfp4":
        return build_mxfp4_weight_only_config(ohb_manifest=ohb_manifest)
    elif quant_format == "nvfp4":
        return build_nvfp4_weight_only_config(ohb_manifest=ohb_manifest)
    else:
        raise ValueError(f"Unknown format: {quant_format}. Choose from {FORMATS}")


def _apply_quantization(policy, quant_format: str, device: str) -> dict:
    import modelopt.torch.quantization as mtq

    ohb_manifest = {}
    if Path(OHB_MANIFEST_PATH).exists():
        with open(OHB_MANIFEST_PATH) as f:
            ohb_manifest = json.load(f)
        print(f"[INFO] OHB: {len(ohb_manifest)} layers protected")

    quant_config = _build_quant_config(quant_format, ohb_manifest)

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
                    except Exception:
                        pass
                    n_done += 1
                    if n_done >= 64:
                        break
            print(f"[INFO] Calibration: {n_done} batches")
        except Exception as e:
            print(f"[WARN] Calibration error: {e}")

    print(f"[INFO] Applying {quant_format} quantization...")
    mtq.quantize(policy, quant_config, forward_loop=forward_loop)
    print(f"[INFO] {quant_format} done.")
    return ohb_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_format", choices=FORMATS, required=True,
                        help="Quantization format to evaluate")
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    out_dir = Path(args.output_dir or f"results/stage5c_{args.quant_format}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading policy from {args.pretrained_path}")
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

    print(f"\n[INFO] === Stage 5c: {args.quant_format} (NFE={NUM_INFERENCE_STEPS}) ===")
    _apply_quarot(policy, args.device)
    ohb_manifest = _apply_quantization(policy, args.quant_format, args.device)

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

    print(f"\n[INFO] Evaluating {len(task_ids)} tasks × {args.n_episodes} ep ({args.quant_format}, NFE={NUM_INFERENCE_STEPS})...")
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
    print(f"[STAGE 5c | {args.quant_format}] pc_success={pc:.1f}%  avg_sum_reward={rw:.4f}")
    print(f"  stage0_baseline:  94.6% (NFE=10, FP16)")
    print(f"  stage4_rotation:  95.0% (NFE=10, R1+R3+OHB, FP16)")
    print(f"  stage5_int4_g128: 40.0% (NFE=10, INT4 g=128)")
    print(f"  stage5c_{args.quant_format}: {pc:.1f}% (NFE={NUM_INFERENCE_STEPS})")

    per_group = eval_info.get("per_group", {})
    if per_group:
        print("\nPer-task breakdown:")
        for tg, agg in sorted(per_group.items()):
            print(f"  {tg}: {agg.get('pc_success', float('nan')):.1f}%  (n={agg.get('n_episodes', 0)})")

    result = {
        "stage": f"stage5c_{args.quant_format}",
        "quant_format": args.quant_format,
        "quarot": {"r1_llm": True, "r3_shared": True},
        "n_ohb_protected": len(ohb_manifest),
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "overall": overall,
        "eval_info": eval_info,
    }
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
