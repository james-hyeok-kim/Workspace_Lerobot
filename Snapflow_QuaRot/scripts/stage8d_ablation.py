"""Stage 8d Ablation — 구성요소별 INT4 g=4 오류 원인 분석.

ablation_mode:
  "llm_only"    : LLM (PaliGemma lang+vision) 만 INT4, gemma_expert FP16
  "expert_only" : gemma_expert 만 INT4, LLM FP16
  "both"        : 전체 INT4 (= stage8d g=4, 82%)
  "fp16"        : 전체 FP16 (= Path A, 100%)
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


def _build_ablation_config(mode: str, group_size: int, quant_type: str = "w4a16") -> dict:
    """mode별 quantization config 생성.
    quant_type: "w4a16" (weight only) or "w4a4" (weight+activation)
    """
    w4 = {"num_bits": 4, "block_sizes": {-1: group_size}, "enable": True}
    # activation quantizer: per-tensor INT4 (no axis to avoid seq-len mismatch)
    a4 = {"num_bits": 4, "enable": True} if quant_type == "w4a4" else {"enable": False}
    fp16 = {"enable": False}

    if mode == "llm_only":
        # LLM (PaliGemma lang+vision) weight [+activation] INT4, gemma_expert FP16
        return {
            "quant_cfg": {
                "*language_model*weight_quantizer": w4,
                "*vision_tower*weight_quantizer": w4,
                "*multi_modal_projector*weight_quantizer": w4,
                "*language_model*input_quantizer": a4,
                "*vision_tower*input_quantizer": a4,
                "*multi_modal_projector*input_quantizer": a4,
                "*gemma_expert*weight_quantizer": fp16,
                "*gemma_expert*input_quantizer": fp16,
                "*action_in_proj*": fp16,
                "*action_out_proj*": fp16,
                "*lm_head*": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "expert_only":
        # gemma_expert INT4, LLM FP16
        return {
            "quant_cfg": {
                "*language_model*weight_quantizer": fp16,
                "*vision_tower*weight_quantizer": fp16,
                "*multi_modal_projector*weight_quantizer": fp16,
                "*gemma_expert*weight_quantizer": w4,
                "*action_in_proj*weight_quantizer": w4,
                "*action_out_proj*weight_quantizer": w4,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "expert_attn":
        # gemma_expert attention (q/k/v/o_proj) weight [+activation] INT4
        return {
            "quant_cfg": {
                "*gemma_expert*q_proj*weight_quantizer": w4,
                "*gemma_expert*k_proj*weight_quantizer": w4,
                "*gemma_expert*v_proj*weight_quantizer": w4,
                "*gemma_expert*o_proj*weight_quantizer": w4,
                "*gemma_expert*q_proj*input_quantizer": a4,
                "*gemma_expert*k_proj*input_quantizer": a4,
                "*gemma_expert*v_proj*input_quantizer": a4,
                "*gemma_expert*o_proj*input_quantizer": a4,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "expert_mlp":
        # gemma_expert MLP (gate/up/down_proj) weight [+activation] INT4
        return {
            "quant_cfg": {
                "*gemma_expert*gate_proj*weight_quantizer": w4,
                "*gemma_expert*up_proj*weight_quantizer": w4,
                "*gemma_expert*down_proj*weight_quantizer": w4,
                "*gemma_expert*gate_proj*input_quantizer": a4,
                "*gemma_expert*up_proj*input_quantizer": a4,
                "*gemma_expert*down_proj*input_quantizer": a4,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "action_proj":
        # action_in_proj + action_out_proj 만 INT4
        return {
            "quant_cfg": {
                "*action_in_proj*weight_quantizer": w4,
                "*action_out_proj*weight_quantizer": w4,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "llm_vision":
        # vision_tower + multi_modal_projector 만 INT4 (LLM lang은 FP16)
        return {
            "quant_cfg": {
                "*vision_tower*weight_quantizer": w4,
                "*multi_modal_projector*weight_quantizer": w4,
                "*language_model*weight_quantizer": fp16,
                "*gemma_expert*weight_quantizer": fp16,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "llm_lang":
        # language_model 만 INT4 (vision은 FP16)
        return {
            "quant_cfg": {
                "*language_model*weight_quantizer": w4,
                "*vision_tower*weight_quantizer": fp16,
                "*multi_modal_projector*weight_quantizer": fp16,
                "*gemma_expert*weight_quantizer": fp16,
                "*lm_head*": fp16,
                "*input_quantizer": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    elif mode == "both":
        # LLM + Expert 모두 INT4 (전체 양자화)
        return {
            "quant_cfg": {
                "*language_model*weight_quantizer": w4,
                "*vision_tower*weight_quantizer": w4,
                "*multi_modal_projector*weight_quantizer": w4,
                "*gemma_expert*weight_quantizer": w4,
                "*action_in_proj*weight_quantizer": w4,
                "*action_out_proj*weight_quantizer": w4,
                "*language_model*input_quantizer": a4,
                "*vision_tower*input_quantizer": a4,
                "*multi_modal_projector*input_quantizer": a4,
                "*gemma_expert*input_quantizer": a4,
                "*lm_head*": fp16,
                "default": fp16,
            },
            "algorithm": "max",
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _apply_ablation_quant(policy, device, mode, group_size, quant_type="w4a16"):
    import modelopt.torch.quantization as mtq

    cfg = _build_ablation_config(mode, group_size, quant_type=quant_type)
    print(f"[INFO] Ablation={mode}: INT4 g={group_size} quant_type={quant_type}")

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
            print(f"[WARN] Calibration error: {e}")

    mtq.quantize(policy, cfg, forward_loop=forward_loop)

    # Patch any uncalibrated quantizers to avoid amax>=0 assert
    import torch
    for name, mod in policy.named_modules():
        for qname in ("input_quantizer", "weight_quantizer"):
            q = getattr(mod, qname, None)
            if q is None:
                continue
            amax = getattr(q, "_amax", None)
            if amax is not None and isinstance(amax, torch.Tensor) and (amax < 0).any():
                q._amax = torch.ones_like(amax).abs_()
    print(f"[INFO] Ablation quant done: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llm_only", "expert_only", "fp16",
                                            "expert_attn", "expert_mlp", "action_proj",
                                            "llm_vision", "llm_lang", "both"],
                        required=True, help="ablation mode")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--quant_type", choices=["w4a16", "w4a4"], default="w4a16",
                        help="w4a16=weight-only INT4, w4a4=weight+activation INT4")
    parser.add_argument("--task_ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    qt_suffix = f"_{args.quant_type}" if args.quant_type != "w4a16" else ""
    out_dir = args.output_dir or f"results/ablation_{args.mode}_g{args.group_size}{qt_suffix}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_cfg.pretrained_path = PRETRAINED_PATH
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.compile_model = False
    policy_cfg.gradient_checkpointing = False
    policy_cfg.n_action_steps = 10
    policy_cfg.num_inference_steps = 1

    env_cfg = LiberoEnv(task="libero_10", task_ids=args.task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    _patch_embed_image(policy)

    sd = torch.load(PATH_A, map_location="cpu")
    policy.load_state_dict(sd, strict=False)
    print(f"[INFO] Path A loaded")

    if args.mode != "fp16":
        _apply_ablation_quant(policy, args.device, args.mode, args.group_size,
                              quant_type=args.quant_type)
    else:
        print("[INFO] FP16 mode — no quantization")

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
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "tokenizer_processor": {"tokenizer_name": "/tmp/paligemma_tok_fast"},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    print(f"\n[INFO] Ablation={args.mode} | {len(args.task_ids)} tasks × {args.n_episodes} ep")
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
    print(f"[ABLATION {args.mode}] pc_success={pc:.1f}%")
    print(f"  fp16 (Path A):        100.0%  (no quant)")
    print(f"  both INT4 g=4:         82.0%  (stage8d)")
    print(f"  {args.mode} INT4 g={args.group_size}: {pc:.1f}%")

    per_task = eval_info.get("per_task", [])
    if per_task:
        print("\nPer-task:")
        for t in per_task:
            succ = t["metrics"]["successes"]
            pct = sum(succ) / len(succ) * 100
            fails = [i for i, s in enumerate(succ) if not s]
            print(f"  task {t['task_id']:2d}: {pct:5.1f}%  fails={fails}")

    result = {
        "ablation_mode": args.mode,
        "group_size": args.group_size,
        "overall": overall,
        "eval_info": eval_info,
        "reference": {"fp16": 100.0, "both_int4_g4": 82.0},
    }
    out_path = Path(out_dir) / f"libero_10_{args.mode}_g{args.group_size}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
