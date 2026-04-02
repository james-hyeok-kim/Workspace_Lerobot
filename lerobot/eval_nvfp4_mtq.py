"""
NVFP4 Quantization Evaluation for LeRobot policies via NVIDIA ModelOpt MTQ.

Applies NVFP4_DEFAULT_CFG to the full policy, dumps a quantized-layer report
and eval metrics (per-episode + aggregated) to a single JSON file.

Usage (TEST MODE – 1 task, 1 episode):
    python eval_nvfp4_mtq.py \
        --pretrained_path lerobot/pi05_libero_finetuned \
        --task libero_10 \
        --n_episodes 1 \
        --output_json results/nvfp4_test.json
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import gc

# ── Path setup ────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _p in [str(_root / "src"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
        print(f"[PATH] {_p}")

# ── ModelOpt ──────────────────────────────────────────────────────────────────
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    print("[OK] NVIDIA ModelOpt loaded")
except ImportError as e:
    print(f"[ERROR] modelopt not found: {e}")
    sys.exit(1)

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_layer_report(model: torch.nn.Module) -> list[dict]:
    """Return a per-leaf-module list with quantization status."""
    report = []
    for name, module in model.named_modules():
        # Skip non-leaf modules
        if list(module.children()):
            continue
        module_type = type(module).__name__
        # MTQ marks quantized Linear layers with quantizer sub-attributes
        is_quantized = (
            hasattr(module, "weight_quantizer")
            or hasattr(module, "input_quantizer")
            or "Quantized" in module_type
            or "quant" in module_type.lower()
        )
        entry: dict = {"layer": name, "type": module_type, "is_quantized": is_quantized}
        if hasattr(module, "weight") and module.weight is not None:
            entry["weight_dtype"] = str(module.weight.dtype)
            entry["weight_bits"] = module.weight.element_size() * 8
        report.append(entry)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NVFP4 MTQ quantization + LiberoEnv eval")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="HF Hub repo ID or local path to the pretrained policy")
    parser.add_argument("--task", type=str, default="libero_10",
                        help="LIBERO suite name (e.g. libero_10, libero_spatial)")
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Number of episodes to evaluate (default: 1 for TEST MODE)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of parallel envs (n_envs)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable AMP (autocast) during inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="nvfp4_result.json",
                        help="Path for the output JSON report")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 0. Skip if result already exists
    if output_path.exists():
        print(f"[SKIP] 결과 파일이 이미 존재합니다 → {output_path}")
        try:
            with open(output_path, encoding="utf-8") as _f:
                _existing = json.load(_f)
            _agg = _existing.get("eval_results", {}).get("aggregated", {})
            _qc  = _existing.get("quantization", {})
            print(f"[SKIP] task={_existing.get('config',{}).get('task','?')}  "
                  f"success={_agg.get('pc_success', float('nan')):.1f}%  "
                  f"reward={_agg.get('avg_sum_reward', float('nan')):.4f}  "
                  f"quant={_qc.get('quantized_count','?')}/{_qc.get('total_leaf_modules','?')}")
        except Exception as _e:
            print(f"[SKIP] 기존 JSON 읽기 실패: {_e}")
        return

    # 1. Load policy config from Hub / local path
    print(f"[INFO] Loading policy config from: {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    # 2. Build LiberoEnv config (TEST MODE: single suite, single task_id)
    env_cfg = LiberoEnv(task=args.task)

    # 3. Create vectorised envs  →  dict[suite_name -> dict[task_id -> VectorEnv]]
    print(f"[INFO] Creating LIBERO env: task={args.task}, n_envs={args.batch_size}")
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    # 4. Load policy weights
    print(f"[INFO] Loading policy weights ...")
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # 5. Build pre/post processors
    preprocessor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # 6. Apply NVFP4_DEFAULT_CFG quantization to the full policy
    print("[INFO] Applying NVFP4_DEFAULT_CFG quantization to the policy ...")
    mtq.quantize(policy, config=NVFP4_DEFAULT_CFG)
    print("[INFO] Quantization complete.")

    # 6b. Disable torch.compile on the inner model — MTQ replaces Linear layers with
    #     QuantizedLinear modules that use _FoldedCallback, which is incompatible with
    #     torch._dynamo tracing. Restore the original (uncompiled) callables instead.
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner_model = getattr(policy, "model", None)
    if inner_model is not None:
        for _attr in ("sample_actions", "forward"):
            _fn = getattr(inner_model, _attr, None)
            if _fn is None:
                continue
            # torch.compile stores the original callable in _torchdynamo_orig_callable
            _orig = getattr(_fn, "_torchdynamo_orig_callable", None) or getattr(_fn, "_orig_mod", None)
            if _orig is not None:
                setattr(inner_model, _attr, _orig)
                print(f"[INFO] torch.compile disabled for inner_model.{_attr}")

    # 7. Collect quantized-layer report
    layer_report = get_layer_report(policy)
    quant_count = sum(1 for layer in layer_report if layer["is_quantized"])
    total_count = len(layer_report)
    print(f"[INFO] Quantized layers: {quant_count} / {total_count}")

    # 8. Evaluate  (TEST MODE: 1 task_id, n_episodes=1)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]
    print(f"[INFO] Evaluating suite='{suite_name}' task_id={task_id}, n_episodes={args.n_episodes}")

    with torch.no_grad(), (torch.autocast(device_type=device.type) if args.use_amp else nullcontext()):
        eval_info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
        )


    # 8. env 닫기 전에 GC 먼저
    gc.collect()
    env.close()


    # 9. Assemble and save JSON
    output = {
        "config": {
            "pretrained_path": args.pretrained_path,
            "task": args.task,
            "suite_name": suite_name,
            "task_id": task_id,
            "n_episodes": args.n_episodes,
            "batch_size": args.batch_size,
            "use_amp": args.use_amp,
            "device": args.device,
            "quant_config": "NVFP4_DEFAULT_CFG",
        },
        "quantization": {
            "total_leaf_modules": total_count,
            "quantized_count": quant_count,
            "unquantized_count": total_count - quant_count,
            "layers": layer_report,
        },
        "eval_results": eval_info,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # 10. Summary
    agg = eval_info.get("aggregated", {})
    print(f"\n{'='*50}")
    print(f"[RESULT] Task          : {args.task} (suite={suite_name}, task_id={task_id})")
    print(f"[RESULT] Episodes      : {args.n_episodes}")
    print(f"[RESULT] Success rate  : {agg.get('pc_success', float('nan')):.1f}%")
    print(f"[RESULT] Avg sum reward: {agg.get('avg_sum_reward', float('nan')):.4f}")
    print(f"[RESULT] Quant layers  : {quant_count} / {total_count}")
    print(f"[RESULT] Saved to      : {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
