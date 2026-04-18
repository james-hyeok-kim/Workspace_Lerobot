"""
LM + DiT NVFP4_DEFAULT_CFG quantization + libero_10 layer-wise capture.

Applies mtq.quantize(NVFP4_DEFAULT_CFG) to:
  - language_model  (LM / PaliGemma backbone)
  - gemma_expert    (DiT / action expert)

For each task_id (0~9):
  - Registers forward hooks on all quantized Linear layers
  - Captures {x, W, y} on the first forward pass
  - Removes hooks
  - Runs eval_policy (n_episodes)
  - Saves layer_captures_task{tid}.pt + eval_task{tid}.json

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_nvfp4_lm_dit.py \\
        --pretrained_path lerobot/pi05_libero_finetuned \\
        --n_episodes 5 --batch_size 5 --output_dir results
"""

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "lerobot"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ModelOpt ──────────────────────────────────────────────────────────────────
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    print("[OK] ModelOpt loaded")
except ImportError as e:
    print(f"[ERROR] modelopt not found: {e}")
    sys.exit(1)

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# Quantization helpers  (from eval_quant_sweep.py:312-342)
# ══════════════════════════════════════════════════════════════════════════════

def apply_nvfp4_lm_dit(policy: torch.nn.Module) -> None:
    """Apply NVFP4_DEFAULT_CFG to LM + DiT components."""
    lm = policy.model.paligemma_with_expert.paligemma.model.language_model
    mtq.quantize(lm, config=NVFP4_DEFAULT_CFG)
    print("[INFO] NVFP4 applied → language_model")

    expert = policy.model.paligemma_with_expert.gemma_expert
    mtq.quantize(expert, config=NVFP4_DEFAULT_CFG)
    print("[INFO] NVFP4 applied → gemma_expert")


def disable_torch_compile(policy: torch.nn.Module) -> None:
    """Disable torch.compile on inner model (required after MTQ)."""
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is None:
        return
    for attr in ("sample_actions", "forward"):
        fn = getattr(inner, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
        if orig is not None:
            setattr(inner, attr, orig)
            print(f"[INFO] torch.compile disabled for model.{attr}")


def build_quant_report(policy: torch.nn.Module) -> dict:
    """Count quantized vs total Linear layers."""
    total, quantized, layers = 0, 0, []
    for name, m in policy.named_modules():
        if list(m.children()):
            continue
        if isinstance(m, torch.nn.Linear) or hasattr(m, "weight_quantizer"):
            total += 1
            is_q = hasattr(m, "weight_quantizer")
            if is_q:
                quantized += 1
            layers.append({"name": name, "type": type(m).__name__, "quantized": is_q})
    return {"total_linear": total, "quantized": quantized, "layers": layers}


# ══════════════════════════════════════════════════════════════════════════════
# Layer capture
# ══════════════════════════════════════════════════════════════════════════════

def register_capture_hooks(policy: torch.nn.Module):
    """Register forward hooks on all quantized Linear layers.

    Returns (captures_dict, hooks_list).
    captures_dict[layer_name] = {"x": tensor, "W": tensor, "y": tensor}
    Call remove_hooks(hooks_list) after the first forward.
    """
    captures = {}
    hooks = []

    for name, module in policy.named_modules():
        if not hasattr(module, "weight_quantizer"):
            continue

        def _make_hook(layer_name, mod):
            def _hook(m, inp, out):
                if layer_name in captures:
                    return  # already captured
                x = inp[0].detach().cpu() if isinstance(inp, (tuple, list)) else inp.detach().cpu()
                # weight: may be plain Parameter or QTensorWrapper
                w_raw = mod.weight
                try:
                    from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
                    if isinstance(w_raw, QTensorWrapper):
                        qt = w_raw.get_qtensor()
                        wq = mod.weight_quantizer
                        w = qt.dequantize(
                            dtype=torch.bfloat16,
                            scale=getattr(wq, "_scale", None),
                            double_scale=getattr(wq, "_double_scale", None),
                            block_sizes={-1: 16},
                        ).detach().cpu()
                    else:
                        # fake-quant path: apply weight_quantizer (quantize→dequantize) to get
                        # the fake-quantized W that was actually used in the forward pass.
                        try:
                            w = mod.weight_quantizer(w_raw).detach().cpu()
                        except Exception:
                            w = w_raw.detach().cpu()
                except Exception:
                    w = w_raw.detach().cpu()
                y = out.detach().cpu()
                captures[layer_name] = {"x": x, "W": w, "y": y}
            return _hook

        h = module.register_forward_hook(_make_hook(name, module))
        hooks.append(h)

    return captures, hooks


def remove_hooks(hooks: list) -> None:
    for h in hooks:
        h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Task description helper
# ══════════════════════════════════════════════════════════════════════════════

def get_task_commands(suite_name: str = "libero_10") -> dict:
    """Return {task_id: task_language_description}."""
    try:
        from libero.libero.benchmark import get_benchmark_dict
        suite = get_benchmark_dict()[suite_name]()
        return {i: suite.get_task(i).language for i in range(suite.n_tasks)}
    except Exception as e:
        print(f"[WARN] Could not load task descriptions: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 LM+DiT quantization + libero_10 layer-wise capture"
    )
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Task IDs to evaluate (default: 0~9 all)")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    print(f"[INFO] task_ids: {task_ids}  n_episodes: {args.n_episodes}  batch_size: {args.batch_size}")

    # ── task commands ─────────────────────────────────────────────────────────
    task_commands = get_task_commands("libero_10")
    (out_dir / "task_commands.json").write_text(
        json.dumps({str(k): v for k, v in task_commands.items()}, indent=2, ensure_ascii=False)
    )
    for tid in task_ids:
        print(f"  task_id={tid}: {task_commands.get(tid, '?')}")

    # ── model load ────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading policy: {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False

    env_cfg = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # ── NVFP4 quantization ────────────────────────────────────────────────────
    print("\n[INFO] Applying NVFP4_DEFAULT_CFG to LM + DiT ...")
    apply_nvfp4_lm_dit(policy)
    disable_torch_compile(policy)

    # ── quant report ──────────────────────────────────────────────────────────
    report = build_quant_report(policy)
    print(f"[INFO] Quantized: {report['quantized']} / {report['total_linear']} Linear layers")
    (out_dir / "quant_report.json").write_text(json.dumps(report, indent=2))

    # ── suite name ────────────────────────────────────────────────────────────
    suite_name = next(iter(envs_dict))

    # ── per-task eval + capture ───────────────────────────────────────────────
    summary = {}

    for tid in task_ids:
        print(f"\n{'='*60}")
        print(f"[INFO] task_id={tid}: {task_commands.get(tid, '?')}")
        print(f"{'='*60}")

        env = envs_dict[suite_name][tid]

        # register hooks
        captures, hooks = register_capture_hooks(policy)
        captured_flag = {"done": False}

        # wrap to remove hooks after first forward
        original_sample = policy.model.sample_actions if hasattr(policy.model, "sample_actions") else None

        def _remove_once(*args, **kwargs):
            if not captured_flag["done"] and len(captures) > 0:
                remove_hooks(hooks)
                captured_flag["done"] = True
                print(f"  [INFO] Captured {len(captures)} layers, hooks removed.")

        # attach a post-forward callback via a thin module hook on policy
        def _policy_hook(m, inp, out):
            _remove_once()

        ph = policy.register_forward_hook(_policy_hook)

        # eval
        with torch.no_grad():
            eval_info = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
            )

        ph.remove()
        # ensure hooks removed even if policy hook didn't fire
        if not captured_flag["done"]:
            remove_hooks(hooks)
            print(f"  [INFO] Captured {len(captures)} layers (fallback removal).")

        agg = eval_info.get("aggregated", {})
        pc = agg.get("pc_success", float("nan"))
        rw = agg.get("avg_sum_reward", float("nan"))
        print(f"  [RESULT] success={pc:.1f}%  reward={rw:.4f}  captured_layers={len(captures)}")

        # save layer captures
        cap_path = out_dir / f"layer_captures_task{tid}.pt"
        torch.save({"task_id": tid, "task_command": task_commands.get(tid, ""), "captures": captures},
                   cap_path)
        print(f"  [SAVED] {cap_path}  ({len(captures)} layers)")

        # save eval result
        eval_out = {
            "task_id": tid,
            "task_command": task_commands.get(tid, ""),
            "n_episodes": args.n_episodes,
            "eval_results": eval_info,
            "pc_success": pc,
            "avg_sum_reward": rw,
            "captured_layers": len(captures),
        }
        eval_path = out_dir / f"eval_task{tid}.json"
        eval_path.write_text(json.dumps(eval_out, indent=2, default=str))

        summary[tid] = {
            "task_command": task_commands.get(tid, ""),
            "pc_success": pc,
            "avg_sum_reward": rw,
            "captured_layers": len(captures),
        }

        del captures
        gc.collect()
        torch.cuda.empty_cache()

    # ── summary ───────────────────────────────────────────────────────────────
    successes = [v["pc_success"] for v in summary.values() if not isinstance(v["pc_success"], float) or not __import__("math").isnan(v["pc_success"])]
    avg_success = sum(successes) / len(successes) if successes else float("nan")

    print(f"\n{'='*60}")
    print(f"[SUMMARY] avg_success={avg_success:.1f}%  tasks={list(summary.keys())}")
    for tid, s in summary.items():
        print(f"  task_id={tid}: {s['pc_success']:.1f}%  ({s['task_command'][:60]})")

    summary_out = {
        "pretrained_path": args.pretrained_path,
        "quant": "NVFP4_DEFAULT_CFG (LM+DiT)",
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "avg_success": avg_success,
        "per_task": {str(k): v for k, v in summary.items()},
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary_out, indent=2, ensure_ascii=False))
    print(f"[SAVED] {out_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
