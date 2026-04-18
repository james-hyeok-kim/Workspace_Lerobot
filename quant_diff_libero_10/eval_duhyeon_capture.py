"""
eval_duhyeon_capture.py
Duhyeon nvfp4_bmm (is_all + BMM) + libero_10 layer-wise capture.

Applies Duhyeon's nvfp4 hook (from e2e_quant_0417.py) to ALL nn.Linear:
  - weight: mx_block_quant(W, nvfp4)
  - activation: mx_block_quant(x, nvfp4)
  - BMM=True: q/k/v projection OUTPUT also quantized

For each task_id (0~9):
  - Runs eval_policy (n_episodes)
  - Captures {x, W_fp, W_dh, y_fp (pre-hook), y_dh (hook output)} on first forward
  - Saves layer_captures_task{tid}.pt + eval_task{tid}.json

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_duhyeon_capture.py \\
        --pretrained_path lerobot/pi05_libero_finetuned \\
        --n_episodes 5 --batch_size 5 --output_dir duhyeon_results
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "lerobot"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# Duhyeon's NVFP4 fake-quant (from e2e_quant_0417.py)
# ══════════════════════════════════════════════════════════════════════════════

_FP8_E4M3FN_MAX      = 448.0
_FP8_E4M3FN_MIN_NORM = 2.0 ** (1 - 7)


def _fp_quant_dh(x: torch.Tensor, exp_bits: int, man_bits: int, fp_max: float) -> torch.Tensor:
    bias     = (1 << (exp_bits - 1)) - 1
    min_exp  = 1 - bias
    max_exp  = int(math.log2(fp_max))
    min_norm = 2.0 ** min_exp
    sub_step = min_norm / (2 ** man_bits)

    x    = x.float()
    sign = x.sign()
    xabs = x.abs().clamp(max=fp_max)

    xsub  = (xabs / sub_step).round() * sub_step
    log2f = xabs.clamp(min=1e-38).log2().floor().clamp(min_exp, max_exp)
    man_ulp = (2.0 ** log2f) / (2 ** man_bits)
    xnorm = ((xabs / man_ulp).round() * man_ulp).clamp(max=fp_max)

    result = torch.where(xabs >= min_norm, xnorm, xsub)
    result = torch.where(xabs == 0, torch.zeros_like(result), result)
    return (sign * result)


def _fp8_e4m3fn_scale_dh(max_abs: torch.Tensor, fp4_max: float, x_amax: float) -> torch.Tensor:
    x_scales_hp = (max_abs * _FP8_E4M3FN_MAX / x_amax).clamp(
        min=_FP8_E4M3FN_MIN_NORM, max=_FP8_E4M3FN_MAX
    )
    return _fp_quant_dh(x_scales_hp, 4, 3, fp_max=_FP8_E4M3FN_MAX)


def duhyeon_nvfp4_fake_quant(t: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """mx_block_quant with scale_type='fp8_e4m3fn', E2M1 format (Duhyeon)."""
    fp_max     = 6.0
    exp_bits, man_bits = 2, 1
    orig_shape = t.shape
    K    = orig_shape[-1]
    x_2d = t.float().reshape(-1, K)
    T    = x_2d.shape[0]

    pad = (-K) % block_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad))

    n_blocks = x_2d.shape[1] // block_size
    x_blocks = x_2d.reshape(T, n_blocks, block_size)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
    x_amax  = x_2d.abs().max().clamp(min=1e-38).item()

    decode_scale = x_amax / (fp_max * _FP8_E4M3FN_MAX)
    x_scales     = _fp8_e4m3fn_scale_dh(max_abs, fp_max, x_amax)

    x_q  = _fp_quant_dh(x_blocks / (decode_scale * x_scales), exp_bits, man_bits, fp_max)
    x_dq = (x_q * decode_scale * x_scales).reshape(T, -1)[:, :K]
    return x_dq.reshape(orig_shape[:-1] + (K,)).to(t.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Hook factories
# ══════════════════════════════════════════════════════════════════════════════

def is_qkv_proj(name: str) -> bool:
    return any(name.endswith(s) for s in (".q_proj", ".k_proj", ".v_proj"))


def _make_duhyeon_nvfp4_hook(module: nn.Linear, quantize_output: bool):
    """Hook that replaces the layer's output with Duhyeon NVFP4 fake-quant result."""
    def hook(m, inp, output):
        x     = inp[0]
        orig  = x.shape
        dtype = x.dtype
        x2d   = x.reshape(-1, x.shape[-1])
        w     = m.weight

        w_q = duhyeon_nvfp4_fake_quant(w.float()).to(dtype)
        x_q = duhyeon_nvfp4_fake_quant(x2d.float()).to(dtype)

        y = x_q @ w_q.T
        if m.bias is not None:
            y = y + m.bias

        result = y.reshape(orig[:-1] + (w.shape[0],))

        if quantize_output:
            r2d    = result.reshape(-1, result.shape[-1])
            result = duhyeon_nvfp4_fake_quant(r2d.float()).to(dtype).reshape(result.shape)

        return result
    return hook


def attach_duhyeon_hooks(policy) -> list:
    """Attach Duhyeon nvfp4_bmm hooks to ALL nn.Linear. Returns list of handles."""
    hooks = []
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        qout = is_qkv_proj(name)
        h    = mod.register_forward_hook(_make_duhyeon_nvfp4_hook(mod, quantize_output=qout))
        hooks.append(h)
    return hooks


# ══════════════════════════════════════════════════════════════════════════════
# Capture hooks (fired AFTER Duhyeon hook, so y = Duhyeon-quantized output)
# ══════════════════════════════════════════════════════════════════════════════

def register_capture_hooks(policy):
    """Register capture hooks on all nn.Linear (first forward only).

    NOTE: register these AFTER Duhyeon hooks so the captured 'y' = Duhyeon output.
    Captures:
      x     : input activation (FP, before quantization — Duhyeon hook quantizes inside)
      W_fp  : original FP weight
      W_dh  : Duhyeon fake-quantized weight (offline)
      y_dh  : hook output (= Duhyeon quantized output since Duhyeon hook already ran)
    """
    captures = {}
    hooks    = []

    for name, module in policy.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def _make_hook(layer_name, mod):
            def _hook(m, inp, out):
                if layer_name in captures:
                    return
                x    = inp[0].detach().cpu()
                W_fp = mod.weight.detach().cpu()
                # Offline compute W_dh
                W_dh = duhyeon_nvfp4_fake_quant(W_fp.float()).to(W_fp.dtype)
                y_dh = out.detach().cpu()
                captures[layer_name] = {
                    "x":    x,
                    "W_fp": W_fp,
                    "W_dh": W_dh,
                    "y_dh": y_dh,
                }
            return _hook

        h = module.register_forward_hook(_make_hook(name, module))
        hooks.append(h)

    return captures, hooks


def remove_hooks(hooks: list) -> None:
    for h in hooks:
        h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Task descriptions
# ══════════════════════════════════════════════════════════════════════════════

def get_task_commands(suite_name: str = "libero_10") -> dict:
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
        description="Duhyeon nvfp4_bmm eval + layer capture on libero_10"
    )
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task_ids",    type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes",  type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=5)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="duhyeon_results")
    args = parser.parse_args()

    device  = torch.device(args.device)
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
    print(f"\n[INFO] Loading policy (no MTQ — Duhyeon hooks only): {args.pretrained_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False

    env_cfg  = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # ── Disable torch.compile (hooks are incompatible with dynamo tracing) ────
    import torch._dynamo as _dynamo
    _dynamo.reset()
    inner = getattr(policy, "model", None)
    if inner is not None:
        for attr in ("sample_actions", "forward"):
            fn = getattr(inner, attr, None)
            if fn is None:
                continue
            orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "_orig_mod", None)
            if orig is not None:
                setattr(inner, attr, orig)
                print(f"[INFO] torch.compile disabled for model.{attr}")

    preprocessor_overrides = {"device_processor": {"device": args.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # ── Attach Duhyeon nvfp4_bmm hooks ───────────────────────────────────────
    print("\n[INFO] Attaching Duhyeon nvfp4_bmm hooks (is_all + BMM) ...")
    total_linear = sum(1 for _, m in policy.named_modules() if isinstance(m, nn.Linear))
    bmm_count    = sum(1 for n, m in policy.named_modules()
                       if isinstance(m, nn.Linear) and is_qkv_proj(n))
    print(f"  Total nn.Linear: {total_linear}")
    print(f"  BMM layers (q/k/v — output also quantized): {bmm_count}")

    dh_quant_hooks = attach_duhyeon_hooks(policy)
    print(f"  Attached {len(dh_quant_hooks)} Duhyeon hooks")

    # Save hooked layer list
    all_linear_names = [n for n, m in policy.named_modules() if isinstance(m, nn.Linear)]
    (out_dir / "hooked_layers.txt").write_text("\n".join(all_linear_names))

    suite_name = next(iter(envs_dict))

    # ── per-task eval + capture ───────────────────────────────────────────────
    summary = {}

    for tid in task_ids:
        print(f"\n{'='*60}")
        print(f"[INFO] task_id={tid}: {task_commands.get(tid, '?')}")
        print(f"{'='*60}")

        env = envs_dict[suite_name][tid]

        # Register capture hooks (AFTER Duhyeon hooks — so y = Duhyeon output)
        captures, cap_hooks = register_capture_hooks(policy)
        captured_flag = {"done": False}

        def _policy_hook(m, inp, out):
            if not captured_flag["done"] and len(captures) > 0:
                remove_hooks(cap_hooks)
                captured_flag["done"] = True
                print(f"  [INFO] Captured {len(captures)} layers, cap hooks removed.")

        ph = policy.register_forward_hook(_policy_hook)

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
        if not captured_flag["done"]:
            remove_hooks(cap_hooks)
            print(f"  [INFO] Captured {len(captures)} layers (fallback removal).")

        agg = eval_info.get("aggregated", {})
        pc  = agg.get("pc_success", float("nan"))
        rw  = agg.get("avg_sum_reward", float("nan"))
        print(f"  [RESULT] success={pc:.1f}%  reward={rw:.4f}  captured_layers={len(captures)}")

        # Save captures
        cap_path = out_dir / f"layer_captures_task{tid}.pt"
        torch.save({
            "task_id": tid,
            "task_command": task_commands.get(tid, ""),
            "captures": captures,
            "method": "duhyeon_nvfp4_bmm",
        }, cap_path)
        print(f"  [SAVED] {cap_path}  ({len(captures)} layers)")

        eval_out = {
            "task_id": tid,
            "task_command": task_commands.get(tid, ""),
            "method": "duhyeon_nvfp4_bmm",
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
    import math as _math
    successes = [v["pc_success"] for v in summary.values()
                 if not _math.isnan(v["pc_success"])]
    avg_success = sum(successes) / len(successes) if successes else float("nan")

    print(f"\n{'='*60}")
    print(f"[SUMMARY] avg_success={avg_success:.1f}%  tasks={list(summary.keys())}")
    for tid, s in summary.items():
        print(f"  task_id={tid}: {s['pc_success']:.1f}%  ({s['task_command'][:60]})")

    summary_out = {
        "pretrained_path": args.pretrained_path,
        "method": "duhyeon_nvfp4_bmm (is_all + BMM)",
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "avg_success": avg_success,
        "per_task": {str(k): v for k, v in summary.items()},
    }
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary_out, indent=2, ensure_ascii=False)
    )
    print(f"[SAVED] {out_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
