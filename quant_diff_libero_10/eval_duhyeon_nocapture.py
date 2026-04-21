"""
eval_duhyeon_nocapture.py
Duhyeon nvfp4_bmm (is_all + BMM) — success rate only, NO layer capture.

Usage:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python eval_duhyeon_nocare.py \
        --pretrained_path lerobot/pi05_libero_finetuned \
        --task_ids 5 6 7 8 9 \
        --n_episodes 10 --batch_size 5 \
        --output_dir results_duhyeon_10ep
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

_ROOT = Path(__file__).resolve().parent.parent
for _p in [
    str(_ROOT / "lerobot" / "src"),
    str(_ROOT / "lerobot"),
    str(_ROOT / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# Duhyeon's NVFP4 fake-quant
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


def is_qkv_proj(name: str) -> bool:
    return any(name.endswith(s) for s in (".q_proj", ".k_proj", ".v_proj"))


def _make_duhyeon_nvfp4_hook(mod: nn.Linear, quantize_output: bool):
    def hook(m, inp, result):
        dtype = result.dtype
        x     = inp[0]
        x_2d  = x.reshape(-1, x.shape[-1])
        x_q   = duhyeon_nvfp4_fake_quant(x_2d.float()).to(dtype).reshape(x.shape)

        W     = mod.weight
        W_q   = duhyeon_nvfp4_fake_quant(W.float()).to(dtype)

        x2d   = x_q.reshape(-1, x_q.shape[-1])
        out   = (x2d @ W_q.T).reshape(result.shape[:-1] + (W_q.shape[0],))
        if mod.bias is not None:
            out = out + mod.bias

        if quantize_output:
            out = duhyeon_nvfp4_fake_quant(out.reshape(-1, out.shape[-1]).float()).to(dtype).reshape(out.shape)

        return out
    return hook


def attach_duhyeon_hooks(policy) -> list:
    hooks = []
    for name, mod in policy.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        qout = is_qkv_proj(name)
        h    = mod.register_forward_hook(_make_duhyeon_nvfp4_hook(mod, quantize_output=qout))
        hooks.append(h)
    return hooks


def _install_init_state_controller(vec_env):
    """Pin episode assignments to init_states[i + batch*n_envs] (from e2e_quant_0417.py).

    External reset (seed != None): uses controlled counter.
    Internal reset (seed=None, from step() autoreset): ignored.
    """
    def _make_controlled_reset(env, orig, counter, stride):
        def _controlled_reset(seed=None, **kwargs):
            if seed is not None:
                env.init_state_id = counter[0]
                result = orig(seed=seed, **kwargs)
                counter[0] += stride
            else:
                result = orig(seed=seed, **kwargs)
            return result
        return _controlled_reset

    counters = []
    for sub_env in vec_env.envs:
        counter = [sub_env.episode_index]
        stride  = sub_env._reset_stride
        sub_env.reset = _make_controlled_reset(sub_env, sub_env.reset, counter, stride)
        counters.append((counter, sub_env.episode_index))

    def reset_counters():
        for counter, initial in counters:
            counter[0] = initial
    return reset_counters


def get_task_commands(suite_name: str = "libero_10") -> dict:
    try:
        from libero.libero.benchmark import get_benchmark_dict
        suite = get_benchmark_dict()[suite_name]()
        return {i: suite.get_task(i).language for i in range(suite.n_tasks)}
    except Exception as e:
        print(f"[WARN] Could not load task descriptions: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Duhyeon nvfp4_bmm eval — success rate only (no capture)"
    )
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task_ids",    type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes",  type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=5)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--output_dir",  type=str, default="results_duhyeon_10ep")
    parser.add_argument("--start_seed",  type=int, default=1000,
                        help="Seed passed to eval_policy for deterministic episode seeding")
    parser.add_argument("--n_action_steps", type=int, default=10,
                        help="Number of action steps per inference (overrides policy default 50)")
    parser.add_argument("--init_control", action="store_true",
                        help="Install init_state_controller for deterministic init state cycling")
    args = parser.parse_args()

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = args.task_ids if args.task_ids is not None else list(range(10))
    print(f"[INFO] task_ids={task_ids}  n_episodes={args.n_episodes}  batch_size={args.batch_size}")

    task_commands = get_task_commands("libero_10")
    for tid in task_ids:
        print(f"  task_id={tid}: {task_commands.get(tid, '?')}")

    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    policy_cfg.n_action_steps = args.n_action_steps

    env_cfg   = LiberoEnv(task="libero_10", task_ids=task_ids)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

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

    print("\n[INFO] Attaching Duhyeon nvfp4_bmm hooks ...")
    dh_hooks = attach_duhyeon_hooks(policy)
    n_bmm    = sum(1 for n, m in policy.named_modules() if isinstance(m, nn.Linear) and is_qkv_proj(n))
    print(f"  Attached {len(dh_hooks)} hooks  (BMM q/k/v: {n_bmm})")

    suite_name = next(iter(envs_dict))
    if args.init_control:
        for tid in task_ids:
            _install_init_state_controller(envs_dict[suite_name][tid])
        print(f"[INFO] init_state_controller installed for {len(task_ids)} tasks")
    summary    = {}

    for tid in task_ids:
        print(f"\n{'='*60}")
        print(f"[INFO] task_id={tid}: {task_commands.get(tid, '?')}")

        env = envs_dict[suite_name][tid]

        with torch.no_grad():
            eval_info = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                start_seed=args.start_seed,
            )

        agg = eval_info.get("aggregated", {})
        pc  = agg.get("pc_success", float("nan"))
        rw  = agg.get("avg_sum_reward", float("nan"))
        print(f"  [RESULT] task={tid}  success={pc:.1f}%  reward={rw:.4f}")

        eval_out = {
            "task_id":       tid,
            "task_command":  task_commands.get(tid, ""),
            "method":        "duhyeon_nvfp4_bmm",
            "n_episodes":    args.n_episodes,
            "eval_results":  eval_info,
            "pc_success":    pc,
            "avg_sum_reward": rw,
        }
        (out_dir / f"eval_task{tid}.json").write_text(json.dumps(eval_out, indent=2, default=str))
        summary[tid] = {"task_command": task_commands.get(tid, ""), "pc_success": pc}

        gc.collect()
        torch.cuda.empty_cache()

    successes = [v["pc_success"] for v in summary.values()]
    avg = sum(successes) / len(successes) if successes else float("nan")
    print(f"\n[SUMMARY] avg={avg:.1f}%  tasks={list(summary.keys())}")
    for tid, s in summary.items():
        print(f"  task {tid}: {s['pc_success']:.1f}%  {s['task_command'][:60]}")

    summary_out = {
        "method": "duhyeon_nvfp4_bmm",
        "task_ids": task_ids,
        "n_episodes": args.n_episodes,
        "batch_size": args.batch_size,
        "start_seed": args.start_seed,
        "n_action_steps": args.n_action_steps,
        "init_control": args.init_control,
        "avg_success": avg,
        "per_task": {str(k): v for k, v in summary.items()},
    }
    (out_dir / "eval_summary_nocarture.json").write_text(json.dumps(summary_out, indent=2, ensure_ascii=False))
    print(f"[SAVED] {out_dir}/eval_summary_nocarture.json")


if __name__ == "__main__":
    main()
