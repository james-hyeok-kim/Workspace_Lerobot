"""
End-to-end quantization evaluation for PI05 — full LIBERO suite.

Skeleton : bmm_libero10.py  (per-task env creation, forward-hook quantization)
Formats  : sensitivity_mx.py-style _FORMATS dict
NVFP4    : FP8-E4M3fn block-scale fake-quant (fouroversix NVFP4 spec reference, no lib dep)

Supported formats (extend _FORMATS to add new schemes):
  int_w8a8  INT8 W/A symmetric fake-quant (per-channel W, per-token A)
  mxfp4     E2M1/E2M1  block=32  E8M0 scale       (OCP MXFP4,  fp_max=6.0)
  mxfp6     E2M3/E3M2  block=32  E8M0 scale       (OCP MXFP6,  fp_max=7.5/28.0)
  mxfp8     E4M3/E5M2  block=32  E8M0 scale       (OCP MXFP8,  fp_max=448/57344)
  nvfp4     E2M1/E2M1  block=16  FP8-E4M3fn scale (OCP NVFP4,  fp_max=6.0)

nan_type convention (three distinct NaN encodings in OCP FP formats):
  "none"     all bit patterns are valid finite numbers (E2M1, MXFP6)
             fp_max = (2 - 2^-m) * 2^max_exp  where max_exp uses all-ones exponent
  "max_man"  only (max_exp, all-ones mantissa) = NaN; all-ones exponent otherwise valid
             OCP E4M3 / FP8-E4M3fn: fp_max = 1.75 * 2^8 = 448  (NOT 240)
  "max_exp"  all-ones exponent = NaN/inf (like IEEE; E5M2)
             OCP E5M2: fp_max = 1.75 * 2^15 = 57344

Default experiments: baseline, int_w8a8_bmm, mxfp4_bmm, nvfp4_bmm
  All use is_all layer filter.  BMM = quantize q/k/v projection outputs.

CLI args (stripped before draccus sees them):
  --suites=all | libero_spatial,libero_10,...   (default: all)
  --exps=baseline,mxfp4_bmm,...                (default: all experiments)

Usage:
python duhyeon/thanos/e2e_quant.py \
    --suites=libero_10 \
    --policy.path=lerobot/pi05_libero_finetuned \
    --env.type=libero --env.task=libero_spatial \
    --eval.batch_size=5 --eval.n_episodes=50 \
    --policy.n_action_steps=10 --policy.device=cuda \
    --policy.compile_model=false
"""

import json
import logging
import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.libero import _get_suite as _libero_get_suite
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
import lerobot.scripts.lerobot_eval as _lerobot_eval
from lerobot.scripts.lerobot_eval import run_one
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from tqdm import trange as _orig_trange


def _patched_trange(*args, desc="", **kwargs):
    if "Running rollout" in str(desc):
        kwargs["disable"] = True
    return _orig_trange(*args, desc=desc, **kwargs)

_lerobot_eval.trange = _patched_trange


# ===========================================================================
# Layer filters
# ===========================================================================

def is_siglip(name):   return "encoder.layers" in name
def is_expert(name):   return "gemma_expert" in name
def is_vlm(name):      return "language_model" in name
def is_gemma(name):    return is_expert(name) or is_vlm(name)
def is_all(name):      return True
def is_qkv_proj(name): return any(name.endswith(s) for s in (".q_proj", ".k_proj", ".v_proj"))


# ===========================================================================
# FP fake-quant primitives  (OCP-correct fp_max)
# ===========================================================================

def _fp_max_val(exp_bits: int, man_bits: int, nan_type: str = "none") -> float:
    """Max representable finite value for a FP format.

    nan_type:
      "none"     all bit patterns valid (E2M1/MXFP4, MXFP6).
                 fp_max = (2 - 2^-m) * 2^max_exp
      "max_man"  only (max_exp, all-ones mantissa) = NaN; all-ones exponent is otherwise valid.
                 OCP E4M3 / FP8-E4M3fn: max mantissa = (2^m - 2), fp_max = 1.75 * 2^8 = 448.
      "max_exp"  all-ones exponent = NaN/inf (IEEE-like; OCP E5M2).
                 fp_max = (2 - 2^-m) * 2^(max_exp - 1), e.g. 1.75 * 2^15 = 57344.
    """
    bias         = (1 << (exp_bits - 1)) - 1
    all_ones_exp = (1 << exp_bits) - 1  # biased

    if nan_type == "max_exp":
        # Reserve all-ones exponent; mantissa can be anything
        max_biased_exp = all_ones_exp - 1
        man_factor     = 2.0 - 2.0 ** (-man_bits)
    elif nan_type == "max_man":
        # All-ones exponent valid; only (max_exp, all-ones mantissa) = NaN
        max_biased_exp = all_ones_exp
        man_factor     = 1.0 + (2 ** man_bits - 2) / 2 ** man_bits  # = 2 - 2^(1-m)
    else:  # "none"
        max_biased_exp = all_ones_exp
        man_factor     = 2.0 - 2.0 ** (-man_bits)

    return man_factor * (2.0 ** (max_biased_exp - bias))


def _fp_quant(x: torch.Tensor, exp_bits: int, man_bits: int, fp_max: float) -> torch.Tensor:
    """Fake-quantize x to a low-precision FP format (normal + subnormal).

    fp_max is passed explicitly so callers control the NaN convention
    (see _fp_max_val for how to compute it per format).
    max_exp for log2 clamping is derived as floor(log2(fp_max)).
    """
    bias     = (1 << (exp_bits - 1)) - 1
    min_exp  = 1 - bias
    max_exp  = int(math.log2(fp_max))   # floor(log2(fp_max)); valid for all OCP formats
    min_norm = 2.0 ** min_exp
    sub_step = min_norm / (2 ** man_bits)

    x    = x.float()
    sign = x.sign()
    xabs = x.abs().clamp(max=fp_max)

    # Subnormal path
    xsub = (xabs / sub_step).round() * sub_step

    # Normal path
    log2f   = xabs.clamp(min=1e-38).log2().floor().clamp(min_exp, max_exp)
    man_ulp = (2.0 ** log2f) / (2 ** man_bits)
    xnorm   = ((xabs / man_ulp).round() * man_ulp).clamp(max=fp_max)

    result = torch.where(xabs >= min_norm, xnorm, xsub)
    result = torch.where(xabs == 0, torch.zeros_like(result), result)
    return (sign * result).to(x.dtype)


def _e8m0_scale(max_abs: torch.Tensor, fp_max: float) -> torch.Tensor:
    """MX E8M0 block scale: smallest power-of-2 >= max_abs / fp_max."""
    ratio = (max_abs / fp_max).clamp(min=2.0 ** -127)
    return 2.0 ** ratio.log2().ceil().clamp(-127, 127)


# FP8-E4M3fn constants (OCP / NVIDIA convention, nan_type="max_man")
#   fp_max = 448  (E=1111,M=111 is the single NaN; E=1111,M=110 = 448 is valid)
_FP8_E4M3FN_MAX      = _fp_max_val(4, 3, nan_type="max_man")  # 448.0
_FP8_E4M3FN_MIN_NORM = 2.0 ** (1 - 7)                         # 2^-6


def _fp8_e4m3fn_scale(max_abs: torch.Tensor, fp4_max: float,
                      x_amax: float) -> torch.Tensor:
    """NVFP4 block scale with tensor-level normalization (fouroversix spec).

    encode_scale = (fp4_max * fp8_max) / x_amax
    x_scales_hp  = (block_amax / fp4_max) * encode_scale
               = block_amax * fp8_max / x_amax   ← always in [0, fp8_max]
    x_scales     = E4M3fn(x_scales_hp)

    Ref: fouroversix quantize_to_nvfp4, ScaleType.nv → torch.float8_e4m3fn (fp_max=448).
    """
    x_scales_hp = (max_abs * _FP8_E4M3FN_MAX / x_amax).clamp(
        min=_FP8_E4M3FN_MIN_NORM, max=_FP8_E4M3FN_MAX
    )
    return _fp_quant(x_scales_hp, 4, 3, fp_max=_FP8_E4M3FN_MAX)


def mx_block_quant(x: torch.Tensor, fmt: dict, block_size: int, scale_type: str) -> torch.Tensor:
    """Block fake-quantize along the last dimension of x.

    fmt: element-format descriptor, e.g. {"exp_bits":2, "man_bits":1, "nan_type":"none"}
    """
    fp_max     = _fp_max_val(fmt["exp_bits"], fmt["man_bits"], fmt["nan_type"])
    orig_shape = x.shape
    K    = orig_shape[-1]
    x_2d = x.float().reshape(-1, K)
    T    = x_2d.shape[0]

    pad = (-K) % block_size
    if pad:
        x_2d = F.pad(x_2d, (0, pad))

    n_blocks = x_2d.shape[1] // block_size
    x_blocks = x_2d.reshape(T, n_blocks, block_size)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-38)
    if scale_type == "e8m0":
        scale = _e8m0_scale(max_abs, fp_max)
    else:  # "fp8_e4m3fn"  (NVFP4)
        x_amax       = x_2d.abs().max().clamp(min=1e-38).item()
        decode_scale = x_amax / (fp_max * _FP8_E4M3FN_MAX)   # x_amax / (6 * 448)
        x_scales     = _fp8_e4m3fn_scale(max_abs, fp_max, x_amax)
        # dequantize with decode_scale to recover absolute scale
        x_q  = _fp_quant(x_blocks / (decode_scale * x_scales),
                         fmt["exp_bits"], fmt["man_bits"], fp_max)
        x_dq = (x_q * decode_scale * x_scales).reshape(T, -1)[:, :K]
        return x_dq.reshape(orig_shape[:-1] + (K,))

    x_q  = _fp_quant(x_blocks / scale, fmt["exp_bits"], fmt["man_bits"], fp_max)
    x_dq = (x_q * scale).reshape(T, -1)[:, :K]
    return x_dq.reshape(orig_shape[:-1] + (K,))


# ===========================================================================
# INT fake-quant primitives
# ===========================================================================

def _sym_fake_quant(x, bits, scale):
    qmax = 2 ** (bits - 1) - 1
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale

def fq_weight_per_channel(w, bits):
    scale = w.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
    return _sym_fake_quant(w, bits, scale.clamp(min=1e-8))

def fq_act_per_token(x2d, bits):
    scale = x2d.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
    return _sym_fake_quant(x2d, bits, scale.clamp(min=1e-8))


def _outlier_mask(x2d, outlier_ratio):
    n_outlier = max(1, int(x2d.shape[1] * outlier_ratio))
    idx = x2d.abs().amax(dim=0).topk(n_outlier).indices
    mask = torch.zeros(x2d.shape[1], dtype=torch.bool, device=x2d.device)
    mask[idx] = True
    return mask


# ===========================================================================
# Format registry
# ===========================================================================
# FP element-format descriptors  — nan_type controls which bit patterns are NaN
_E4M3 = {"exp_bits": 4, "man_bits": 3, "nan_type": "max_man"}  # fp_max=448   (OCP MXFP8 W, FP8-E4M3fn)
_E5M2 = {"exp_bits": 5, "man_bits": 2, "nan_type": "max_exp"}  # fp_max=57344 (OCP MXFP8 A)
_E2M3 = {"exp_bits": 2, "man_bits": 3, "nan_type": "none"}     # fp_max=7.5   (OCP MXFP6 W)
_E3M2 = {"exp_bits": 3, "man_bits": 2, "nan_type": "none"}     # fp_max=28.0  (OCP MXFP6 A)
_E2M1 = {"exp_bits": 2, "man_bits": 1, "nan_type": "none"}     # fp_max=6.0   (OCP MXFP4/NVFP4)

# Format dict — add new schemes here (no other changes needed)
_FORMATS: dict[str, dict] = {
    "int_w8a8": {
        "kind":   "int",
        "w_bits": 8, "a_bits": 8,
    },
    "int_w4a8": {
        "kind":   "int",
        "w_bits": 4, "a_bits": 8,
    },
    "int_w4a4": {
        "kind":   "int",
        "w_bits": 4, "a_bits": 4,
    },
    "int_w4a4_op1": {
        "kind":   "int",
        "w_bits": 4, "a_bits": 4, "outlier_ratio": 0.01,
    },
    "mxfp8": {
        "kind":       "mx",
        "w_fmt":      _E4M3, "a_fmt": _E5M2,
        "block_size": 32, "scale_type": "e8m0",
    },
    "mxfp6": {
        "kind":       "mx",
        "w_fmt":      _E2M3, "a_fmt": _E3M2,
        "block_size": 32, "scale_type": "e8m0",
    },
    "mxfp4": {
        "kind":       "mx",
        "w_fmt":      _E2M1, "a_fmt": _E2M1,
        "block_size": 32, "scale_type": "e8m0",
    },
    "nvfp4": {
        "kind":       "mx",
        "w_fmt":      _E2M1, "a_fmt": _E2M1,
        "block_size": 16, "scale_type": "fp8_e4m3fn",
    },
}


# ===========================================================================
# Hook factories
# ===========================================================================

def _make_hook_int(w_bits, a_bits, quantize_output, outlier_ratio=None):
    def hook(module, inp, output):
        x         = inp[0]
        orig      = x.shape
        dtype     = x.dtype
        x2d       = x.float().reshape(-1, x.shape[-1])
        w         = module.weight.float()
        if outlier_ratio is not None:
            omask     = _outlier_mask(x2d, outlier_ratio)
            x_normal  = fq_act_per_token(x2d[:, ~omask], a_bits)
            x_outlier = x2d[:, omask]
            w_normal  = fq_weight_per_channel(w[:, ~omask], w_bits) if w_bits else w[:, ~omask]
            w_outlier = w[:, omask]
            y = x_normal @ w_normal.T + x_outlier @ w_outlier.T
        else:
            x_q = fq_act_per_token(x2d, a_bits)
            w_q = fq_weight_per_channel(w, w_bits)
            y   = x_q @ w_q.T
        if module.bias is not None:
            y = y + module.bias.float()
        result = y.to(dtype).reshape(orig[:-1] + (w.shape[0],))
        if quantize_output:
            r2d    = result.float().reshape(-1, result.shape[-1])
            result = fq_act_per_token(r2d, a_bits).to(dtype).reshape(result.shape)
        return result
    return hook


def _make_hook_mx(w_fmt, a_fmt, block_size, scale_type, quantize_output):
    def hook(module, inp, output):
        x     = inp[0]
        orig  = x.shape
        dtype = x.dtype
        x2d   = x.reshape(-1, x.shape[-1])
        w     = module.weight
        w_q   = mx_block_quant(w.float(),   w_fmt, block_size, scale_type).to(dtype)
        x_q   = mx_block_quant(x2d.float(), a_fmt, block_size, scale_type).to(dtype)
        y     = x_q @ w_q.T
        if module.bias is not None:
            y = y + module.bias
        result = y.reshape(orig[:-1] + (w.shape[0],))
        if quantize_output:
            r2d    = result.reshape(-1, result.shape[-1])
            r_q    = mx_block_quant(r2d.float(), a_fmt, block_size, scale_type).to(dtype)
            result = r_q.reshape(result.shape)
        return result
    return hook


def attach_hooks(model, fmt_name, bmm=True, layer_filter=is_all):
    """Attach quantization hooks; returns list of hook handles."""
    fmt   = _FORMATS[fmt_name]
    kind  = fmt["kind"]
    hooks = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not layer_filter(name):
            continue
        qout = bmm and is_qkv_proj(name)
        if kind == "int":
            h = _make_hook_int(fmt["w_bits"], fmt["a_bits"], quantize_output=qout,
                               outlier_ratio=fmt.get("outlier_ratio"))
        elif kind == "mx":
            h = _make_hook_mx(fmt["w_fmt"], fmt["a_fmt"],
                              fmt["block_size"], fmt["scale_type"],
                              quantize_output=qout)
        else:
            raise ValueError(f"Unknown format kind: {kind!r}")
        hooks.append(mod.register_forward_hook(h))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ===========================================================================
# Init-state reproducibility
# ===========================================================================

def _install_init_state_controller(vec_env):
    """Pin episode assignments to init_states[i + batch*n_envs] for all sub-envs.

    External reset (seed != None, from rollout): uses controlled counter.
    Internal reset (seed=None, from step() or NEXT_STEP autoreset): ignored;
    the next external reset overwrites init_state_id anyway.

    Returns reset_fn() that rewinds all counters to episode 0, allowing the
    same vec_env to be reused across experiments.
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


# ===========================================================================
# Diagnostics helpers
# ===========================================================================

_logged_hook_formats: set[str] = set()


def _save_hooked_layers(policy: nn.Module, fmt_name: str,
                        layer_filter, out_dir: str) -> None:
    """Save list of hooked layer names to out_dir/{fmt_name}.txt (once per format)."""
    if fmt_name in _logged_hook_formats:
        return
    _logged_hook_formats.add(fmt_name)
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n, m in policy.named_modules()
             if isinstance(m, nn.Linear) and layer_filter(n)]
    path = os.path.join(out_dir, f"{fmt_name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(names))
    logging.info(f"Hooked layers → {path}  ({len(names)} layers)")


# ===========================================================================
# Suites & Experiments
# ===========================================================================

_ALL_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    # "libero_90",
]

# Episode length = longest training demo * 1.2 (per-suite margin)
_SUITE_EPISODE_LENGTH = {
    "libero_spatial": 386,  # 193 * 2
    "libero_object":  508,  # 254 * 2
    "libero_goal":    540,  # 270 * 2
    "libero_10":      989,  # 505 * 2 — capped at robosuite default horizon (1000)
    "libero_90":      746,  # 373 * 2
}

_ALL_EXPERIMENTS = [
    {"name": "baseline",      "fmt": None,       "layer_filter": None,   "bmm": False},
    {"name": "int_w8a8_bmm",  "fmt": "int_w8a8", "layer_filter": is_all, "bmm": True},
    {"name": "mxfp4_bmm",     "fmt": "mxfp4",    "layer_filter": is_all, "bmm": True},
    {"name": "nvfp4_bmm",     "fmt": "nvfp4",    "layer_filter": is_all, "bmm": True},
    # {"name": "int_w4a8",      "fmt": "int_w4a8", "layer_filter": is_all, "bmm": False},
    {"name": "int_w4a8_bmm",  "fmt": "int_w4a8", "layer_filter": is_all, "bmm": True},
    # {"name": "int_w4a4",      "fmt": "int_w4a4", "layer_filter": is_all, "bmm": False},
    {"name": "int_w4a4_bmm",     "fmt": "int_w4a4",     "layer_filter": is_all, "bmm": True},
    {"name": "int_w4a4_op1_bmm", "fmt": "int_w4a4_op1", "layer_filter": is_all, "bmm": True},
]


def _pop_arg(name: str) -> str | None:
    """Strip --name=val or --name val from sys.argv; return val or None."""
    for i, a in enumerate(sys.argv):
        if a.startswith(f"--{name}="):
            sys.argv.pop(i)
            return a[len(f"--{name}="):]
        if a == f"--{name}" and i + 1 < len(sys.argv):
            sys.argv.pop(i)
            return sys.argv.pop(i)
    return None


_suites_val = _pop_arg("suites") or "all"
SUITES = _ALL_SUITES if _suites_val == "all" else [s.strip() for s in _suites_val.split(",")]

_exps_val = _pop_arg("exps")
EXPERIMENTS = (
    [e for e in _ALL_EXPERIMENTS if e["name"] in {s.strip() for s in _exps_val.split(",")}]
    if _exps_val is not None else _ALL_EXPERIMENTS
)


# ===========================================================================
# Main
# ===========================================================================

OUT_FILE = "e2e_quant_results.json"


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    init_logging()

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    base_out_dir = os.environ.get("E2E_OUT_DIR", "duhyeon/thanos/results")

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    # CUDA warmup: pre-allocate float32 weight copies to prevent OOM on first MX experiment.
    # MX hooks call module.weight.float() on every forward pass; cold-start allocator spikes
    # can OOM even when steady-state memory fits. Warming up forces the allocator to reserve
    # a sufficient pool which is then reused for all subsequent experiments.
    if device.type == "cuda":
        logging.info("Warming up CUDA memory allocator (float32 weight copies)...")
        with torch.no_grad():
            _warmup = [m.weight.float() for m in policy.modules() if isinstance(m, nn.Linear)]
            del _warmup
            torch.cuda.synchronize()
        logging.info("CUDA warmup done.")

    preprocessor_overrides = {
        "device_processor":              {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    for suite in SUITES:
        logging.info(f"\n{'#'*60}")
        logging.info(f"[SUITE] {suite}")

        suite_dir    = os.path.join(base_out_dir, suite)
        os.makedirs(suite_dir, exist_ok=True)
        results_path = os.path.join(suite_dir, OUT_FILE)

        results, done = [], set()
        if os.path.exists(results_path):
            with open(results_path) as f:
                for r in json.load(f):
                    results.append(r)
                    done.add(r["name"])
            logging.info(f"Resuming: {len(done)} already done: {sorted(done)}")

        if done.issuperset(exp["name"] for exp in EXPERIMENTS):
            logging.info(f"[SKIP SUITE] all {len(done)} experiments already done")
            continue

        cfg.env.task = suite
        cfg.env.episode_length = _SUITE_EPISODE_LENGTH[suite]
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

        _suite_obj   = _libero_get_suite(suite)
        all_task_ids = list(range(len(_suite_obj.tasks)))
        del _suite_obj
        logging.info(f"Suite {suite}: {len(all_task_ids)} tasks.")

        # Create envs once per suite; reuse across all experiments (structural
        # guarantee that every scheme runs on identical env instances).
        envs = make_env(cfg.env, n_envs=cfg.eval.batch_size,
                        use_async_envs=cfg.eval.use_async_envs,
                        trust_remote_code=cfg.trust_remote_code)
        task_reset_fns = {
            tid: _install_init_state_controller(envs[suite][tid])
            for tid in all_task_ids
        }

        for exp in EXPERIMENTS:
            name = exp["name"]
            if name in done:
                logging.info(f"[SKIP] {name}")
                continue

            # Attach hooks
            hooks = []
            if exp["fmt"] is not None:
                n_layers = sum(
                    1 for n, m in policy.named_modules()
                    if isinstance(m, nn.Linear) and exp["layer_filter"](n)
                )
                logging.info(f"\n{'='*60}")
                logging.info(f"[EXP] {name}  fmt={exp['fmt']}  bmm={exp['bmm']}"
                             f"  filter={exp['layer_filter'].__name__}  ({n_layers} layers)")
                hooks = attach_hooks(policy, exp["fmt"],
                                     bmm=exp["bmm"],
                                     layer_filter=exp["layer_filter"])
                _save_hooked_layers(policy, exp["fmt"], exp["layer_filter"],
                                    os.path.join(base_out_dir, "hooked_layers"))
            else:
                logging.info(f"\n{'='*60}")
                logging.info(f"[EXP] {name}  (baseline — no quantization)")

            per_task_infos, all_successes = [], []
            for reset_fn in task_reset_fns.values():
                reset_fn()
            for tid in all_task_ids:
                set_seed(cfg.seed)
                with torch.no_grad(), (
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp
                    else nullcontext()
                ):
                    _, _, metrics = run_one(
                        suite, tid, envs[suite][tid],
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        max_episodes_rendered=0,
                        videos_dir=None,
                        return_episode_data=False,
                        start_seed=cfg.seed,
                    )
                per_task_infos.append({"task_group": suite, "task_id": tid, "metrics": metrics})
                all_successes.extend(metrics["successes"])

            remove_hooks(hooks)

            overall = float(np.nanmean(all_successes)) * 100
            per_task = [
                {
                    "task_group": t["task_group"],
                    "task_id":    t["task_id"],
                    "pc_success": float(np.nanmean(t["metrics"]["successes"]) * 100),
                }
                for t in per_task_infos
            ]

            record = {
                "name":               name,
                "fmt":                exp["fmt"],
                "bmm":                exp["bmm"],
                "n_quantized_layers": (
                    sum(1 for n, m in policy.named_modules()
                        if isinstance(m, nn.Linear) and exp["layer_filter"](n))
                    if exp["fmt"] is not None else 0
                ),
                "overall_pc_success": overall,
                "per_task":           per_task,
            }
            results.append(record)
            done.add(name)

            on_disk = []
            if os.path.exists(results_path):
                with open(results_path) as f:
                    try: on_disk = json.load(f)
                    except json.JSONDecodeError: pass
            disk_names = {r["name"] for r in on_disk}
            merged = on_disk + [r for r in results if r["name"] not in disk_names]
            with open(results_path, "w") as f:
                json.dump(merged, f, indent=2)

            baseline = next((r for r in results if r["name"] == "baseline"), None)
            base_sr  = baseline["overall_pc_success"] if baseline else float("nan")
            delta    = overall - base_sr
            logging.info(f"      success={overall:.1f}%  "
                         f"(delta {'+' if delta >= 0 else ''}{delta:.1f}%)")

        close_envs(envs)
        logging.info(f"\n[SUITE DONE] {suite} — results at {results_path}")
        _print_summary(results)


def _print_summary(results: list):
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    base_sr  = baseline["overall_pc_success"] if baseline else float("nan")
    print(f"\nBaseline: {base_sr:.1f}%")
    hdr = f"  {'Experiment':<22} {'Success%':>9} {'Delta':>8}  {'Format':>10} {'BMM':>5} {'#Layers':>7}"
    print(hdr)
    print("  " + "-" * 65)
    for r in results:
        if r["name"] == "baseline":
            continue
        d = r["overall_pc_success"] - base_sr
        print(f"  {r['name']:<22} {r['overall_pc_success']:>8.1f}%"
              f"  {'+' if d >= 0 else ''}{d:>5.1f}%"
              f"  {str(r['fmt']):>10} {str(r['bmm']):>5} {r['n_quantized_layers']:>6}")
    print()


if __name__ == "__main__":
    main()
