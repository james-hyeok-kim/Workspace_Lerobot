"""
LM vs DiT Quantization Sweep for LeRobot pi05.

RPCA 없이 균일 양자화(INT8/INT4/INT3/NVFP4)를 LM / DiT / 전체에 적용하고
libero_spatial / libero_10에서 success rate 비교.

Schemes  : fp32, int8wa, int4wa, int3wa, nvfp4wa,
           int3wa_bw  (INT3 block-wise, block=16),
           int4wa_svd (truncated SVD rank=32 + INT4 block-wise, no residual)
Targets  : lm_only, dit_only, all
Part B   : action step sweep (num_inference_steps = 1,3,5,10) on fp32
Part C   : action step sweep on int8wa (all)

Usage (TEST MODE):
    python eval_quant_sweep.py --task libero_spatial --n_episodes 2 \\
        --schemes fp32 int8wa --targets all --output_dir logs/quant_test

Usage (FULL SWEEP):
    bash run_quant_sweep.sh
"""

import copy
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _p in [str(_root / "src"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ModelOpt (NVFP4) ──────────────────────────────────────────────────────────
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    _MODELOPT_AVAILABLE = True
    print("[OK] NVIDIA ModelOpt loaded")
except ImportError as e:
    _MODELOPT_AVAILABLE = False
    print(f"[WARN] modelopt not available ({e}); nvfp4wa scheme disabled")

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# 양자화 함수 (균일, RPCA 없음)
# ══════════════════════════════════════════════════════════════════════════════

def quant_int_perchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric INT-N. INT8, INT3에 사용."""
    n_pos = 2 ** (bits - 1) - 1  # e.g. 127 for INT8, 3 for INT3
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / n_pos
    W_q = (W / scale).round().clamp(-n_pos - 1, n_pos)
    return (W_q * scale).to(W.dtype)


def quant_int_blockwise(W: torch.Tensor, bits: int, block_size: int = 16) -> torch.Tensor:
    """Block-wise symmetric INT-N. INT4에 사용."""
    out_f, in_f = W.shape
    n_pos = 2 ** (bits - 1) - 1
    pad = (-in_f) % block_size
    W_p = F.pad(W, (0, pad)) if pad else W
    W_b = W_p.reshape(out_f, -1, block_size)
    scale = W_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / n_pos
    W_q = (W_b / scale).round().clamp(-n_pos - 1, n_pos)
    return ((W_q * scale).reshape(out_f, -1)[:, :in_f]).to(W.dtype)


def quant_act_dynamic(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Dynamic per-tensor symmetric activation quantization."""
    n_pos = 2 ** (bits - 1) - 1
    scale = x.abs().amax().clamp(min=1e-8) / n_pos
    return ((x / scale).round().clamp(-n_pos - 1, n_pos) * scale).to(x.dtype)


def svd_then_quantize(
    W: torch.Tensor,
    bits: int = 4,
    rank: int = 32,
    block_size: int = 16,
) -> torch.Tensor:
    """
    Truncated SVD → INT-N blockwise quantization. 잔차(S) 없이 재구성값만 양자화.

    W ≈ U_k · diag(S_k) · V_k^T  → Q(W_lr) [INT-N blockwise]

    큰 행렬은 torch.svd_lowrank(randomized)로 처리.
    """
    W_f = W.float()
    m, n = W_f.shape
    k = min(rank, min(m, n))
    try:
        # 랜덤화 SVD (대형 행렬 효율적)
        U, S, V = torch.svd_lowrank(W_f, q=min(k + 4, min(m, n)), niter=4)
        W_lr = (U[:, :k] * S[:k].unsqueeze(0)) @ V[:, :k].T
    except Exception:
        # Fallback: full SVD
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        W_lr = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
    return quant_int_blockwise(W_lr, bits, block_size).to(W.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# QuantizedLinear — RPCA 없는 균일 양자화
# ══════════════════════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """nn.Linear 교체용. weight를 양자화 후 dequant 값으로 저장, activation 선택적 양자화."""

    def __init__(
        self,
        W_dequant: torch.Tensor,
        bias: torch.Tensor | None,
        act_bits: int | None = None,
    ):
        super().__init__()
        self.register_buffer("W_dequant", W_dequant.half())
        self.bias = nn.Parameter(bias.half(), requires_grad=False) if bias is not None else None
        self.act_bits = act_bits

    @property
    def weight(self) -> torch.Tensor:
        return self.W_dequant

    @property
    def out_features(self) -> int:
        return self.W_dequant.shape[0]

    @property
    def in_features(self) -> int:
        return self.W_dequant.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_dequant
        x_in = x.to(W.dtype)
        if self.act_bits is not None:
            x_in = quant_act_dynamic(x_in, self.act_bits)
        out = F.linear(x_in, W, self.bias)
        return out.to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 컴포넌트 분류
# ══════════════════════════════════════════════════════════════════════════════

# LM: PaliGemma 언어 모델 (Gemma 2B) — attention + MLP layers
# DiT: Action Expert (Gemma 300M) + action projection + time MLP
LM_PATTERNS = [
    "paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
]
DIT_PATTERNS = [
    "paligemma_with_expert.gemma_expert",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
]
# LM + DiT expert만 (Vision, Action Head 제외)
LM_DIT_PATTERNS = [
    "paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
    "paligemma_with_expert.gemma_expert",
]
# Vision tower는 fp32 고정이라 all에서도 skip
SKIP_PATTERNS = [
    "vision_tower",
    "embed_tokens",
]
# Action Head: lm_dit 타겟에서 제외
ACTION_HEAD_PATTERNS = [
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
]


def _is_target_layer(full_name: str, target: str) -> bool:
    """레이어 이름이 target 컴포넌트에 속하는지 판단."""
    # 항상 skip할 레이어
    for pat in SKIP_PATTERNS:
        if pat in full_name:
            return False

    if target == "lm":
        return any(pat in full_name for pat in LM_PATTERNS)
    elif target == "dit":
        return any(pat in full_name for pat in DIT_PATTERNS)
    elif target == "lm_dit":
        # Vision + Action Head 제외, LM + Expert만
        return any(pat in full_name for pat in LM_DIT_PATTERNS)
    elif target == "all":
        return True  # skip 이외 모두
    return False


# ══════════════════════════════════════════════════════════════════════════════
# INT quantization 적용
# ══════════════════════════════════════════════════════════════════════════════

def _replace_linear_recursive(
    module: nn.Module,
    bits: int,
    act_bits: int | None,
    target: str,
    report: list[dict],
    prefix: str = "",
    blockwise: bool = False,   # True → always block-wise (INT3_bw 등)
    svd_rank: int | None = None,  # None → 일반 quant, int → SVD+quant
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            if not _is_target_layer(full_name, target):
                continue

            out_f, in_f = child.weight.shape
            W = child.weight.data.float()
            bias = child.bias.data.float() if child.bias is not None else None

            if svd_rank is not None:
                # SVD → INT-N blockwise (잔차 없음)
                W_q = svd_then_quantize(W, bits=bits, rank=svd_rank, block_size=16)
            elif blockwise or bits == 4:
                # block-wise 양자화 (INT4 기본, INT3_bw 포함)
                W_q = quant_int_blockwise(W, bits, block_size=16)
            else:
                # per-channel 양자화 (INT8, INT3 기본)
                W_q = quant_int_perchannel(W, bits)

            new_layer = QuantizedLinear(W_q, bias, act_bits=act_bits)
            setattr(module, name, new_layer)
            mode = "svd_bw" if svd_rank else ("bw" if blockwise else "pc")
            report.append({"layer": full_name, "bits": bits, "mode": mode, "shape": [out_f, in_f]})
        else:
            _replace_linear_recursive(
                child, bits, act_bits, target, report, prefix=full_name,
                blockwise=blockwise, svd_rank=svd_rank,
            )


def apply_int_quant(
    policy: nn.Module,
    bits: int,
    target: str,
    weight_only: bool = False,
    blockwise: bool = False,
    svd_rank: int | None = None,
) -> list[dict]:
    """모든 nn.Linear (target 컴포넌트 내)를 QuantizedLinear로 교체."""
    act_bits = None if weight_only else bits
    report: list[dict] = []
    _replace_linear_recursive(
        policy, bits, act_bits, target, report, prefix="",
        blockwise=blockwise, svd_rank=svd_rank,
    )
    return report


# ══════════════════════════════════════════════════════════════════════════════
# NVFP4 quantization (MTQ)
# ══════════════════════════════════════════════════════════════════════════════

def apply_nvfp4_quant(policy: nn.Module, target: str) -> list[dict]:
    """MTQ NVFP4을 target 컴포넌트에 적용."""
    if not _MODELOPT_AVAILABLE:
        raise RuntimeError("modelopt not available; cannot apply nvfp4wa")

    report: list[dict] = []

    if target == "all":
        mtq.quantize(policy, config=NVFP4_DEFAULT_CFG)
        report.append({"target_module": "policy (all)", "scheme": "nvfp4"})

    elif target == "lm":
        # LM sub-module에만 MTQ 적용
        lm = policy.model.paligemma_with_expert.paligemma.model.language_model
        mtq.quantize(lm, config=NVFP4_DEFAULT_CFG)
        report.append({"target_module": "language_model", "scheme": "nvfp4"})

    elif target == "dit":
        # Action expert (Gemma 300M)
        expert = policy.model.paligemma_with_expert.gemma_expert
        mtq.quantize(expert, config=NVFP4_DEFAULT_CFG)
        report.append({"target_module": "gemma_expert", "scheme": "nvfp4"})

        # Action head (개별 Linear → INT4 blockwise fallback, MTQ는 sub-module 단위 필요)
        inner = policy.model
        for attr in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
            layer = getattr(inner, attr, None)
            if isinstance(layer, nn.Linear):
                out_f, in_f = layer.weight.shape
                W_q = quant_int_blockwise(layer.weight.data.float(), 4, 16)
                bias = layer.bias.data.float() if layer.bias is not None else None
                setattr(inner, attr, QuantizedLinear(W_q, bias, act_bits=4))
                report.append({"target_module": f"model.{attr}", "scheme": "int4_fallback"})

    elif target == "lm_dit":
        # LM + Action Expert 양자화 (Vision tower + Action Head 제외)
        lm = policy.model.paligemma_with_expert.paligemma.model.language_model
        mtq.quantize(lm, config=NVFP4_DEFAULT_CFG)
        report.append({"target_module": "language_model", "scheme": "nvfp4"})

        expert = policy.model.paligemma_with_expert.gemma_expert
        mtq.quantize(expert, config=NVFP4_DEFAULT_CFG)
        report.append({"target_module": "gemma_expert", "scheme": "nvfp4"})

    return report


# ══════════════════════════════════════════════════════════════════════════════
# torch.compile 비활성화
# ══════════════════════════════════════════════════════════════════════════════

def disable_torch_compile(policy: nn.Module) -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# 커버리지 리포트
# ══════════════════════════════════════════════════════════════════════════════

def build_quant_report(policy: nn.Module) -> dict:
    """모델 전체를 스캔해 양자화된/미양자화 Linear 레이어 수 집계."""
    total = 0
    quantized = 0
    for name, mod in policy.named_modules():
        if list(mod.children()):
            continue  # non-leaf skip
        if isinstance(mod, (nn.Linear, QuantizedLinear)):
            total += 1
            if isinstance(mod, QuantizedLinear):
                quantized += 1
        elif isinstance(mod, nn.Module):
            # MTQ QuantizedLinear 탐지
            mtype = type(mod).__name__
            if "Quantized" in mtype or "quant" in mtype.lower():
                if hasattr(mod, "weight_quantizer") or hasattr(mod, "input_quantizer"):
                    total += 1
                    quantized += 1
    return {"total_linear": total, "quantized": quantized, "unquantized": total - quantized}


# ══════════════════════════════════════════════════════════════════════════════
# 단일 실험 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    pretrained_path: str,
    task: str,
    n_episodes: int,
    batch_size: int,
    device_str: str,
    scheme: str,
    target: str,
    num_inference_steps: int | None,
    env_cfg,
    envs_dict: dict,
    use_amp: bool = False,
    **kwargs,
) -> dict:
    """
    단일 (scheme, target, num_inference_steps) 조합 평가.
    매번 새 policy를 로드해 양자화 상태가 독립적으로 유지됨.
    """
    print(f"\n[RUN] scheme={scheme}  target={target}  steps={num_inference_steps}")

    device = torch.device(device_str)

    # 정책 로드
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    policy_cfg.pretrained_path = pretrained_path
    policy_cfg.device = device_str
    policy_cfg.use_amp = use_amp

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    # 프로세서
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device_str}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # num_inference_steps 오버라이드
    if num_inference_steps is not None:
        policy.model.config.num_inference_steps = num_inference_steps

    quant_report: dict = {}

    svd_rank: int = kwargs.get("svd_rank", 32)

    # Quantization 적용
    if scheme == "fp32":
        pass  # 베이스라인: 양자화 없음
    elif scheme == "int8wa":
        layer_report = apply_int_quant(policy, bits=8, target=target, weight_only=False)
        quant_report["layer_report"] = layer_report
    elif scheme == "int4wa":
        layer_report = apply_int_quant(policy, bits=4, target=target, weight_only=False)
        quant_report["layer_report"] = layer_report
    elif scheme == "int3wa":
        # 기본 per-channel
        layer_report = apply_int_quant(policy, bits=3, target=target, weight_only=False)
        quant_report["layer_report"] = layer_report
    elif scheme == "int3wa_bw":
        # INT3 block-wise (block=16)
        layer_report = apply_int_quant(policy, bits=3, target=target,
                                        weight_only=False, blockwise=True)
        quant_report["layer_report"] = layer_report
    elif scheme == "int4wa_svd":
        # Truncated SVD (rank=svd_rank) + INT4 block-wise, 잔차 없음
        layer_report = apply_int_quant(policy, bits=4, target=target,
                                        weight_only=False, svd_rank=svd_rank)
        quant_report["layer_report"] = layer_report
        quant_report["svd_rank"] = svd_rank
    elif scheme == "nvfp4wa":
        layer_report = apply_nvfp4_quant(policy, target=target)
        quant_report["layer_report"] = layer_report
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # torch.compile 비활성화 (MTQ 또는 QuantizedLinear 사용 후 필요)
    if scheme != "fp32":
        disable_torch_compile(policy)

    # 커버리지
    coverage = build_quant_report(policy)
    quant_report["coverage"] = coverage
    print(f"[INFO] Quantized: {coverage['quantized']} / {coverage['total_linear']}")

    # 환경
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # 평가
    with torch.no_grad(), (
        torch.autocast(device_type=device.type) if use_amp else nullcontext()
    ):
        eval_info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
        )

    agg = eval_info.get("aggregated", {})
    pc_success = agg.get("pc_success", float("nan"))
    avg_reward = agg.get("avg_sum_reward", float("nan"))
    print(f"[RESULT] success={pc_success:.1f}%  reward={avg_reward:.4f}")

    # 메모리 정리
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "scheme": scheme,
        "target": target,
        "num_inference_steps": num_inference_steps,
        "quantization": quant_report,
        "eval_results": eval_info,
        "pc_success": pc_success,
        "avg_sum_reward": avg_reward,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Summary 출력 & 저장
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v):
    return f"{v:.1f}%" if v == v else "  N/A "


def print_summary_table(scheme_results: list[dict], step_results: list[dict]) -> None:
    W = 72
    sep = "─" * W
    print(f"\n{'='*W}")
    print(" [Part A] Scheme × Target 결과")
    print(sep)
    print(f"  {'Scheme':<12} {'Target':<12} {'Success%':>9} {'AvgReward':>11}")
    print(sep)
    for r in scheme_results:
        print(f"  {r['scheme']:<12} {r['target']:<12} "
              f"{_fmt(r['pc_success']):>9} {r['avg_sum_reward']:>11.4f}")
    print(sep)

    if step_results:
        print(f"\n{'='*W}")
        print(" [Part B/C] Action step (num_inference_steps) 영향")
        print(sep)
        print(f"  {'Scheme':<12} {'Steps':>6} {'Success%':>9} {'AvgReward':>11}")
        print(sep)
        for r in step_results:
            print(f"  {r['scheme']:<12} {r['num_inference_steps']:>6} "
                  f"{_fmt(r['pc_success']):>9} {r['avg_sum_reward']:>11.4f}")
        print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LM vs DiT quantization sweep for LeRobot pi05"
    )
    parser.add_argument("--pretrained_path", type=str,
                        default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task", type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1, help="n_envs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="logs/quant_sweep")
    parser.add_argument("--use_amp", action="store_true")

    # 스위프 범위 (기본: 전체 실험)
    parser.add_argument(
        "--schemes", type=str, nargs="+",
        default=["fp32", "int8wa", "nvfp4wa", "int4wa", "int3wa",
                 "int3wa_bw", "int4wa_svd"],
        help="평가할 양자화 스킴",
    )
    parser.add_argument(
        "--svd_rank", type=int, default=32,
        help="int4wa_svd 스킴의 truncated SVD rank (default: 32)",
    )
    parser.add_argument(
        "--targets", type=str, nargs="+",
        default=["lm_only", "dit_only", "all"],
        help="양자화 대상 컴포넌트. fp32는 target 무관 (all로 고정)",
    )
    # Action step sweep
    parser.add_argument(
        "--step_sweep", action="store_true",
        help="Part B: num_inference_steps sweep on fp32 + int8wa",
    )
    parser.add_argument(
        "--step_values", type=int, nargs="+",
        default=[1, 3, 5, 10],
        help="num_inference_steps 값 목록 (step_sweep 시 사용)",
    )
    args = parser.parse_args()

    # 검증
    VALID_SCHEMES = {"fp32", "int8wa", "int4wa", "int3wa", "nvfp4wa",
                     "int3wa_bw", "int4wa_svd"}
    VALID_TARGETS = {"lm_only", "dit_only", "all", "lm_dit"}
    for s in args.schemes:
        if s not in VALID_SCHEMES:
            print(f"[ERROR] 알 수 없는 scheme: {s}. 가능: {sorted(VALID_SCHEMES)}")
            sys.exit(1)
    for t in args.targets:
        if t not in VALID_TARGETS:
            print(f"[ERROR] 알 수 없는 target: {t}. 가능: {sorted(VALID_TARGETS)}")
            sys.exit(1)
    if "nvfp4wa" in args.schemes and not _MODELOPT_AVAILABLE:
        print("[ERROR] nvfp4wa 스킴은 modelopt 필요. PYTHONPATH 확인")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 환경 생성 (모든 실험에서 재사용)
    print(f"[INFO] Creating LIBERO env: task={args.task}, n_envs={args.batch_size}")
    env_cfg = LiberoEnv(task=args.task)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    print(f"[INFO] suite={suite_name}  task_id={task_id}")

    scheme_results: list[dict] = []
    step_results: list[dict] = []

    # ── Part A: Scheme × Target ────────────────────────────────────────────────
    # fp32는 target 무관 — 'all'로 한 번만 실행
    experiments: list[tuple[str, str]] = []
    for scheme in args.schemes:
        if scheme == "fp32":
            experiments.append(("fp32", "all"))
        else:
            for target in args.targets:
                experiments.append((scheme, target))

    # 중복 제거 (fp32 여러 번 지정 방지)
    seen = set()
    experiments_dedup = []
    for e in experiments:
        if e not in seen:
            seen.add(e)
            experiments_dedup.append(e)
    experiments = experiments_dedup

    print(f"\n[INFO] Part A: {len(experiments)} 실험 실행 예정")
    for scheme, target in experiments:
        exp_key = f"{scheme}_{target}"
        out_path = output_dir / f"{exp_key}.json"

        # Skip if exists
        if out_path.exists():
            print(f"[SKIP] {exp_key}: 이미 존재 → {out_path}")
            try:
                with open(out_path, encoding="utf-8") as f:
                    existing = json.load(f)
                r = {
                    "scheme": scheme, "target": target,
                    "num_inference_steps": existing.get("num_inference_steps"),
                    "pc_success": existing.get("pc_success", float("nan")),
                    "avg_sum_reward": existing.get("avg_sum_reward", float("nan")),
                }
                scheme_results.append(r)
            except Exception as e:
                print(f"[WARN] 기존 JSON 로드 실패 ({e}), 재실행")
            else:
                continue

        result = run_experiment(
            pretrained_path=args.pretrained_path,
            task=args.task,
            n_episodes=args.n_episodes,
            batch_size=args.batch_size,
            device_str=args.device,
            scheme=scheme,
            target=target,
            num_inference_steps=None,  # 기본값 사용
            env_cfg=env_cfg,
            envs_dict=envs_dict,
            use_amp=args.use_amp,
            svd_rank=args.svd_rank,
        )
        scheme_results.append({
            "scheme": scheme, "target": target,
            "num_inference_steps": result["num_inference_steps"],
            "pc_success": result["pc_success"],
            "avg_sum_reward": result["avg_sum_reward"],
        })

        # 개별 JSON 저장
        out_data = {
            "config": {
                "pretrained_path": args.pretrained_path,
                "task": args.task,
                "suite_name": suite_name,
                "task_id": task_id,
                "n_episodes": args.n_episodes,
                "batch_size": args.batch_size,
                "scheme": scheme,
                "target": target,
                "num_inference_steps": result["num_inference_steps"],
                "device": args.device,
            },
            "scheme": scheme,
            "target": target,
            "num_inference_steps": result["num_inference_steps"],
            "pc_success": result["pc_success"],
            "avg_sum_reward": result["avg_sum_reward"],
            "quantization": result["quantization"],
            "eval_results": result["eval_results"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"[SAVED] {out_path}")

    # ── Part B/C: Action step sweep ────────────────────────────────────────────
    if args.step_sweep:
        step_schemes = ["fp32"]
        if "int8wa" in args.schemes:
            step_schemes.append("int8wa")

        print(f"\n[INFO] Part B/C: step sweep {args.step_values} × {step_schemes}")
        for scheme in step_schemes:
            for steps in args.step_values:
                # fp32 baseline의 steps=10은 이미 Part A에서 실행됨 (skip 가능)
                exp_key = f"steps_{scheme}_step{steps}"
                out_path = output_dir / f"{exp_key}.json"

                if out_path.exists():
                    print(f"[SKIP] {exp_key}: 이미 존재")
                    try:
                        with open(out_path, encoding="utf-8") as f:
                            existing = json.load(f)
                        step_results.append({
                            "scheme": scheme,
                            "num_inference_steps": steps,
                            "pc_success": existing.get("pc_success", float("nan")),
                            "avg_sum_reward": existing.get("avg_sum_reward", float("nan")),
                        })
                    except Exception:
                        pass
                    else:
                        continue

                target = "all"
                result = run_experiment(
                    pretrained_path=args.pretrained_path,
                    task=args.task,
                    n_episodes=args.n_episodes,
                    batch_size=args.batch_size,
                    device_str=args.device,
                    scheme=scheme,
                    target=target,
                    num_inference_steps=steps,
                    env_cfg=env_cfg,
                    envs_dict=envs_dict,
                    use_amp=args.use_amp,
                    svd_rank=args.svd_rank,
                )
                step_results.append({
                    "scheme": scheme,
                    "num_inference_steps": steps,
                    "pc_success": result["pc_success"],
                    "avg_sum_reward": result["avg_sum_reward"],
                })

                out_data = {
                    "config": {
                        "pretrained_path": args.pretrained_path,
                        "task": args.task,
                        "scheme": scheme,
                        "target": target,
                        "num_inference_steps": steps,
                        "n_episodes": args.n_episodes,
                        "device": args.device,
                    },
                    "scheme": scheme,
                    "target": target,
                    "num_inference_steps": steps,
                    "pc_success": result["pc_success"],
                    "avg_sum_reward": result["avg_sum_reward"],
                    "quantization": result["quantization"],
                    "eval_results": result["eval_results"],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_data, f, indent=2, default=str)
                print(f"[SAVED] {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary_table(scheme_results, step_results)

    # Ranking (success rate 기준)
    all_results = scheme_results + [
        {**r, "target": "all"} for r in step_results
    ]
    ranking = sorted(
        all_results,
        key=lambda x: (float("nan") if x["pc_success"] != x["pc_success"] else -x["pc_success"]),
    )

    summary = {
        "config": {
            "pretrained_path": args.pretrained_path,
            "task": args.task,
            "n_episodes": args.n_episodes,
            "batch_size": args.batch_size,
            "schemes": args.schemes,
            "targets": args.targets,
            "step_sweep": args.step_sweep,
            "step_values": args.step_values if args.step_sweep else [],
        },
        "scheme_target_results": scheme_results,
        "action_step_results": step_results,
        "ranking": ranking,
    }
    summary_path = output_dir / "quant_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SUMMARY] 저장: {summary_path}")


if __name__ == "__main__":
    main()
