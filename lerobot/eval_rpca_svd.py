"""
RPCA + Quantization Evaluation for LeRobot pi05 policy.

각 Linear 레이어의 가중치 W를 RPCA로 W = L + S 분해(L: 저랭크, S: 희소 아웃라이어)한 뒤,
L을 목표 bit width로 양자화하고 S는 fp16 잔차로 유지합니다.

지원 양자화 스킴:
  int8_w     - INT8 weight-only (per-channel symmetric)
  int8_wa    - INT8 weight + activation
  int4_w     - INT4 weight-only (block-wise, block=16)
  int4_wa    - INT4 weight + activation
  int2_w     - INT2 weight-only (per-channel symmetric)
  int2_wa    - INT2 weight + activation
  ternary_w  - Ternary weight-only (TWN: {-α, 0, +α})
  nvfp4_wa   - NVFP4 weight + activation via MTQ (+ RPCA 잔차 보정)

Usage (TEST MODE):
    python eval_rpca_svd.py --schemes int8_w nvfp4_wa --n_episodes 1
"""

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent
for _p in [str(_root / "src"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
        print(f"[PATH] {_p}")

# ── ModelOpt (NVFP4 scheme에서만 사용) ───────────────────────────────────────
try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    _MODELOPT_AVAILABLE = True
    print("[OK] NVIDIA ModelOpt loaded")
except ImportError as e:
    _MODELOPT_AVAILABLE = False
    print(f"[WARN] modelopt not available ({e}); nvfp4_wa scheme disabled")

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy


# ══════════════════════════════════════════════════════════════════════════════
# RPCA 분해 (Inexact ALM with randomized truncated SVD)
# ══════════════════════════════════════════════════════════════════════════════

def rpca_ialm(
    M: torch.Tensor,
    lam: float | None = None,
    mu_init: float | None = None,
    max_iter: int = 15,
    tol: float = 1e-4,
    max_rank: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust PCA via Inexact Augmented Lagrangian Method (IALM).
    M = L + S,  L: 저랭크,  S: 희소

    큰 행렬에서는 torch.svd_lowrank(randomized)를 사용해 속도 확보.
    """
    M_f = M.float()
    m, n = M_f.shape

    if lam is None:
        lam = 1.0 / (max(m, n) ** 0.5)
    if mu_init is None:
        frobenius = M_f.norm(p="fro")
        mu_init = (m * n) / (4.0 * M_f.abs().sum().clamp(min=1e-8))

    mu = mu_init
    rho = 1.5
    mu_max = mu * 1e6

    S = torch.zeros_like(M_f)
    Y = torch.zeros_like(M_f)
    M_norm = M_f.norm(p="fro").clamp(min=1e-8)

    for _ in range(max_iter):
        # ── L 업데이트: singular value thresholding (randomized SVD) ──────────
        T = M_f - S + Y / mu
        rank_q = min(max_rank + 4, min(m, n))  # slight oversampling
        try:
            U, sigma, V = torch.svd_lowrank(T, q=rank_q, niter=2)
        except Exception:
            break

        # Soft threshold on singular values
        sv_thresh = (sigma - 1.0 / mu).clamp(min=0.0)
        k = max(1, int((sv_thresh > 0).sum().item()))
        k = min(k, max_rank)
        L = (U[:, :k] * sv_thresh[:k].unsqueeze(0)) @ V[:, :k].T

        # ── S 업데이트: element-wise soft thresholding ────────────────────────
        T_s = M_f - L + Y / mu
        S = T_s.sign() * (T_s.abs() - lam / mu).clamp(min=0.0)

        # ── Dual variable 업데이트 ─────────────────────────────────────────────
        residual = M_f - L - S
        Y = Y + mu * residual

        if residual.norm(p="fro") / M_norm < tol:
            break

        mu = min(rho * mu, mu_max)

    return L.to(M.dtype), S.to(M.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 양자화 함수들 (모두 dequantized 텐서 반환)
# ══════════════════════════════════════════════════════════════════════════════

def quant_int_perchannel(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric INT-N. (INT8, INT2에 적합)"""
    n_pos = 2 ** (bits - 1) - 1  # e.g. 127 for INT8
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / n_pos
    W_q = (W / scale).round().clamp(-n_pos - 1, n_pos)
    return (W_q * scale).to(W.dtype)


def quant_int_blockwise(W: torch.Tensor, bits: int, block_size: int = 16) -> torch.Tensor:
    """Block-wise symmetric INT-N. (INT4, NVFP4 근사에 적합; block=16)"""
    out_f, in_f = W.shape
    n_pos = 2 ** (bits - 1) - 1
    pad = (-in_f) % block_size
    W_p = F.pad(W, (0, pad)) if pad else W
    W_b = W_p.reshape(out_f, -1, block_size)
    scale = W_b.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / n_pos
    W_q = (W_b / scale).round().clamp(-n_pos - 1, n_pos)
    return ((W_q * scale).reshape(out_f, -1)[:, :in_f]).to(W.dtype)


def quant_ternary(W: torch.Tensor) -> torch.Tensor:
    """Ternary Weight Networks (TWN): values in {-α, 0, +α} per row."""
    thresh = 0.7 * W.abs().mean(dim=1, keepdim=True)
    pos = W > thresh
    neg = W < -thresh
    W_t = torch.zeros_like(W)
    W_t[pos] = 1.0
    W_t[neg] = -1.0
    nonzero = (pos | neg).float()
    n = nonzero.sum(dim=1, keepdim=True).clamp(min=1.0)
    alpha = (W.abs() * nonzero).sum(dim=1, keepdim=True) / n
    return (W_t * alpha).to(W.dtype)


def quant_act_dynamic(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Dynamic per-tensor symmetric activation quantization."""
    n_pos = 2 ** (bits - 1) - 1
    scale = x.abs().amax().clamp(min=1e-8) / n_pos
    return ((x / scale).round().clamp(-n_pos - 1, n_pos) * scale).to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 모듈: RPCAQuantizedLinear  (INT/Ternary 스킴)
# ══════════════════════════════════════════════════════════════════════════════

class RPCAQuantizedLinear(nn.Module):
    """RPCA + manual quantization linear layer.

    forward(x) = linear(x, Q(L) + S, bias)
    - Q(L): 양자화된 저랭크 성분 (fp16으로 저장)
    - S    : 희소 아웃라이어 잔차 (fp16으로 저장)
    - act_bits: None이면 weight-only, int면 activation도 양자화
    """

    def __init__(
        self,
        L_dequant: torch.Tensor,
        S: torch.Tensor,
        bias: torch.Tensor | None,
        act_bits: int | None = None,
    ):
        super().__init__()
        self.register_buffer("L_dequant", L_dequant.half())
        self.register_buffer("S", S.half())
        self.bias = nn.Parameter(bias.half(), requires_grad=False) if bias is not None else None
        self.act_bits = act_bits

    @property
    def weight(self) -> torch.Tensor:
        """호환성: 원본 nn.Linear처럼 .weight 접근 허용."""
        return self.L_dequant + self.S

    @property
    def out_features(self) -> int:
        return self.L_dequant.shape[0]

    @property
    def in_features(self) -> int:
        return self.L_dequant.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.L_dequant + self.S
        x_in = x.to(W_eff.dtype)
        if self.act_bits is not None:
            x_in = quant_act_dynamic(x_in, self.act_bits)
        out = F.linear(x_in, W_eff, self.bias)
        return out.to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 모듈: RPCANvfp4Wrapper  (NVFP4 스킴 - MTQ가 내부 nn.Linear를 양자화)
# ══════════════════════════════════════════════════════════════════════════════

class RPCANvfp4Wrapper(nn.Module):
    """RPCA + NVFP4 wrapper.

    내부 self.linear (nn.Linear, weight=L)은 MTQ가 NVFP4로 양자화.
    self.S (fp16)는 잔차로 forward에서 더해짐.

    forward(x) = linear_nvfp4(x) + F.linear(x, S)
    """

    def __init__(
        self,
        L: torch.Tensor,
        S: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        out_f, in_f = L.shape
        # L이 있는 device/dtype으로 직접 생성 (CPU/CUDA 불일치 방지)
        self.linear = nn.Linear(in_f, out_f, bias=bias is not None,
                                device=L.device, dtype=L.dtype)
        self.linear.weight.data.copy_(L)
        if bias is not None:
            self.linear.bias.data.copy_(bias)
        self.register_buffer("S", S.half())

    @property
    def weight(self) -> torch.Tensor:
        """호환성: 원본 nn.Linear처럼 .weight 접근 허용 (MTQ 이후엔 QuantizedLinear.weight)."""
        return self.linear.weight

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.linear(x)
        s_out = F.linear(x.to(self.S.dtype), self.S).to(x.dtype)
        return main_out + s_out


# ══════════════════════════════════════════════════════════════════════════════
# 양자화 스킴 레지스트리
# ══════════════════════════════════════════════════════════════════════════════

# fmt: (bits, weight_only, quant_format)
#   bits        - INT bit width (None for ternary/nvfp4)
#   weight_only - True: W만,  False: W+A 모두
#   quant_fmt   - "int_pc" | "int_bw" | "ternary" | "nvfp4_mtq"
QUANT_SCHEMES: dict[str, tuple] = {
    "int8_w":    (8,    True,  "int_pc"),
    "int8_wa":   (8,    False, "int_pc"),
    "int4_w":    (4,    True,  "int_bw"),
    "int4_wa":   (4,    False, "int_bw"),
    "int2_w":    (2,    True,  "int_pc"),
    "int2_wa":   (2,    False, "int_pc"),
    "ternary_w": (None, True,  "ternary"),
    "nvfp4_wa":  (None, False, "nvfp4_mtq"),
}

# TEST MODE 기본 스킴 (빠른 검증용)
TEST_SCHEMES = ["int8_w", "nvfp4_wa"]
# FULL MODE 전체 스킴
FULL_SCHEMES = list(QUANT_SCHEMES.keys())


# ══════════════════════════════════════════════════════════════════════════════
# RPCA 양자화 적용
# ══════════════════════════════════════════════════════════════════════════════

def _replace_recursive(
    module: nn.Module,
    scheme: str,
    rpca_rank: int,
    rpca_lam_scale: float,
    max_rpca_dim: int,
    report: list[dict],
    prefix: str = "",
) -> None:
    """모델을 재귀 탐색하며 nn.Linear를 RPCA 양자화 모듈로 교체."""
    bits, weight_only, fmt = QUANT_SCHEMES[scheme]
    act_bits = None if weight_only else bits

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            out_f, in_f = child.weight.shape
            W = child.weight.data.float()
            bias = child.bias.data.float() if child.bias is not None else None

            # 너무 큰 레이어는 RPCA 건너뜀 (직접 양자화)
            use_rpca = max(out_f, in_f) <= max_rpca_dim
            if use_rpca:
                lam = rpca_lam_scale / (max(out_f, in_f) ** 0.5)
                L, S = rpca_ialm(W, lam=lam, max_rank=rpca_rank)
            else:
                L, S = W, torch.zeros_like(W)

            # 양자화
            if fmt == "int_pc":
                L_q = quant_int_perchannel(L, bits)
            elif fmt == "int_bw":
                L_q = quant_int_blockwise(L, bits, block_size=16)
            elif fmt == "ternary":
                L_q = quant_ternary(L)
            elif fmt == "nvfp4_mtq":
                # MTQ가 이후에 양자화; 여기서는 RPCANvfp4Wrapper로만 교체
                new_layer = RPCANvfp4Wrapper(L, S, bias)
                setattr(module, name, new_layer)
                report.append({
                    "layer": full_name, "scheme": scheme, "fmt": fmt,
                    "out_f": out_f, "in_f": in_f, "use_rpca": use_rpca,
                    "S_nnz_ratio": round((S.abs() > 1e-8).float().mean().item(), 4),
                    "S_norm_ratio": round((S.norm() / W.norm().clamp(min=1e-8)).item(), 4),
                })
                continue
            else:
                L_q = L  # fallback

            new_layer = RPCAQuantizedLinear(L_q, S, bias, act_bits=act_bits)
            setattr(module, name, new_layer)

            report.append({
                "layer": full_name, "scheme": scheme, "fmt": fmt,
                "out_f": out_f, "in_f": in_f, "bits": bits,
                "use_rpca": use_rpca,
                "S_nnz_ratio": round((S.abs() > 1e-8).float().mean().item(), 4),
                "S_norm_ratio": round((S.norm() / W.norm().clamp(min=1e-8)).item(), 4),
            })

        else:
            _replace_recursive(child, scheme, rpca_rank, rpca_lam_scale,
                                max_rpca_dim, report, prefix=full_name)


def apply_rpca_quant(
    policy: nn.Module,
    scheme: str,
    rpca_rank: int = 32,
    rpca_lam_scale: float = 1.0,
    max_rpca_dim: int = 1024,
) -> list[dict]:
    """정책 모델의 모든 Linear 레이어에 RPCA 양자화 적용."""
    report: list[dict] = []
    _replace_recursive(policy, scheme, rpca_rank, rpca_lam_scale,
                       max_rpca_dim, report, prefix="")

    # nvfp4_mtq 스킴: MTQ로 RPCANvfp4Wrapper 내부 nn.Linear를 NVFP4 양자화
    if scheme == "nvfp4_wa":
        if not _MODELOPT_AVAILABLE:
            print("[ERROR] modelopt 없음 — nvfp4_wa 스킴 건너뜀")
            return report
        print("[INFO] Applying MTQ NVFP4 to inner nn.Linear modules ...")
        mtq.quantize(policy, config=NVFP4_DEFAULT_CFG)
        print("[INFO] MTQ NVFP4 quantization complete.")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 레이어 분류 및 양자화 커버리지 분석
# ══════════════════════════════════════════════════════════════════════════════

def _classify_layer(full_name: str) -> tuple[str, str]:
    """
    레이어 전체 경로명으로 (component, role) 반환.

    component: vision | llm | expert | projector | head | action_head | other
    role:      attn_Q | attn_K | attn_V | attn_O |
               mlp_gate_up | mlp_down | mlp_fc |
               head | projector | action_in | action_out | time_mlp | other_fc
    """
    # ── Component ──────────────────────────────────────────────────────────
    if "vision_tower" in full_name or "siglip" in full_name:
        comp = "vision"
    elif "gemma_expert" in full_name:
        comp = "expert"
    elif (
        ("paligemma.model.layers" in full_name)
        or ("language_model.model.layers" in full_name)
        or ("paligemma" in full_name and ".layers." in full_name
            and "vision_tower" not in full_name)
    ):
        comp = "llm"
    elif "multi_modal_projector" in full_name or "mm_projector" in full_name:
        comp = "projector"
    elif "lm_head" in full_name:
        comp = "head"
    elif (
        "action_in_proj" in full_name
        or "action_out_proj" in full_name
        or "time_mlp" in full_name
    ):
        comp = "action_head"
    else:
        comp = "other"

    # ── Role (마지막 segment 기준) ──────────────────────────────────────────
    last = full_name.split(".")[-1]
    if last in ("q_proj", "query", "wq"):
        role = "attn_Q"
    elif last in ("k_proj", "key", "wk"):
        role = "attn_K"
    elif last in ("v_proj", "value", "wv"):
        role = "attn_V"
    elif last in ("out_proj", "o_proj", "wo"):
        role = "attn_O"
    elif last in ("gate_proj", "up_proj"):
        role = "mlp_gate_up"
    elif last == "down_proj":
        role = "mlp_down"
    elif last in ("fc1", "fc2", "fc"):
        role = "mlp_fc"
    elif "lm_head" in last:
        role = "head"
    elif "projector" in last or (last == "linear" and "projector" in full_name):
        role = "projector"
    elif "action_in" in full_name:
        role = "action_in"
    elif "action_out" in full_name:
        role = "action_out"
    elif "time_mlp" in full_name:
        role = "time_mlp"
    else:
        role = "other_fc"

    return comp, role


def build_quant_coverage(policy: nn.Module) -> dict:
    """
    양자화 후 모델 전체를 스캔해 컴포넌트/역할/타입별 커버리지 통계 반환.

    quant_status:
      quant_rpca   - RPCA 분해 후 양자화 (S ≠ 0)
      quant_direct - S=0으로 직접 양자화 (max_rpca_dim 초과)
      not_quant    - 양자화되지 않은 nn.Linear
    """
    from collections import defaultdict

    _zero = lambda: {"total": 0, "quant_rpca": 0, "quant_direct": 0, "not_quant": 0}
    by_comp: dict[str, dict] = defaultdict(_zero)
    by_role: dict[str, dict] = defaultdict(_zero)
    by_type: dict[str, int] = defaultdict(int)
    rows: list[dict] = []

    QUANT_TYPES = (RPCAQuantizedLinear, RPCANvfp4Wrapper)

    for full_name, mod in policy.named_modules():
        mtype = type(mod).__name__

        # RPCANvfp4Wrapper: 내부에 self.linear가 있어 leaf가 아니지만 여기서 직접 집계
        if isinstance(mod, RPCANvfp4Wrapper):
            pass  # 아래 공통 처리로 진행
        else:
            # 리프(leaf)가 아닌 컨테이너는 스킵 (중복 카운팅 방지)
            if list(mod.children()):
                continue

        if isinstance(mod, nn.Embedding):
            by_type["nn.Embedding"] += 1
            continue

        if not isinstance(mod, (nn.Linear, *QUANT_TYPES)):
            by_type[mtype] += 1
            continue

        # ── shape 추출 ──────────────────────────────────────────────────────
        if isinstance(mod, RPCAQuantizedLinear):
            out_f, in_f = mod.L_dequant.shape
        elif isinstance(mod, RPCANvfp4Wrapper):
            out_f, in_f = mod.out_features, mod.in_features
        else:
            out_f, in_f = mod.weight.shape

        # ── 양자화 상태 판별 ────────────────────────────────────────────────
        if isinstance(mod, (RPCAQuantizedLinear, RPCANvfp4Wrapper)):
            s_buf = mod.S if isinstance(mod, RPCAQuantizedLinear) else mod.S
            status = "quant_rpca" if s_buf.norm().item() > 1e-8 else "quant_direct"
        else:
            status = "not_quant"

        comp, role = _classify_layer(full_name)
        by_type[mtype] += 1
        by_comp[comp]["total"] += 1
        by_comp[comp][status] += 1
        by_role[role]["total"] += 1
        by_role[role][status] += 1
        rows.append({
            "name": full_name, "type": mtype,
            "component": comp, "role": role,
            "out_f": out_f, "in_f": in_f,
            "quant_status": status,
        })

    totals = _zero()
    for v in by_comp.values():
        for k in totals:
            totals[k] += v[k]

    return {
        "by_component": dict(by_comp),
        "by_role": dict(by_role),
        "by_type": dict(by_type),
        "totals": totals,
        "all_layers": rows,
    }


def print_coverage_table(cov: dict, scheme: str = "") -> None:
    """커버리지 통계를 터미널에 표 형태로 출력."""
    totals = cov["totals"]
    total_quant = totals["quant_rpca"] + totals["quant_direct"]

    def pct(n, d):
        return f"{n / d * 100:.1f}%" if d > 0 else "   - "

    W = 74
    sep = "─" * W

    # ── 컴포넌트별 ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    title = f" [{scheme}] 컴포넌트별 양자화 커버리지" if scheme else " 컴포넌트별 양자화 커버리지"
    print(title)
    print(sep)
    print(f"  {'Component':<20} {'Total':>6} {'RPCA':>7} {'Direct':>8} {'NotQuant':>9} {'Quant%':>8}")
    print(sep)

    COMP_ORDER = ["vision", "llm", "expert", "projector", "head", "action_head", "other"]
    for comp in COMP_ORDER:
        if comp not in cov["by_component"]:
            continue
        v = cov["by_component"][comp]
        q = v["quant_rpca"] + v["quant_direct"]
        print(f"  {comp:<20} {v['total']:>6} {v['quant_rpca']:>7} {v['quant_direct']:>8} "
              f"{v['not_quant']:>9} {pct(q, v['total']):>8}")
    print(sep)
    print(f"  {'TOTAL':<20} {totals['total']:>6} {totals['quant_rpca']:>7} "
          f"{totals['quant_direct']:>8} {totals['not_quant']:>9} "
          f"{pct(total_quant, totals['total']):>8}")

    # ── 역할별 ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(" 레이어 역할별 양자화 커버리지")
    print(sep)
    print(f"  {'Role':<20} {'Total':>6} {'RPCA':>7} {'Direct':>8} {'NotQuant':>9} {'Quant%':>8}")
    print(sep)

    ROLE_GROUPS = [
        ("Attention",  ["attn_Q", "attn_K", "attn_V", "attn_O"]),
        ("MLP",        ["mlp_gate_up", "mlp_down", "mlp_fc"]),
        ("Head/Proj",  ["head", "projector"]),
        ("ActionHead", ["action_in", "action_out", "time_mlp"]),
        ("Other FC",   ["other_fc"]),
    ]
    for grp_label, roles in ROLE_GROUPS:
        grp_t = grp_r = grp_d = grp_n = 0
        grp_rows = []
        for role in roles:
            if role not in cov["by_role"]:
                continue
            v = cov["by_role"][role]
            q = v["quant_rpca"] + v["quant_direct"]
            grp_rows.append((role, v))
            grp_t += v["total"]; grp_r += v["quant_rpca"]
            grp_d += v["quant_direct"]; grp_n += v["not_quant"]
        if grp_t == 0:
            continue
        print(f"  ── {grp_label} ──")
        for role, v in grp_rows:
            q = v["quant_rpca"] + v["quant_direct"]
            print(f"    {role:<18} {v['total']:>6} {v['quant_rpca']:>7} {v['quant_direct']:>8} "
                  f"{v['not_quant']:>9} {pct(q, v['total']):>8}")
        gq = grp_r + grp_d
        print(f"  {'  subtotal':<20} {grp_t:>6} {grp_r:>7} {grp_d:>8} "
              f"{grp_n:>9} {pct(gq, grp_t):>8}")

    # ── 모듈 타입 분포 ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(" 모듈 타입별 leaf count (전체 모델)")
    print(sep)
    for mtype, cnt in sorted(cov["by_type"].items(), key=lambda x: -x[1]):
        tag = ""
        if mtype in ("RPCAQuantizedLinear", "RPCANvfp4Wrapper"):
            tag = "  ← quantized"
        elif mtype == "Linear":
            tag = "  ← NOT quantized"
        print(f"  {mtype:<42} {cnt:>6}{tag}")

    print(f"\n  [참고] torch.matmul / torch.bmm (attention score·value 연산) 은")
    print(f"         nn.Module이 아니라 이 통계에 포함되지 않음.")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# torch.compile 비활성화 (eval_nvfp4_mtq.py 와 동일 수정)
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RPCA + Quantization eval for LeRobot pi05")
    parser.add_argument("--pretrained_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--task", type=str, default="libero_10")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="n_envs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="logs/rpca_test")
    parser.add_argument("--schemes", type=str, nargs="+", default=TEST_SCHEMES,
                        help=f"양자화 스킴 목록. 선택지: {list(QUANT_SCHEMES.keys())}")
    parser.add_argument("--rpca_rank", type=int, default=32,
                        help="RPCA 저랭크 성분의 최대 rank")
    parser.add_argument("--rpca_lam_scale", type=float, default=1.0,
                        help="RPCA lambda 스케일 (lambda = scale/sqrt(max(m,n)))")
    parser.add_argument("--max_rpca_dim", type=int, default=1024,
                        help="RPCA 적용 최대 레이어 차원. 초과 레이어는 직접 양자화")
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    # 스킴 검증
    for s in args.schemes:
        if s not in QUANT_SCHEMES:
            print(f"[ERROR] 알 수 없는 스킴 '{s}'. 가능한 스킴: {sorted(QUANT_SCHEMES.keys())}")
            sys.exit(1)
        if s == "nvfp4_wa" and not _MODELOPT_AVAILABLE:
            print("[ERROR] nvfp4_wa 스킴은 modelopt 필요. PYTHONPATH 확인")
            sys.exit(1)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 평가 스킴: {args.schemes}")
    print(f"[INFO] Creating LIBERO env: task={args.task}, n_envs={args.batch_size}")

    env_cfg = LiberoEnv(task=args.task)
    envs_dict = make_env(env_cfg, n_envs=args.batch_size)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    all_results: dict[str, dict] = {}

    for scheme in args.schemes:
        print(f"\n{'='*60}")
        print(f"[SCHEME] {scheme}")
        print(f"{'='*60}")

        # ── 정책 로드 ──────────────────────────────────────────────────────────
        policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
        policy_cfg.pretrained_path = args.pretrained_path
        policy_cfg.device = args.device
        policy_cfg.use_amp = args.use_amp

        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        # ── 프로세서 빌드 ──────────────────────────────────────────────────────
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )

        # ── RPCA 양자화 적용 ───────────────────────────────────────────────────
        print(f"[INFO] Applying RPCA + {scheme} ...")
        layer_report = apply_rpca_quant(
            policy, scheme,
            rpca_rank=args.rpca_rank,
            rpca_lam_scale=args.rpca_lam_scale,
            max_rpca_dim=args.max_rpca_dim,
        )

        # ── 커버리지 분석 & 출력 ───────────────────────────────────────────────
        coverage = build_quant_coverage(policy)
        print_coverage_table(coverage, scheme=scheme)

        # ── torch.compile 비활성화 ─────────────────────────────────────────────
        disable_torch_compile(policy)

        # ── 평가 ───────────────────────────────────────────────────────────────
        print(f"[INFO] Evaluating suite='{suite_name}' task_id={task_id}, "
              f"n_episodes={args.n_episodes}")
        with torch.no_grad(), (
            torch.autocast(device_type=device.type) if args.use_amp else nullcontext()
        ):
            eval_info = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
            )

        agg = eval_info.get("aggregated", {})
        pc_success = agg.get("pc_success", float("nan"))
        avg_reward = agg.get("avg_sum_reward", float("nan"))
        print(f"[RESULT] {scheme}: success={pc_success:.1f}%  avg_reward={avg_reward:.4f}")

        # ── 결과 저장 ──────────────────────────────────────────────────────────
        totals = coverage["totals"]
        n_rpca = totals["quant_rpca"]
        n_direct = totals["quant_direct"]
        n_not = totals["not_quant"]
        out = {
            "config": {
                "pretrained_path": args.pretrained_path,
                "task": args.task,
                "suite_name": suite_name,
                "task_id": task_id,
                "n_episodes": args.n_episodes,
                "batch_size": args.batch_size,
                "scheme": scheme,
                "rpca_rank": args.rpca_rank,
                "rpca_lam_scale": args.rpca_lam_scale,
                "max_rpca_dim": args.max_rpca_dim,
                "device": args.device,
            },
            "quantization": {
                "totals": coverage["totals"],
                "by_component": coverage["by_component"],
                "by_role": coverage["by_role"],
                "by_type": coverage["by_type"],
                "layer_detail": coverage["all_layers"],   # 개별 레이어 상세
            },
            "eval_results": eval_info,
        }
        out_path = output_dir / f"rpca_{scheme}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[INFO] 저장: {out_path}")

        all_results[scheme] = {
            "pc_success": pc_success,
            "avg_sum_reward": avg_reward,
            "quant_rpca": n_rpca,
            "quant_direct": n_direct,
            "not_quant": n_not,
            "total_linear": totals["total"],
        }

        # 메모리 정리
        del policy
        torch.cuda.empty_cache()

    env.close()

    # ── 요약 JSON ──────────────────────────────────────────────────────────────
    summary = {
        "config": vars(args),
        "suite_name": suite_name,
        "task_id": task_id,
        "results": all_results,
    }
    summary_path = output_dir / "rpca_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── 최종 요약 출력 ─────────────────────────────────────────────────────────
    W = 74
    print(f"\n{'═'*W}")
    print("[FINAL SUMMARY]")
    print(f"  {'Scheme':<15} {'Success':>8} {'Reward':>8} {'Total':>7} {'RPCA':>7} {'Direct':>8} {'NotQ':>6}")
    print(f"  {'─'*(W-2)}")
    for s, r in all_results.items():
        print(f"  {s:<15} {r['pc_success']:>7.1f}% {r['avg_sum_reward']:>8.4f} "
              f"{r['total_linear']:>7} {r['quant_rpca']:>7} {r['quant_direct']:>8} {r['not_quant']:>6}")
    print(f"[INFO] Summary: {summary_path}")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()
