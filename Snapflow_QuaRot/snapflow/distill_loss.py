"""SnapFlow distillation loss — 논문 수식 그대로 구현.

논문 (Eq. 12): L = α·L_FM + (1-α)·λ·L_shortcut   [α=0.5, λ=0.1]

L_FM  (표준 flow matching):
  t ~ U(0,1), x_t = (1-t)*x0 + t*ε
  L_FM = ||Fθ(x_t, s=t, t=t) - (ε - x0)||²

L_shortcut (2-step Euler shortcut, self-distillation):
  x1 = ε  (pure noise)
  [stop-grad] v1     = Fθ(x1,   s=1,   t=1)
  [stop-grad] x0.5   = x1 - 0.5·v1
  [stop-grad] v0.5   = Fθ(x0.5, s=0.5, t=0.5)
  v_target = (v1 + v0.5) / 2   [trapezoidal Euler average]
  [with-grad] v_1nfe = Fθ(x1,   s=0,   t=1)
  L_shortcut = ||v_1nfe - v_target||²

target-time embedding φs (TargetTimeEmbedding):
  adarms_cond += φs(s) — 모델이 s=t (FM) vs s=0 (1-NFE)를 구별하게 함
  zero-init이므로 학습 초기에는 기존 모델 동작 보존.

피험자 pi0.5 flow convention:
  x_t = t * noise + (1 - t) * actions   (t=1: pure noise, t=0: clean action)
  velocity target = noise - actions
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ── forward 헬퍼 ────────────────────────────────────────────────────────────

def get_velocity(
    inner,
    images,
    img_masks,
    tokens,
    masks,
    x_t: Tensor,
    t: Tensor,
    s: Tensor | None = None,
    target_time_emb: nn.Module | None = None,
) -> Tensor:
    """pi0.5 action expert로부터 velocity 계산.

    Args:
        inner: PI05Pytorch instance (policy.model)
        images, img_masks, tokens, masks: VLM prefix inputs
        x_t: [B, chunk, max_action_dim] noisy action
        t:   [B] current time (0=clean, 1=noise)
        s:   [B] target time. None이면 t와 동일 (FM mode)
        target_time_emb: TargetTimeEmbedding | None
    Returns:
        v_t: [B, chunk, max_action_dim] predicted velocity
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = inner.embed_prefix(
        images, img_masks, tokens, masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = inner.embed_suffix(x_t, t)

    # V4: adarms_cond은 항상 clean FM conditioning으로 유지 (phi_s 주입 X)
    # phi_s는 action output space에 주입 (AdaRMS 오염 방지)

    model_dtype = (
        inner.paligemma_with_expert.paligemma.model.language_model
        .layers[0].self_attn.q_proj.weight.dtype
    )
    if model_dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = inner._prepare_attention_masks_4d(att_2d_masks)

    (_, suffix_out), _ = inner.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    suffix_out = suffix_out[:, -inner.config.chunk_size:]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v = inner.action_out_proj(suffix_out)  # [B, chunk, max_action_dim]

    # V4: phi_s는 action output에 더함 (conditioning이 아닌 action space에 주입)
    # adarms_cond는 항상 FM conditioning으로 유지 → FM mode (10-NFE) 절대 오염 안 됨
    if target_time_emb is not None and s is not None:
        phi_s = target_time_emb(s.to(dtype=v.dtype))  # [B, out_dim]
        v = v + phi_s.unsqueeze(1)  # broadcast: [B, 1, out_dim] → [B, chunk, out_dim]

    return v


# ── 손실 함수 ────────────────────────────────────────────────────────────────

class SnapFlowLoss(nn.Module):
    """SnapFlow 논문 수식 그대로 구현한 손실 함수.

    L = α·L_FM + (1-α)·λ·L_shortcut   (α=0.5, λ=0.1)
    """

    def __init__(
        self,
        action_dim: int = 7,
        alpha: float = 0.5,
        lam: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.alpha = alpha
        self.lam = lam

    def forward(
        self,
        policy,
        batch: dict,
        target_time_emb: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        """SnapFlow mixed loss.

        Args:
            policy: trainable PI05Policy (VLM frozen, action expert trainable)
            batch: {image, language, action, ...}
            target_time_emb: TargetTimeEmbedding (optional, adds s-encoding)
        Returns:
            dict with 'loss' and diagnostics
        """
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

        images, img_masks = policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        x0 = policy.prepare_action(batch)           # [B, T, D] clean action
        B, T, D = x0.shape
        device = x0.device
        d = self.action_dim

        noise = policy.model.sample_noise(x0.shape, device)   # [B, T, D]

        # ── FM Loss ──────────────────────────────────────────────────────────
        # 랜덤 t에서 표준 flow matching: Fθ(x_t, s=t, t=t) → (ε - x0)
        t_fm = torch.rand(B, device=device, dtype=x0.dtype)         # [B]
        t_bc = t_fm[:, None, None]
        x_t  = t_bc * noise + (1.0 - t_bc) * x0                    # [B, T, D]
        target_fm = noise - x0                                       # [B, T, D]

        # FM loss: phi_s를 사용하지 않음 — action expert가 FM behavior를 유지하도록
        # phi_s는 shortcut student에서만 gradient를 받아야 안정적으로 학습됨
        v_fm = get_velocity(
            policy.model, images, img_masks, tokens, masks,
            x_t, t_fm, s=t_fm,
            target_time_emb=None,   # FM loss는 phi_s 없이 — circular dependency 방지
        )
        loss_fm = F.mse_loss(v_fm[:, :, :d], target_fm[:, :, :d])

        # lam=0이면 shortcut forward pass 전부 skip (3번 forward 절약 → ~4x speedup)
        if self.lam == 0.0:
            return {
                "loss": loss_fm,
                "loss_fm": loss_fm.detach(),
                "loss_shortcut": torch.zeros((), device=device),
                "v_fm_norm": v_fm[:, :, :d].norm().detach(),
                "v_1nfe_norm": torch.zeros((), device=device),
                "v_target_norm": torch.zeros((), device=device),
            }

        # ── Shortcut Loss ─────────────────────────────────────────────────────
        # x1 = ε (pure noise), 2-step Euler self-distillation
        x1 = noise
        t1   = torch.ones(B,  device=device, dtype=x0.dtype)       # t=1
        t05  = torch.full((B,), 0.5, device=device, dtype=x0.dtype)  # t=0.5
        s_zero = torch.zeros(B, device=device, dtype=x0.dtype)      # s=0 (1-NFE mode)

        with torch.no_grad():
            # Teacher: pure FM trajectory (phi_s 없음) — stable target 보장
            # phi_s를 teacher에 쓰면 teacher target 자체가 흔들리는 moving-target 문제 발생
            v1 = get_velocity(
                policy.model, images, img_masks, tokens, masks,
                x1, t1, s=t1,
                target_time_emb=None,   # teacher는 phi_s 없이
            )
            # midpoint via Euler half-step
            x05 = x1 - 0.5 * v1                                     # [B, T, D]

            v05 = get_velocity(
                policy.model, images, img_masks, tokens, masks,
                x05, t05, s=t05,
                target_time_emb=None,   # teacher는 phi_s 없이
            )
            # trapezoidal average → better marginal velocity estimate
            v_target = 0.5 * (v1 + v05)                             # [B, T, D]

        # 1-NFE prediction: phi_s(s=0)만 gradient 받음 — s=0만 학습
        v_1nfe = get_velocity(
            policy.model, images, img_masks, tokens, masks,
            x1, t1, s=s_zero,   # s=0 → 1-NFE mode
            target_time_emb=target_time_emb,   # student에만 phi_s 사용
        )
        loss_shortcut = F.mse_loss(v_1nfe[:, :, :d], v_target[:, :, :d])

        # ── Combined Loss ─────────────────────────────────────────────────────
        alpha, lam = self.alpha, self.lam
        loss = alpha * loss_fm + (1.0 - alpha) * lam * loss_shortcut

        return {
            "loss": loss,
            "loss_fm": loss_fm.detach(),
            "loss_shortcut": loss_shortcut.detach(),
            "v_fm_norm": v_fm[:, :, :d].norm().detach(),
            "v_1nfe_norm": v_1nfe[:, :, :d].norm().detach(),
            "v_target_norm": v_target[:, :, :d].norm().detach(),
        }
