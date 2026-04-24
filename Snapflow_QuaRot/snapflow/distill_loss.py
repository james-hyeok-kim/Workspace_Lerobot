"""SnapFlow distillation loss for pi0.5.

pi0.5 flow convention (from modeling_pi05.py::PI05Pytorch.forward):
  x_t = t * noise + (1 - t) * actions   (t=1: pure noise, t=0: clean)
  u_t = noise - actions                  (velocity target)
  v_t = net(x_t, t)                      (predicted velocity)

SnapFlow distillation loss:
  1. Sample t ~ U(0, 1), noise ~ N(0, I)
  2. x_t = t * noise + (1 - t) * actions
  3. v_T = teacher(x_t, t)              [no_grad]
  4. a_T = x_t - t * v_T               [teacher's predicted clean action, Euler to t=0]
  5. v_S = student(noise, t=1)          [1-NFE from pure noise]
  6. a_S = noise - v_S                  [student's predicted clean action]
  7. loss = MSE(a_S, a_T)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_velocity(inner, images, img_masks, tokens, masks, x_t: Tensor, t: Tensor) -> Tensor:
    """Compute flow velocity v_t without KV cache or deepcopy.

    Replicates PI05Pytorch.forward() internals but returns v_t directly.
    Compatible with gradient flow (student) and no_grad (teacher) contexts.

    Args:
        inner: PI05Pytorch instance (policy.model)
        images: list of [B, H, W, C] or [B, C, H, W] float32 tensors
        img_masks: list of [B] bool tensors
        tokens: [B, seq_len] int64
        masks: [B, seq_len] bool
        x_t: [B, chunk_size, max_action_dim] noisy action at time t
        t: [B] timestep values in [0, 1]

    Returns:
        v_t: [B, chunk_size, max_action_dim] predicted velocity
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = inner.embed_prefix(
        images, img_masks, tokens, masks
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = inner.embed_suffix(x_t, t)

    # Match dtype to model weights
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
    return inner.action_out_proj(suffix_out)


class SnapFlowLoss(nn.Module):
    """SnapFlow distillation loss for pi0.5."""

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim

    def forward(
        self,
        teacher_policy,
        student_policy,
        batch: dict,
    ) -> dict[str, Tensor]:
        """Compute SnapFlow distillation loss.

        Args:
            teacher_policy: frozen PI05Policy
            student_policy: trainable PI05Policy (student)
            batch: dict with image/language/action keys

        Returns:
            dict with 'loss' and diagnostic tensors
        """
        # Preprocess images through the student policy (both teacher/student have same preprocessing)
        images, img_masks = student_policy._preprocess_images(batch)

        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Get padded actions (chunk_size, max_action_dim)
        actions = student_policy.prepare_action(batch)  # [B, chunk_size, max_action_dim]
        B, T, D = actions.shape
        device = actions.device

        # Sample noise and interpolation time
        noise = student_policy.model.sample_noise(actions.shape, device)  # [B, T, D]
        t = torch.rand(B, device=device, dtype=actions.dtype)             # [B]
        t_bcast = t[:, None, None]

        # x_t: pi0.5 convention — t*noise + (1-t)*actions
        x_t = t_bcast * noise + (1.0 - t_bcast) * actions

        # Teacher velocity at (x_t, t) — no gradient
        t_teacher = t.clone()
        with torch.no_grad():
            v_teacher = get_velocity(
                teacher_policy.model, images, img_masks, tokens, masks, x_t, t_teacher
            )

        # Teacher's clean action (Euler step from t to 0)
        a_teacher = x_t - t_bcast * v_teacher  # [B, T, D]

        # Student velocity at (noise, t=1) — 1 NFE, with gradient
        t_one = torch.ones(B, device=device, dtype=actions.dtype)
        v_student = get_velocity(
            student_policy.model, images, img_masks, tokens, masks, noise, t_one
        )

        # Student's clean action
        a_student = noise - v_student  # [B, T, D]

        # Loss only on true action dimensions (not padding)
        d = self.action_dim
        loss = F.mse_loss(a_student[:, :, :d], a_teacher[:, :, :d])

        return {
            "loss": loss,
            "v_teacher_norm": v_teacher[:, :, :d].norm().detach(),
            "v_student_norm": v_student[:, :, :d].norm().detach(),
            "a_mse_vs_gt": F.mse_loss(a_student[:, :, :d].detach(), actions[:, :, :d]).detach(),
        }
