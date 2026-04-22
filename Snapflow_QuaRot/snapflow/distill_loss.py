"""SnapFlow distillation loss.

References:
  - SnapFlow (2024): 1-step flow matching via trajectory distillation.
  - π0 paper appendix (Physical Intelligence, 2024).

Loss formulation:
  Given a ground-truth action a and noise ε ~ N(0, I):
    1. Sample t ~ U(0, 1)
    2. Compute interpolated x_t = (1 - t) * ε + t * a
    3. Teacher predicts velocity: v_T = teacher_net(x_t, t)  (NFE=1 at given t)
    4. Teacher's predicted clean action: a_T = x_t + (0 - t) * v_T = x_t - t * v_T
    5. Student predicts from pure noise in 1 step: v_S = student_net(x_1=ε, t=1)
    6. Student's predicted clean action: a_S = ε + (-1) * v_S = ε - v_S
    7. Loss = MSE(a_S, a_T)  (student learns to imitate teacher's predicted endpoint)

This bypasses the need for real action labels during distillation — only the
teacher's flow prediction is used as supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SnapFlowLoss(nn.Module):
    """Compute the SnapFlow distillation loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, model, batch: dict) -> dict[str, Tensor]:
        """Compute loss.

        Args:
            model: TeacherStudent instance
            batch: lerobot-style batch dict with "action" and observations

        Returns:
            dict with "loss" and diagnostic scalars
        """
        action = batch.get("action")
        if action is None:
            raise KeyError("batch must contain 'action' key for SnapFlow distillation")

        B, T, D = action.shape
        device = action.device
        dtype = action.dtype

        # 1. Sample t and noise
        t = torch.rand(B, device=device, dtype=dtype)  # [B]
        noise = torch.randn_like(action)  # [B, T, D]

        # 2. Interpolate
        t_bcast = t[:, None, None]  # [B, 1, 1]
        x_t = (1 - t_bcast) * noise + t_bcast * action  # [B, T, D]

        # 3. Teacher velocity at (x_t, t) — single NFE at given t
        with torch.no_grad():
            v_teacher = _get_velocity_at(model.teacher, batch, x_t, t)

        # 4. Teacher's predicted clean action
        a_teacher = x_t - t_bcast * v_teacher  # [B, T, D]

        # 5. Student velocity at (noise, t=1) — 1 NFE from pure noise
        t_one = torch.ones(B, device=device, dtype=dtype)
        v_student = _get_velocity_at(model.student, batch, noise, t_one)

        # 6. Student's predicted clean action
        a_student = noise - v_student  # [B, T, D]

        # 7. Loss
        if self.reduction == "mean":
            loss = F.mse_loss(a_student, a_teacher)
        else:
            loss = F.mse_loss(a_student, a_teacher, reduction="none")

        return {
            "loss": loss,
            "v_teacher_norm": v_teacher.norm().item(),
            "v_student_norm": v_student.norm().item(),
            "action_mse": F.mse_loss(a_student.detach(), action).item(),
        }


def _get_velocity_at(policy: nn.Module, batch: dict, x_t: Tensor, t: Tensor) -> Tensor:
    """Call the policy's internal denoise_step to get velocity at given (x_t, t).

    This accesses the inner PI05Pytorch.denoise_step method directly.
    Requires the policy's KV cache (prefix) to already be computed.

    NOTE: This is an approximation — we call denoise_step once per batch
    with a single shared t value (using t[0] as representative).
    A proper implementation would compute the prefix cache once and
    reuse it for all B samples.
    """
    inner = getattr(policy, "model", policy)

    # We need to embed the prefix first (this mirrors modeling_pi05.py:sample_actions)
    # For now, use the policy's forward pass structure
    # In production, extract prefix_embs once and call denoise_step in a loop
    raise NotImplementedError(
        "SnapFlow distillation requires direct access to PI05Pytorch internals. "
        "Implement _get_velocity_at by calling model.embed_prefix + model.denoise_step "
        "from modeling_pi05.py. See TODO in snapflow/trainer.py."
    )
