"""TeacherStudent wrapper for SnapFlow 1-NFE distillation.

The teacher is a frozen copy of pi0.5 (NFE=10).
The student shares all weights initially and is fine-tuned to match
the teacher's integrated trajectory in a single step.

Key idea (SnapFlow):
  At training time, sample a random timestep t ∈ [0, 1].
  Compute: x_t = (1 - t) * noise + t * action_chunk   (flow interpolation)
  Teacher predicts: v_teacher = net(x_t, t)
  Integrate teacher trajectory: x_0_teacher = x_t - t * v_teacher  (single Euler step)
  Student predicts: v_student = net_student(x_1, t=1)  (start from pure noise, 1 step)
  Loss: MSE(student_predicted_x0, teacher_predicted_x0)

After distillation, the student can produce a clean action with NFE=1:
  x_0 = x_1 + (-1) * v_student(x_1, t=1)   (dt = -1/1 = -1)
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


class TeacherStudent(nn.Module):
    """Wraps PI05Policy: teacher = frozen original, student = trainable copy."""

    def __init__(self, base_policy: nn.Module, share_weights: bool = False):
        super().__init__()
        self.student = base_policy

        if share_weights:
            # Share weights: teacher IS the student (EMA or same params)
            # This is unusual; typically teacher is frozen separately
            self.teacher = base_policy
        else:
            self.teacher = copy.deepcopy(base_policy)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        log.info(
            f"TeacherStudent initialized. "
            f"Teacher params: {sum(p.numel() for p in self.teacher.parameters()):,} (frozen). "
            f"Student params: {sum(p.numel() for p in self.student.parameters() if p.requires_grad):,} (trainable)."
        )

    @property
    def config(self):
        return self.student.config

    def teacher_inference(self, batch: dict, num_steps: int = 10) -> Tensor:
        """Run teacher with multi-step NFE to get reference action chunks."""
        with torch.no_grad():
            return self.teacher.select_action(batch)

    def student_inference(self, batch: dict, num_steps: int = 1) -> Tensor:
        """Run student with 1-NFE."""
        # Temporarily override num_inference_steps
        orig = self.student.config.num_inference_steps
        self.student.config.num_inference_steps = num_steps
        result = self.student.select_action(batch)
        self.student.config.num_inference_steps = orig
        return result

    def forward(self, batch: dict) -> dict:
        """Compute distillation loss (used during training)."""
        from snapflow.distill_loss import SnapFlowLoss

        loss_fn = SnapFlowLoss()
        return loss_fn(self, batch)
