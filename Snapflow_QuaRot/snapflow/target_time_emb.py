"""Target-time embedding φs for SnapFlow.

논문: "a zero-initialized two-layer MLP that encodes s and adds to the
existing time embedding before each transformer block. Zero initialization
preserves the teacher at step 0."

V4 fix: phi_s is injected into action output space (max_action_dim=32),
NOT into adarms_cond (2048-dim conditioning). This prevents AdaRMS corruption.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TargetTimeEmbedding(nn.Module):
    """Zero-initialized MLP: target time s → action velocity offset.

    V4: out_dim defaults to max_action_dim (32) for action-space injection.
    sinusoidal_dim controls the richness of s encoding (independent of out_dim).
    """

    def __init__(
        self,
        out_dim: int = 32,
        sinusoidal_dim: int = 256,
        hidden_dim: int = 256,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        # Legacy compat: embed_dim was the single-dim parameter in v3
        embed_dim: int | None = None,
    ):
        super().__init__()
        if embed_dim is not None:
            # Legacy v3 mode: all dims equal embed_dim (adarms_cond space injection)
            out_dim = embed_dim
            sinusoidal_dim = embed_dim
            hidden_dim = embed_dim

        self.out_dim = out_dim
        self.sinusoidal_dim = sinusoidal_dim
        self.hidden_dim = hidden_dim
        self.min_period = min_period
        self.max_period = max_period

        self.mlp_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)

        # Zero-init: 학습 시작 시 φs(s)=0 → 사전학습 모델 동작 유지
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    @property
    def embed_dim(self):
        """Legacy compat: embed_dim == out_dim for v3 checkpoints."""
        return self.out_dim

    def forward(self, s: Tensor) -> Tensor:
        """
        Args:
            s: [B] target times in [0, 1]
        Returns:
            [B, out_dim] offset to add to action velocity (or adarms_cond in legacy mode)
        """
        from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
        s_emb = create_sinusoidal_pos_embedding(
            s, self.sinusoidal_dim,
            min_period=self.min_period,
            max_period=self.max_period,
            device=s.device,
        ).to(dtype=s.dtype)
        x = F.silu(self.mlp_in(s_emb))
        x = self.mlp_out(x)   # zero-init이므로 초기에는 0 출력
        return x
