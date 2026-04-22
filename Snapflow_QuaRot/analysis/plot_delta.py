"""Plot activation delta between two stages (before/after rotation or quantization)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def plot_delta_stats(
    stats_a_path: str,
    stats_b_path: str,
    output_path: str,
    metric: str = "max_abs",
    label_a: str = "FP16",
    label_b: str = "Rotated",
):
    """Load two stat files and plot side-by-side delta."""
    from analysis.activation_stats import compare_stats

    if not Path(stats_a_path).exists():
        log.error(f"Stats A not found: {stats_a_path}")
        return
    if not Path(stats_b_path).exists():
        log.error(f"Stats B not found: {stats_b_path}")
        return

    stats_a = torch.load(stats_a_path, weights_only=False)
    stats_b = torch.load(stats_b_path, weights_only=False)

    compare_stats(
        stats_before=stats_a,
        stats_after=stats_b,
        metric=metric,
        output_path=output_path,
        title=f"{metric}: {label_a} vs {label_b}",
    )
