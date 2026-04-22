"""Activation statistics visualization.

Generates max-abs and kurtosis heatmaps from calibration stats.
Reuses plotting patterns from dist_analysis_cross_model/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def plot_activation_heatmap(
    stats: dict,
    metric: str = "max_abs",
    output_path: str | None = None,
    title: str = "",
    top_n: int = 50,
):
    """Plot a bar chart of top-N layers by chosen metric.

    Args:
        stats: dict from calib_capture.py (layer_name -> {max_abs, kurtosis, ...})
        metric: "max_abs" or "kurtosis"
        output_path: if provided, save figure to this path
        title: plot title
        top_n: how many layers to show
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot.")
        return

    names = []
    values = []
    for name, s in stats.items():
        v = s.get(metric)
        if v is not None:
            names.append(name)
            values.append(float(v))

    # Sort descending
    pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Top-{top_n} layers by {metric}")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        log.info(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    plt.close()


def compare_stats(
    stats_before: dict,
    stats_after: dict,
    metric: str = "max_abs",
    output_path: str | None = None,
    title: str = "",
):
    """Overlay before/after stats for the same set of layers."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not installed; skipping plot.")
        return

    common_keys = sorted(set(stats_before.keys()) & set(stats_after.keys()))
    before_vals = [stats_before[k].get(metric, 0.0) for k in common_keys]
    after_vals = [stats_after[k].get(metric, 0.0) for k in common_keys]

    x = np.arange(len(common_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width / 2, before_vals, width, label="Before")
    ax.bar(x + width / 2, after_vals, width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(common_keys, rotation=90, fontsize=5)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} before vs after")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        log.info(f"Saved comparison plot to {output_path}")
    else:
        plt.show()
    plt.close()


def load_and_plot(stats_path: str, output_dir: str, stage: str = "stage0"):
    stats = torch.load(stats_path, weights_only=False)
    for metric in ["max_abs", "kurtosis"]:
        plot_activation_heatmap(
            stats,
            metric=metric,
            output_path=f"{output_dir}/{stage}_{metric}.png",
            title=f"[{stage}] {metric}",
        )
