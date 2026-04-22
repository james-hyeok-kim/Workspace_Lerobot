"""Outlier Head Bypass (OHB) — keep top-K outlier heads / layers in FP16.

Algorithm:
1. Load calibration stats (from Stage 0 calib_capture.py).
2. Rank all Linear layers by chosen metric (kurtosis or max_abs).
3. Select top-K% as "outlier" layers.
4. When building the W4A4 quantization recipe, mark these layers with disable=True
   so ModelOpt's MTQ keeps them in FP16.

Also saves the manifest so later stages can reload it without re-computing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import torch
from torch import nn

log = logging.getLogger(__name__)

MetricType = Literal["kurtosis", "max_abs"]


def _score_layers(
    calib_stats: dict[str, dict],
    metric: MetricType = "kurtosis",
) -> list[tuple[str, float]]:
    """Return [(layer_name, score), ...] sorted descending."""
    scored = []
    for name, s in calib_stats.items():
        val = s.get(metric, 0.0)
        if val is None:
            val = 0.0
        scored.append((name, float(val)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def build_ohb_manifest(
    calib_stats_path: str,
    top_k_pct: float = 5.0,
    metric: MetricType = "kurtosis",
    output_path: str = "artifacts/stage4_ohb_manifest.json",
) -> dict[str, bool]:
    """Compute OHB manifest and save it.

    Returns:
        dict[layer_name -> True] for layers that should be kept in FP16.
    """
    if not Path(calib_stats_path).exists():
        raise FileNotFoundError(f"Calib stats not found: {calib_stats_path}. Run Stage 0 first.")

    stats = torch.load(calib_stats_path, weights_only=False)
    scored = _score_layers(stats, metric)

    n_total = len(scored)
    n_keep = max(1, int(n_total * top_k_pct / 100.0))
    outlier_layers = {name for name, _ in scored[:n_keep]}

    log.info(
        f"OHB: keeping {n_keep}/{n_total} layers in FP16 "
        f"(top {top_k_pct:.1f}% by {metric})"
    )
    for name, score in scored[:n_keep]:
        log.info(f"  FP16 (outlier): {name}  {metric}={score:.4f}")

    manifest = {name: True for name in outlier_layers}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"OHB manifest saved to {output_path}")

    return manifest


def load_ohb_manifest(path: str) -> dict[str, bool]:
    if not Path(path).exists():
        raise FileNotFoundError(f"OHB manifest not found: {path}. Run Stage 4 first.")
    with open(path) as f:
        return json.load(f)


def apply_ohb(policy: nn.Module, ohb_cfg, calib_stats_path: str):
    """Attach OHB manifest to the policy (used by W4A4 quantization bridge)."""
    manifest_path = ohb_cfg.manifest_path

    if not Path(manifest_path).exists():
        log.info("OHB manifest not found — building from calib stats...")
        build_ohb_manifest(
            calib_stats_path=calib_stats_path,
            top_k_pct=ohb_cfg.top_k_pct,
            metric=ohb_cfg.metric,
            output_path=manifest_path,
        )

    manifest = load_ohb_manifest(manifest_path)
    # Attach manifest to policy so modelopt_bridge can read it
    policy._ohb_manifest = manifest
    log.info(f"OHB manifest loaded: {len(manifest)} layers protected.")
