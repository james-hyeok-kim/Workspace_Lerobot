"""AdaLN (Adaptive Layer Norm) handling for pi0.5 action expert.

pi0.5 uses `use_adarms` / `use_adarms_for_expert` (modeling_pi05.py L351, L383)
to conditionally apply AdaRMS (adaptive RMS norm) for timestep conditioning.

The AdaLN modulation pattern:
  scale, shift = Linear(timestep_emb).chunk(2)
  x = norm(x) * (1 + scale) + shift
or:
  x = norm(x) * scale  (AdaRMS, scale-only)

This breaks the standard RMSNorm fusion because the gain is input-dependent
(it varies per timestep token).  The strategy here:

1. Detect whether AdaLN is active (by checking config.use_adarms fields).
2. If AdaLN is inactive: standard RMSNorm fusion applies.
3. If AdaLN IS active:
   - Keep the modulation Linear (timestep_emb -> scale/shift) in FP16.
   - Apply rotation AFTER the scale/shift broadcast, not before.
   - This means the norm can still be fused (gain=1 after fusion), but
     the modulation scale must be re-applied after rotation.
   - Concretely: we insert a thin wrapper around the AdaRMS path that
     applies the Hadamard to x AFTER scaling, so the downstream projection
     sees rotated activations.

For now this module:
- Detects AdaLN usage.
- Reports which modules have AdaLN.
- Marks AdaLN Linear modules to exclude from quantization (keep FP16).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from torch import nn

log = logging.getLogger(__name__)


class AdaLNInfo(NamedTuple):
    use_adarms_llm: bool
    use_adarms_expert: bool
    adaln_module_names: list[str]  # module names to keep FP16


def detect_adaln(policy: nn.Module) -> AdaLNInfo:
    """Inspect pi0.5 policy config and model to detect AdaLN usage."""
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return AdaLNInfo(False, False, [])

    # Check config flags (may be on the inner model's config)
    inner = getattr(policy, "model", policy)
    inner_cfg = getattr(inner, "config", cfg)

    use_adarms_llm = bool(getattr(inner_cfg, "use_adarms", False))
    use_adarms_expert = bool(getattr(inner_cfg, "use_adarms_for_expert", False))

    adaln_names: list[str] = []
    if use_adarms_llm or use_adarms_expert:
        # Find all modules whose name suggests AdaRMS / AdaLN modulation
        for name, module in inner.named_modules():
            n_lower = name.lower()
            if any(kw in n_lower for kw in ["adarms", "adaln", "adanorm", "time_emb", "timestep"]):
                adaln_names.append(name)

    log.info(
        f"AdaLN detection: use_adarms_llm={use_adarms_llm}, "
        f"use_adarms_expert={use_adarms_expert}, "
        f"found {len(adaln_names)} candidate modules."
    )
    return AdaLNInfo(use_adarms_llm, use_adarms_expert, adaln_names)


def get_adaln_protected_modules(info: AdaLNInfo, policy: nn.Module) -> set[nn.Module]:
    """Return the set of nn.Module objects that should be kept in FP16."""
    protected: set[nn.Module] = set()
    inner = getattr(policy, "model", policy)
    for name, module in inner.named_modules():
        if name in info.adaln_module_names:
            protected.add(module)
    return protected
