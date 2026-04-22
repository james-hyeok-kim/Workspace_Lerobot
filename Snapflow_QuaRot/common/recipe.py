"""Stage recipe dataclass — declarative per-stage config."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SnapFlowConfig:
    enabled: bool = False
    # Path to distilled student checkpoint (.safetensors).
    # If None and enabled=True, the baseline model is used at NFE=1 (no distillation).
    student_ckpt: str | None = None


@dataclass
class QuaRotConfig:
    enabled: bool = False
    # "llm" | "llm+dit"
    scope: str = "llm"
    r1: bool = True   # embed + residual stream rotation
    r2: bool = True   # o_proj <-> down_proj offline rotation
    r3: bool = False  # V <-> o_proj offline rotation (requires joint attention care)
    r4: bool = False  # online Hadamard at down_proj input (runtime)
    fuse_rmsnorm: bool = True


@dataclass
class OHBConfig:
    enabled: bool = False
    # Fraction of outlier heads/layers to keep in FP16 (top-K by kurtosis)
    top_k_pct: float = 5.0
    metric: str = "kurtosis"  # "kurtosis" | "max_abs"
    manifest_path: str = "artifacts/stage4_ohb_manifest.json"


@dataclass
class AdaLNConfig:
    auto_detect: bool = True  # auto-detect use_adarms from PI05Config


@dataclass
class W4A4Config:
    enabled: bool = False
    weight_bits: int = 4
    act_bits: int = 4
    group_size: int = 128
    online_hadamard: bool = True  # R4 at down_proj inputs via ModelOpt RotateConfig
    quant_state_path: str = "artifacts/stage5_quant_state.pt"


@dataclass
class EvalConfig:
    task: str = "libero_10"
    n_episodes: int = 100
    batch_size: int = 1  # n parallel envs
    device: str = "cuda"
    use_amp: bool = False
    start_seed: int = 0


@dataclass
class PolicyOverride:
    n_action_steps: int = 10
    num_inference_steps: int = 10


@dataclass
class Recipe:
    """Declarative per-stage configuration."""

    name: str = "unnamed"
    policy: PolicyOverride = field(default_factory=PolicyOverride)
    eval: EvalConfig = field(default_factory=EvalConfig)
    snapflow: SnapFlowConfig = field(default_factory=SnapFlowConfig)
    quarot: QuaRotConfig = field(default_factory=QuaRotConfig)
    ohb: OHBConfig = field(default_factory=OHBConfig)
    adaln: AdaLNConfig = field(default_factory=AdaLNConfig)
    w4a4: W4A4Config = field(default_factory=W4A4Config)

    # Paths
    pretrained_path: str = "lerobot/pi05_libero_finetuned"
    calib_stats_path: str = "artifacts/stage0_calib_stats.pt"
    output_dir: str = "results/stage0"

    @classmethod
    def from_yaml(cls, path: str) -> "Recipe":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "Recipe":
        recipe = cls()
        recipe.name = d.get("name", recipe.name)
        recipe.pretrained_path = d.get("pretrained_path", recipe.pretrained_path)
        recipe.calib_stats_path = d.get("calib_stats_path", recipe.calib_stats_path)
        recipe.output_dir = d.get("output_dir", recipe.output_dir)

        if "policy" in d:
            p = d["policy"]
            recipe.policy.n_action_steps = p.get("n_action_steps", recipe.policy.n_action_steps)
            recipe.policy.num_inference_steps = p.get("num_inference_steps", recipe.policy.num_inference_steps)

        if "eval" in d:
            e = d["eval"]
            for k, v in e.items():
                if hasattr(recipe.eval, k):
                    setattr(recipe.eval, k, v)

        if "snapflow" in d:
            for k, v in d["snapflow"].items():
                if hasattr(recipe.snapflow, k):
                    setattr(recipe.snapflow, k, v)

        if "quarot" in d:
            for k, v in d["quarot"].items():
                if hasattr(recipe.quarot, k):
                    setattr(recipe.quarot, k, v)

        if "ohb" in d:
            for k, v in d["ohb"].items():
                if hasattr(recipe.ohb, k):
                    setattr(recipe.ohb, k, v)

        if "adaln" in d:
            for k, v in d["adaln"].items():
                if hasattr(recipe.adaln, k):
                    setattr(recipe.adaln, k, v)

        if "w4a4" in d:
            for k, v in d["w4a4"].items():
                if hasattr(recipe.w4a4, k):
                    setattr(recipe.w4a4, k, v)

        return recipe
