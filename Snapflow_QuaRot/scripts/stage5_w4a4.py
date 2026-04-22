"""Stage 5 — W4A4 aggressive quantization.

Applies W4A4 (INT4 weight + INT4 activation) with:
- R4 online Hadamard at down_proj inputs (via ModelOpt RotateConfig)
- OHB manifest to keep outlier layers in FP16
- AdaLN modules in FP16

Pipeline: load → QuaRot(R1,R2,R3,R4) → OHB → W4A4 calibration → eval
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

for _p in [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2] / "lerobot" / "src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(cfg_path: str):
    from common.recipe import Recipe
    from common.policy_loader import load_policy
    from common.smoke import run_smoke
    from common.eval_driver import run_eval
    from common.results_db import ResultsDB

    recipe = Recipe.from_yaml(cfg_path)
    log.info(f"Stage 5 — W4A4. Config: {cfg_path}")

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    log.info("=== Smoke Test (W4A4 sanity check) ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.warning("Smoke failed — W4A4 may cause instability. Check layer-wise MSE.")

    log.info("=== Full Evaluation (NFE=10, W4A4) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    # Save quant state for Stage 6 reuse
    if recipe.w4a4.quant_state_path:
        from quant.modelopt_bridge import save_quant_state
        save_quant_state(policy, recipe.w4a4.quant_state_path)

    db = ResultsDB()
    db.append("stage5_w4a4", result, recipe)
    log.info(f"[Stage 5] pc_success={result['aggregated'].get('pc_success', 'N/A'):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5_w4a4.yaml")
    args = parser.parse_args()
    main(args.config)
