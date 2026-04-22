"""Stage 4 — Outlier Head Bypass + AdaLN handling.

Builds OHB manifest from Stage 0 calib stats.
Detects AdaLN (use_adarms) and marks AdaLN Linear modules as FP16-protected.
Validates that remaining activations are within bounds.
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
    from quarot.adaln_handler import detect_adaln

    recipe = Recipe.from_yaml(cfg_path)
    log.info(f"Stage 4 — OHB + AdaLN. Config: {cfg_path}")

    # Ensure calib stats exist
    if not Path(recipe.calib_stats_path).exists():
        log.error(
            f"Calib stats not found: {recipe.calib_stats_path}\n"
            "Run Stage 0 first: bash scripts/run_stage.sh 0"
        )
        sys.exit(1)

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    # Detect AdaLN after loading
    adaln_info = detect_adaln(policy)
    log.info(
        f"AdaLN: use_adarms_llm={adaln_info.use_adarms_llm}, "
        f"use_adarms_expert={adaln_info.use_adarms_expert}, "
        f"protected={len(adaln_info.adaln_module_names)} modules"
    )

    log.info("=== Smoke Test ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke failed — OHB or AdaLN handling may be broken.")
        sys.exit(1)

    log.info("=== Full Evaluation (NFE=10, with OHB) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    db = ResultsDB()
    db.append("stage4_ohb_adaln", result, recipe)
    log.info(f"[Stage 4] pc_success={result['aggregated'].get('pc_success', 'N/A'):.1f}%")
    log.info(f"[Stage 4] OHB manifest: {recipe.ohb.manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_ohb_adaln.yaml")
    args = parser.parse_args()
    main(args.config)
