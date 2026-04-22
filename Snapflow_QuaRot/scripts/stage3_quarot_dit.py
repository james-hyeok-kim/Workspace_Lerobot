"""Stage 3 — QuaRot extended to DiT (action expert) with R3.

Adds R3 (V <-> o_proj) to both LLM and DiT.
Handles cross-attention: LLM KV rotation is linked to expert Q rotation.
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
    log.info(f"Stage 3 — QuaRot LLM+DiT. Config: {cfg_path}")

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    log.info("=== Smoke Test (joint attention R3 invariance check) ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke failed — R3 cross-attention correction may be incorrect.")
        sys.exit(1)

    log.info("=== Full Evaluation (NFE=10, LLM+DiT rotated) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    db = ResultsDB()
    db.append("stage3_quarot_llm_dit", result, recipe)
    log.info(f"[Stage 3] pc_success={result['aggregated'].get('pc_success', 'N/A'):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_quarot_llm_dit.yaml")
    args = parser.parse_args()
    main(args.config)
