"""Stage 1 — Evaluate SnapFlow student (1-NFE).

Requires artifacts/stage1_student.safetensors to exist.
Run stage1_snapflow_distill.py first.
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
    log.info(f"Stage 1 — SnapFlow student eval. Config: {cfg_path}")

    ckpt = recipe.snapflow.student_ckpt or "artifacts/stage1_student.safetensors"
    if not Path(ckpt).exists():
        log.error(
            f"Student checkpoint not found: {ckpt}\n"
            "Run: python scripts/stage1_snapflow_distill.py --config configs/stage1_snapflow.yaml"
        )
        sys.exit(1)

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    log.info("=== Smoke Test ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke failed — aborting.")
        sys.exit(1)

    log.info("=== Full Evaluation (NFE=1) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    db = ResultsDB()
    db.append("stage1_snapflow", result, recipe)

    agg = result.get("aggregated", {})
    log.info(f"[Stage 1] pc_success={agg.get('pc_success', 'N/A'):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_snapflow.yaml")
    args = parser.parse_args()
    main(args.config)
