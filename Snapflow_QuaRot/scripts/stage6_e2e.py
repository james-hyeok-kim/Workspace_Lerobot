"""Stage 6 — End-to-end integration.

Combines:
  SnapFlow (NFE=1) + QuaRot(LLM+DiT, R1-R4) + OHB + W4A4

Prerequisite artifacts:
  /data/jameskimh/james_lerobot_results/artifacts/stage1_student.safetensors  (from stage1_snapflow_distill.py)
  /data/jameskimh/james_lerobot_results/artifacts/stage4_ohb_manifest.json    (from stage4_ohb_adaln.py or auto-built)
  /data/jameskimh/james_lerobot_results/artifacts/stage0_calib_stats.pt       (from stage0_baseline.py)

If stage1 student is not yet trained, consider running Stage 5 eval
as E2E proxy (no SnapFlow, but all quant) to validate the quant pipeline.
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
    log.info(f"Stage 6 — E2E Integration. Config: {cfg_path}")

    # Pre-flight checks
    student_ckpt = recipe.snapflow.student_ckpt or "/data/jameskimh/james_lerobot_results/artifacts/stage1_student.safetensors"
    if recipe.snapflow.enabled and not Path(student_ckpt).exists():
        log.error(
            f"SnapFlow student checkpoint not found: {student_ckpt}\n"
            "Run stage1_snapflow_distill.py first, or set snapflow.enabled=false\n"
            "to evaluate Stage 6 without SnapFlow."
        )
        sys.exit(1)

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    log.info("=== Smoke Test (E2E) ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.warning("Smoke failed — proceeding with full eval anyway (check results carefully).")

    log.info("=== Full Evaluation (NFE=1, W4A4, SnapFlow+QuaRot+OHB) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    db = ResultsDB()
    db.append("stage6_e2e", result, recipe)

    agg = result.get("aggregated", {})
    lat = result.get("latency", {})
    log.info(f"\n{'='*60}")
    log.info(f"[Stage 6 E2E]  pc_success  = {agg.get('pc_success', 'N/A'):.1f}%")
    log.info(f"[Stage 6 E2E]  latency_p50 = {lat.get('p50_ms', 'N/A'):.1f} ms")
    log.info(f"[Stage 6 E2E]  NFE = {recipe.policy.num_inference_steps}")
    log.info(f"[Stage 6 E2E]  See: {recipe.output_dir}/libero_10.json")
    log.info(f"{'='*60}")
    log.info("Full leaderboard: cat results/leaderboard.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage6_e2e.yaml")
    args = parser.parse_args()
    main(args.config)
