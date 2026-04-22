"""Stage 2 — QuaRot on LLM only (R1, R2; no DiT, no quant).

Applies offline Hadamard rotation to PaliGemma LLM backbone.
DiT (action expert) stays in FP16.
Expected result: pc_success ≈ Stage 0 (rotation is FP-lossless).
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
    from common.calib_capture import capture_calib_stats, save_calib_stats
    from analysis.activation_stats import load_and_plot
    from lerobot.envs.configs import LiberoEnv

    recipe = Recipe.from_yaml(cfg_path)
    log.info(f"Stage 2 — QuaRot LLM only. Config: {cfg_path}")

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    log.info("=== Smoke Test ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke failed — rotation may have broken the model.")
        sys.exit(1)

    # Capture post-rotation activation stats for comparison
    env_cfg = LiberoEnv(task=recipe.eval.task)
    stats = capture_calib_stats(policy, env_cfg, n_chunks=64, device=recipe.eval.device)
    stage_stats_path = "artifacts/stage2_calib_stats.pt"
    if stats:
        save_calib_stats(stats, stage_stats_path)

    log.info("=== Full Evaluation (NFE=10, LLM rotated) ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    # Plot delta vs Stage 0
    if stats and Path(recipe.calib_stats_path).exists():
        from analysis.plot_delta import plot_delta_stats
        plots_dir = Path(recipe.output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_delta_stats(
            recipe.calib_stats_path, stage_stats_path,
            str(plots_dir / "delta_max_abs.png"),
            metric="max_abs", label_a="FP16", label_b="QuaRot-LLM",
        )

    db = ResultsDB()
    db.append("stage2_quarot_llm", result, recipe)
    log.info(f"[Stage 2] pc_success={result['aggregated'].get('pc_success', 'N/A'):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_quarot_llm.yaml")
    args = parser.parse_args()
    main(args.config)
