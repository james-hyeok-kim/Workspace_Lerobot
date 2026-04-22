"""Stage 0 — FP16 Baseline.

Actions:
1. Load pi0.5 (FP16, NFE=10, n_action_steps=10).
2. Run LIBERO-10 evaluation (100 episodes).
3. Collect calibration stats (64 chunks) → artifacts/stage0_calib_stats.pt
4. Plot activation heatmaps → results/stage0/plots/
5. Append to leaderboard.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Path setup
_root = Path(__file__).resolve().parents[2]
for _p in [str(_root / "Snapflow_QuaRot"), str(_root / "lerobot" / "src")]:
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
    log.info(f"Stage 0 — Baseline. Config: {cfg_path}")

    policy, pre, post, env_pre, env_post = load_policy(recipe)

    # 1. Smoke test
    log.info("=== Smoke Test ===")
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke test failed — aborting Stage 0.")
        sys.exit(1)

    # 2. Full evaluation
    log.info("=== Full LIBERO-10 Evaluation ===")
    result = run_eval(policy, pre, post, env_pre, env_post, recipe)

    # 3. Calibration stat capture
    log.info("=== Calibration Stat Capture ===")
    env_cfg = LiberoEnv(task=recipe.eval.task)
    stats = capture_calib_stats(
        policy=policy,
        env_cfg=env_cfg,
        n_chunks=64,
        device=recipe.eval.device,
    )
    if stats:
        save_calib_stats(stats, recipe.calib_stats_path)

    # 4. Activation heatmaps
    if stats:
        plots_dir = Path(recipe.output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        load_and_plot(recipe.calib_stats_path, str(plots_dir), stage="stage0")

    # 5. Leaderboard
    db = ResultsDB()
    db.append("stage0_baseline", result, recipe)

    agg = result.get("aggregated", {})
    log.info(f"\n{'='*50}")
    log.info(f"[Stage 0] pc_success={agg.get('pc_success', 'N/A'):.1f}%")
    log.info(f"[Stage 0] avg_sum_reward={agg.get('avg_sum_reward', 'N/A'):.4f}")
    log.info(f"[Stage 0] Results → {recipe.output_dir}/libero_10.json")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage0_baseline.yaml")
    args = parser.parse_args()
    main(args.config)
