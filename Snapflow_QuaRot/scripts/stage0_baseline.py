"""Stage 0 — FP16 Baseline (2-GPU parallel on GPU 2,3).

Actions:
1. Load pi0.5 (FP16, NFE=10, n_action_steps=50).
2. Smoke test on GPU 2.
3. Run LIBERO-10 eval in parallel across GPU 2 and 3 (50 episodes each).
4. Collect calibration stats (64 chunks, GPU 2) → artifacts/stage0_calib_stats.pt
5. Plot activation heatmaps → results/stage0/plots/
6. Append to leaderboard.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[2]
for _p in [
    str(_root / "Snapflow_QuaRot"),
    str(_root / "lerobot" / "src"),
    str(_root / "TensorRT-Model-Optimizer"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


_DEFAULT_GPU_IDS = [2, 3]


def main(cfg_path: str, n_gpus: int = 2):
    from common.recipe import Recipe
    from common.policy_loader import load_policy
    from common.smoke import run_smoke
    from common.parallel_eval import run_parallel_eval
    from common.results_db import ResultsDB
    from common.calib_capture import capture_calib_stats, save_calib_stats
    from lerobot.envs.configs import LiberoEnv

    recipe = Recipe.from_yaml(cfg_path)
    recipe.output_dir = str(Path(recipe.output_dir).resolve())
    recipe.calib_stats_path = str(Path(recipe.calib_stats_path).resolve())
    log.info(f"Stage 0 — Baseline ({n_gpus}-GPU parallel). Config: {cfg_path}")
    log.info(f"  NFE={recipe.policy.num_inference_steps}, n_action_steps={recipe.policy.n_action_steps}")
    log.info(f"  n_episodes={recipe.eval.n_episodes}, task={recipe.eval.task}")

    # 1. Smoke test (single GPU 2)
    log.info("=" * 60)
    log.info("STEP 1/4: Smoke Test (GPU 2)")
    log.info("=" * 60)
    import os
    # GPU 0 for EGL (required for headless rendering); GPU 2 for CUDA inference (cuda:1).
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["LD_LIBRARY_PATH"] = (
        "/home/jovyan/egl_libs:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    recipe.eval.device = "cuda:1"
    policy, pre, post, env_pre, env_post = load_policy(recipe)
    ok = run_smoke(policy, pre, post, env_pre, env_post, recipe)
    if not ok:
        log.error("Smoke test FAILED — aborting Stage 0.")
        sys.exit(1)
    log.info("[Smoke] PASSED")

    # 2. Parallel eval across 4 GPUs
    log.info("=" * 60)
    log.info(f"STEP 2/4: Full LIBERO-10 Eval ({n_gpus} GPUs × {recipe.eval.n_episodes // n_gpus} episodes)")
    log.info("=" * 60)
    result = run_parallel_eval(recipe, n_gpus=n_gpus, gpu_ids=_DEFAULT_GPU_IDS[:n_gpus])

    # 3. Calibration stat capture (on GPU 2 sequentially)
    log.info("=" * 60)
    log.info("STEP 3/4: Calibration Stat Capture (GPU 2, 64 chunks)")
    log.info("=" * 60)
    # Reload policy for fresh GPU 0 session (previous may be freed)
    del policy, pre, post, env_pre, env_post
    import torch; torch.cuda.empty_cache()

    policy, pre, post, env_pre, env_post = load_policy(recipe)
    env_cfg = LiberoEnv(task=recipe.eval.task)
    stats = capture_calib_stats(
        policy=policy,
        env_cfg=env_cfg,
        n_chunks=64,
        device=recipe.eval.device,
    )
    if stats:
        save_calib_stats(stats, recipe.calib_stats_path)
        log.info(f"Calib stats → {recipe.calib_stats_path}")

    # 4. Plots
    log.info("=" * 60)
    log.info("STEP 4/4: Activation Heatmaps")
    log.info("=" * 60)
    if stats:
        from analysis.activation_stats import load_and_plot
        plots_dir = Path(recipe.output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        load_and_plot(recipe.calib_stats_path, str(plots_dir), stage="stage0")
        log.info(f"Plots → {plots_dir}")

    # 5. Leaderboard
    db = ResultsDB()
    db.append("stage0_baseline", result, recipe)

    agg = result.get("aggregated", {})
    lat = result.get("latency", {})
    log.info("=" * 60)
    log.info(f"[Stage 0] pc_success    = {agg.get('pc_success', 'N/A'):.1f}%")
    log.info(f"[Stage 0] avg_sum_reward = {agg.get('avg_sum_reward', 'N/A'):.4f}")
    log.info(f"[Stage 0] latency_p50   = {lat.get('p50_ms', 'N/A')} ms")
    log.info(f"[Stage 0] n_gpus used   = {n_gpus}")
    log.info(f"[Stage 0] Results       → {recipe.output_dir}/libero_10.json")
    log.info("=" * 60)
    log.info("Leaderboard: cat results/leaderboard.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage0_baseline.yaml")
    parser.add_argument("--n_gpus", type=int, default=2)
    args = parser.parse_args()
    main(args.config, n_gpus=args.n_gpus)
