"""4-GPU parallel evaluation harness (task-based).

Distributes LIBERO-10 task_ids across N GPUs.
Each subprocess handles a subset of tasks and runs n_episodes_per_task episodes per task.
Main process aggregates pc_success and latency across all shards.

Usage (from stage scripts):
    from common.parallel_eval import run_parallel_eval
    result = run_parallel_eval(recipe, n_gpus=4)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_N_LIBERO10_TASKS = 10  # LIBERO-10 always has 10 tasks (indices 0..9)


def run_parallel_eval(
    recipe,
    n_gpus: int = 4,
    smoke_only: bool = False,
    gpu_ids: list[int] | None = None,
) -> dict:
    """Run eval_policy_all in parallel across n_gpus GPUs.

    Distributes task_ids [0..9] across GPUs.  Each GPU runs all its assigned
    tasks for n_episodes_per_task episodes (default 10).

    Args:
        gpu_ids: Physical GPU IDs to use (e.g. [2, 3]).  Defaults to [0..n_gpus-1].

    Returns aggregated results dict with 'aggregated' and 'latency' keys.
    """
    if gpu_ids is None:
        gpu_ids = list(range(n_gpus))
    n_gpus = len(gpu_ids)

    if smoke_only:
        # Smoke: 1 task, 1 episode on first available GPU
        task_shards = [(gpu_ids[0], [0])]
        n_ep = 1
    else:
        all_tasks = list(range(_N_LIBERO10_TASKS))
        # Distribute tasks round-robin across GPUs
        task_shards = []
        for i, gpu_id in enumerate(gpu_ids):
            assigned = all_tasks[i::n_gpus]  # stride by n_gpus
            if assigned:
                task_shards.append((gpu_id, assigned))
        n_ep = recipe.eval.n_episodes_per_task

    log.info(
        f"Parallel eval: {_N_LIBERO10_TASKS} tasks across {len(task_shards)} GPUs "
        f"({n_ep} ep/task)"
    )
    for gpu_id, tasks in task_shards:
        log.info(f"  GPU{gpu_id}: tasks={tasks}")

    import yaml

    result_files = []
    procs = []

    for gpu_id, task_ids in task_shards:
        result_path = tempfile.mktemp(suffix=f"_gpu{gpu_id}.json")
        result_files.append(result_path)

        cfg_override = _recipe_to_dict(recipe)
        cfg_override["eval"]["task_ids"] = task_ids
        cfg_override["eval"]["n_episodes_per_task"] = n_ep
        cfg_override["eval"]["device"] = "cuda:1"  # cuda:1=gpu_id within CUDA_VISIBLE_DEVICES=0,gpu_id
        cfg_override["output_dir"] = str(Path(recipe.output_dir) / f"gpu{gpu_id}")

        tmp_cfg = tempfile.mktemp(suffix=".yaml")
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg_override, f)

        worker_script = Path(__file__).parent / "_eval_worker.py"
        env = os.environ.copy()
        # EGL requires GPU 0 for device initialization; CUDA inference runs on gpu_id.
        # We expose GPU 0 (EGL) and the target GPU via CUDA_VISIBLE_DEVICES.
        # Within the subprocess cuda:0=GPU0(EGL only), cuda:1=gpu_id(inference).
        env["CUDA_VISIBLE_DEVICES"] = f"0,{gpu_id}"
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"

        lerobot_src = str(Path(__file__).resolve().parents[3] / "lerobot" / "src")
        modelopt_dir = str(Path(__file__).resolve().parents[3] / "TensorRT-Model-Optimizer")
        sf_dir = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = f"{sf_dir}:{lerobot_src}:{modelopt_dir}:{env.get('PYTHONPATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"/home/jovyan/egl_libs:/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["LIBERO_DATASET_PATH"] = env.get(
            "LIBERO_DATASET_PATH", "/data/jameskimh/james_libero_datasets"
        )

        cmd = [sys.executable, str(worker_script), tmp_cfg, result_path]
        log.info(f"  Spawning GPU{gpu_id}: tasks={task_ids}, n_ep={n_ep}")
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)

    exit_codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(exit_codes) if c != 0]
    if failed:
        log.error(f"GPU shards {failed} exited with non-zero code.")

    return _aggregate_results(result_files, recipe)


def _aggregate_results(result_files: list[str], recipe) -> dict:
    """Merge per-shard JSON results into a single aggregated result."""
    all_successes = []
    all_rewards = []
    all_max_rewards = []
    all_latencies = []

    for rfile in result_files:
        if not Path(rfile).exists():
            log.warning(f"Result file not found: {rfile}")
            continue
        with open(rfile) as f:
            r = json.load(f)

        agg = r.get("aggregated", {})
        if "pc_success" in agg and agg["pc_success"] is not None:
            all_successes.append(agg["pc_success"])
        if "avg_sum_reward" in agg and agg["avg_sum_reward"] is not None:
            all_rewards.append(agg["avg_sum_reward"])
        if "avg_max_reward" in agg and agg["avg_max_reward"] is not None:
            all_max_rewards.append(agg["avg_max_reward"])

        lat = r.get("latency", {})
        if lat.get("p50_ms") is not None:
            all_latencies.append(lat["p50_ms"])

    aggregated = {
        "pc_success": float(np.mean(all_successes)) if all_successes else None,
        "avg_sum_reward": float(np.mean(all_rewards)) if all_rewards else None,
        "avg_max_reward": float(np.mean(all_max_rewards)) if all_max_rewards else None,
    }
    latency = {
        "p50_ms": float(np.mean(all_latencies)) if all_latencies else None,
        "p95_ms": None,
    }

    result = {
        "config": {
            "stage": recipe.name,
            "n_episodes": recipe.eval.n_episodes,
            "nfe": recipe.policy.num_inference_steps,
            "n_action_steps": recipe.policy.n_action_steps,
            "n_gpus": len(result_files),
            "snapflow": recipe.snapflow.enabled,
            "quarot": recipe.quarot.enabled,
            "ohb": recipe.ohb.enabled,
            "w4a4": recipe.w4a4.enabled,
        },
        "aggregated": aggregated,
        "latency": latency,
    }

    out_dir = Path(recipe.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "libero_10.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    pc = aggregated.get("pc_success")
    pc_str = f"{pc:.1f}%" if pc is not None else "N/A"
    lat_str = latency.get("p50_ms")
    log.info(
        f"[{recipe.name}] AGGREGATED: pc_success={pc_str}  "
        f"latency_p50={lat_str}"
    )
    return result


def _recipe_to_dict(recipe) -> dict:
    """Serialize Recipe to plain dict for YAML serialization."""
    import dataclasses
    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj
    return to_dict(recipe)
