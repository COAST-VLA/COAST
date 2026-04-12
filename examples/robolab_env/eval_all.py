"""Evaluate all (or a filtered subset of) RoboLab tasks in parallel.

Each task runs as a separate ``main.py`` subprocess — each subprocess boots
its own Isaac Sim process with its own GPU context. RoboLab natively supports
``num_envs > 1`` *within* a single Isaac Sim, so each subprocess can run
multiple parallel episodes per task. The default ``--num_envs 1`` matches the
robocasa/libero flow; bump it to 4–8 for faster per-task throughput if GPU
memory allows.

Unlike robocasa/libero (which use MuJoCo + EGL), Isaac Sim subprocesses are
heavy (~35 s boot, 10+ GB VRAM each). Running more than 1–2 ``--num_workers``
concurrently will almost certainly OOM a single GPU.
**Default ``--num_workers 1``** (sequential) is recommended unless you have
multiple GPUs and assign one per subprocess via ``CUDA_VISIBLE_DEVICES``.

Usage:

    # All 120 tasks, one at a time (recommended):
    OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --headless

    # Only tasks tagged "simple":
    OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --headless --tag simple

    # Specific tasks:
    OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --headless \\
        --tasks BananaInBowlTask RubiksCubeTask

    # 4 parallel envs per task, 2 runs → 8 episodes each:
    OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --headless \\
        --num_envs 4 --num_runs 2

Output:
    examples/robolab_env/output/<output_name>/
    ├── results.json
    ├── parallel_logs/task_NNN_<TaskName>.log
    └── <TaskName>/run_NN_envNN.mp4
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import re
import subprocess
import sys
from typing import Optional

import numpy as np
import tyro

from tasks import ALL_TAGS, TASKS, tasks_with_all_tags, tasks_with_tag

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Task selection — exactly one of these should be set (or none for all).
    # --tasks: explicit task names. --tag: include tasks with this tag.
    tasks: Optional[list[str]] = None
    tag: Optional[str] = None

    instruction_type: str = "default"
    # Parallel envs inside each subprocess's Isaac Sim process.
    num_envs: int = 1
    # Sequential runs per task; total episodes = num_runs * num_envs.
    num_runs: int = 1
    max_steps: Optional[int] = None
    replan_steps: int = 8
    resize_size: int = 224
    fps: Optional[int] = None
    seed: int = 7
    device: str = "cuda:0"

    collect: bool = False

    # Concurrent subprocess limit. Isaac Sim is heavy — keep at 1 for a
    # single GPU. Increase only if you have multiple GPUs.
    num_workers: int = 1

    # Override the top-level output directory.
    output_dir: Optional[str] = None


def _resolve_task_list(args: Args) -> list[str]:
    """Return the list of task names to evaluate, based on CLI flags."""
    if args.tasks is not None:
        unknown = set(args.tasks) - set(TASKS)
        if unknown:
            raise ValueError(
                f"Unknown task names: {sorted(unknown)}. "
                f"See tasks.py for the full list."
            )
        return sorted(args.tasks)

    if args.tag is not None:
        if args.tag not in ALL_TAGS:
            raise ValueError(
                f"Unknown tag {args.tag!r}. Available: {sorted(ALL_TAGS)}"
            )
        return tasks_with_tag(args.tag)

    # No filter → all tasks.
    return sorted(TASKS.keys())


def _build_command(
    args: Args,
    task_name: str,
    output_dir: str,
) -> list[str]:
    """Build the subprocess argv for ``main.py``."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "main.py"),
        "--headless",
        "--host", args.host,
        "--port", str(args.port),
        "--task-name", task_name,
        "--instruction-type", args.instruction_type,
        "--num-envs", str(args.num_envs),
        "--num-runs", str(args.num_runs),
        "--replan-steps", str(args.replan_steps),
        "--resize-size", str(args.resize_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output-dir", output_dir,
    ]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.fps is not None:
        cmd += ["--fps", str(args.fps)]
    if args.collect:
        cmd.append("--collect")
    return cmd


def _run_one_task(
    args: Args,
    task_name: str,
    task_idx: int,
    log_dir: str,
    script_dir: str,
    output_dir: str,
) -> dict:
    """Run main.py for one task in a subprocess, return a result dict."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", task_name.strip()).strip("-")
    log_path = os.path.join(log_dir, f"task_{task_idx:03d}_{sanitized}.log")

    cmd = _build_command(args, task_name, output_dir)

    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    logger.info("[%03d] start  %s", task_idx, task_name)

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=script_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

    # Parse success_rate from the subprocess log.
    success_rate = float("nan")
    try:
        with open(log_path) as f:
            text = f.read()
        matches = re.findall(r"success_rate=([0-9.]+)", text)
        if matches:
            success_rate = float(matches[-1])
    except Exception:  # noqa: BLE001
        pass

    total_episodes = args.num_runs * args.num_envs
    logger.info(
        "[%03d] done   %-40s  success=%.2f  rc=%d",
        task_idx,
        task_name,
        success_rate,
        proc.returncode,
    )

    return {
        "task_name": task_name,
        "task_idx": task_idx,
        "tags": sorted(TASKS.get(task_name, set())),
        "instruction_type": args.instruction_type,
        "num_episodes": total_episodes,
        "success_rate": success_rate,
        "returncode": proc.returncode,
        "log_path": log_path,
    }


def _save_results(
    results: list[dict],
    results_path: str,
    args: Args,
    task_names: list[str],
    *,
    final: bool = False,
) -> None:
    """Write results.json — interim (sorted by task_idx) or final (enriched)."""
    results_sorted = sorted(results, key=lambda r: r["task_idx"])

    if not final:
        json.dump(results_sorted, open(results_path, "w"), indent=2)
        return

    valid = [
        r for r in results_sorted
        if isinstance(r["success_rate"], float)
        and not math.isnan(r["success_rate"])
    ]
    mean_success = float(np.mean([r["success_rate"] for r in valid])) if valid else 0.0

    per_task = sorted(results_sorted, key=lambda r: r["success_rate"], reverse=True)

    tag_filter = args.tag if args.tag else ("custom" if args.tasks else "all")
    payload = {
        "filter": tag_filter,
        "instruction_type": args.instruction_type,
        "num_envs": args.num_envs,
        "num_runs": args.num_runs,
        "total_episodes_per_task": args.num_runs * args.num_envs,
        "num_tasks": len(task_names),
        "mean_success_rate": round(mean_success, 4),
        "per_task": [
            {
                "task_name": r["task_name"],
                "tags": r["tags"],
                "success_rate": r["success_rate"],
                "num_episodes": r["num_episodes"],
            }
            for r in per_task
        ],
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = tyro.cli(Args)
    task_names = _resolve_task_list(args)

    if not task_names:
        logger.error("No tasks matched the given filter.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    tag_filter = args.tag if args.tag else ("custom" if args.tasks else "all")
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(
            script_dir, "output", f"{tag_filter}-{args.instruction_type}"
        )
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "parallel_logs")
    os.makedirs(log_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")

    total_episodes_per_task = args.num_runs * args.num_envs
    logger.info("=" * 60)
    logger.info(
        "RoboLab eval: %d tasks, %d episodes each "
        "(num_envs=%d × num_runs=%d), %d workers",
        len(task_names),
        total_episodes_per_task,
        args.num_envs,
        args.num_runs,
        args.num_workers,
    )
    logger.info("  filter: %s", tag_filter)
    logger.info("  output: %s", output_dir)
    logger.info("=" * 60)

    task_metadata = {name: idx for idx, name in enumerate(task_names)}
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as pool:
        futures = {
            pool.submit(
                _run_one_task,
                args,
                task_name,
                task_metadata[task_name],
                log_dir,
                script_dir,
                output_dir,
            ): task_name
            for task_name in task_names
        }

        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.exception("  %s raised an exception", task_name)
                result = {
                    "task_name": task_name,
                    "task_idx": task_metadata[task_name],
                    "tags": sorted(TASKS.get(task_name, set())),
                    "instruction_type": args.instruction_type,
                    "num_episodes": total_episodes_per_task,
                    "success_rate": float("nan"),
                    "returncode": -1,
                    "log_path": "",
                }

            results.append(result)
            _save_results(results, results_path, args, task_names, final=False)

    _save_results(results, results_path, args, task_names, final=True)

    valid = [
        r for r in results
        if isinstance(r["success_rate"], float)
        and not math.isnan(r["success_rate"])
    ]
    mean_success = float(np.mean([r["success_rate"] for r in valid])) if valid else 0.0

    logger.info("=" * 60)
    logger.info(
        "[%s/%s] mean success rate: %.2f (%.0f%%)  (%d/%d tasks valid)",
        tag_filter,
        args.instruction_type,
        mean_success,
        mean_success * 100.0,
        len(valid),
        len(results),
    )
    for r in sorted(results, key=lambda r: r["success_rate"], reverse=True):
        rate_str = f"{r['success_rate']:.2f}" if not math.isnan(r["success_rate"]) else " NaN"
        logger.info("  %-45s %s", r["task_name"], rate_str)
    logger.info("=" * 60)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
