"""
Evaluate all tasks in an ML45 split (train or test).

Normal eval (WebSocket server):
    MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

Activation collection, single GPU:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \\
        --collect --split train --num_envs 16 \\
        --policy.config=pi05_metaworld \\
        --policy.dir=checkpoints/openpi-metaworld-5000

Activation collection, multi-GPU (round-robin task sharding):
    MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \\
        --collect --split train --num_envs 16 --gpus 0 1 \\
        --policy.config=pi05_metaworld \\
        --policy.dir=checkpoints/openpi-metaworld-5000
"""

import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import sys
from typing import Literal

# Reuse helpers from main.py so single-task and multi-task eval cannot diverge.
# main.py is a sibling module; running via "uv run examples/metaworld/eval_all.py"
# puts its directory on sys.path[0], so "from main import ..." resolves correctly.
from main import CAMERA_IDS  # noqa: F401
from main import TASK_TO_PROMPT  # noqa: F401
from main import Args as _MainArgs
from main import MetaworldCollectState  # noqa: F401
from main import MultiCameraWrapper  # noqa: F401
from main import PolicyArgs
from main import load_policy
from main import make_env
from main import run_episode
from main import tile_frames  # noqa: F401
import metaworld
import numpy as np
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)


# Curated subset of 10 ML45-train tasks (tasks whose success rate varies
# meaningfully across training checkpoints — used for faster iteration).
SUBSET = [
    "assembly-v3",
    "disassemble-v3",
    "hammer-v3",
    "handle-pull-side-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "stick-push-v3",
]


@dataclasses.dataclass
class Args:
    # WebSocket server (ignored when --collect is set).
    host: str = "0.0.0.0"
    port: int = 8000

    # Which ML45 split or curated subset to evaluate (ignored if --tasks is non-empty).
    split: Literal["train", "test", "subset"] = "subset"
    # Subset of task names to evaluate. If empty, uses --split.
    tasks: list[str] = dataclasses.field(default_factory=list)

    # Number of parallel environments per task.
    num_envs: int = 15
    # Number of episodes per task.
    num_episodes: int = 1
    # Maximum steps per episode.
    max_steps: int = 300
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 10

    width: int = 224
    height: int = 224

    # Cameras to use for policy input.
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    # The camera used for rendering the video output (must be one of the policy cameras).
    render_camera: str = "corner"

    fps: int = 24
    seed: int = 69_420

    # Override the eval-artifact directory (videos, results.json). If None, defaults to
    # ``examples/metaworld/output/ML45-{split}/``. Relative paths are resolved against
    # the user's shell cwd, matching the libero and robocasa examples.
    output_dir: str | None = None

    # --- Activation collection ---
    # When True, load the policy in-process (no WebSocket) and save intermediate
    # activations during rollout. Requires a PyTorch checkpoint.
    collect: bool = False
    # In-process policy config and checkpoint. Only consulted when --collect.
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)
    # Root directory for saved activations; see main.py for the full path template.
    collect_output_dir: str = "./activations"
    # Round-robin task sharding across GPUs. Each subprocess loads its own policy
    # copy on one GPU. Only valid with --collect. Use CUDA_VISIBLE_DEVICES to pin
    # a single GPU for the normal-eval path.
    gpus: list[int] = dataclasses.field(default_factory=list)


def _resolve_tasks(args: Args) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    if args.split == "subset":
        return list(SUBSET)
    ml45 = metaworld.ML45()
    return list(ml45.train_classes.keys()) if args.split == "train" else list(ml45.test_classes.keys())


def _per_task_args(env_name: str, args: Args, task_output_dir: str) -> _MainArgs:
    """Build a ``main.Args`` for ``run_episode`` to consume for a given task."""
    return _MainArgs(
        host=args.host,
        port=args.port,
        env_name=env_name,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        width=args.width,
        height=args.height,
        policy_cameras=list(args.policy_cameras),
        render_camera=args.render_camera,
        fps=args.fps,
        seed=args.seed,
        output_dir=task_output_dir,
        collect=args.collect,
        policy=args.policy,
        collect_output_dir=args.collect_output_dir,
    )


def eval_task(env_name: str, policy, args: Args, output_dir: str, collect_extras: dict) -> dict[str, float]:
    """Evaluate a single task over ``num_episodes`` and return mean success rate."""
    env = make_env(
        env_name=env_name,
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_names=args.policy_cameras,
    )

    task_output_dir = os.path.join(output_dir, env_name)
    os.makedirs(task_output_dir, exist_ok=True)

    task_args = _per_task_args(env_name, args, task_output_dir)

    episode_success_rates: list[float] = []
    try:
        for episode in range(args.num_episodes):
            _, success = run_episode(env, policy, task_args, episode, task_output_dir, collect_extras)
            episode_success_rates.append(float(success.mean()))
    finally:
        env.close()

    return {"success_rate": float(np.mean(episode_success_rates))}


def run_local(args: Args) -> None:
    """Single-process task loop (normal eval, or single-GPU / pinned-GPU collection)."""
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "output", f"ML45-{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    env_names = _resolve_tasks(args)
    logger.info(f"Evaluating {len(env_names)} tasks")


    policy, collect_extras = load_policy(args)

    results_path = os.path.join(output_dir, "results.json")
    results: dict[str, float] = {}
    for env_name in tqdm(env_names, desc=f"ML45-{args.split}"):
        task_result = eval_task(env_name, policy, args, output_dir, collect_extras)
        results[env_name] = task_result["success_rate"]
        logger.info(f"[{env_name}] success_rate={results[env_name]:.2f}")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    mean_success = float(np.mean(list(results.values()))) if results else 0.0
    summary = {
        "mean_success_rate": mean_success,
        "per_task": dict(sorted(results.items(), key=lambda x: x[1], reverse=True)),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 60)
    logger.info(f"Overall mean success rate: {mean_success:.2f} ({mean_success:.0%})")
    logger.info("Per-task results:")
    for env_name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {env_name:<40s} {rate:.2f}")

    if args.collect:
        logger.info(f"Activations saved under {collect_extras['base_output_dir']}")


def run_multi_gpu(args: Args) -> None:
    """Launch one subprocess per GPU, each running this script on a subset of tasks.

    ``bash -c`` is used so that ``CUDA_VISIBLE_DEVICES`` is exported before any
    Python/CUDA initialization in the subprocess.
    """
    tasks = _resolve_tasks(args)

    # Ensure PyTorch checkpoint exists before spawning; avoids N subprocesses
    # racing on the one-time conversion step.
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

    ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)

    task_chunks: list[list[str]] = [[] for _ in args.gpus]
    for i, task in enumerate(tasks):
        task_chunks[i % len(args.gpus)].append(task)

    log_dir = pathlib.Path(args.collect_output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for gpu_id, chunk in zip(args.gpus, task_chunks, strict=True):
        logger.info(f"GPU {gpu_id}: {len(chunk)} tasks — {chunk}")

    processes: list[tuple[int, subprocess.Popen, object]] = []
    for gpu_id, chunk in zip(args.gpus, task_chunks, strict=True):
        if not chunk:
            continue
        tasks_str = " ".join(chunk)
        inner_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} MUJOCO_GL=egl"
            f" {sys.executable} {__file__}"
            f" --collect"
            f" --policy.config={args.policy.config}"
            f" --policy.dir={args.policy.dir}"
            f" --num_envs={args.num_envs}"
            f" --num_episodes={args.num_episodes}"
            f" --max_steps={args.max_steps}"
            f" --replan_steps={args.replan_steps}"
            f" --collect_output_dir={args.collect_output_dir}"
            f" --seed={args.seed}"
            f" --split={args.split}"
            f" --gpus {gpu_id}"
            f" --tasks {tasks_str}"
        )
        log_file = log_dir / f"gpu_{gpu_id}.log"
        log_fh = open(log_file, "w")  # noqa: SIM115
        logger.info(f"Starting GPU {gpu_id} (log: {log_file})")
        proc = subprocess.Popen(["bash", "-c", inner_cmd], stdout=log_fh, stderr=log_fh)
        processes.append((gpu_id, proc, log_fh))

    for gpu_id, proc, log_fh in processes:
        proc.wait()
        log_fh.close()
        if proc.returncode != 0:
            logger.error(f"GPU {gpu_id} exited with code {proc.returncode}. See {log_dir / f'gpu_{gpu_id}.log'}")
        else:
            logger.info(f"GPU {gpu_id} completed successfully")


def main(args: Args) -> None:
    if args.gpus and not args.collect:
        raise ValueError(
            "--gpus is only valid with --collect. For normal eval, pin a single GPU with CUDA_VISIBLE_DEVICES instead."
        )

    if len(args.gpus) > 1:
        run_multi_gpu(args)
        return

    if len(args.gpus) == 1:
        # Pin before any torch import (load_policy does lazy import inside run_local).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    run_local(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
