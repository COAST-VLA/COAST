"""
Evaluate every environment in a RoboCasa task set using a policy server.

Available task sets come from ``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``,
notably:
- ``atomic_seen``         — 18 atomic target tasks
- ``composite_seen``      — 16 seen composite target tasks
- ``composite_unseen``    — 16 unseen composite target tasks
- ``target50``            — atomic_seen + composite_seen + composite_unseen
- ``pretrain50`` / ``pretrain100`` / ``pretrain200`` / ``pretrain300`` — pretraining task sets

Examples:
    MUJOCO_GL=egl uv run examples/robocasa_env/eval_all.py --task_set atomic_seen
    MUJOCO_GL=egl uv run examples/robocasa_env/eval_all.py --task_set composite_seen --split target

For evaluating a single env instead, use main.py:
    MUJOCO_GL=egl uv run examples/robocasa_env/main.py --env_name CloseBlenderLid
"""

import dataclasses
import json
import logging
import os

from main import eval_task
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # Task set name. Must be a key of ``TASK_SET_REGISTRY``.
    task_set: str = "atomic_seen"
    # Dataset split: "pretrain" (in-distribution object instances) or "target" (held-out).
    split: str = "pretrain"
    # Number of episodes to run per task.
    num_episodes: int = 1
    # Override the maximum steps per episode. If None, uses 1.5 * task horizon.
    max_steps: int | None = None
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into the video output (must be keys of ``main.CAMERA_KEYS``).
    render_cameras: list[str] = dataclasses.field(
        default_factory=lambda: ["agentview_left", "agentview_right", "eye_in_hand"]
    )

    fps: int = 24
    seed: int = 7


def main(args: Args) -> None:
    if args.task_set not in TASK_SET_REGISTRY:
        raise ValueError(f"Unknown task_set '{args.task_set}'. Available: {sorted(TASK_SET_REGISTRY.keys())}")
    env_names = TASK_SET_REGISTRY[args.task_set]

    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    output_dir = os.path.join(os.path.dirname(__file__), "output", f"{args.task_set}-{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Evaluating {len(env_names)} tasks from {args.task_set} (split={args.split})")

    results: dict[str, float] = {}
    results_path = os.path.join(output_dir, "results.json")

    for env_name in tqdm(env_names, desc=f"{args.task_set}/{args.split}"):
        task_result = eval_task(env_name, policy, args, output_dir)
        results[env_name] = task_result["success_rate"]
        logger.info(f"[{env_name}] success_rate={results[env_name]:.2f}")

        # Save incrementally so progress isn't lost on early exit.
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    mean_success = float(np.mean(list(results.values()))) if results else 0.0
    summary = {
        "task_set": args.task_set,
        "split": args.split,
        "mean_success_rate": mean_success,
        "per_task": dict(sorted(results.items(), key=lambda x: x[1], reverse=True)),
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 60)
    logger.info(f"[{args.task_set}/{args.split}] mean success rate: {mean_success:.2f} ({mean_success:.0%})")
    for env_name, rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {env_name:<40s} {rate:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
