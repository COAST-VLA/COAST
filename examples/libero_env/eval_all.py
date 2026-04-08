"""
Evaluate every task in a LIBERO suite using a policy server.

Examples:
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial
    MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_10 --num_episodes 10
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm

from collection_session import CollectionSession
from main import eval_task, get_task_suite

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # LIBERO suite name.
    task_suite_name: str = "libero_spatial"
    # Number of episodes / initial states per task.
    num_episodes: int = 2
    # Override the suite default max steps. If None, uses main.SUITE_MAX_STEPS.
    max_steps: Optional[int] = None
    # Number of settling steps before policy actions.
    num_steps_wait: int = 10
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 5

    # Image resize size for the policy input.
    resize_size: int = 224

    # Cameras to tile into the video output.
    render_cameras: List[str] = dataclasses.field(
        default_factory=lambda: ["agentview", "eye_in_hand"]
    )

    fps: int = 10
    seed: int = 7

    # If True, attach activation-collection metadata to every infer call so the
    # server (started with --collect_activations) saves intermediates to its disk.
    collect: bool = False


def main(args: Args) -> None:
    np.random.seed(args.seed)

    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info("Server metadata: %s", policy.get_server_metadata())

    task_suite = get_task_suite(args.task_suite_name)
    output_dir = os.path.join(os.path.dirname(__file__), "output", args.task_suite_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Evaluating %d tasks from %s", task_suite.n_tasks, args.task_suite_name)

    collect_session = CollectionSession(policy) if args.collect else None

    results = []  # type: List[Dict[str, object]]
    results_path = os.path.join(output_dir, "results.json")

    for task_id in tqdm(range(task_suite.n_tasks), desc=args.task_suite_name):
        task_result = eval_task(
            args.task_suite_name,
            task_id,
            policy,
            args,
            output_dir,
            collect_session=collect_session,
        )
        task_summary = {
            "task_id": task_id,
            "task_name": task_result["task_name"],
            "task_description": task_result["task_description"],
            "success_rate": task_result["success_rate"],
        }
        results.append(task_summary)
        logger.info(
            "[task_%02d/%s] success_rate=%.2f",
            task_id,
            task_result["task_name"],
            task_result["success_rate"],
        )

        # Save incrementally so progress isn't lost on early exit.
        with open(results_path, "w") as file_handle:
            json.dump(results, file_handle, indent=2)

    mean_success = (
        float(np.mean([task["success_rate"] for task in results])) if results else 0.0
    )
    summary = {
        "task_suite_name": args.task_suite_name,
        "mean_success_rate": mean_success,
        "per_task": sorted(
            results, key=lambda item: item["success_rate"], reverse=True
        ),
    }
    with open(results_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    logger.info("Results saved to %s", results_path)
    logger.info("=" * 60)
    logger.info(
        "[%s] mean success rate: %.2f (%.0f%%)",
        args.task_suite_name,
        mean_success,
        mean_success * 100.0,
    )
    for task in summary["per_task"]:
        logger.info(
            "  task_%02d %-35s %.2f",
            task["task_id"],
            task["task_name"],
            task["success_rate"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
