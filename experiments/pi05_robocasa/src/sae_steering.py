#!/usr/bin/env python3
"""SAE-ActAdd steering baseline for pi0.5 RoboCasa.

Loads v_sae from $OPENPI_DATA_HOME/robocasa_pi05_sae_vectors.npz (built by
experiments/sae/src/fit_sae_vectors.py) and runs steered eval reusing the
LinearSteeringHook already defined in conceptor_steering.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys
from typing import List

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conceptor_steering import (  # noqa: E402
    ROBOCASA_TASKS, LinearSteeringHook,
    SteeredPolicyWrapper, run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
DEFAULT_SAE_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "robocasa_pi05_sae_vectors.npz"


@dataclasses.dataclass
class Args:
    task: str = ROBOCASA_TASKS[0]
    config: str = "pi05_robocasa"
    checkpoint_dir: str = "checkpoints/pi05_pretrain_human300/multitask_learning/75000"
    layer: int = 11
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    sae_vectors_npz: str = str(DEFAULT_SAE_NPZ)
    num_episodes: int = 15
    port: int = 8000
    output_dir: str = "experiments/pi05_robocasa/sae_steering_results"
    skip_baseline: bool = False


def main(args: Args):
    task_output_dir = pathlib.Path(args.output_dir) / args.task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Task: %s  Output: %s", args.task, task_output_dir)

    sae_path = pathlib.Path(args.sae_vectors_npz)
    if not sae_path.is_file():
        sys.exit(f"sae_vectors_npz not found: {sae_path}")
    key = f"{args.task}__L{args.layer}__sae__V_contrastive"
    with np.load(sae_path) as nv:
        if key not in nv.files:
            sys.exit(f"key not in NPZ: {key}")
        v_sae = np.asarray(nv[key], dtype=np.float32)
    logger.info("Loaded v_sae  key=%s  ||v||=%.4f", key, float(np.linalg.norm(v_sae)))

    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    summary_path = task_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f).get("conditions", [])
        all_results = [r for r in existing if r is not None]
        done = {r["condition"] for r in all_results}
    else:
        all_results, done = [], set()

    def save_progress():
        sorted_r = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
        tmp = summary_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"task": args.task, "conditions": sorted_r}, f, indent=2)
        tmp.replace(summary_path)

    logger.info("Loading policy...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001

    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    if not args.skip_baseline and "baseline" not in done:
        logger.info("\n[1] Baseline...")
        wrapper.update_hooks(None)
        all_results.append(run_condition(
            args.task, args.port, args.num_episodes, "baseline", task_output_dir,
        ))
        done.add("baseline"); save_progress()

    logger.info("\n[2] SAE steering (α sweep)")
    for la in args.alphas:
        cond = f"sae_L{args.layer}_la{la}"
        if cond in done:
            continue
        hook = LinearSteeringHook(v_sae, alpha=float(la), device=device)
        wrapper.update_hooks([(args.layer, hook)])
        all_results.append(run_condition(
            args.task, args.port, args.num_episodes, cond, task_output_dir,
        ))
        done.add(cond); save_progress()

    logger.info("Saved → %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
