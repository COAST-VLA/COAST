#!/usr/bin/env python3
"""SAE-ActAdd steering baseline for pi0-FAST MetaWorld.

Loads v_sae from $OPENPI_DATA_HOME/pi0fast_metaworld_sae_vectors.npz (built by
experiments/sae/src/fit_sae_vectors.py) and runs steered eval mirroring the
linear_final path in conceptor_steering.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys
from typing import List

import jax.numpy as jnp
import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conceptor_steering import (  # noqa: E402
    METAWORLD_TASKS,
    SteeredFastPolicyWrapper, run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
DEFAULT_SAE_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "pi0fast_metaworld_sae_vectors.npz"


@dataclasses.dataclass
class Args:
    task: str = METAWORLD_TASKS[0]
    config: str = "pi0_fast_metaworld"
    checkpoint_dir: str = "checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500/"
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    sae_vectors_npz: str = str(DEFAULT_SAE_NPZ)
    num_envs: int = 16
    num_episodes: int = 16
    max_steps: int = 500
    replan_steps: int = 5
    seed: int = 7
    port: int = 8000
    max_decoding_steps: int = 256
    output_dir: str = "experiments/pi0_fast_metaworld/sae_steering_results"
    skip_baseline: bool = False


def main(args: Args):
    task_output_dir = pathlib.Path(args.output_dir) / args.task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Task: %s  Output: %s", args.task, task_output_dir)

    sae_path = pathlib.Path(args.sae_vectors_npz)
    if not sae_path.is_file():
        sys.exit(f"sae_vectors_npz not found: {sae_path}")
    with np.load(sae_path) as nv:
        key = f"{args.task}__sae__V_contrastive"
        if key not in nv.files:
            sys.exit(f"key not in NPZ: {key}")
        v_sae = np.asarray(nv[key], dtype=np.float32)
    logger.info("Loaded v_sae from %s  (||v||=%.4f)", sae_path, float(np.linalg.norm(v_sae)))

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
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    wrapper = SteeredFastPolicyWrapper(policy)
    start_server_background(wrapper, args.port)

    d = v_sae.shape[-1]
    identity_stack = np.broadcast_to(np.eye(d, dtype=np.float32), (3, d, d))
    C_stack_dev = jnp.asarray(identity_stack)
    step_idx_dev = jnp.zeros((args.max_decoding_steps,), dtype=jnp.int32)
    v_dev = jnp.asarray(v_sae)

    def _run(name):
        return run_condition(
            args.task, args.port, args.num_envs, args.num_episodes,
            name, task_output_dir, seed=args.seed,
            max_steps=args.max_steps, replan_steps=args.replan_steps,
        )

    if not args.skip_baseline and "baseline" not in done:
        logger.info("\n[1] Baseline...")
        wrapper.disarm()
        all_results.append(_run("baseline"))
        done.add("baseline"); save_progress()

    logger.info("\n[2] SAE steering (α sweep)")
    for la in args.alphas:
        cond = f"sae_la{la}"
        if cond in done:
            continue
        wrapper.arm(C_stack_dev, 0.0, step_idx_dev, add_bias=v_dev, add_alpha=float(la))
        all_results.append(_run(cond))
        done.add(cond); save_progress()

    logger.info("Saved → %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
