#!/usr/bin/env python3
"""SAE-ActAdd steering baseline for pi0-FAST LIBERO.

Loads v_sae from $OPENPI_DATA_HOME/pi0fast_libero_sae_vectors.npz (built by
experiments/sae/src/fit_sae_vectors.py) and runs steered eval with one α
condition at a time, mirroring the linear_final path in conceptor_steering.py.

The injected vector is ``v_sae``; it slots into the existing JAX
``infer_with_steering_fast`` ``add_bias`` / ``add_alpha`` channel — no
JIT-cache divergence, no new model code.

Usage:
    uv run experiments/pi0_fast_libero/src/sae_steering.py \\
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \\
        --alphas 0.5 1.0
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

# Reuse infra from the existing conceptor_steering driver (server, eval client,
# wrapper). Keeping import path local to this experiment so SLURM jobs work.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conceptor_steering import (  # noqa: E402
    LIBERO_TASK_IDS, LIBERO_TASKS,
    SteeredFastPolicyWrapper, run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
DEFAULT_SAE_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "pi0fast_libero_sae_vectors.npz"


@dataclasses.dataclass
class Args:
    task: str = LIBERO_TASKS[0]
    config: str = "pi0_fast_libero"
    checkpoint_dir: str = "checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000/"
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    sae_vectors_npz: str = str(DEFAULT_SAE_NPZ)
    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000
    output_dir: str = "experiments/pi0_fast_libero/sae_steering_results"
    max_decoding_steps: int = 256
    skip_baseline: bool = False


def main(args: Args):
    task_short = args.task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Task: %s", args.task)
    logger.info("Output: %s", task_output_dir)

    sae_path = pathlib.Path(args.sae_vectors_npz)
    if not sae_path.is_file():
        sys.exit(f"sae_vectors_npz not found: {sae_path}")
    with np.load(sae_path) as nv:
        key = f"{args.task}__sae__V_contrastive"
        if key not in nv.files:
            sys.exit(f"key not in NPZ: {key}")
        v_sae = np.asarray(nv[key], dtype=np.float32)
    logger.info("Loaded v_sae from %s  (||v||=%.4f, expected ~1.0)", sae_path, float(np.linalg.norm(v_sae)))

    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    # Resume support: existing summary.json
    summary_path = task_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f).get("conditions", [])
        all_results = [r for r in existing if r is not None]
        done = {r["condition"] for r in all_results}
    else:
        all_results = []
        done = set()

    def save_progress():
        sorted_results = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
        tmp = summary_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"task": args.task, "conditions": sorted_results}, f, indent=2)
        tmp.replace(summary_path)

    # Load policy ONCE.
    logger.info("Loading policy (one-time cost)...")
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    logger.info("Policy loaded.")

    wrapper = SteeredFastPolicyWrapper(policy)
    start_server_background(wrapper, args.port)

    # Pad C_stack to identity so JIT signature matches the conceptor sweeps;
    # β=0 means the matrix term contributes nothing.
    d = v_sae.shape[-1]
    identity_stack = np.broadcast_to(np.eye(d, dtype=np.float32), (3, d, d))
    C_stack_dev = jnp.asarray(identity_stack)
    step_idx_dev = jnp.zeros((args.max_decoding_steps,), dtype=jnp.int32)  # uniform → all zeros
    v_dev = jnp.asarray(v_sae)

    if not args.skip_baseline and "baseline" not in done:
        logger.info("\n[1] Baseline (no steering)...")
        wrapper.disarm()
        r = run_condition(args.task, args.port, args.task_suite_name, args.num_episodes,
                          "baseline", task_output_dir)
        all_results.append(r)
        done.add("baseline")
        save_progress()

    logger.info("\n[2] SAE steering (α sweep)")
    for la in args.alphas:
        cond_name = f"sae_la{la}"
        if cond_name in done:
            logger.info("  %s — already done, skip", cond_name)
            continue
        wrapper.arm(C_stack_dev, 0.0, step_idx_dev, add_bias=v_dev, add_alpha=float(la))
        r = run_condition(args.task, args.port, args.task_suite_name, args.num_episodes,
                          cond_name, task_output_dir)
        all_results.append(r)
        done.add(cond_name)
        save_progress()

    logger.info("\n%s", "=" * 70)
    for r in sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True):
        logger.info("%-30s %.3f", r["condition"], r["success_rate"])
    logger.info("Saved → %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
