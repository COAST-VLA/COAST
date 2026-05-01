#!/usr/bin/env python3
"""Reference SAE-ActAdd steering driver — DO NOT RUN DIRECTLY.

This file is the *reference* / canonical version of the SAE steering driver.
The actual per-experiment drivers live at:

    experiments/pi0_fast_libero/src/sae_steering.py        (JAX, additive via add_bias/add_alpha)
    experiments/pi0_fast_metaworld/src/sae_steering.py     (JAX, same path as libero)
    experiments/pi05_libero/src/sae_steering.py            (PyTorch, hook on layer 11)
    experiments/pi05_robocasa/src/sae_steering.py          (PyTorch, hook on layer 11)

Each one imports the policy/server/eval-subprocess infrastructure from its
sibling ``conceptor_steering.py``, so they cannot be reduced to a single file
without cross-experiment imports. The differences between the four are small;
this reference shows the JAX (pi0-fast) variant which is the simplest. For the
PyTorch variant, the only changes are:

  * Replace ``SteeredFastPolicyWrapper`` with the per-experiment
    ``SteeredPolicyWrapper`` (PyTorch hook variant).
  * Replace ``wrapper.arm(C_stack_dev, 0.0, step_idx_dev, add_bias=v_dev,
    add_alpha=la)`` with ``wrapper.update_hooks([(layer, LinearSteeringHook(v,
    alpha=la, device=device))])``.
  * NPZ key includes ``__L{layer}`` because pi0.5 has a layer axis;
    pi0-fast doesn't.

End-to-end pipeline:
  1. Train a per-task SAE: experiments/sae/src/train_sae.py
  2. Build v_sae: experiments/sae/src/fit_sae_vectors.py
  3. Run this driver (the per-experiment version) — one α condition at a time.

The injected vector is ``v_sae``; for pi0-fast it slots into the existing JAX
``infer_with_steering_fast`` ``add_bias`` / ``add_alpha`` channel — no
JIT-cache divergence, no new model code. For pi0.5 it goes through a
PyTorch forward hook ``h ← h + α·v`` registered on the chosen layer.
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

# Per-experiment driver imports the corresponding conceptor_steering symbols:
#   from conceptor_steering import (LIBERO_TASKS, SteeredFastPolicyWrapper,
#                                   run_condition, start_server_background)
# This reference file does not run, so we don't.

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))


@dataclasses.dataclass
class Args:
    task: str = ""
    config: str = "pi0_fast_libero"
    checkpoint_dir: str = "checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000/"
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    sae_vectors_npz: str = ""  # default: $OPENPI_DATA_HOME/{exp}_sae_vectors.npz
    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000
    output_dir: str = "experiments/{exp}/sae_steering_results"
    max_decoding_steps: int = 256
    skip_baseline: bool = False


def main(args: Args):
    """Reference flow. Per-experiment file does the same with concrete imports."""
    raise SystemExit(
        "This is a reference file. Run the per-experiment driver instead:\n"
        "  experiments/pi0_fast_libero/src/sae_steering.py\n"
        "  experiments/pi0_fast_metaworld/src/sae_steering.py\n"
        "  experiments/pi05_libero/src/sae_steering.py\n"
        "  experiments/pi05_robocasa/src/sae_steering.py"
    )

    task_short = args.task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load v_sae ──
    sae_path = pathlib.Path(args.sae_vectors_npz)
    if not sae_path.is_file():
        sys.exit(f"sae_vectors_npz not found: {sae_path}")
    with np.load(sae_path) as nv:
        # Key naming differs between pi0-fast and pi05:
        #   pi0-fast:  {task}__sae__V_contrastive
        #   pi05:      {task}__L{layer}__sae__V_contrastive
        key = f"{args.task}__sae__V_contrastive"
        if key not in nv.files:
            sys.exit(f"key not in NPZ: {key}")
        v_sae = np.asarray(nv[key], dtype=np.float32)
    logger.info("Loaded v_sae from %s  (||v||=%.4f, expected ~1.0)", sae_path, float(np.linalg.norm(v_sae)))

    # ── Resume support ──
    summary_path = task_output_dir / "summary.json"
    all_results: list = []
    done: set = set()
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f).get("conditions", [])
        all_results = [r for r in existing if r is not None]
        done = {r["condition"] for r in all_results}

    def save_progress():
        sorted_results = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
        tmp = summary_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"task": args.task, "conditions": sorted_results}, f, indent=2)
        tmp.replace(summary_path)

    # ── Load policy ONCE ──
    # from openpi.policies import policy_config as _policy_config
    # from openpi.training import config as _config
    # train_config = _config.get_config(args.config)
    # policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    # wrapper = SteeredFastPolicyWrapper(policy)
    # start_server_background(wrapper, args.port)

    # ── Build steering payload ──
    # For pi0-fast: pad C_stack to identity so JIT signature matches the
    # conceptor sweep (β=0 means matrix term contributes nothing).
    d = v_sae.shape[-1]
    identity_stack = np.broadcast_to(np.eye(d, dtype=np.float32), (3, d, d))
    C_stack_dev = jnp.asarray(identity_stack)
    step_idx_dev = jnp.zeros((args.max_decoding_steps,), dtype=jnp.int32)
    v_dev = jnp.asarray(v_sae)

    # if not args.skip_baseline and "baseline" not in done:
    #     wrapper.disarm()
    #     r = run_condition(args.task, args.port, args.task_suite_name,
    #                       args.num_episodes, "baseline", task_output_dir)
    #     all_results.append(r); done.add("baseline"); save_progress()

    for la in args.alphas:
        cond = f"sae_la{la}"            # pi05 variant: f"sae_L{layer}_la{la}"
        if cond in done:
            continue
        # wrapper.arm(C_stack_dev, 0.0, step_idx_dev, add_bias=v_dev, add_alpha=float(la))
        # r = run_condition(args.task, args.port, args.task_suite_name,
        #                   args.num_episodes, cond, task_output_dir)
        # all_results.append(r); done.add(cond); save_progress()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
