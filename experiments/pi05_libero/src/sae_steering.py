#!/usr/bin/env python3
"""SAE-ActAdd steering baseline for pi0.5 LIBERO.

Loads v_sae from $OPENPI_DATA_HOME/libero_sae_vectors.npz (built by
experiments/sae/src/fit_sae_vectors.py) and runs steered eval mirroring the
linear ActAdd path in pi05_robocasa/src/conceptor_steering.py:LinearSteeringHook.

Single layer per condition (default L=11, matching SWEEP_LAYER in
run_linear_only_sweep.sh). The injected vector is the SAE-feature-aggregated
contrastive direction from fit_sae_vectors.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import sys
from typing import List, Optional

import numpy as np
import torch
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conceptor_steering import (  # noqa: E402
    LIBERO_TASK_IDS, LIBERO_TASKS,
    SteeredPolicyWrapper, run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
DEFAULT_SAE_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "libero_sae_vectors.npz"


class LinearSteeringHook:
    """ActAdd-style PyTorch forward hook: h' = h + α·v.  Time-invariant."""

    def __init__(self, v: np.ndarray, alpha: float, device: str = "cuda"):
        self.v = torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
        self.alpha = float(alpha)
        self.intervention_norms: list[float] = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        v = self.v.to(dtype=h.dtype)
        h_steered = h + self.alpha * v
        self.intervention_norms.append(torch.norm(h_steered - h).item())
        return (h_steered,) + rest if rest is not None else h_steered

    def set_denoise_step(self, t):
        pass

    def reset_logs(self):
        self.intervention_norms = []


@dataclasses.dataclass
class Args:
    task: str = LIBERO_TASKS[0]
    config: str = "pi05_libero"
    checkpoint_dir: str = "checkpoints/pi05_libero/libero_b200_bs512/2000"
    layer: int = 11
    """Single steering layer. Matches SWEEP_LAYER convention in linear-only sweeps."""
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    sae_vectors_npz: str = str(DEFAULT_SAE_NPZ)
    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000
    output_dir: str = "experiments/pi05_libero/sae_steering_results"
    skip_baseline: bool = False


def main(args: Args):
    task_short = args.task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
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
    logger.info("Loaded v_sae from %s  key=%s  ||v||=%.4f", sae_path, key, float(np.linalg.norm(v_sae)))

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
    logger.info("Policy on %s", device)

    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    if not args.skip_baseline and "baseline" not in done:
        logger.info("\n[1] Baseline...")
        wrapper.update_hooks(None)
        all_results.append(run_condition(
            args.task, args.port, args.task_suite_name, args.num_episodes,
            "baseline", task_output_dir,
        ))
        done.add("baseline"); save_progress()

    logger.info("\n[2] SAE steering (α sweep)")
    for la in args.alphas:
        cond = f"sae_L{args.layer}_la{la}"
        if cond in done:
            logger.info("  %s — already done, skip", cond); continue
        hook = LinearSteeringHook(v_sae, alpha=float(la), device=device)
        wrapper.update_hooks([(args.layer, hook)])
        all_results.append(run_condition(
            args.task, args.port, args.task_suite_name, args.num_episodes,
            cond, task_output_dir,
        ))
        done.add(cond); save_progress()

    logger.info("Saved → %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
