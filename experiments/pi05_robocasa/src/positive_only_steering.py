#!/usr/bin/env python3
"""
Positive-Only Conceptor Steering for pi0.5 RoboCasa
====================================================

Uses C_success directly (no contrastive NOT C_failure) as the steering
conceptor. Minimal sweep to fill the "Pos.-Only" column in the results table.

Usage (from repo root):
    uv run experiments/pi05_robocasa/src/positive_only_steering.py \
        --task CloseFridge
"""

import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import threading
import time
from typing import List

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
CONCEPTOR_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "robocasa_conceptors.npz"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # openpi-new/

ROBOCASA_TASKS = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]


# ──────────────────────────────────────────────────────────────────────────────
# Load Pre-Computed Conceptors
# ──────────────────────────────────────────────────────────────────────────────


def load_npz():
    if not CONCEPTOR_NPZ.exists():
        raise FileNotFoundError(f"Conceptor file not found: {CONCEPTOR_NPZ}")
    return np.load(CONCEPTOR_NPZ, allow_pickle=True)


def get_success_conceptor(npz, task, layer, alpha):
    """Load C_success (positive-only, no contrastive NOT)."""
    key = f"{task}__L{layer}__{alpha}__C_success"
    if key not in npz:
        raise KeyError(f"Key not found in npz: {key}")
    return npz[key]


# ──────────────────────────────────────────────────────────────────────────────
# Steering Hook
# ──────────────────────────────────────────────────────────────────────────────


class ConceptorSteeringHook:
    """PyTorch forward hook: h' = (1-beta)*h + beta*(C @ h)."""

    def __init__(self, conceptor_matrix, beta=0.3, device="cuda"):
        self.beta = beta
        self.current_denoise_step = 0
        d = conceptor_matrix.shape[0]
        I = torch.eye(d, dtype=torch.float32).to(device)
        C = torch.tensor(conceptor_matrix, dtype=torch.float32).to(device)
        self.M = (1 - beta) * I + beta * C
        self.intervention_norms = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        M = self.M.to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(torch.norm(h_steered - h).item())
        return (h_steered,) + rest if rest is not None else h_steered

    def set_denoise_step(self, t):
        self.current_denoise_step = t

    def reset_logs(self):
        self.intervention_norms = []


# ──────────────────────────────────────────────────────────────────────────────
# Server + Eval
# ──────────────────────────────────────────────────────────────────────────────


class SteeredPolicyWrapper:
    def __init__(self, policy, steering_hooks):
        self._policy = policy
        self._steering_hooks = steering_hooks

    def update_hooks(self, steering_hooks):
        self._steering_hooks = steering_hooks

    def infer(self, obs):
        if self._steering_hooks:
            for _, hook_fn in self._steering_hooks:
                hook_fn.reset_logs()
            result, _ = self._policy.infer_with_steering(
                obs, steering_hooks=self._steering_hooks
            )
            return result
        return self._policy.infer(obs)

    @property
    def metadata(self):
        return self._policy.metadata


SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


def run_single_task_eval(task_name, num_episodes, port, output_dir):
    robocasa_env_dir = REPO_ROOT / "examples" / "robocasa_env"
    abs_output_dir = str(pathlib.Path(output_dir).resolve())
    cmd = [
        str(robocasa_env_dir / ".venv" / "bin" / "python"),
        str(robocasa_env_dir / "main.py"),
        "--env_name", task_name,
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", abs_output_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info(f"Eval: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(robocasa_env_dir), env=env,
                          capture_output=True, text=True, timeout=14400)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error(f"Eval failed (rc={proc.returncode}):\n{log_text[-3000:]}")
        return None
    matches = SUCCESS_RATE_RE.findall(log_text)
    if matches:
        return float(matches[-1])
    logger.error(f"No success_rate found in main.py output:\n{log_text[-2000:]}")
    return None


def start_server_background(wrapper, port):
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(5)
    logger.info(f"Server started on port {port} (daemon thread)")
    return t


def run_condition(task_name, port, num_episodes,
                  condition_name, task_output_dir):
    cond_dir = task_output_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)

    sr = run_single_task_eval(task_name, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")

    logger.info(f"  {condition_name}: SR={sr:.3f}")
    return {"condition": condition_name, "success_rate": sr}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Run positive-only conceptor steering for one task."""
    task: str = ROBOCASA_TASKS[0]

    # Policy
    config: str = "pi05_robocasa"
    checkpoint_dir: str = "/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"

    # Minimal sweep
    layers: List[int] = dataclasses.field(default_factory=lambda: [5, 11])
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0])
    betas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3])

    # Eval
    num_episodes: int = 15
    port: int = 8000

    output_dir: str = "experiments/pi05_robocasa/steering_results"


def main(args: Args):
    task = args.task
    task_output_dir = pathlib.Path(args.output_dir) / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.layers) * len(args.alphas) * len(args.betas)
    logger.info(f"Task: {task}")
    logger.info(f"Output: {task_output_dir}")
    logger.info(f"Positive-only sweep: {total} conditions "
                f"({len(args.layers)} layers × {len(args.alphas)} alphas × {len(args.betas)} betas)")

    # Load model ONCE
    logger.info("Loading policy...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # Load conceptors
    npz = load_npz()

    # Load existing results to append to
    summary_path = task_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        all_results = existing.get("conditions", [])
        existing_conds = {r["condition"] for r in all_results}
        logger.info(f"Loaded {len(all_results)} existing conditions from summary.json")
    else:
        all_results = []
        existing_conds = set()

    def save_progress():
        """Merge-from-disk save to avoid concurrent-write clobber across jobs."""
        merged = {}
        if summary_path.exists():
            try:
                existing_ = json.load(open(summary_path))
                for r in existing_.get("conditions", []):
                    if r is not None and "condition" in r:
                        merged[r["condition"]] = r
            except (json.JSONDecodeError, OSError):
                pass
        for r in all_results:
            if r is not None and "condition" in r:
                merged[r["condition"]] = r
        sorted_results = sorted(merged.values(),
                                key=lambda x: x.get("success_rate", 0), reverse=True)
        tmp = summary_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"task": task, "conditions": sorted_results}, f, indent=2)
        tmp.replace(summary_path)

    # Start server
    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    # ── Positive-only conditions ──
    idx = 0
    for layer in args.layers:
        for alpha in args.alphas:
            C_succ = get_success_conceptor(npz, task, layer, alpha)
            for beta in args.betas:
                idx += 1
                cond_name = f"pos_only_L{layer}_a{alpha}_b{beta}"

                if cond_name in existing_conds:
                    logger.info(f"  [{idx}/{total}] {cond_name} — already exists, skipping")
                    continue

                logger.info(f"  [{idx}/{total}] {cond_name}")
                hook = ConceptorSteeringHook(C_succ, beta=beta, device=device)
                wrapper.update_hooks([(layer, hook)])
                r = run_condition(task, args.port, args.num_episodes,
                                  cond_name, task_output_dir)
                all_results.append(r)
                existing_conds.add(cond_name)
                save_progress()

    # Final summary
    save_progress()
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Condition':<45s} {'SR':>8s}")
    logger.info(f"{'-'*55}")
    for r in sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True):
        if r["condition"].startswith("pos_only"):
            logger.info(f"{r['condition']:<45s} {r['success_rate']:>8.3f}")
    logger.info(f"{'='*70}")
    logger.info(f"Results appended to {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
