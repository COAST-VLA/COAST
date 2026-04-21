#!/usr/bin/env python3
"""
Conceptor-Based Steering for pi0.5 LIBERO
==========================================

Loads pre-computed conceptors from libero_conceptors.npz and runs steered
policy evaluation on LIBERO via the WebSocket server + eval_all.py pattern.

One SLURM job per task.  Each job:
  1. Loads the model ONCE
  2. Runs baseline (no steering)
  3. Sweeps all (layer × alpha × beta × strategy) using that task's conceptors
  4. Runs random-conceptor controls
  5. Saves everything under  output_dir/{task_short_name}/

Pre-computed conceptor file:
    $OPENPI_DATA_HOME/libero_conceptors.npz
    Key pattern: {task}__L{layer}__{alpha_or_per_step_N}__{C_contrastive|C_success|C_failure}

Usage (from repo root):
    uv run experiments/pi05_libero/src/conceptor_steering.py \
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
"""

import dataclasses
import json
import logging
import os
import pathlib
import re
import socket
import subprocess
import sys
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
CONCEPTOR_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "libero_conceptors.npz"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # openpi-new/

# Task name → libero_10 task_id mapping (from benchmark registry)
LIBERO_TASK_IDS = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 0,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 1,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 2,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 3,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 4,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 5,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 6,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 7,
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 8,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 9,
}

LIBERO_TASKS = list(LIBERO_TASK_IDS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Load Pre-Computed Conceptors from .npz
# ──────────────────────────────────────────────────────────────────────────────


def load_npz():
    """Load the pre-computed conceptor .npz."""
    if not CONCEPTOR_NPZ.exists():
        raise FileNotFoundError(f"Conceptor file not found: {CONCEPTOR_NPZ}")
    return np.load(CONCEPTOR_NPZ, allow_pickle=True)


def get_global_contrastive(npz, task, layer, alpha):
    key = f"{task}__L{layer}__{alpha}__C_contrastive"
    if key not in npz:
        raise KeyError(f"Key not found in npz: {key}")
    return npz[key]


def get_per_step_contrastive(npz, task, layer, step):
    key = f"{task}__L{layer}__per_step_{step}__C_contrastive"
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


def compute_random_conceptor(d=1024, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Steered Server + LIBERO Client
# ──────────────────────────────────────────────────────────────────────────────


class SteeredPolicyWrapper:
    """Wraps a Policy to route infer() through infer_with_steering()."""

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


def run_single_task_eval(task_name, task_suite_name, num_episodes, port, output_dir):
    """Launch main.py for a SINGLE task via subprocess. Returns success_rate float."""
    task_id = LIBERO_TASK_IDS[task_name]
    libero_env_dir = REPO_ROOT / "examples" / "libero_env"
    abs_output_dir = str(pathlib.Path(output_dir).resolve())
    cmd = [
        str(libero_env_dir / ".venv" / "bin" / "python"),
        str(libero_env_dir / "main.py"),
        "--task_suite_name", task_suite_name,
        "--task_id", str(task_id),
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", abs_output_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info(f"Eval: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(libero_env_dir), env=env,
                          capture_output=True, text=True, timeout=7200)
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
    """Start the WebSocket server in a daemon thread. Returns the thread (for reference only)."""
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(5)  # wait for bind
    logger.info(f"Server started on port {port} (daemon thread)")
    return t


def run_condition(task_name, port, task_suite, num_episodes,
                  condition_name, task_output_dir):
    """Run one eval against the already-running server. Returns result dict."""
    cond_dir = task_output_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)

    sr = run_single_task_eval(task_name, task_suite, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")

    logger.info(f"  {condition_name}: SR={sr:.3f}")
    return {"condition": condition_name, "success_rate": sr}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Run all steering conditions for one task. Model loads once."""
    # Which task's conceptors to use for steering
    task: str = LIBERO_TASKS[0]

    # Policy
    config: str = "pi05_libero"
    checkpoint_dir: str = "/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_libero/libero_b200_bs512/2000"

    # Sweep axes
    layers: List[int] = dataclasses.field(default_factory=lambda: [5, 11, 17])
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0])
    betas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3, 0.5])
    strategies: List[str] = dataclasses.field(default_factory=lambda: ["global", "per_step_0", "per_step_9"])

    # Eval
    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000
    n_random_controls: int = -1  # -1 = full layer×beta grid, 0 = skip

    output_dir: str = "experiments/pi05_libero/steering_results"


def main(args: Args):
    task = args.task
    # Short name for folder: e.g. "KITCHEN_SCENE3_turn_on_the_stove..."
    task_short = task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {task}")
    logger.info(f"Output: {task_output_dir}")
    logger.info(f"Sweep: {len(args.layers)} layers × {len(args.alphas)} alphas × "
                f"{len(args.betas)} betas × {len(args.strategies)} strategies")

    # Save args
    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    # Load model ONCE
    logger.info("Loading policy (one-time cost)...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # Load conceptors ONCE
    npz = load_npz()
    all_results = []

    def save_progress():
        """Incremental save so progress isn't lost on early exit."""
        sorted_results = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
        with open(task_output_dir / "summary.json", "w") as f:
            json.dump({"task": task, "conditions": sorted_results}, f, indent=2)

    # Reusable wrapper — we swap hooks between conditions.
    # Server starts ONCE and stays alive; we just mutate the wrapper's hooks.
    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    # ── 1. Baseline ──
    logger.info("\n[1/4] Baseline (no steering)...")
    wrapper.update_hooks(None)
    r = run_condition(task, args.port, args.task_suite_name,
                      args.num_episodes,
                      "baseline", task_output_dir)
    all_results.append(r)
    save_progress()

    # ── 2. Steered conditions ──
    total = len(args.layers) * len(args.alphas) * len(args.betas) * len(args.strategies)
    logger.info(f"\n[2/4] Steered conditions ({total} total)...")
    idx = 0
    for layer in args.layers:
        for alpha in args.alphas:
            C_global = get_global_contrastive(npz, task, layer, alpha)
            for beta in args.betas:
                for strategy in args.strategies:
                    idx += 1
                    cond_name = f"{strategy}_L{layer}_a{alpha}_b{beta}"
                    logger.info(f"  [{idx}/{total}] {cond_name}")

                    if strategy == "global":
                        C = C_global
                    elif strategy.startswith("per_step_"):
                        step = int(strategy.split("_")[-1])
                        C = get_per_step_contrastive(npz, task, layer, step)
                    else:
                        logger.warning(f"  Unknown strategy {strategy}, skipping")
                        continue

                    hook = ConceptorSteeringHook(C, beta=beta, device=device)
                    wrapper.update_hooks([(layer, hook)])
                    r = run_condition(task, args.port, args.task_suite_name,
                                      args.num_episodes,
                                      cond_name, task_output_dir)
                    all_results.append(r)
                    save_progress()

    # ── 3. Random controls ──
    random_pairs = [(l, b) for l in args.layers for b in args.betas]
    if args.n_random_controls >= 0:
        random_pairs = random_pairs[: args.n_random_controls]
    n_rand = len(random_pairs)
    logger.info(f"\n[3/4] Random controls ({n_rand})...")
    for layer, beta in random_pairs:
        cond_name = f"random_L{layer}_b{beta}"
        C_rand = compute_random_conceptor(seed=layer * 100 + int(beta * 10))
        hook = ConceptorSteeringHook(C_rand, beta=beta, device=device)
        wrapper.update_hooks([(layer, hook)])
        r = run_condition(task, args.port, args.task_suite_name,
                          args.num_episodes,
                          cond_name, task_output_dir)
        all_results.append(r)
        save_progress()

    # ── 4. Final summary ──
    logger.info(f"\n[4/4] Saving final summary...")
    save_progress()

    logger.info(f"\n{'='*70}")
    logger.info(f"{'Condition':<45s} {'SR':>8s}")
    logger.info(f"{'-'*55}")
    for r in all_results:
        logger.info(f"{r['condition']:<45s} {r['success_rate']:>8.3f}")
    logger.info(f"{'='*70}")
    logger.info(f"All results for {task_short} saved to {task_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
