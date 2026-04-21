#!/usr/bin/env python3
"""
Conceptor-Based Steering for pi0-fast LIBERO
=============================================
Loads pre-computed conceptors from ``pi0fast_libero_conceptors.npz`` and runs
steered policy evaluation on LIBERO via the WebSocket server + eval_all.py
pattern.  One SLURM job per task.

Key differences from ``experiments/pi05_libero/src/conceptor_steering.py``:

- pi0-fast is JAX, so there is no PyTorch forward hook; steering is applied
  via the JIT-compiled fast kernel ``Policy.infer_with_steering_fast``, which
  takes a small ``C_stack: (K, d, d)`` plus scalar ``beta`` plus
  ``step_to_C_idx: (max_decoding_steps,)`` — M = (1-β)I + βC is materialized
  lazily inside the jitted decode so we avoid a (max_decoding_steps, d, d)
  host buffer. The driver pads K=3 across all strategies so the JIT cache
  stays hot throughout the sweep.

- There is no ``layer`` axis — only one intervention point.  The sweep is
  ``alpha × beta × strategy``.

- ``strategy`` options are:
    * ``global``        — one C applied to every generated token.
    * ``per_token_first``/``mid``/``last`` — C fit from tokens at that
      position within each inference step, still applied every token.
    * ``linear``        — interpolate β from β_start at token 0 to β_end at
      token max_decoding_steps with the same ``C = C_contrastive``.
    * (positive-only steering lives in ``positive_only_steering.py``.)

Usage:
    uv run experiments/pi0_fast_libero/src/conceptor_steering.py \
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
from typing import List, Optional

import jax.numpy as jnp
import numpy as np
import tyro

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
CONCEPTOR_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "pi0fast_libero_conceptors.npz"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

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

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


# ── Steering matrix construction ─────────────────────────────────────────


def load_npz():
    if not CONCEPTOR_NPZ.exists():
        raise FileNotFoundError(f"Conceptor file not found: {CONCEPTOR_NPZ}")
    return np.load(CONCEPTOR_NPZ, allow_pickle=True)


def get_conceptor(npz, task: str, strategy: str, alpha: float, kind: str) -> np.ndarray:
    key = f"{task}__{strategy}__{alpha}__{kind}"
    if key not in npz:
        raise KeyError(f"Conceptor not in npz: {key}")
    return npz[key]


def _uniform_step_idx(max_steps: int) -> np.ndarray:
    return np.zeros((max_steps,), dtype=np.int32)


def _combined_step_idx(max_steps: int) -> np.ndarray:
    """[0]*T/3 + [1]*T/3 + [2]*(T - 2*T/3) — per-token conceptor indices."""
    t1 = max_steps // 3
    t2 = 2 * max_steps // 3
    idx = np.empty((max_steps,), dtype=np.int32)
    idx[:t1] = 0
    idx[t1:t2] = 1
    idx[t2:] = 2
    return idx


def build_steering_single(C: np.ndarray, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """One conceptor used at every step.

    Padded to K=3 so the JIT-cache signature is identical across global,
    per-step-combined, and positive-only sweeps (same C_stack rank and shape).
    """
    Cf = C.astype(np.float32)
    C_stack = np.stack([Cf, Cf, Cf], axis=0)
    return C_stack, _uniform_step_idx(max_steps)


def build_steering_combined(
    npz, task: str, alpha: float, max_steps: int, kind: str = "C_contrastive",
) -> tuple[np.ndarray, np.ndarray]:
    """Position-aware conceptor stack: first/mid/last by token index third."""
    C_first = get_conceptor(npz, task, "per_token_first", alpha, kind).astype(np.float32)
    C_mid = get_conceptor(npz, task, "per_token_mid", alpha, kind).astype(np.float32)
    C_last = get_conceptor(npz, task, "per_token_last", alpha, kind).astype(np.float32)
    C_stack = np.stack([C_first, C_mid, C_last], axis=0)
    return C_stack, _combined_step_idx(max_steps)


# ── Steered policy wrapper ───────────────────────────────────────────────


class SteeredFastPolicyWrapper:
    """Wraps a pi0-fast JAX Policy to route ``infer`` through
    ``infer_with_steering_fast`` when steering is armed.

    Stores device-resident ``C_stack`` / ``beta`` / ``step_to_C_idx`` so the
    hot path avoids host-to-device transfer per websocket call. The driver
    calls ``arm()`` once per sweep condition; ``disarm()`` falls back to the
    unsteered path (used for baseline inference, if ever needed)."""

    def __init__(self, policy):
        self._policy = policy
        self._C_stack = None
        self._beta = None
        self._step_to_C_idx = None

    def arm(self, C_stack, beta, step_to_C_idx) -> None:
        self._C_stack = C_stack
        self._beta = beta
        self._step_to_C_idx = step_to_C_idx

    def disarm(self) -> None:
        self._C_stack = None
        self._beta = None
        self._step_to_C_idx = None

    def infer(self, obs):
        if self._C_stack is None:
            return self._policy.infer(obs)
        return self._policy.infer_with_steering_fast(
            obs,
            C_stack=self._C_stack,
            beta=self._beta,
            step_to_C_idx=self._step_to_C_idx,
        )

    @property
    def metadata(self):
        return self._policy.metadata


# ── Eval via main.py subprocess ──────────────────────────────────────────


def run_single_task_eval(task_name: str, task_suite_name: str, num_episodes: int,
                         port: int, output_dir: str, seed: int = 7) -> Optional[float]:
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
        "--seed", str(seed),
        "--output_dir", abs_output_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info("Eval: %s", " ".join(cmd[-12:]))
    proc = subprocess.run(
        cmd, cwd=str(libero_env_dir), env=env,
        capture_output=True, text=True, timeout=7200, check=False,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error("Eval failed (rc=%d):\n%s", proc.returncode, log_text[-3000:])
        return None
    matches = SUCCESS_RATE_RE.findall(log_text)
    if not matches:
        logger.error("No success_rate in output:\n%s", log_text[-2000:])
        return None
    return float(matches[-1])


def start_server_background(wrapper: SteeredFastPolicyWrapper, port: int) -> threading.Thread:
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # Wait for the socket to actually bind before returning.
    for _ in range(60):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError(f"Server never bound port {port}")
    logger.info("Steering server bound on port %d", port)
    return t


def run_condition(task_name: str, port: int, task_suite: str, num_episodes: int,
                  condition_name: str, task_output_dir: pathlib.Path) -> dict:
    cond_dir = task_output_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    sr = run_single_task_eval(task_name, task_suite, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")
    logger.info("  %s: SR=%.3f", condition_name, sr)
    return {"condition": condition_name, "success_rate": sr}


# ── CLI ──────────────────────────────────────────────────────────────────


def baseline_from_activations(task: str, activations_dir: str) -> Optional[float]:
    """Read episode metadata to get baseline success rate without running inference."""
    task_dir = pathlib.Path(activations_dir) / task
    if not task_dir.exists():
        return None
    n_success, n_total = 0, 0
    for ep_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n_total += 1
        if meta.get("episode_success", False):
            n_success += 1
    if n_total == 0:
        return None
    return n_success / n_total


@dataclasses.dataclass
class Args:
    task: str = LIBERO_TASKS[0]

    config: str = "pi0_fast_libero"
    checkpoint_dir: str = "checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000/"

    activations_dir: str = ""
    """Path to activations/{ckpt_step}/ for baseline SR. Defaults to
    $OPENPI_DATA_HOME/pi0fast-libero-activations-v1-2000-15env/2000."""

    global_alphas: List[float] = dataclasses.field(default_factory=lambda: [1.0, 2.0, 10.0])
    per_step_combined_alphas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0])
    positive_only_alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0, 2.0])
    betas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.2, 0.3])
    max_decoding_steps: int = 256

    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000

    output_dir: str = "experiments/pi0_fast_libero/steering_results"


def main(args: Args) -> None:
    task = args.task
    if task not in LIBERO_TASK_IDS:
        raise ValueError(f"Unknown task: {task}")

    act_dir = args.activations_dir or str(
        pathlib.Path(OPENPI_DATA_HOME) / "pi0fast-libero-activations-v1-2000-15env" / "2000"
    )

    task_short = task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Task: %s", task)
    logger.info("Output: %s", task_output_dir)

    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    logger.info("Loading pi0-fast policy...")
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    logger.info("Policy loaded (JAX).")

    npz = load_npz()

    all_results: List[dict] = []
    summary_path = task_output_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        all_results = prev.get("conditions", [])
        done = {r["condition"] for r in all_results}
        logger.info("Resuming with %d previously completed conditions", len(done))
    else:
        done = set()

    def save_progress() -> None:
        sorted_results = sorted(
            all_results, key=lambda x: x.get("success_rate", 0), reverse=True
        )
        with open(summary_path, "w") as f:
            json.dump({"task": task, "conditions": sorted_results}, f, indent=2)

    wrapper = SteeredFastPolicyWrapper(policy)
    start_server_background(wrapper, args.port)

    step_idx_single_dev = jnp.asarray(_uniform_step_idx(args.max_decoding_steps))
    step_idx_combined_dev = jnp.asarray(_combined_step_idx(args.max_decoding_steps))

    # ── 1. Baseline (from activation data — no GPU inference) ──
    if "baseline" not in done:
        logger.info("\n[1] Baseline (from activation metadata)...")
        baseline_sr = baseline_from_activations(task, act_dir)
        if baseline_sr is not None:
            logger.info("  baseline: SR=%.3f (from activation data)", baseline_sr)
            all_results.append({"condition": "baseline", "success_rate": baseline_sr})
        else:
            logger.warning("  No activation data for baseline — running inference.")
            wrapper.disarm()
            all_results.append(
                run_condition(
                    task, args.port, args.task_suite_name, args.num_episodes,
                    "baseline", task_output_dir,
                )
            )
        save_progress()

    # ── 2. Global (contrastive) ──
    logger.info("\n[2] Global contrastive conditions...")
    for alpha in args.global_alphas:
        try:
            C = get_conceptor(npz, task, "global", alpha, "C_contrastive")
        except KeyError:
            logger.warning("No global contrastive for %s/a=%s — skipping.", task[:40], alpha)
            continue
        C_stack_np, _ = build_steering_single(C, args.max_decoding_steps)
        C_stack_dev = jnp.asarray(C_stack_np)
        for beta in args.betas:
            cond_name = f"global_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            wrapper.arm(C_stack_dev, float(beta), step_idx_single_dev)
            all_results.append(
                run_condition(
                    task, args.port, args.task_suite_name, args.num_episodes,
                    cond_name, task_output_dir,
                )
            )
            save_progress()

    # ── 3. Per-step combined (contrastive, first/mid/last by position) ──
    logger.info("\n[3] Per-step combined conditions...")
    for alpha in args.per_step_combined_alphas:
        try:
            C_stack_np, _ = build_steering_combined(
                npz, task, alpha, args.max_decoding_steps, kind="C_contrastive",
            )
        except KeyError as e:
            logger.warning("Missing per_token conceptor for combined: %s — skipping.", e)
            continue
        C_stack_dev = jnp.asarray(C_stack_np)
        for beta in args.betas:
            cond_name = f"per_step_combined_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            wrapper.arm(C_stack_dev, float(beta), step_idx_combined_dev)
            all_results.append(
                run_condition(
                    task, args.port, args.task_suite_name, args.num_episodes,
                    cond_name, task_output_dir,
                )
            )
            save_progress()

    # ── 4. Positive-only (C_success, global) ──
    logger.info("\n[4] Positive-only conditions...")
    for alpha in args.positive_only_alphas:
        try:
            C = get_conceptor(npz, task, "global", alpha, "C_success")
        except KeyError:
            logger.warning("No C_success for %s/a=%s — skipping.", task[:40], alpha)
            continue
        C_stack_np, _ = build_steering_single(C, args.max_decoding_steps)
        C_stack_dev = jnp.asarray(C_stack_np)
        for beta in args.betas:
            cond_name = f"positive_only_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            wrapper.arm(C_stack_dev, float(beta), step_idx_single_dev)
            all_results.append(
                run_condition(
                    task, args.port, args.task_suite_name, args.num_episodes,
                    cond_name, task_output_dir,
                )
            )
            save_progress()

    # ── Final ──
    save_progress()
    logger.info("\n%s", "=" * 70)
    logger.info("%-45s %8s", "Condition", "SR")
    for r in sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True):
        logger.info("%-45s %8.3f", r["condition"], r["success_rate"])
    logger.info("Saved to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
