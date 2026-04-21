#!/usr/bin/env python3
"""
Collect activations during steered pi0.5 LIBERO rollouts.

Starts a collection-mode server with per-task conceptor steering hooks,
then runs eval_all.py --collect to execute rollouts and save activations.

The server applies the best global contrastive steering config per task
(from steering_collection_config.json) while simultaneously capturing
intermediates via the standard CollectingPolicy hooks.

Usage (from repo root):
    export CUDA_VISIBLE_DEVICES=<gpu_id>
    uv run experiments/pi05_libero/src/collect_steered_activations.py \
        --num_episodes 15

Activations are saved to:
    $OPENPI_DATA_HOME/activations/pi05_steered_activations/pi05_libero/<task_name>/...
"""

import argparse
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
OUTPUT_DIR = pathlib.Path(OPENPI_DATA_HOME) / "activations" / "pi05_steered_activations"
STEERING_CONFIG = REPO_ROOT / "experiments" / "pi05_libero" / "steering_collection_config.json"

# LIBERO task_id mapping (must match LIBERO benchmark registry order for libero_10)
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

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_steering_hooks_fn(steering_config_path):
    """Build a task_name -> steering_hooks resolver from a JSON config file.

    Returns a callable: task_name -> list[(layer_idx, hook)] or None.
    """
    import json

    import numpy as np
    import torch

    with open(steering_config_path) as f:
        task_configs = json.load(f)

    npz_path = pathlib.Path(OPENPI_DATA_HOME) / "libero_conceptors.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Conceptor file not found: {npz_path}")
    conceptors_npz = np.load(npz_path, allow_pickle=True)
    logger.info("Loaded conceptors from %s (%d arrays)", npz_path, len(conceptors_npz.files))

    task_hooks = {}
    for task_name, cfg in task_configs.items():
        layer = int(cfg["layer"])
        alpha = float(cfg["alpha"])
        beta = float(cfg["beta"])
        key = f"{task_name}__L{layer}__{alpha}__C_contrastive"
        if key not in conceptors_npz:
            logger.warning("Conceptor key %s not found, skipping task %s", key, task_name)
            continue
        C = conceptors_npz[key]
        d = C.shape[0]
        I = torch.eye(d, dtype=torch.float32, device="cuda")
        C_t = torch.tensor(C, dtype=torch.float32, device="cuda")
        M = (1 - beta) * I + beta * C_t

        class _Hook:
            """Forward hook: h' = h @ M^T (same as ConceptorSteeringHook)."""

            def __init__(self, M_matrix):
                self.M = M_matrix
                self.current_denoise_step = 0

            def __call__(self, module, input, output):
                if isinstance(output, tuple):
                    h, rest = output[0], output[1:]
                else:
                    h, rest = output, None
                h_steered = torch.matmul(h, self.M.to(dtype=h.dtype).T)
                return (h_steered,) + rest if rest is not None else h_steered

            def set_denoise_step(self, t):
                self.current_denoise_step = t

        hook = _Hook(M)
        task_hooks[task_name] = [(layer, hook)]
        logger.info("Steering hook: %s -> L%d, alpha=%.1f, beta=%.2f", task_name[:50], layer, alpha, beta)

    def resolve(task_name):
        return task_hooks.get(task_name)

    return resolve


def start_server(port, num_episodes):
    """Start the collection-mode server with steering hooks."""
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.serving.activation_collector import CollectingPolicy
    from openpi.training import config as _config

    # Load policy — use the local fine-tuned checkpoint (same as steering eval)
    config_name = "pi05_libero"
    checkpoint_dir = "/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_libero/libero_b200_bs512/2000"

    logger.info("Loading pi05_libero policy from %s (PyTorch)...", checkpoint_dir)
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    ensure_pytorch_checkpoint(checkpoint_dir, config_name)
    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name), checkpoint_dir
    )

    # Build per-task steering hooks
    steering_hooks_fn = build_steering_hooks_fn(str(STEERING_CONFIG))

    # Wrap in CollectingPolicy with steering
    # checkpoint_step determines the subdirectory: OUTPUT_DIR/checkpoint_step/task_name/...
    # Use "pi05_libero" to match STEERED_ACTIVATIONS_DIR in the visualization script.
    collecting_policy = CollectingPolicy(
        policy=policy,
        output_root=OUTPUT_DIR,
        checkpoint_step="pi05_libero",
        policy_dir=checkpoint_dir,
        config_name=config_name,
        steering_hooks_fn=steering_hooks_fn,
    )

    logger.info("Starting collection server on port %d (output: %s)", port, OUTPUT_DIR)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=collecting_policy,
        host="0.0.0.0",
        port=port,
        metadata=collecting_policy.metadata,
    )
    server.serve_forever()


def run_task_eval(task_name, task_id, port, num_episodes):
    """Run eval for a single task via subprocess (libero_env venv)."""
    libero_env_dir = REPO_ROOT / "examples" / "libero_env"
    cmd = [
        str(libero_env_dir / ".venv" / "bin" / "python"),
        str(libero_env_dir / "main.py"),
        "--task_suite_name", "libero_10",
        "--task_id", str(task_id),
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--collect",
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info("Running: %s", " ".join(cmd[-8:]))
    proc = subprocess.run(
        cmd, cwd=str(libero_env_dir), env=env,
        capture_output=True, text=True, timeout=7200,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error("Task %s failed (rc=%d):\n%s", task_name, proc.returncode, log_text[-3000:])
        return None
    matches = SUCCESS_RATE_RE.findall(log_text)
    sr = float(matches[-1]) if matches else None
    logger.info("Task %s: success_rate=%s", task_name, sr)
    return sr


def main():
    parser = argparse.ArgumentParser(description="Collect steered activations for pi0.5 LIBERO")
    parser.add_argument("--num_episodes", type=int, default=15,
                        help="Number of episodes per task (default: 15)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (default: auto-detect free port)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific task names to collect (default: all 10)")
    args = parser.parse_args()

    port = args.port or find_free_port()

    # Start server in background thread
    server_error = [None]  # mutable container for thread exception

    def _start_server_wrapper(port, num_episodes):
        try:
            start_server(port, num_episodes)
        except Exception as e:
            server_error[0] = e
            logger.error("Server thread crashed: %s", e)

    server_thread = threading.Thread(
        target=_start_server_wrapper, args=(port, args.num_episodes), daemon=True
    )
    server_thread.start()

    # Wait for server to be ready (poll the port)
    import socket as _socket
    logger.info("Waiting for server on port %d...", port)
    for attempt in range(120):  # up to 10 minutes
        if server_error[0] is not None:
            logger.error("Server failed to start: %s", server_error[0])
            sys.exit(1)
        try:
            with _socket.create_connection(("localhost", port), timeout=2):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(5)
    else:
        logger.error("Server did not start within 10 minutes. Aborting.")
        sys.exit(1)
    logger.info("Server is ready on port %d", port)

    # Determine which tasks to run
    if args.tasks:
        tasks = {t: LIBERO_TASK_IDS[t] for t in args.tasks}
    else:
        tasks = LIBERO_TASK_IDS

    # Run each task sequentially
    results = {}
    for task_name, task_id in tasks.items():
        logger.info("=" * 70)
        logger.info("Collecting: %s (task_id=%d)", task_name, task_id)
        logger.info("=" * 70)
        sr = run_task_eval(task_name, task_id, port, args.num_episodes)
        results[task_name] = sr

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 70)
    for task_name, sr in results.items():
        logger.info("  %s: %s", task_name[:60], sr)
    logger.info("Activations saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
