"""Sweep conceptor steering hyperparameters for MetaWorld ML45 tasks.

Loads the policy once, spins up a steering-capable WebSocket server in a
background thread, and runs ``examples/metaworld/main.py`` as a subprocess for
each (task, condition) pair. Parses ``success_rate=X.YY`` from the subprocess
log, picks the argmax per task, and writes ``best_configs.json``.

This is the ONLY place sweeps happen. Normal users never run this script —
they just call ``examples/metaworld/eval_all.py --steer --steering_config
experiments/metaworld/best_configs.json`` with the file this script produces.

Usage (from repo root)::

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python experiments/metaworld/find_best_configs.py
"""

# ruff: noqa: DTZ003, DTZ005, E741, FBT001, FBT002, N806, PT018, RUF001, RUF002, RUF003
from __future__ import annotations

import dataclasses
import datetime
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
from typing import Any

import tyro

logger = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONCEPTOR_NPZ = REPO_ROOT / "conceptors" / "metaworld_conceptors.npz"

# Default subset — a small set of ML45 train tasks with decent baseline SR.
# Pass --tasks to override with any env_name present in your conceptor NPZ.
DEFAULT_TASKS: tuple[str, ...] = (
    "reach-v3",
    "pick-place-v3",
    "push-v3",
    "door-open-v3",
    "drawer-open-v3",
)

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


@dataclasses.dataclass
class Args:
    config: str = "pi05_metaworld"
    checkpoint_dir: str = "checkpoints/openpi-metaworld-5000"

    tasks: tuple[str, ...] = DEFAULT_TASKS

    layers: tuple[int, ...] = (11,)
    alphas: tuple[float, ...] = (0.1, 0.5, 1.0)
    betas: tuple[float, ...] = (0.1, 0.3)
    strategies: tuple[str, ...] = (
        "global",
        "per_step",
        "positive_only",
        "random_matched",
        "linear",
    )

    num_episodes: int = 10
    num_envs: int = 10
    max_steps: int = 300
    port: int = 8103

    output_dir: pathlib.Path = pathlib.Path("experiments/metaworld/steering_results")
    best_configs_path: pathlib.Path = pathlib.Path("experiments/metaworld/best_configs.json")


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"server never bound to {host}:{port}")


def _start_server_background(policy: Any, port: int) -> threading.Thread:
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    _wait_for_port("127.0.0.1", port)
    logger.info("Steering server bound on port %d", port)
    return t


def _run_one_eval(
    env_name: str,
    num_episodes: int,
    num_envs: int,
    max_steps: int,
    port: int,
    output_dir: pathlib.Path,
    steer: bool,
    layer: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    strategy: str | None = None,
) -> float:
    """Launch examples/metaworld/main.py for one (task, condition) and parse SR."""
    main_py = REPO_ROOT / "examples" / "metaworld" / "main.py"
    cmd = [
        sys.executable,
        str(main_py),
        "--env_name",
        env_name,
        "--num_episodes",
        str(num_episodes),
        "--num_envs",
        str(num_envs),
        "--max_steps",
        str(max_steps),
        "--port",
        str(port),
        "--output_dir",
        str(output_dir.resolve()),
    ]
    if steer:
        assert layer is not None and alpha is not None and beta is not None and strategy is not None
        cmd.extend(
            [
                "--steer",
                "--steering_layer",
                str(layer),
                "--steering_alpha",
                str(alpha),
                "--steering_beta",
                str(beta),
                "--steering_strategy",
                strategy,
                "--steering_task",
                env_name,
            ]
        )

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False, timeout=7200)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error("Eval returncode=%d. Tail:\n%s", proc.returncode, log_text[-2000:])
        return float("nan")
    matches = SUCCESS_RATE_RE.findall(log_text)
    if not matches:
        logger.error("No success_rate in output. Tail:\n%s", log_text[-1500:])
        return float("nan")
    return float(matches[-1])


def main(args: Args) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    logger.info("Loading policy (one-time cost)...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.serving.steering import SteeredPolicyWrapper
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, torch_compile=False)
    device = str(policy._pytorch_device)  # noqa: SLF001

    wrapper = SteeredPolicyWrapper(policy, conceptor_npz_path=CONCEPTOR_NPZ, device=device)
    _start_server_background(wrapper, args.port)

    partial_results_path = run_dir / "partial_results.jsonl"
    per_task_results: dict[str, dict[str, float]] = {task: {} for task in args.tasks}

    total_conditions = len(args.layers) * len(args.alphas) * len(args.betas) * len(args.strategies) + 1
    for task in args.tasks:
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 70)
        logger.info("TASK: %s", task)

        sr = _run_one_eval(
            task,
            args.num_episodes,
            args.num_envs,
            args.max_steps,
            args.port,
            task_dir / "baseline",
            steer=False,
        )
        per_task_results[task]["baseline"] = sr
        with open(partial_results_path, "a") as f:
            f.write(json.dumps({"task": task, "condition": "baseline", "success_rate": sr}) + "\n")
        logger.info("[1/%d] baseline: SR=%.3f", total_conditions, sr)

        cond_idx = 1
        for layer in args.layers:
            for alpha in args.alphas:
                for beta in args.betas:
                    for strategy in args.strategies:
                        cond_idx += 1
                        cond_name = f"{strategy}_L{layer}_a{alpha}_b{beta}"
                        sr = _run_one_eval(
                            task,
                            args.num_episodes,
                            args.num_envs,
                            args.max_steps,
                            args.port,
                            task_dir / cond_name,
                            steer=True,
                            layer=layer,
                            alpha=alpha,
                            beta=beta,
                            strategy=strategy,
                        )
                        per_task_results[task][cond_name] = sr
                        with open(partial_results_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "task": task,
                                        "condition": cond_name,
                                        "layer": layer,
                                        "alpha": alpha,
                                        "beta": beta,
                                        "strategy": strategy,
                                        "success_rate": sr,
                                    }
                                )
                                + "\n"
                            )
                        logger.info("[%d/%d] %s: SR=%.3f", cond_idx, total_conditions, cond_name, sr)

        with open(run_dir / "per_task_results.json", "w") as f:
            json.dump(per_task_results, f, indent=2)

    best_configs: dict[str, Any] = {
        "task_suite": "metaworld_ml45_train",
        "source_sweep": str(run_dir),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_episodes_per_condition": args.num_episodes,
        "defaults": {
            "layer": args.layers[0],
            "alpha": args.alphas[0],
            "beta": args.betas[0],
            "strategy": args.strategies[0],
        },
        "tasks": {},
    }
    for task in args.tasks:
        results = per_task_results[task]
        baseline_sr = results.get("baseline", 0.0)
        steered = {k: v for k, v in results.items() if k != "baseline" and v == v}
        if not steered:
            logger.warning("No valid steered results for %s; skipping", task)
            continue
        best_cond, best_sr = max(steered.items(), key=lambda kv: kv[1])
        m = re.match(r"^(.+)_L(\d+)_a([0-9.]+)_b([0-9.]+)$", best_cond)
        if not m:
            logger.warning("Could not parse condition %s", best_cond)
            continue
        strategy, layer, alpha, beta = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        best_configs["tasks"][task] = {
            "layer": layer,
            "alpha": alpha,
            "beta": beta,
            "strategy": strategy,
            "baseline_sr": round(baseline_sr, 3),
            "steered_sr": round(best_sr, 3),
        }

    args.best_configs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.best_configs_path, "w") as f:
        json.dump(best_configs, f, indent=2)

    logger.info("=" * 70)
    logger.info("Best configs written to %s", args.best_configs_path)
    logger.info(
        "Summary: %d tasks, mean baseline=%.3f, mean best=%.3f",
        len(best_configs["tasks"]),
        sum(t["baseline_sr"] for t in best_configs["tasks"].values()) / max(1, len(best_configs["tasks"])),
        sum(t["steered_sr"] for t in best_configs["tasks"].values()) / max(1, len(best_configs["tasks"])),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
