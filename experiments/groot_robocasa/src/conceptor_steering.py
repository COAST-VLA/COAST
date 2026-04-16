#!/usr/bin/env python3
"""
Conceptor + Linear Steering for GR00T N1.5 RoboCasa
===================================================

Loads pre-computed conceptors (and linear-steering directions) from
`$OPENPI_DATA_HOME/groot_n15_robocasa_conceptors.npz` and evaluates steered
policy rollouts on RoboCasa via the in-process WebSocket server + subprocess
client pattern (mirrors `pi05_robocasa/src/conceptor_steering.py`).

One job per task — each job:
    1. Loads the GR00T N1.5 policy ONCE on GPU.
    2. Runs baseline (no steering) once.
    3. Sweeps all (strategy × layer × α × β) conditions against the one
       loaded policy by mutating the shared `SteeredPolicyWrapper`.
    4. Runs random-conceptor controls.
    5. Writes results incrementally to `output_dir/{task}/summary.json`.

Strategies (all reuse the same npz):
    - global         : single conceptor, applied at every denoising step
    - per_step       : a DIFFERENT conceptor at each of 4 denoising steps
                        (the per_step_{ds}__C_contrastive matrices).
    - linear         : ActAdd-style h' = h + α·v, v = unit(mean_s - mean_f),
                        one vector per (task, layer) applied at every step.
    - linear_per_step: linear, but per-denoising-step direction vector.

Wire protocol matches `groot_env/serve.py`. Client is
`examples/robocasa_env/main.py`, run via its own Python 3.11 venv.

Run (from `openpi-groot/groot_env`):

    cd groot_env
    CUDA_VISIBLE_DEVICES=0 uv run python \\
        ../experiments/groot_robocasa/src/conceptor_steering.py \\
        --task CloseFridge \\
        --layers 5 8 11 \\
        --strategies global per_step linear
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np
import tyro

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = pathlib.Path(
    os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
)
DEFAULT_CONCEPTOR_NPZ = OPENPI_DATA_HOME / "groot_n15_robocasa_conceptors.npz"

# This file lives at:  <REPO>/experiments/groot_robocasa/src/conceptor_steering.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Mixed-outcome tasks — the other two (PickPlaceCounterToCabinet, ToStove) have
# no failure episodes in the current activation collection, so contrastive
# conceptors cannot be built. Flag them to upstream collection if needed.
GROOT_ROBOCASA_TASKS = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]

NUM_DENOISING_STEPS = 4
HIDDEN_DIM = 1536

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor / direction loaders
# ──────────────────────────────────────────────────────────────────────────────

def _require_key(npz, key: str) -> np.ndarray:
    if key not in npz:
        raise KeyError(f"Conceptor npz missing key: {key}")
    return npz[key]


def load_global_conceptor(npz, task: str, layer: int, alpha: float) -> np.ndarray:
    return _require_key(npz, f"{task}__L{layer}__{alpha}__C_contrastive")


def load_per_step_conceptors(npz, task: str, layer: int) -> list[np.ndarray]:
    return [
        _require_key(npz, f"{task}__L{layer}__per_step_{t}__C_contrastive")
        for t in range(NUM_DENOISING_STEPS)
    ]


def load_linear_direction(npz, task: str, layer: int) -> np.ndarray:
    return _require_key(npz, f"{task}__L{layer}__linear__V_contrastive")


def load_per_step_linear_directions(npz, task: str, layer: int) -> list[np.ndarray]:
    return [
        _require_key(npz, f"{task}__L{layer}__linear_per_step_{t}__V_contrastive")
        for t in range(NUM_DENOISING_STEPS)
    ]


def load_positive_only_conceptor(npz, task: str, layer: int, alpha: float) -> np.ndarray:
    """Load C_success (positive-only, no contrastive NOT)."""
    return _require_key(npz, f"{task}__L{layer}__{alpha}__C_success")


def compute_random_conceptor(d: int = HIDDEN_DIM, alpha: float = 1.0, seed: int = 42) -> np.ndarray:
    """Random conceptor control: PSD matrix with a conceptor-like spectrum."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper that routes infer() through infer_with_steering()
# ──────────────────────────────────────────────────────────────────────────────

class SteeredPolicyWrapper:
    """Wraps `GR00TAdapterPolicy` so the WebSocket server can call `.infer(obs)`
    while we mutate the steering spec between conditions."""

    def __init__(self, policy, steering_spec=None, metadata: dict[str, Any] | None = None):
        self._policy = policy
        self._spec = steering_spec or {}
        self._extra_metadata = metadata or {}

    def update_spec(self, spec) -> None:
        self._spec = spec or {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._spec:
            return self._policy.infer_with_steering(obs, self._spec)
        return self._policy.infer(obs)

    @property
    def metadata(self) -> dict[str, Any]:
        base = dict(self._extra_metadata)
        base["steering_active"] = bool(self._spec)
        return base


def start_server_background(wrapper: SteeredPolicyWrapper, port: int) -> threading.Thread:
    import websocket_policy_server  # provided by groot_env; same as serve.py

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(5)  # let the socket bind
    logger.info(f"GR00T N1.5 steering server started on port {port} (daemon thread)")
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Eval subprocess (robocasa_env/.venv Python 3.11 client)
# ──────────────────────────────────────────────────────────────────────────────

def run_single_task_eval(task_name: str, num_episodes: int, port: int, output_dir: str) -> float | None:
    robocasa_env_dir = REPO_ROOT / "examples" / "robocasa_env"
    client_python = robocasa_env_dir / ".venv" / "bin" / "python"
    if not client_python.is_file():
        raise FileNotFoundError(
            f"robocasa client venv not found at {client_python}. "
            f"Run  cd {robocasa_env_dir} && uv sync  first."
        )
    cmd = [
        str(client_python),
        str(robocasa_env_dir / "main.py"),
        "--env_name", task_name,
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", str(pathlib.Path(output_dir).resolve()),
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info(f"  eval: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, cwd=str(robocasa_env_dir), env=env,
        capture_output=True, text=True, timeout=14400,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error(f"eval subprocess failed (rc={proc.returncode}):\n{log_text[-3000:]}")
        return None
    matches = SUCCESS_RATE_RE.findall(log_text)
    if matches:
        return float(matches[-1])
    logger.error(f"no success_rate found in client output:\n{log_text[-2000:]}")
    return None


def run_condition(task: str, port: int, num_episodes: int,
                  condition_name: str, task_output_dir: pathlib.Path) -> dict[str, Any]:
    cond_dir = task_output_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    sr = run_single_task_eval(task, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")
    logger.info(f"  {condition_name}: SR={sr:.3f}")
    return {"condition": condition_name, "success_rate": sr}


# ──────────────────────────────────────────────────────────────────────────────
# Spec builders — one per strategy
# ──────────────────────────────────────────────────────────────────────────────

def build_conceptor_spec(layer: int, C: np.ndarray, beta: float):
    from groot_steering import LayerSteeringSpec
    return {layer: LayerSteeringSpec(C=C, beta=beta)}


def build_per_step_conceptor_spec(layer: int, Cs: list[np.ndarray], beta: float):
    from groot_steering import LayerSteeringSpec
    return {layer: LayerSteeringSpec(C_per_step=Cs, beta=beta)}


def build_linear_spec(layer: int, v: np.ndarray, alpha: float):
    from groot_steering import LayerSteeringSpec
    return {layer: LayerSteeringSpec(v=v, alpha=alpha)}


def build_per_step_linear_spec(layer: int, vs: list[np.ndarray], alpha: float):
    from groot_steering import LayerSteeringSpec
    return {layer: LayerSteeringSpec(v_per_step=vs, alpha=alpha)}


# ──────────────────────────────────────────────────────────────────────────────
# Summary I/O (merge-from-disk to survive concurrent jobs on the same task)
# ──────────────────────────────────────────────────────────────────────────────

def _load_summary(summary_path: pathlib.Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    try:
        with open(summary_path) as fh:
            return json.load(fh).get("conditions", []) or []
    except (json.JSONDecodeError, OSError):
        return []


def _save_summary(summary_path: pathlib.Path, task: str, results: list[dict[str, Any]]) -> None:
    # Merge with any concurrent writer's copy on disk.
    merged: dict[str, dict[str, Any]] = {}
    for r in _load_summary(summary_path):
        if r and "condition" in r:
            merged[r["condition"]] = r
    for r in results:
        if r and "condition" in r:
            merged[r["condition"]] = r
    sorted_results = sorted(merged.values(),
                            key=lambda x: x.get("success_rate") or -1.0, reverse=True)
    tmp = summary_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"task": task, "conditions": sorted_results}, fh, indent=2)
    tmp.replace(summary_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    """Run all steering conditions for ONE task. GR00T N1.5 policy loaded once."""

    # Which task's conceptors to use, and to eval on.
    task: str = GROOT_ROBOCASA_TASKS[0]

    # Policy bring-up (matches groot_env/serve.py defaults).
    checkpoint_dir: str = (
        "../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"
    )
    device: str = "cuda:0"
    denoising_steps: int = NUM_DENOISING_STEPS

    # Conceptor npz (must be produced by build_conceptors.py).
    conceptor_npz: str = str(DEFAULT_CONCEPTOR_NPZ)

    # Sweep axes — keep defaults narrow. In practice, `select_parameters.py`
    # writes a JSON with one layer + a few alphas; pass those here.
    layers: list[int] = dataclasses.field(default_factory=lambda: [5, 8, 11])
    alphas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0])
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3])
    # Scale factors for linear (ActAdd-style) steering control.
    linear_alphas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0])
    # Which strategies to sweep. Conceptor: {global, per_step, positive_only}.
    # Linear (control): {linear, linear_per_step}. Random is a separate phase.
    strategies: list[str] = dataclasses.field(
        default_factory=lambda: ["global", "per_step", "positive_only", "linear", "linear_per_step"]
    )

    # Eval.
    num_episodes: int = 15
    port: int = 8000

    # Random-conceptor control count. -1 = full layer×beta grid, 0 = skip, N = first N.
    n_random_controls: int = -1

    output_dir: str = "experiments/groot_robocasa/steering_results"


def _log_progress(label: str, i: int, total: int) -> None:
    logger.info(f"  [{i}/{total}] {label}")


def main(args: Args) -> None:
    task = args.task
    if task not in GROOT_ROBOCASA_TASKS:
        logger.warning(
            f"task={task!r} is not in the default mixed-outcome list "
            f"{GROOT_ROBOCASA_TASKS}. Proceeding anyway."
        )

    task_output_dir = pathlib.Path(args.output_dir) / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task:    {task}")
    logger.info(f"Output:  {task_output_dir}")
    logger.info(
        f"Sweep:   layers={args.layers}  alphas={args.alphas}  betas={args.betas}  "
        f"linear_alphas={args.linear_alphas}  strategies={args.strategies}"
    )

    with open(task_output_dir / "sweep_args.json", "w") as fh:
        json.dump(dataclasses.asdict(args), fh, indent=2, default=str)

    # ── Load GR00T N1.5 policy (once) ────────────────────────────────────────
    logger.info("Loading GR00T N1.5 policy (one-time cost) ...")
    # groot_adapter.py lives in groot_env/ (not on sys.path by default when this
    # script is invoked via `uv run python .../src/conceptor_steering.py`).
    import sys as _sys
    _groot_env_dir = str(REPO_ROOT / "groot_env")
    if _groot_env_dir not in _sys.path:
        _sys.path.insert(0, _groot_env_dir)
    import groot_adapter  # runs inside groot_env venv

    policy = groot_adapter.make_robocasa_policy(
        model_path=args.checkpoint_dir,
        device=args.device,
        denoising_steps=args.denoising_steps,
    )
    metadata = {
        "backend": "groot_n15",
        "model_path": args.checkpoint_dir,
        "embodiment": "robocasa",
        "denoising_steps": args.denoising_steps,
        "steering": True,
    }
    logger.info(f"Policy loaded on {args.device}")

    # ── Load conceptor npz ───────────────────────────────────────────────────
    conceptor_npz_path = pathlib.Path(args.conceptor_npz)
    if not conceptor_npz_path.is_file():
        raise FileNotFoundError(f"conceptor npz not found: {conceptor_npz_path}")
    logger.info(f"Loading conceptors from {conceptor_npz_path}")
    npz = np.load(conceptor_npz_path, allow_pickle=False)

    # ── Start server (reusable wrapper; we swap specs between conditions) ────
    wrapper = SteeredPolicyWrapper(policy, steering_spec=None, metadata=metadata)
    start_server_background(wrapper, args.port)

    # ── Load prior progress (resume-friendly) ────────────────────────────────
    summary_path = task_output_dir / "summary.json"
    all_results = _load_summary(summary_path)
    done = {r["condition"] for r in all_results if r and "condition" in r}
    if done:
        logger.info(f"Resuming — {len(done)} conditions already on disk.")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    def record(r: dict[str, Any]) -> None:
        nonlocal consecutive_failures
        all_results.append(r)
        done.add(r["condition"])
        _save_summary(summary_path, task, all_results)
        sr = r.get("success_rate")
        if sr is None or (isinstance(sr, float) and np.isnan(sr)):
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    f"Aborting: {consecutive_failures} consecutive eval failures "
                    f"— server is likely dead. Check logs."
                )
                raise RuntimeError(
                    f"{consecutive_failures} consecutive NaN results — aborting sweep"
                )
        else:
            consecutive_failures = 0

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    if "baseline" in done:
        logger.info("\n[1/4] Baseline already present — skipping.")
    else:
        logger.info("\n[1/4] Baseline (no steering) ...")
        wrapper.update_spec(None)
        record(run_condition(task, args.port, args.num_episodes, "baseline", task_output_dir))

    # ── 2. Steered conditions ────────────────────────────────────────────────
    # Precount total for progress logging. per_step's α is baked into the npz
    # (all per_step_ds conceptors are stored at build_conceptors' per_step_alpha,
    # default 1.0), so per_step sweeps only (layer × β), not (layer × α × β).
    total = 0
    for s in args.strategies:
        if s == "global":
            total += len(args.layers) * len(args.alphas) * len(args.betas)
        elif s == "per_step":
            total += len(args.layers) * len(args.betas)
        elif s == "positive_only":
            total += len(args.layers) * len(args.alphas) * len(args.betas)
        elif s in ("linear", "linear_per_step"):
            total += len(args.layers) * len(args.linear_alphas)
    logger.info(f"\n[2/4] Steered conditions ({total} total) ...")

    idx = 0
    for layer in args.layers:
        # Lazy-load directions/conceptors only once per layer.
        cached: dict[str, Any] = {}

        for strategy in args.strategies:
            if strategy == "global":
                for alpha in args.alphas:
                    for beta in args.betas:
                        idx += 1
                        cond = f"global_L{layer}_a{alpha}_b{beta}"
                        if cond in done:
                            _log_progress(f"{cond} — skip (already done)", idx, total)
                            continue
                        _log_progress(cond, idx, total)
                        C = cached.get(("global", alpha))
                        if C is None:
                            C = load_global_conceptor(npz, task, layer, alpha)
                            cached[("global", alpha)] = C
                        wrapper.update_spec(build_conceptor_spec(layer, C, beta))
                        record(run_condition(task, args.port, args.num_episodes,
                                             cond, task_output_dir))

            elif strategy == "per_step":
                Cs = cached.get("per_step")
                if Cs is None:
                    Cs = load_per_step_conceptors(npz, task, layer)
                    cached["per_step"] = Cs
                for beta in args.betas:
                    idx += 1
                    cond = f"per_step_L{layer}_b{beta}"
                    if cond in done:
                        _log_progress(f"{cond} — skip (already done)", idx, total)
                        continue
                    _log_progress(cond, idx, total)
                    wrapper.update_spec(build_per_step_conceptor_spec(layer, Cs, beta))
                    record(run_condition(task, args.port, args.num_episodes,
                                         cond, task_output_dir))

            elif strategy == "positive_only":
                for alpha in args.alphas:
                    for beta in args.betas:
                        idx += 1
                        cond = f"pos_only_L{layer}_a{alpha}_b{beta}"
                        if cond in done:
                            _log_progress(f"{cond} — skip (already done)", idx, total)
                            continue
                        _log_progress(cond, idx, total)
                        C = cached.get(("pos_only", alpha))
                        if C is None:
                            C = load_positive_only_conceptor(npz, task, layer, alpha)
                            cached[("pos_only", alpha)] = C
                        wrapper.update_spec(build_conceptor_spec(layer, C, beta))
                        record(run_condition(task, args.port, args.num_episodes,
                                             cond, task_output_dir))

            elif strategy == "linear":
                v = cached.get("linear")
                if v is None:
                    v = load_linear_direction(npz, task, layer)
                    cached["linear"] = v
                for la in args.linear_alphas:
                    idx += 1
                    cond = f"linear_L{layer}_la{la}"
                    if cond in done:
                        _log_progress(f"{cond} — skip (already done)", idx, total)
                        continue
                    _log_progress(cond, idx, total)
                    wrapper.update_spec(build_linear_spec(layer, v, la))
                    record(run_condition(task, args.port, args.num_episodes,
                                         cond, task_output_dir))

            elif strategy == "linear_per_step":
                vs = cached.get("linear_per_step")
                if vs is None:
                    vs = load_per_step_linear_directions(npz, task, layer)
                    cached["linear_per_step"] = vs
                for la in args.linear_alphas:
                    idx += 1
                    cond = f"linear_per_step_L{layer}_la{la}"
                    if cond in done:
                        _log_progress(f"{cond} — skip (already done)", idx, total)
                        continue
                    _log_progress(cond, idx, total)
                    wrapper.update_spec(build_per_step_linear_spec(layer, vs, la))
                    record(run_condition(task, args.port, args.num_episodes,
                                         cond, task_output_dir))

            else:
                logger.warning(f"unknown strategy {strategy!r}, skipping")

    # ── 3. Random-conceptor control ──────────────────────────────────────────
    random_pairs = [(L, b) for L in args.layers for b in args.betas]
    if args.n_random_controls == 0:
        random_pairs = []
    elif args.n_random_controls > 0:
        random_pairs = random_pairs[: args.n_random_controls]
    logger.info(f"\n[3/4] Random controls ({len(random_pairs)}) ...")
    from groot_steering import LayerSteeringSpec
    for layer, beta in random_pairs:
        cond = f"random_L{layer}_b{beta}"
        if cond in done:
            logger.info(f"  {cond} — skip (already done)")
            continue
        C_rand = compute_random_conceptor(seed=layer * 100 + int(beta * 10))
        wrapper.update_spec({layer: LayerSteeringSpec(C=C_rand, beta=beta)})
        record(run_condition(task, args.port, args.num_episodes, cond, task_output_dir))

    # ── 4. Final summary ─────────────────────────────────────────────────────
    logger.info("\n[4/4] Done. Summary:")
    logger.info("=" * 66)
    logger.info(f"{'Condition':<48s} {'SR':>8s}")
    logger.info("-" * 58)
    for r in sorted(all_results, key=lambda x: x.get("success_rate") or -1.0, reverse=True):
        sr = r.get("success_rate")
        sr_str = "nan" if sr is None or (isinstance(sr, float) and np.isnan(sr)) else f"{sr:.3f}"
        logger.info(f"{r['condition']:<48s} {sr_str:>8s}")
    logger.info("=" * 66)
    logger.info(f"All results for {task} saved to {task_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
