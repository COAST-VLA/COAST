#!/usr/bin/env python3
"""
Conceptor + Linear Steering for pi0.5 LIBERO
=============================================

Runs one task's full steering sweep across all five strategies. The pi0.5
policy is loaded ONCE per job; the WebSocket server is spun up ONCE as a
daemon thread inside the same process, and conditions are run back-to-back by
swapping the wrapper's active steering spec. The LIBERO client is launched as
a subprocess per condition (it lives in a separate Python venv).

Supported strategies
    linear            ActAdd-style: h' = h + α·v  (v = unit(mean_s − mean_f)).
    global            Contrastive conceptor, one matrix applied at every
                      denoising step:  h' = (1−β) h + β (h @ C^T),
                      where C = C_success · (I − C_failure).
    per_step          Same as `global` but a DIFFERENT conceptor at each
                      denoising step 0..9 — swapped in each iteration of the
                      flow-matching loop via an external step counter.
    positive_only     Same as `global` but C = C_success (no contrastive NOT).
    random            Random conceptor control with matched quota; isolates
                      the effect of structure vs. any random rotation.

Required on-disk inputs
    1. Conceptor npz produced by `build_conceptors.py`  →  $OPENPI_DATA_HOME/libero_conceptors.npz
    2. pi0.5 LIBERO checkpoint directory (--checkpoint-dir)

Per-step mechanism
    Each `ConceptorSteeringHook` / `LinearSteeringHook` exposes
    `.set_denoise_step(t)`. The policy's `infer_with_steering` implementation
    must call this once at the top of each denoising iteration so the hook
    knows which of the 10 per-step matrices / vectors to apply. A helper
    `install_per_step_patch()` monkey-patches the suffix-model sampler in
    place and is applied automatically below when `--strategies per_step
    linear_per_step` are requested.

Usage (from repo root, inside the main openpi venv):

    uv run experiments/pi05_libero/for_subin/conceptor_steering.py \\
        --task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \\
        --strategies linear global per_step positive_only random \\
        --layers 11 17 \\
        --alphas 0.5 1.0 2.0 \\
        --betas 0.1 0.3 \\
        --linear-alphas 0.1 0.5 1.0 \\
        --num-episodes 15
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
import torch
import tyro

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths / task registry
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = pathlib.Path(
    os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
)
DEFAULT_CONCEPTOR_NPZ = OPENPI_DATA_HOME / "libero_conceptors.npz"

# experiments/pi05_libero/for_subin/conceptor_steering.py  →  repo root 3 levels up.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

NUM_DENOISING_STEPS = 10
HIDDEN_DIM = 1024

# LIBERO-10 task registry (name → task_id in the libero_10 benchmark suite).
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
LIBERO_TASKS = list(LIBERO_TASK_IDS)


# ──────────────────────────────────────────────────────────────────────────────
# Steering specs + hooks
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LayerSteeringSpec:
    """Describes the steering applied at a single layer.

    Exactly one of these four fields should be set per spec:
        C            → one conceptor matrix (d,d), applied every denoising step
        C_per_step   → list of 10 conceptor matrices, one per denoising step
        v            → one linear direction (d,), applied every denoising step
        v_per_step   → list of 10 linear directions, one per denoising step
    """
    C: np.ndarray | None = None
    C_per_step: list[np.ndarray] | None = None
    v: np.ndarray | None = None
    v_per_step: list[np.ndarray] | None = None
    beta: float = 0.3     # for conceptor  (h' = (1-β)h + β(h@C^T))
    alpha: float = 1.0    # for linear     (h' = h + α·v)

    def mode(self) -> str:
        if self.C is not None:           return "conceptor_global"
        if self.C_per_step is not None:  return "conceptor_per_step"
        if self.v is not None:           return "linear_global"
        if self.v_per_step is not None:  return "linear_per_step"
        raise ValueError("empty LayerSteeringSpec")


class ConceptorSteeringHook:
    """Forward hook applying h' = (1-β)h + β(h @ C^T). Supports per-step C."""

    def __init__(self, spec: LayerSteeringSpec, device):
        self.spec = spec
        self.beta = spec.beta
        self.device = device
        self.current_step = 0
        self._cache_M: dict[int, torch.Tensor] = {}
        if spec.C is not None:
            self._cache_M[-1] = self._make_M(spec.C)
        elif spec.C_per_step is not None:
            for i, Ci in enumerate(spec.C_per_step):
                self._cache_M[i] = self._make_M(Ci)
        else:
            raise ValueError("ConceptorSteeringHook requires spec.C or spec.C_per_step")
        self.intervention_norms: list[float] = []

    def _make_M(self, C: np.ndarray) -> torch.Tensor:
        d = C.shape[0]
        I = torch.eye(d, dtype=torch.float32, device=self.device)
        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        return (1 - self.beta) * I + self.beta * C_t

    def set_denoise_step(self, t: int) -> None:
        self.current_step = t

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __call__(self, module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        key = self.current_step if self.spec.C_per_step is not None else -1
        M = self._cache_M[key].to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
        return (h_steered,) + rest if rest is not None else h_steered


class LinearSteeringHook:
    """Forward hook applying h' = h + α·v. Supports per-step v."""

    def __init__(self, spec: LayerSteeringSpec, device):
        self.spec = spec
        self.alpha = spec.alpha
        self.device = device
        self.current_step = 0
        self._cache_v: dict[int, torch.Tensor] = {}
        if spec.v is not None:
            self._cache_v[-1] = torch.tensor(spec.v, dtype=torch.float32, device=device)
        elif spec.v_per_step is not None:
            for i, vi in enumerate(spec.v_per_step):
                self._cache_v[i] = torch.tensor(vi, dtype=torch.float32, device=device)
        else:
            raise ValueError("LinearSteeringHook requires spec.v or spec.v_per_step")
        self.intervention_norms: list[float] = []

    def set_denoise_step(self, t: int) -> None:
        self.current_step = t

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __call__(self, module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        key = self.current_step if self.spec.v_per_step is not None else -1
        v = self._cache_v[key].to(dtype=h.dtype)
        h_steered = h + self.alpha * v
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
        return (h_steered,) + rest if rest is not None else h_steered


def compute_random_conceptor(d: int = HIDDEN_DIM, alpha: float = 1.0, seed: int = 42) -> np.ndarray:
    """Random SPD matrix with a conceptor-shaped eigenvalue profile. Matched quota control."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper — swap steering spec without restarting the server
# ──────────────────────────────────────────────────────────────────────────────

class SteeredPolicyWrapper:
    """Wraps a pi0.5 policy to route infer() through infer_with_steering()."""

    def __init__(self, policy, device):
        self._policy = policy
        self._device = device
        self._spec: dict[int, LayerSteeringSpec] | None = None
        self._hooks: list[tuple[int, Any]] = []

    def update_spec(self, spec: dict[int, LayerSteeringSpec] | None) -> None:
        self._spec = spec
        self._hooks = []
        if spec is None:
            return
        for layer, layer_spec in spec.items():
            mode = layer_spec.mode()
            if mode.startswith("conceptor"):
                hook = ConceptorSteeringHook(layer_spec, device=self._device)
            else:
                hook = LinearSteeringHook(layer_spec, device=self._device)
            self._hooks.append((layer, hook))

    def infer(self, obs):
        if not self._hooks:
            return self._policy.infer(obs)
        for _, h in self._hooks:
            h.reset_logs()
        result, _ = self._policy.infer_with_steering(obs, steering_hooks=self._hooks)
        return result

    @property
    def metadata(self):
        return self._policy.metadata


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess-based LIBERO client
# ──────────────────────────────────────────────────────────────────────────────

SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


def run_single_task_eval(task_name: str, task_suite: str, num_episodes: int,
                         port: int, output_dir: str) -> float | None:
    task_id = LIBERO_TASK_IDS[task_name]
    libero_env_dir = REPO_ROOT / "examples" / "libero_env"
    abs_output_dir = str(pathlib.Path(output_dir).resolve())
    cmd = [
        str(libero_env_dir / ".venv" / "bin" / "python"),
        str(libero_env_dir / "main.py"),
        "--task_suite_name", task_suite,
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
    logger.error(f"No success_rate parsed from client output:\n{log_text[-2000:]}")
    return None


def start_server_background(wrapper: SteeredPolicyWrapper, port: int):
    from openpi.serving import websocket_policy_server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(5)
    logger.info(f"Server on port {port} (daemon thread)")
    return t


def run_condition(task: str, task_suite: str, port: int, num_episodes: int,
                  cond_name: str, task_output_dir: pathlib.Path) -> dict[str, Any]:
    cond_dir = task_output_dir / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)
    sr = run_single_task_eval(task, task_suite, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")
    logger.info(f"  {cond_name}: SR={sr:.3f}")
    return {"condition": cond_name, "success_rate": sr}


# ──────────────────────────────────────────────────────────────────────────────
# Summary I/O  (merge-from-disk so concurrent writers don't clobber each other)
# ──────────────────────────────────────────────────────────────────────────────

def _load_summary(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            return json.load(fh).get("conditions", []) or []
    except (json.JSONDecodeError, OSError):
        return []


def _save_summary(path: pathlib.Path, task: str, results: list[dict[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for r in _load_summary(path):
        if r and "condition" in r:
            merged[r["condition"]] = r
    for r in results:
        if r and "condition" in r:
            merged[r["condition"]] = r
    sorted_results = sorted(merged.values(),
                            key=lambda x: x.get("success_rate") or -1.0, reverse=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"task": task, "conditions": sorted_results}, fh, indent=2)
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor loaders
# ──────────────────────────────────────────────────────────────────────────────

def get_contrastive(npz, task, layer, alpha) -> np.ndarray:
    return npz[f"{task}__L{layer}__{alpha}__C_contrastive"]


def get_success_only(npz, task, layer, alpha) -> np.ndarray:
    return npz[f"{task}__L{layer}__{alpha}__C_success"]


def get_per_step_contrastive_all(npz, task, layer) -> list[np.ndarray]:
    return [npz[f"{task}__L{layer}__per_step_{t}__C_contrastive"]
            for t in range(NUM_DENOISING_STEPS)]


def get_linear_contrastive(npz, task, layer) -> np.ndarray:
    return npz[f"{task}__L{layer}__linear__V_contrastive"]


def get_per_step_linear_contrastive_all(npz, task, layer) -> list[np.ndarray]:
    return [npz[f"{task}__L{layer}__linear_per_step_{t}__V_contrastive"]
            for t in range(NUM_DENOISING_STEPS)]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    """Run all requested steering conditions for ONE LIBERO task. Policy loaded once."""

    task: str = LIBERO_TASKS[0]
    config: str = "pi05_libero"
    checkpoint_dir: str = "/path/to/pi05_libero_checkpoint"

    # Strategies to sweep. Valid: linear, global, per_step, positive_only, random.
    strategies: list[str] = dataclasses.field(
        default_factory=lambda: ["linear", "global", "per_step", "positive_only", "random"]
    )

    # Sweep axes.
    layers: list[int] = dataclasses.field(default_factory=lambda: [5, 11, 17])
    alphas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0])
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3])
    linear_alphas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0])

    # Conceptor source.
    conceptor_npz: str = str(DEFAULT_CONCEPTOR_NPZ)

    # Eval.
    task_suite_name: str = "libero_10"
    num_episodes: int = 15
    port: int = 8000

    # Random-control count: -1 = full layer×beta grid; 0 = skip; N = first N.
    n_random_controls: int = -1

    output_dir: str = "experiments/pi05_libero/steering_results"


def _log(i: int, total: int, cond: str) -> None:
    logger.info(f"  [{i}/{total}] {cond}")


def main(args: Args) -> None:
    task = args.task
    if task not in LIBERO_TASK_IDS:
        logger.error(f"unknown task {task!r}. Valid: {LIBERO_TASKS}")
        raise SystemExit(1)

    # Folder name matches existing pi05 convention: first 60 chars of task name.
    task_short = task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task:       {task}")
    logger.info(f"Output:     {task_output_dir}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Sweep:      layers={args.layers}  alphas={args.alphas}  "
                f"betas={args.betas}  linear_alphas={args.linear_alphas}")

    with open(task_output_dir / "sweep_args.json", "w") as fh:
        json.dump(dataclasses.asdict(args), fh, indent=2, default=str)

    # ── Load policy ONCE ─────────────────────────────────────────────────────
    logger.info("Loading pi0.5 LIBERO policy (one-time cost) ...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # ── Load conceptor npz ───────────────────────────────────────────────────
    npz_path = pathlib.Path(args.conceptor_npz)
    if not npz_path.is_file():
        raise FileNotFoundError(f"conceptor npz not found: {npz_path}")
    logger.info(f"Loading conceptors from {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)

    # ── Server (persistent; swap spec between conditions) ───────────────────
    wrapper = SteeredPolicyWrapper(policy, device=device)
    start_server_background(wrapper, args.port)

    # ── Resume-friendly summary ─────────────────────────────────────────────
    summary_path = task_output_dir / "summary.json"
    all_results = _load_summary(summary_path)
    done = {r["condition"] for r in all_results if r and "condition" in r}
    if done:
        logger.info(f"Resuming — {len(done)} conditions on disk.")

    def record(r):
        all_results.append(r)
        done.add(r["condition"])
        _save_summary(summary_path, task, all_results)

    def run(cond_name, spec):
        if cond_name in done:
            logger.info(f"  {cond_name} already present, skipping")
            return
        wrapper.update_spec(spec)
        record(run_condition(task, args.task_suite_name, args.port,
                             args.num_episodes, cond_name, task_output_dir))

    # ── 1. Baseline ─────────────────────────────────────────────────────────
    logger.info("\n[baseline] no steering")
    run("baseline", None)

    # ── 2. Count total conditions for progress logging ──────────────────────
    total = 0
    if "linear" in args.strategies:         total += len(args.layers) * len(args.linear_alphas)
    if "global" in args.strategies:         total += len(args.layers) * len(args.alphas) * len(args.betas)
    if "per_step" in args.strategies:       total += len(args.layers) * len(args.betas)
    if "positive_only" in args.strategies:  total += len(args.layers) * len(args.alphas) * len(args.betas)
    if "random" in args.strategies:
        if args.n_random_controls < 0:
            total += len(args.layers) * len(args.betas)
        else:
            total += args.n_random_controls
    logger.info(f"\nTotal steered conditions to run: {total}")
    idx = 0

    # ── 3. linear (ActAdd-style, control) ────────────────────────────────────
    if "linear" in args.strategies:
        logger.info("\n[linear]  h' = h + α·v")
        for layer in args.layers:
            try:
                v = get_linear_contrastive(npz, task, layer)
            except KeyError:
                logger.warning(f"  no linear V at L{layer}, skipping"); continue
            for a in args.linear_alphas:
                idx += 1
                cond = f"linear_L{layer}_la{a}"
                _log(idx, total, cond)
                spec = {layer: LayerSteeringSpec(v=v, alpha=a)}
                run(cond, spec)

    # ── 4. global conceptor ──────────────────────────────────────────────────
    if "global" in args.strategies:
        logger.info("\n[global]  h' = (1-β)h + β(h @ C_contr^T)")
        for layer in args.layers:
            for alpha in args.alphas:
                try:
                    C = get_contrastive(npz, task, layer, alpha)
                except KeyError:
                    logger.warning(f"  no C_contrastive at L{layer}/α={alpha}, skipping"); continue
                for beta in args.betas:
                    idx += 1
                    cond = f"global_L{layer}_a{alpha}_b{beta}"
                    _log(idx, total, cond)
                    spec = {layer: LayerSteeringSpec(C=C, beta=beta)}
                    run(cond, spec)

    # ── 5. per_step conceptor (one per denoising step 0..9) ─────────────────
    if "per_step" in args.strategies:
        logger.info("\n[per_step]  Swaps C_contr every denoising step (0..9)")
        for layer in args.layers:
            try:
                Cs = get_per_step_contrastive_all(npz, task, layer)
            except KeyError as e:
                logger.warning(f"  per-step missing for L{layer}: {e}. Skipping."); continue
            for beta in args.betas:
                idx += 1
                cond = f"per_step_L{layer}_b{beta}"
                _log(idx, total, cond)
                spec = {layer: LayerSteeringSpec(C_per_step=Cs, beta=beta)}
                run(cond, spec)

    # ── 6. positive_only (C = C_success, no NOT C_failure) ──────────────────
    if "positive_only" in args.strategies:
        logger.info("\n[positive_only]  C = C_success (no contrastive NOT)")
        for layer in args.layers:
            for alpha in args.alphas:
                try:
                    C = get_success_only(npz, task, layer, alpha)
                except KeyError:
                    logger.warning(f"  no C_success at L{layer}/α={alpha}, skipping"); continue
                for beta in args.betas:
                    idx += 1
                    cond = f"posonly_L{layer}_a{alpha}_b{beta}"
                    _log(idx, total, cond)
                    spec = {layer: LayerSteeringSpec(C=C, beta=beta)}
                    run(cond, spec)

    # ── 7. random control ───────────────────────────────────────────────────
    if "random" in args.strategies:
        logger.info("\n[random]  matched-quota random conceptor control")
        random_pairs = [(L, b) for L in args.layers for b in args.betas]
        if args.n_random_controls >= 0:
            random_pairs = random_pairs[: args.n_random_controls]
        for layer, beta in random_pairs:
            idx += 1
            cond = f"random_L{layer}_b{beta}"
            _log(idx, total, cond)
            C_rand = compute_random_conceptor(d=HIDDEN_DIM,
                                              seed=layer * 100 + int(beta * 10))
            spec = {layer: LayerSteeringSpec(C=C_rand, beta=beta)}
            run(cond, spec)

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{'Condition':<45s} {'SR':>8s}")
    logger.info(f"{'-' * 55}")
    for r in sorted(all_results, key=lambda x: x.get("success_rate") or -1.0, reverse=True):
        sr = r.get("success_rate")
        sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) and not np.isnan(sr) else "  nan"
        logger.info(f"{r['condition']:<45s} {sr_str:>8s}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Results: {task_output_dir}/summary.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
