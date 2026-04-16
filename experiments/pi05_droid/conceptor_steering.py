#!/usr/bin/env python3
"""
Conceptor + Linear Steering for pi0.5 DROID (real robot)
=========================================================

Loads a pi0.5 DROID policy, installs a conceptor / linear steering hook on the
selected suffix-model layer, and serves the steered policy over WebSocket for
the real DROID client to connect to.

Unlike the LIBERO version, there is no subprocess eval loop — DROID evaluation
is manual (a human runs the robot client), so this script configures ONE
steering condition and blocks on the WebSocket server. To run another
condition, stop the server and relaunch.

Supported strategies
    baseline          No steering (plain policy serve). Same as running
                      `serve_policy.py` directly, included for parity.
    linear            ActAdd-style: h' = h + α·v  (v = unit(mean_s − mean_f)).
    global            Contrastive conceptor applied at every denoising step:
                      h' = (1−β) h + β (h @ C^T),
                      where C = C_success · (I − C_failure).
    per_step          Same as `global` but a DIFFERENT conceptor at each
                      denoising step 0..9 — swapped in each iteration of the
                      flow-matching loop via an external step counter.
    positive_only     Same as `global` but C = C_success (no contrastive NOT).
    random            Random conceptor control with matched quota.

Required on-disk inputs
    1. Conceptor npz produced by `build_conceptors.py`  →  $OPENPI_DATA_HOME/droid_conceptors.npz
    2. pi0.5 DROID checkpoint directory (--checkpoint-dir), default:
       $HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid

Usage (from repo root, inside the main openpi venv):

    uv run experiments/pi05_droid/conceptor_steering.py \\
        --task PickUpPineapple \\
        --strategy global \\
        --layer 11 \\
        --alpha 1.0 \\
        --beta 0.3
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
from typing import Any

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = pathlib.Path(
    os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
)
DEFAULT_CONCEPTOR_NPZ = OPENPI_DATA_HOME / "droid_conceptors.npz"
DEFAULT_CHECKPOINT_DIR = OPENPI_DATA_HOME / "openpi-assets" / "checkpoints" / "pi05_droid"

NUM_DENOISING_STEPS = 10
HIDDEN_DIM = 1024

STRATEGIES = ("baseline", "linear", "global", "per_step", "positive_only", "random")


# ──────────────────────────────────────────────────────────────────────────────
# Steering specs + hooks  (same as the LIBERO implementation)
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
    """Random SPD matrix with a conceptor-shaped eigenvalue profile."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper — routes infer() through infer_with_steering()
# ──────────────────────────────────────────────────────────────────────────────

class SteeredPolicyWrapper:
    """Wraps a pi0.5 policy so every infer() call applies the active hooks."""

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


def build_spec(strategy: str, npz, task: str, layer: int,
               alpha: float, beta: float, linear_alpha: float,
               random_seed: int) -> dict[int, LayerSteeringSpec] | None:
    if strategy == "baseline":
        return None
    if strategy == "linear":
        v = get_linear_contrastive(npz, task, layer)
        return {layer: LayerSteeringSpec(v=v, alpha=linear_alpha)}
    if strategy == "global":
        C = get_contrastive(npz, task, layer, alpha)
        return {layer: LayerSteeringSpec(C=C, beta=beta)}
    if strategy == "per_step":
        Cs = get_per_step_contrastive_all(npz, task, layer)
        return {layer: LayerSteeringSpec(C_per_step=Cs, beta=beta)}
    if strategy == "positive_only":
        C = get_success_only(npz, task, layer, alpha)
        return {layer: LayerSteeringSpec(C=C, beta=beta)}
    if strategy == "random":
        C_rand = compute_random_conceptor(d=HIDDEN_DIM, alpha=alpha, seed=random_seed)
        return {layer: LayerSteeringSpec(C=C_rand, beta=beta)}
    raise ValueError(f"unknown strategy {strategy!r}. Valid: {STRATEGIES}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Args:
    """Serve pi0.5 DROID with ONE steering condition applied."""

    # Task for which the conceptors were built (key into the npz).
    task: str = "PickUpPineapple"

    # Checkpoint & config.
    config: str = "pi05_droid"
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)

    # Steering strategy (see STRATEGIES).
    strategy: str = "baseline"

    # Single-condition hyperparameters.
    layer: int = 11
    alpha: float = 1.0          # conceptor aperture (global / positive_only / random)
    beta: float = 0.3           # conceptor mix weight
    linear_alpha: float = 0.5   # ActAdd scale for `linear`

    # Conceptor source.
    conceptor_npz: str = str(DEFAULT_CONCEPTOR_NPZ)

    # Serving.
    host: str = "0.0.0.0"
    port: int = 8000

    # Deterministic seed for the `random` control.
    random_seed: int = 42


def main(args: Args) -> None:
    if args.strategy not in STRATEGIES:
        raise SystemExit(f"unknown strategy {args.strategy!r}. Valid: {STRATEGIES}")

    logger.info(f"Task:       {args.task}")
    logger.info(f"Strategy:   {args.strategy}  (layer={args.layer})")

    # ── Load policy ──────────────────────────────────────────────────────────
    logger.info("Loading pi0.5 DROID policy ...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # ── Build the steering spec (if any) ─────────────────────────────────────
    spec: dict[int, LayerSteeringSpec] | None = None
    if args.strategy != "baseline":
        npz_path = pathlib.Path(args.conceptor_npz)
        if not npz_path.is_file():
            raise FileNotFoundError(f"conceptor npz not found: {npz_path}")
        logger.info(f"Loading conceptors from {npz_path}")
        npz = np.load(npz_path, allow_pickle=False)
        spec = build_spec(
            strategy=args.strategy,
            npz=npz,
            task=args.task,
            layer=args.layer,
            alpha=args.alpha,
            beta=args.beta,
            linear_alpha=args.linear_alpha,
            random_seed=args.random_seed,
        )
        logger.info(f"Steering spec: {args.strategy}  "
                    f"L{args.layer}  α={args.alpha}  β={args.beta}  "
                    f"linear_α={args.linear_alpha}")

    wrapper = SteeredPolicyWrapper(policy, device=device)
    wrapper.update_spec(spec)

    # ── Serve (blocking) ─────────────────────────────────────────────────────
    from openpi.serving import websocket_policy_server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host=args.host, port=args.port, metadata=wrapper.metadata,
    )
    logger.info(f"Serving on {args.host}:{args.port} — connect the DROID client now. Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
