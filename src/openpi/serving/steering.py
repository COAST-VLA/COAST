"""Conceptor-based activation steering for pi0.5 policies.

This module is the single source of truth for steering primitives:

    ConceptorSteeringHook      PyTorch forward hook h' = (1-β)h + β(C @ h)
    SteeredPolicyWrapper       Policy wrapper dispatching on an obs["__steering__"] key
    load_conceptor_npz         Load the pre-computed conceptor .npz
    get_conceptor_matrix       Look up a (task, layer, alpha, strategy) conceptor
    validate_steering_payload  Schema check for the on-wire __steering__ dict
    validate_best_configs_json Schema check for experiments/{env}/best_configs.json

Both the server (scripts/serve_policy.py --steer) and the experiment sweep
driver (experiments/{env}/find_best_configs.py) import from here.

The on-wire protocol mirrors activation_collector.py's __collect__ magic key:
the client attaches obs["__steering__"] = {...}; the wrapper pops it off and
routes through Policy.infer_with_steering.
"""

# ruff: noqa: E741, N806, RUF001, RUF002, RUF003
from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import pathlib
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults — single source of truth
# Duplicated in examples/{libero,robocasa}_env/main.py because those scripts
# run in sub-venvs without openpi. Keep the values in sync.
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_STEERING_LAYER = 11
DEFAULT_STEERING_ALPHA = 0.1
DEFAULT_STEERING_BETA = 0.3
DEFAULT_STEERING_STRATEGY = "global"
ALLOWED_STRATEGIES: tuple[str, ...] = ("global", "per_step_0", "per_step_9")

# On-wire magic key — matches the __collect__ / __finalize_episode__ pattern.
STEERING_KEY = "__steering__"


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor NPZ loading
# ──────────────────────────────────────────────────────────────────────────────


def load_conceptor_npz(path: str | pathlib.Path) -> Any:
    """Load a pre-computed conceptor .npz (memory-mapped NpzFile)."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Conceptor file not found: {path}. "
            "Download from brandonyang/libero-conceptors or brandonyang/robocasa-conceptors."
        )
    return np.load(path, allow_pickle=True)


def _conceptor_key(task: str, layer: int, alpha_or_step: str, kind: str) -> str:
    return f"{task}__L{layer}__{alpha_or_step}__{kind}"


def get_conceptor_matrix(
    npz: Any,
    task: str,
    layer: int,
    alpha: float,
    strategy: str,
) -> np.ndarray:
    """Look up a conceptor matrix from the .npz.

    For ``strategy == "global"``, the key is ``{task}__L{layer}__{alpha}__C_contrastive``.
    For ``strategy == "per_step_N"``, the key is ``{task}__L{layer}__per_step_N__C_contrastive``
    (alpha is ignored per the miranda-v2 convention — per-step conceptors are
    computed at a fixed alpha baked into the NPZ).
    """
    if strategy == "global":
        key = _conceptor_key(task, layer, str(alpha), "C_contrastive")
    elif strategy.startswith("per_step_"):
        step = strategy.split("_")[-1]
        key = _conceptor_key(task, layer, f"per_step_{step}", "C_contrastive")
    else:
        raise ValueError(f"Unknown steering strategy {strategy!r}. Allowed: {ALLOWED_STRATEGIES}")

    if key not in npz:
        raise KeyError(
            f"Conceptor key {key!r} not in NPZ. "
            f"Available keys starting with {task!r}: "
            f"{[k for k in npz.files if k.startswith(task)][:5]}... (truncated)"
        )
    return np.asarray(npz[key])


def available_tasks(npz: Any) -> set[str]:
    """Return the set of task names present in the NPZ (derived from key prefixes)."""
    tasks: set[str] = set()
    for key in npz.files:
        # Keys are "<task>__L<layer>__<alpha_or_per_step_N>__C_{contrastive|success|failure}"
        # Task can contain underscores, so split on "__" (double underscore).
        task, _, _ = key.partition("__L")
        if task:
            tasks.add(task)
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Steering hook
# ──────────────────────────────────────────────────────────────────────────────


class ConceptorSteeringHook:
    """PyTorch forward hook that applies h' = (1-β)h + β(C @ h).

    Pre-computes the interpolated matrix ``M = (1-β)I + β·C`` once at
    construction so each forward pass is a single matmul.
    """

    def __init__(self, conceptor_matrix: np.ndarray, beta: float = 0.3, device: str = "cuda") -> None:
        self.beta = float(beta)
        self.current_denoise_step = 0
        d = conceptor_matrix.shape[0]
        I = torch.eye(d, dtype=torch.float32, device=device)
        C = torch.from_numpy(np.ascontiguousarray(conceptor_matrix)).to(dtype=torch.float32, device=device)
        self.M = (1.0 - self.beta) * I + self.beta * C
        self.intervention_norms: list[float] = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        M = self.M.to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(torch.norm(h_steered - h).item())
        if rest is not None:
            return (h_steered, *rest)
        return h_steered

    def set_denoise_step(self, t: int) -> None:
        self.current_denoise_step = int(t)

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __repr__(self) -> str:
        d = self.M.shape[0]
        return f"ConceptorSteeringHook(dim={d}, beta={self.beta})"


def compute_random_conceptor(d: int = 1024, alpha: float = 0.5, seed: int = 42) -> np.ndarray:
    """Random symmetric conceptor for control experiments."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha**-2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Steering payload validation (server-side)
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_PAYLOAD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "task": str,
    "layer": int,
    "alpha": (int, float),
    "beta": (int, float),
    "strategy": str,
}


def validate_steering_payload(payload: Any, available: Iterable[str]) -> None:
    """Raise ``ValueError`` if the client-supplied __steering__ dict is malformed.

    Checked fields: presence, types, ``strategy in ALLOWED_STRATEGIES``,
    ``task in available`` (the NPZ's task set).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"__steering__ payload must be a dict, got {type(payload).__name__}")

    missing = [k for k in _REQUIRED_PAYLOAD_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"__steering__ payload missing required fields: {missing}")

    for key, expected in _REQUIRED_PAYLOAD_FIELDS.items():
        if not isinstance(payload[key], expected):  # type: ignore[arg-type]
            raise ValueError(f"__steering__.{key} must be {expected}, got {type(payload[key]).__name__}")

    if payload["strategy"] not in ALLOWED_STRATEGIES:
        raise ValueError(f"__steering__.strategy {payload['strategy']!r} not in {ALLOWED_STRATEGIES}")

    available_set = set(available)
    if payload["task"] not in available_set:
        # Show a few candidates to aid debugging.
        hint = sorted(available_set)[:3]
        raise ValueError(f"__steering__.task {payload['task']!r} not found in conceptor NPZ (example keys: {hint})")


# ──────────────────────────────────────────────────────────────────────────────
# best_configs.json validation (client-side, startup-time)
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_CONFIG_FIELDS: dict[str, type | tuple[type, ...]] = {
    "layer": int,
    "alpha": (int, float),
    "beta": (int, float),
    "strategy": str,
}


def validate_best_configs_json(path: str | pathlib.Path) -> dict:
    """Load and validate a ``best_configs.json`` file.

    Returns the parsed dict. Raises ``ValueError`` with a specific message on
    schema violations so the eval_all caller can fail fast before spawning any
    subprocesses.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"best_configs file not found: {path}")

    with open(path) as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"best_configs.json root must be a dict, got {type(cfg).__name__}")
    if "tasks" not in cfg or not isinstance(cfg["tasks"], dict):
        raise ValueError("best_configs.json must have a 'tasks' dict")

    for task_name, task_cfg in cfg["tasks"].items():
        if not isinstance(task_cfg, dict):
            raise ValueError(f"tasks[{task_name!r}] must be a dict")
        for key, expected in _REQUIRED_CONFIG_FIELDS.items():
            if key not in task_cfg:
                raise ValueError(f"tasks[{task_name!r}] missing field {key!r}")
            if not isinstance(task_cfg[key], expected):  # type: ignore[arg-type]
                raise ValueError(f"tasks[{task_name!r}].{key} must be {expected}, got {type(task_cfg[key]).__name__}")
        if task_cfg["strategy"] not in ALLOWED_STRATEGIES:
            raise ValueError(f"tasks[{task_name!r}].strategy {task_cfg['strategy']!r} not in {ALLOWED_STRATEGIES}")

    if "defaults" in cfg:
        for key, expected in _REQUIRED_CONFIG_FIELDS.items():
            if key not in cfg["defaults"]:
                raise ValueError(f"defaults missing field {key!r}")
            if not isinstance(cfg["defaults"][key], expected):  # type: ignore[arg-type]
                raise ValueError(f"defaults.{key} must be {expected}, got {type(cfg['defaults'][key]).__name__}")
        if cfg["defaults"]["strategy"] not in ALLOWED_STRATEGIES:
            raise ValueError(f"defaults.strategy {cfg['defaults']['strategy']!r} not in {ALLOWED_STRATEGIES}")

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# SteeredPolicyWrapper — on-wire __steering__ dispatch
# ──────────────────────────────────────────────────────────────────────────────


class SteeredPolicyWrapper:
    """Wrap a ``Policy`` so it honors an obs[\"__steering__\"] dict.

    On every ``infer(obs)`` call:
      - If obs has no ``__steering__`` key: passes through to ``Policy.infer``.
      - If obs has a ``__steering__`` dict: validates it, looks up (or builds)
        the cached ``ConceptorSteeringHook`` for that (task, layer, alpha, beta,
        strategy) tuple, and calls ``Policy.infer_with_steering``.

    Hooks are cached on the wrapper instance so repeated configs — common
    within a single-task eval loop — don't rebuild the 1024×1024 M matrix.
    """

    def __init__(self, policy: Any, conceptor_npz_path: str | pathlib.Path, device: str) -> None:
        self._policy = policy
        self._npz = load_conceptor_npz(conceptor_npz_path)
        self._available_tasks = available_tasks(self._npz)
        self._device = device
        self._hook_cache: dict[tuple[str, int, float, float, str], ConceptorSteeringHook] = {}

    def _get_or_build_hook(self, payload: dict) -> tuple[int, ConceptorSteeringHook]:
        key = (
            payload["task"],
            int(payload["layer"]),
            float(payload["alpha"]),
            float(payload["beta"]),
            payload["strategy"],
        )
        hook = self._hook_cache.get(key)
        if hook is None:
            C = get_conceptor_matrix(
                self._npz,
                task=key[0],
                layer=key[1],
                alpha=key[2],
                strategy=key[4],
            )
            hook = ConceptorSteeringHook(C, beta=key[3], device=self._device)
            self._hook_cache[key] = hook
            logger.info("Built steering hook %s (cache size=%d)", key, len(self._hook_cache))
        else:
            hook.reset_logs()
        return key[1], hook

    def infer(self, obs: dict) -> dict:
        payload = obs.pop(STEERING_KEY, None) if isinstance(obs, dict) else None
        if payload is None:
            return self._policy.infer(obs)
        validate_steering_payload(payload, self._available_tasks)
        layer, hook = self._get_or_build_hook(payload)
        result, _ = self._policy.infer_with_steering(obs, steering_hooks=[(layer, hook)])
        return result

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def metadata(self) -> dict:
        underlying = getattr(self._policy, "metadata", {}) or {}
        return {**underlying, "steering_enabled": True, "num_conceptor_tasks": len(self._available_tasks)}
