"""Conceptor-based activation steering for pi0.5 policies.

This module is the single source of truth for steering primitives:

    ConceptorSteeringHook      PyTorch forward hook h' = (1-β)h + β(C @ h)
    LinearSteeringHook         PyTorch forward hook h' = h + α · v
    SteeredPolicyWrapper       Policy wrapper dispatching on an obs["__steering__"] key
    load_conceptor_npz         Load the pre-computed conceptor .npz
    get_conceptor_matrix       Look up a (task, layer, alpha, strategy) conceptor
    get_linear_direction       Look up the unit-direction vector for the linear strategy
    validate_steering_payload  Schema check for the on-wire __steering__ dict

The on-wire protocol mirrors activation_collector.py's __collect__ magic key:
the client attaches obs["__steering__"] = {...}; the wrapper pops it off and
routes through Policy.infer_with_steering.
"""

# ruff: noqa: E741, N806, RUF001, RUF002, RUF003
from __future__ import annotations

from collections.abc import Iterable
import logging
import pathlib
from typing import Any

import numpy as np

# Re-export the single source of truth for the protocol (defined in
# openpi_client.steering so sub-venv clients can import the same values).
# See that module for the math docstring per strategy.
from openpi_client.steering import ALLOWED_STRATEGIES
from openpi_client.steering import STEERING_KEY as _STEERING_KEY
import torch

logger = logging.getLogger(__name__)


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
    *,
    random_seed: int | None = None,
) -> np.ndarray:
    """Look up or derive a conceptor matrix from the .npz.

    Keyed lookups (zero compute):
      - ``global``          → ``{task}__L{layer}__{alpha}__C_contrastive``
      - ``positive_only``   → ``{task}__L{layer}__{alpha}__C_success``

    Derived at call time:
      - ``random_matched``  → build from ``C_contrastive`` at (task, layer, α)
                              via ``random_matched_conceptor``. Spectrum
                              matches, eigenvectors are random. The caller
                              must pass ``random_seed`` to get deterministic
                              output; SteeredPolicyWrapper derives the seed
                              from the full cache key.

    ``linear`` and ``per_step`` are not single-matrix strategies — use
    ``get_linear_direction`` and ``get_per_step_conceptor_matrices`` instead.
    """
    from openpi.serving.conceptors import random_matched_conceptor

    if strategy == "global":
        key = _conceptor_key(task, layer, str(alpha), "C_contrastive")
    elif strategy == "per_step":
        raise ValueError("strategy='per_step' returns a LIST of matrices; call get_per_step_conceptor_matrices instead")
    elif strategy == "positive_only":
        key = _conceptor_key(task, layer, str(alpha), "C_success")
    elif strategy == "random_matched":
        # Derived from C_contrastive — spectrum match, random eigenvectors.
        ref_key = _conceptor_key(task, layer, str(alpha), "C_contrastive")
        if ref_key not in npz:
            raise KeyError(
                f"random_matched: reference key {ref_key!r} not in NPZ. "
                f"Available: {[k for k in npz.files if k.startswith(task)][:5]}..."
            )
        if random_seed is None:
            raise ValueError("random_matched strategy requires a random_seed")
        return random_matched_conceptor(np.asarray(npz[ref_key]), seed=random_seed)
    elif strategy == "linear":
        raise ValueError("strategy='linear' returns a vector; call get_linear_direction instead")
    else:
        raise ValueError(f"Unknown steering strategy {strategy!r}. Allowed: {ALLOWED_STRATEGIES}")

    if key not in npz:
        raise KeyError(
            f"Conceptor key {key!r} not in NPZ. "
            f"Available keys starting with {task!r}: "
            f"{[k for k in npz.files if k.startswith(task)][:5]}... (truncated)"
        )
    return np.asarray(npz[key])


def get_per_step_conceptor_matrices(
    npz: Any,
    task: str,
    layer: int,
) -> list[np.ndarray]:
    """Load all 10 per-step contrastive conceptors for the ``per_step`` strategy.

    pi0.5 uses a 10-step flow-matching schedule. The NPZ built by
    ``compute_conceptors.py`` (with ``DEFAULT_PER_STEP_INDICES = tuple(range(10))``)
    ships ``per_step_0`` … ``per_step_9`` for each (task, layer). The caller
    wraps the returned list in a ``ConceptorSteeringHook`` so it can swap the
    active conceptor at each denoising step via ``set_denoise_step(t)``.

    Raises ``KeyError`` if any of the 10 keys is missing (legacy NPZs built
    before ``DEFAULT_PER_STEP_INDICES`` covered all 10 steps must be rebuilt).
    """
    _PI05_NUM_STEPS = 10
    matrices: list[np.ndarray] = []
    for t in range(_PI05_NUM_STEPS):
        key = _conceptor_key(task, layer, f"per_step_{t}", "C_contrastive")
        if key not in npz.files:
            raise KeyError(
                f"per_step: missing NPZ key {key!r}. The NPZ must contain "
                f"per_step_0..per_step_{_PI05_NUM_STEPS - 1} for this task/layer. "
                "Rebuild via experiments/{env}/compute_conceptors.py."
            )
        matrices.append(np.asarray(npz[key]))
    return matrices


def get_linear_direction(npz: Any, task: str, layer: int) -> np.ndarray:
    """Look up the linear-steering direction for a (task, layer).

    Returns a unit vector (shape ``(d,)``) produced by
    ``compute_linear_direction`` at NPZ build time. Used by the ``linear``
    strategy: at inference ``h' = h + α · v``.
    """
    key = f"{task}__L{layer}__linear_direction"
    if key not in npz:
        raise KeyError(
            f"Linear direction key {key!r} not in NPZ. "
            f"This NPZ may predate the linear strategy — rebuild with "
            f"experiments/{{env}}/compute_conceptors.py."
        )
    v = np.asarray(npz[key])
    if v.ndim != 1:
        raise ValueError(f"Expected 1-D linear direction, got shape {v.shape}")
    return v


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

    Two modes:
      - Single-matrix (default): pass one ``conceptor_matrix`` — the hook
        pre-computes ``M = (1-β)I + β·C`` and applies it at every denoising
        step. Used by strategies ``global``, ``positive_only``, and
        ``random_matched`` (the caller chooses which C to build).
      - Per-step (``matrices_per_step`` arg): pass a list of 10 conceptor
        matrices — the hook pre-computes one ``M_t`` per entry and selects
        ``M_t`` at forward time based on ``self.current_denoise_step`` (set by
        the sampler via ``set_denoise_step(t)``). Used by the ``per_step``
        strategy. List length must equal the pi0.5 schedule's 10 steps.
    """

    def __init__(
        self,
        conceptor_matrix: np.ndarray | None = None,
        beta: float = 0.3,
        device: str = "cuda",
        *,
        matrices_per_step: list[np.ndarray] | None = None,
    ) -> None:
        self.beta = float(beta)
        self.current_denoise_step = 0
        self._device = device
        self.intervention_norms: list[float] = []

        if (conceptor_matrix is None) == (matrices_per_step is None):
            raise ValueError(
                "Provide exactly one of conceptor_matrix or matrices_per_step "
                f"(got both={conceptor_matrix is not None and matrices_per_step is not None})"
            )

        if matrices_per_step is not None:
            if not matrices_per_step:
                raise ValueError("matrices_per_step must be non-empty")
            # Pre-build one M per denoising step — indexed by current_denoise_step.
            self._Ms: list[torch.Tensor] | None = [self._build_M(C) for C in matrices_per_step]
            self.M: torch.Tensor | None = None
        else:
            self._Ms = None
            self.M = self._build_M(conceptor_matrix)

    def _build_M(self, C: np.ndarray) -> torch.Tensor:  # noqa: N802, N803
        d = C.shape[0]
        I = torch.eye(d, dtype=torch.float32, device=self._device)
        Ct = torch.from_numpy(np.ascontiguousarray(C)).to(dtype=torch.float32, device=self._device)
        return (1.0 - self.beta) * I + self.beta * Ct

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        # Per-step mode picks M_t; single-matrix mode uses self.M. If
        # current_denoise_step is out of bounds, Python's IndexError surfaces —
        # that's a wiring bug (sampler and hook list length don't agree).
        M = self._Ms[self.current_denoise_step] if self._Ms is not None else self.M
        M = M.to(dtype=h.dtype)
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
        if self._Ms is not None:
            d = self._Ms[0].shape[0]
            return f"ConceptorSteeringHook(dim={d}, beta={self.beta}, per_step={len(self._Ms)})"
        d = self.M.shape[0]
        return f"ConceptorSteeringHook(dim={d}, beta={self.beta})"


def compute_random_conceptor(d: int = 1024, alpha: float = 0.5, seed: int = 42) -> np.ndarray:
    """Random symmetric conceptor for control experiments."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha**-2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


class LinearSteeringHook:
    """PyTorch forward hook that applies h' = h + alpha * v.

    Additive ActAdd-style intervention, in contrast to the multiplicative
    matrix blending in ``ConceptorSteeringHook``. ``v`` is a unit direction
    vector (e.g. from ``compute_linear_direction``); ``alpha`` is the
    magnitude. No β parameter — the interpolation weight is baked into ``alpha``.
    """

    def __init__(self, direction: np.ndarray, alpha: float = 1.0, device: str = "cuda") -> None:
        if direction.ndim != 1:
            raise ValueError(f"LinearSteeringHook expects 1-D direction, got shape {direction.shape}")
        self.alpha = float(alpha)
        self.current_denoise_step = 0
        # Broadcasts against (batch, seq, d) additions.
        self.v = torch.from_numpy(np.ascontiguousarray(direction)).to(dtype=torch.float32, device=device)
        self.intervention_norms: list[float] = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        v = self.v.to(dtype=h.dtype)
        delta = self.alpha * v  # (d,); broadcasts to h
        h_steered = h + delta
        self.intervention_norms.append(torch.norm(h_steered - h).item())
        if rest is not None:
            return (h_steered, *rest)
        return h_steered

    def set_denoise_step(self, t: int) -> None:
        self.current_denoise_step = int(t)

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __repr__(self) -> str:
        d = self.v.shape[0]
        return f"LinearSteeringHook(dim={d}, alpha={self.alpha})"


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
        # Cache key = (task, layer, alpha, beta, strategy).
        self._hook_cache: dict[tuple[str, int, float, float, str], Any] = {}

    def _get_or_build_hook(self, payload: dict) -> tuple[int, Any]:
        key = (
            payload["task"],
            int(payload["layer"]),
            float(payload["alpha"]),
            float(payload["beta"]),
            payload["strategy"],
        )
        hook = self._hook_cache.get(key)
        if hook is None:
            strategy = key[4]
            if strategy == "linear":
                # Linear: h' = h + alpha * v. Beta is ignored (but kept in the cache
                # key for consistency so (linear, α=0.1, β=0.3) and (linear, α=0.1, β=0.1)
                # share the hook via re-entry — which is fine because the hook's
                # behavior only depends on (α, v)).
                v = get_linear_direction(self._npz, task=key[0], layer=key[1])
                hook = LinearSteeringHook(v, alpha=key[2], device=self._device)
            elif strategy == "random_matched":
                # Deterministic seed from the full cache key so repeat requests
                # produce the same random matrix.
                seed = abs(hash(key)) % (2**31)
                C = get_conceptor_matrix(
                    self._npz,
                    task=key[0],
                    layer=key[1],
                    alpha=key[2],
                    strategy=strategy,
                    random_seed=seed,
                )
                hook = ConceptorSteeringHook(C, beta=key[3], device=self._device)
            elif strategy == "per_step":
                # Load all 10 per-step conceptors, build a list of 10 M_t matrices
                # aligned to the pi0.5 sampler's step counter 0..9.
                matrices = get_per_step_conceptor_matrices(
                    self._npz,
                    task=key[0],
                    layer=key[1],
                )
                hook = ConceptorSteeringHook(
                    beta=key[3],
                    device=self._device,
                    matrices_per_step=matrices,
                )
            else:
                # global, positive_only — single-matrix strategies.
                C = get_conceptor_matrix(
                    self._npz,
                    task=key[0],
                    layer=key[1],
                    alpha=key[2],
                    strategy=strategy,
                )
                hook = ConceptorSteeringHook(C, beta=key[3], device=self._device)
            self._hook_cache[key] = hook
            logger.info("Built steering hook %s [%s] (cache size=%d)", key, type(hook).__name__, len(self._hook_cache))
        else:
            hook.reset_logs()
        return key[1], hook

    def infer(self, obs: dict) -> dict:
        payload = obs.pop(_STEERING_KEY, None) if isinstance(obs, dict) else None
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
