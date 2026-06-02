"""Conceptor steering for the GR00T N1.5 RoboCasa server.

This module intentionally lives in ``groot_env`` instead of
``src/openpi/serving/steering.py`` because GR00T uses an isolated Python 3.10
venv and does not import the root ``openpi`` package. The wire protocol is still
shared through ``openpi_client.steering``.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from openpi_client.steering import ALLOWED_STRATEGIES, STEERING_KEY

logger = logging.getLogger(__name__)


_REQUIRED_PAYLOAD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "task": str,
    "layer": int,
    "alpha": (int, float),
    "beta": (int, float),
    "strategy": str,
}


def load_conceptor_npz(path: str | pathlib.Path) -> Any:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Conceptor file not found: {path}. Build GR00T conceptors from "
            "groot_v1 activations or pass a compatible NPZ."
        )
    return np.load(path, allow_pickle=True)


def _conceptor_key(task: str, layer: int, alpha_or_step: str, kind: str) -> str:
    return f"{task}__L{layer}__{alpha_or_step}__{kind}"


def available_tasks(npz: Any) -> set[str]:
    tasks: set[str] = set()
    for key in npz.files:
        if "__L" in key:
            task, _, _ = key.partition("__L")
            if task:
                tasks.add(task)
    return tasks


def _stable_random_seed(seed_key: tuple) -> int:
    return int(
        hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=4).hexdigest(),
        16,
    )


def random_matched_conceptor(C_reference: np.ndarray, seed: int) -> np.ndarray:
    if C_reference.ndim != 2 or C_reference.shape[0] != C_reference.shape[1]:
        raise ValueError(
            f"random_matched_conceptor expects square matrix, got {C_reference.shape}"
        )
    d = C_reference.shape[0]
    Cd = C_reference.astype(np.float64, copy=False)
    Cd = 0.5 * (Cd + Cd.T)
    eigvals = np.linalg.eigvalsh(Cd)

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    C_rand = (Q * eigvals[None, :]) @ Q.T
    return C_rand.astype(np.float32)


def get_conceptor_matrix(
    npz: Any,
    task: str,
    layer: int,
    alpha: float,
    strategy: str,
    *,
    random_seed: int | None = None,
) -> np.ndarray:
    if strategy == "global":
        key = _conceptor_key(task, layer, str(alpha), "C_contrastive")
    elif strategy == "positive_only":
        key = _conceptor_key(task, layer, str(alpha), "C_success")
    elif strategy == "random_matched":
        ref_key = _conceptor_key(task, layer, str(alpha), "C_contrastive")
        if ref_key not in npz.files:
            raise KeyError(
                f"random_matched: reference key {ref_key!r} not in NPZ. "
                f"Available keys starting with {task!r}: "
                f"{[k for k in npz.files if k.startswith(task)][:5]}... (truncated)"
            )
        if random_seed is None:
            raise ValueError("random_matched strategy requires a random_seed")
        return random_matched_conceptor(np.asarray(npz[ref_key]), seed=random_seed)
    elif strategy == "per_step":
        raise ValueError(
            "strategy='per_step' returns a list of matrices; call "
            "get_per_step_conceptor_matrices instead"
        )
    elif strategy == "linear":
        raise ValueError(
            "strategy='linear' returns a vector; call get_linear_direction instead"
        )
    else:
        raise ValueError(
            f"Unknown steering strategy {strategy!r}. Allowed: {ALLOWED_STRATEGIES}"
        )

    if key not in npz.files:
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
    num_denoising_steps: int,
) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for t in range(num_denoising_steps):
        key = _conceptor_key(task, layer, f"per_step_{t}", "C_contrastive")
        if key not in npz.files:
            raise KeyError(
                f"GR00T per_step: missing NPZ key {key!r}. The NPZ must contain "
                f"per_step_0..per_step_{num_denoising_steps - 1} keys for this "
                "task/layer."
            )
        matrices.append(np.asarray(npz[key]))
    return matrices


def get_linear_direction(npz: Any, task: str, layer: int) -> np.ndarray:
    key = f"{task}__L{layer}__linear_direction"
    if key not in npz.files:
        raise KeyError(
            f"Linear direction key {key!r} not in NPZ. This NPZ may predate the "
            "linear strategy."
        )
    v = np.asarray(npz[key])
    if v.ndim != 1:
        raise ValueError(f"Expected 1-D linear direction, got shape {v.shape}")
    return v


def validate_steering_payload(payload: Any, available: Iterable[str]) -> None:
    if not isinstance(payload, dict):
        raise ValueError(
            f"__steering__ payload must be a dict, got {type(payload).__name__}"
        )

    missing = [k for k in _REQUIRED_PAYLOAD_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"__steering__ payload missing required fields: {missing}")

    for key, expected in _REQUIRED_PAYLOAD_FIELDS.items():
        if not isinstance(payload[key], expected) or isinstance(payload[key], bool):
            raise ValueError(
                f"__steering__.{key} must be {expected}, "
                f"got {type(payload[key]).__name__}"
            )

    for numeric_key in ("alpha", "beta"):
        if not np.isfinite(payload[numeric_key]):
            raise ValueError(
                f"__steering__.{numeric_key} must be finite, "
                f"got {payload[numeric_key]!r}"
            )

    if payload["strategy"] not in ALLOWED_STRATEGIES:
        raise ValueError(
            f"__steering__.strategy {payload['strategy']!r} not in {ALLOWED_STRATEGIES}"
        )

    available_set = set(available)
    if payload["task"] not in available_set:
        hint = sorted(available_set)[:3]
        raise ValueError(
            f"__steering__.task {payload['task']!r} not found in conceptor NPZ "
            f"(example keys: {hint})"
        )


class GrootConceptorSteeringHook:
    """PyTorch forward hook for GR00T DiT block residuals."""

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
                "Provide exactly one of conceptor_matrix or matrices_per_step"
            )

        if matrices_per_step is not None:
            if not matrices_per_step:
                raise ValueError("matrices_per_step must be non-empty")
            self._Ms: list[torch.Tensor] | None = [
                self._build_M(C) for C in matrices_per_step
            ]
            self.M: torch.Tensor | None = None
        else:
            self._Ms = None
            self.M = self._build_M(conceptor_matrix)

    def _build_M(self, C: np.ndarray) -> torch.Tensor:  # noqa: N803
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"Conceptor matrix must be square, got {C.shape}")
        d = C.shape[0]
        I = torch.eye(d, dtype=torch.float32, device=self._device)
        Ct = torch.from_numpy(np.ascontiguousarray(C)).to(
            dtype=torch.float32, device=self._device
        )
        return (1.0 - self.beta) * I + self.beta * Ct

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        M = self._Ms[self.current_denoise_step] if self._Ms is not None else self.M
        M = M.to(device=h.device, dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
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
            return (
                f"GrootConceptorSteeringHook(dim={d}, beta={self.beta}, "
                f"per_step={len(self._Ms)})"
            )
        d = self.M.shape[0]
        return f"GrootConceptorSteeringHook(dim={d}, beta={self.beta})"


class GrootLinearSteeringHook:
    """Additive ActAdd-style hook for GR00T DiT block residuals."""

    def __init__(self, direction: np.ndarray, alpha: float = 1.0, device: str = "cuda"):
        if direction.ndim != 1:
            raise ValueError(
                f"GrootLinearSteeringHook expects 1-D direction, got {direction.shape}"
            )
        self.alpha = float(alpha)
        self.current_denoise_step = 0
        self.v = torch.from_numpy(np.ascontiguousarray(direction)).to(
            dtype=torch.float32, device=device
        )
        self.intervention_norms: list[float] = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        v = self.v.to(device=h.device, dtype=h.dtype)
        h_steered = h + self.alpha * v
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
        if rest is not None:
            return (h_steered, *rest)
        return h_steered

    def set_denoise_step(self, t: int) -> None:
        self.current_denoise_step = int(t)

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __repr__(self) -> str:
        return f"GrootLinearSteeringHook(dim={self.v.shape[0]}, alpha={self.alpha})"


class SteeredGrootPolicyWrapper:
    """Wrap a GR00TAdapterPolicy so it honors obs["__steering__"]."""

    def __init__(
        self,
        policy: Any,
        conceptor_npz_path: str | pathlib.Path,
        device: str,
        *,
        num_denoising_steps: int,
    ) -> None:
        self._policy = policy
        self._npz = load_conceptor_npz(conceptor_npz_path)
        self._available_tasks = available_tasks(self._npz)
        self._device = device
        self._num_denoising_steps = int(num_denoising_steps)
        self._hook_cache: dict[tuple[str, int, float, float, str], Any] = {}

    @staticmethod
    def _cache_key(payload: dict) -> tuple[str, int, float, float, str]:
        strategy = payload["strategy"]
        alpha = 0.0 if strategy == "per_step" else float(payload["alpha"])
        beta = 0.0 if strategy == "linear" else float(payload["beta"])
        return (payload["task"], int(payload["layer"]), alpha, beta, strategy)

    def _get_or_build_hook(self, payload: dict) -> tuple[int, Any]:
        key = self._cache_key(payload)
        hook = self._hook_cache.get(key)
        if hook is None:
            strategy = key[4]
            if strategy == "linear":
                v = get_linear_direction(self._npz, task=key[0], layer=key[1])
                hook = GrootLinearSteeringHook(
                    v, alpha=float(payload["alpha"]), device=self._device
                )
            elif strategy == "per_step":
                matrices = get_per_step_conceptor_matrices(
                    self._npz,
                    task=key[0],
                    layer=key[1],
                    num_denoising_steps=self._num_denoising_steps,
                )
                hook = GrootConceptorSteeringHook(
                    beta=float(payload["beta"]),
                    device=self._device,
                    matrices_per_step=matrices,
                )
            else:
                random_seed = None
                if strategy == "random_matched":
                    random_seed = _stable_random_seed((key[0], key[1], key[2], key[4]))
                C = get_conceptor_matrix(
                    self._npz,
                    task=key[0],
                    layer=key[1],
                    alpha=float(payload["alpha"]),
                    strategy=strategy,
                    random_seed=random_seed,
                )
                hook = GrootConceptorSteeringHook(
                    C, beta=float(payload["beta"]), device=self._device
                )
            self._hook_cache[key] = hook
            logger.info(
                "Built GR00T steering hook %s [%s] (cache size=%d)",
                key,
                type(hook).__name__,
                len(self._hook_cache),
            )
        else:
            hook.reset_logs()
        return key[1], hook

    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            return self._policy.infer(obs)
        payload = obs.get(STEERING_KEY)
        if payload is None:
            return self._policy.infer(obs)
        validate_steering_payload(payload, self._available_tasks)
        clean_obs = {k: v for k, v in obs.items() if k != STEERING_KEY}
        layer, hook = self._get_or_build_hook(payload)
        result, _ = self._policy.infer_with_steering(
            clean_obs, steering_hooks=[(layer, hook)]
        )
        return result

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def metadata(self) -> dict:
        underlying = getattr(self._policy, "metadata", {}) or {}
        return {
            **underlying,
            "steering_enabled": True,
            "num_conceptor_tasks": len(self._available_tasks),
            "steering_model_type": "groot_n15",
            "steering_backend": "groot_dit_hooks",
            "steering_target": "dit_hidden_states",
        }
