"""Conceptor computation pipeline.

Consumes the on-disk activation tree written by ``activation_collector.py``
and produces the conceptor ``.npz`` that ``steering.py`` consumes. The key
format matches the miranda-v2 collaborator branch so NPZs produced here are
drop-in compatible with the ``brandonyang/libero-conceptors`` and
``brandonyang/robocasa-conceptors`` datasets.

Math reference (Jaeger 2014 + standard Boolean-AND identity):

    R = X^T X / N                                      (correlation, no centering)
    C(R, α) = R @ inv(R + α^{-2} I)                   (aperture-α conceptor)
    C_AND(C1, C2) = C1 @ inv(C1 + C2 - C1 @ C2) @ C2   (proper Boolean AND)

The "contrastive" conceptor for a task is ``C_AND(C_success, C_failure_complement)``
where ``C_failure_complement = I - C_failure`` so the final matrix projects onto
directions that are both (in success subspace) AND (not in failure subspace).

Shape conventions for the flattening step:
    suffix_residual.npz:all_suffix_residual has shape
        (num_denoise_steps=10, num_collected_layers, num_tokens=32, hidden_dim=1024)

    For a *global* conceptor at layer L, alpha α:
        1. For each episode: slice axis-1 to layer L, yielding (10, 32, 1024)
        2. Concatenate across all episodes of one success-class, reshape to
           (N_eps * 10 * 32, 1024), i.e. treat every token at every denoise step
           as an independent sample (flattening, not mean-pooling — preserves rank)
        3. Compute R then C(R, α)
        4. Combine success+failure via Boolean AND

    For a *per-step-t* conceptor at layer L, t ∈ {0..9}:
        1. Slice axis-0 to t and axis-1 to L, yielding (32, 1024) per episode
        2. Concatenate across episodes, reshape to (N_eps * 32, 1024)
        3. Same C, AND steps

Output NPZ keys (string-exact for miranda-v2 compat):
    {task}__L{layer}__{alpha}__C_success         (global)
    {task}__L{layer}__{alpha}__C_failure
    {task}__L{layer}__{alpha}__C_contrastive
    {task}__L{layer}__per_step_{t}__C_success   (per-step)
    {task}__L{layer}__per_step_{t}__C_failure
    {task}__L{layer}__per_step_{t}__C_contrastive
    {task}__L{layer}__linear_direction           (unit vector, linear strategy)
"""

# ruff: noqa: E741, N803, N806, RUF001, RUF002, RUF003
from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

# pi0.5 default collect_layers (from pi0_pytorch.py:462).
# Axis-1 index into all_suffix_residual → real transformer layer index.
DEFAULT_COLLECT_LAYERS: tuple[int, ...] = (0, 5, 11, 17)
DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 10.0)
# All 10 per-step indices for pi0.5's flow-matching schedule. The ``per_step``
# strategy requires the NPZ to contain ``per_step_0`` .. ``per_step_9`` so the
# sampler's per-step hook can look up a distinct conceptor at each denoising
# iteration. Legacy NPZs built with a narrower tuple must be rebuilt to support
# ``per_step``.
DEFAULT_PER_STEP_INDICES: tuple[int, ...] = tuple(range(10))


# ──────────────────────────────────────────────────────────────────────────────
# Pure math primitives
# ──────────────────────────────────────────────────────────────────────────────


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Uncentered correlation matrix R = X^T X / N.

    Args:
        X: shape (N, d)
    Returns:
        R: shape (d, d), symmetric
    """
    if X.ndim != 2:
        raise ValueError(f"correlation_matrix expects 2D input, got shape {X.shape}")
    N = X.shape[0]
    if N == 0:
        raise ValueError("correlation_matrix: N=0 (empty input)")
    # Cast to float64 for numerical stability before the matmul — we're summing
    # potentially millions of bfloat16-sourced products.
    Xd = X.astype(np.float64, copy=False)
    return (Xd.T @ Xd) / float(N)


def conceptor(R: np.ndarray, alpha: float) -> np.ndarray:
    """Conceptor matrix C = R @ inv(R + α^{-2} I).

    Args:
        R: correlation matrix, shape (d, d)
        alpha: aperture; α→∞ makes C→I, α→0 makes C→0
    Returns:
        C: shape (d, d), same dtype as input (promoted to float64 internally)
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"conceptor expects square R, got shape {R.shape}")
    if alpha <= 0:
        raise ValueError(f"conceptor alpha must be positive, got {alpha}")
    d = R.shape[0]
    Rd = R.astype(np.float64, copy=False)
    inv_term = Rd + (alpha**-2) * np.eye(d, dtype=np.float64)
    # np.linalg.solve is numerically preferable to explicit inv().
    # Rd @ inv(M) == solve(M.T, Rd.T).T
    return np.linalg.solve(inv_term.T, Rd.T).T


def boolean_and(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Proper Boolean AND of two conceptors: C1 @ inv(C1 + C2 - C1 @ C2) @ C2.

    Symmetric up to numerical error. Result projects onto the intersection of
    the two conceptor subspaces.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"boolean_and shape mismatch: {C1.shape} vs {C2.shape}")
    C1d = C1.astype(np.float64, copy=False)
    C2d = C2.astype(np.float64, copy=False)
    # Add tiny ridge for numerical stability if the sum is near-singular.
    M = C1d + C2d - C1d @ C2d
    d = M.shape[0]
    # solve(M, C2d) computes inv(M) @ C2d. We want C1d @ inv(M) @ C2d.
    try:
        inv_M_C2 = np.linalg.solve(M, C2d)
    except np.linalg.LinAlgError:
        M = M + 1e-8 * np.eye(d, dtype=np.float64)
        inv_M_C2 = np.linalg.solve(M, C2d)
    return C1d @ inv_M_C2


def boolean_not(C: np.ndarray) -> np.ndarray:
    """Boolean NOT of a conceptor: I - C.

    For a conceptor C with eigenvalues in [0, 1], NOT(C) has eigenvalues 1 - λ.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"boolean_not expects square matrix, got {C.shape}")
    return np.eye(C.shape[0], dtype=np.float64) - C.astype(np.float64, copy=False)


def contrastive_conceptor(C_success: np.ndarray, C_failure: np.ndarray) -> np.ndarray:
    """Return AND(C_success, NOT(C_failure)).

    This is the key primitive for success-vs-failure steering: project onto the
    subspace that is in successful rollouts AND not in failed rollouts.
    """
    return boolean_and(C_success, boolean_not(C_failure))


def compute_linear_direction(X_success: np.ndarray, X_failure: np.ndarray) -> np.ndarray:
    """Unit vector pointing from mean_failure to mean_success.

    Used by the ``linear`` steering strategy (ActAdd-style): at inference we apply
    ``h' = h + alpha * v`` where ``v`` is this unit vector and ``alpha`` controls
    the intervention magnitude. Baseline against the full conceptor machinery.

    Args:
        X_success: (N_s, d) — flattened success-class activations
        X_failure: (N_f, d) — flattened failure-class activations
    Returns:
        v: (d,) float32 unit vector. Zero vector if the mean difference is
           numerically zero (degenerate case).
    """
    if X_success.ndim != 2 or X_failure.ndim != 2:
        raise ValueError(f"compute_linear_direction expects 2D inputs, got {X_success.shape} and {X_failure.shape}")
    if X_success.shape[1] != X_failure.shape[1]:
        raise ValueError(f"hidden dim mismatch: success d={X_success.shape[1]}, failure d={X_failure.shape[1]}")
    if X_success.shape[0] == 0 or X_failure.shape[0] == 0:
        raise ValueError("compute_linear_direction: empty success or failure set")

    mean_s = X_success.astype(np.float64, copy=False).mean(axis=0)
    mean_f = X_failure.astype(np.float64, copy=False).mean(axis=0)
    diff = mean_s - mean_f
    norm = float(np.linalg.norm(diff))
    if norm < 1e-12:
        logger.warning("compute_linear_direction: mean difference has near-zero norm; returning zero vector")
        return np.zeros_like(diff, dtype=np.float32)
    return (diff / norm).astype(np.float32)


def random_matched_conceptor(C_reference: np.ndarray, seed: int) -> np.ndarray:
    """Build a random-eigenvector conceptor with the same eigenvalue spectrum as C_reference.

    Control baseline: keeps the "strength" (eigenvalue spectrum) of a reference
    conceptor but replaces the eigenvectors with a random orthogonal basis. If
    steering with this random matrix helps the task, the benefit was not from
    the learned direction — it was from the matrix's overall shape.

    Args:
        C_reference: (d, d) symmetric conceptor matrix
        seed: random seed for the orthogonal basis
    Returns:
        C_random: (d, d) float32 symmetric matrix with same eigenvalues as C_reference
    """
    if C_reference.ndim != 2 or C_reference.shape[0] != C_reference.shape[1]:
        raise ValueError(f"random_matched_conceptor expects square matrix, got {C_reference.shape}")
    d = C_reference.shape[0]
    # Symmetrize before eigendecomposition to suppress float32 asymmetry.
    Cd = C_reference.astype(np.float64, copy=False)
    Cd = 0.5 * (Cd + Cd.T)
    eigvals = np.linalg.eigvalsh(Cd)  # ascending order, real

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # Q is orthogonal; C_rand = Q @ diag(eigvals) @ Q.T has same spectrum.
    C_rand = (Q * eigvals[None, :]) @ Q.T
    return C_rand.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# On-disk activation tree walk
# ──────────────────────────────────────────────────────────────────────────────


def iter_episode_dirs(root: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield every ``episode_NNN_env_NNN`` directory under ``root``.

    Layout (per activation_collector.py):
        root/<checkpoint_step>/<task_name>/episode_NNN_env_NNN/
            step_NNNN/suffix_residual.npz
            rewards.npz
            metadata.json
    """
    for ckpt_dir in sorted(root.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        for task_dir in sorted(ckpt_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for ep_dir in sorted(task_dir.iterdir()):
                if ep_dir.is_dir() and ep_dir.name.startswith("episode_"):
                    yield ep_dir


def episode_task_name(ep_dir: pathlib.Path) -> str:
    """Task name is the parent directory name."""
    return ep_dir.parent.name


def episode_is_success(ep_dir: pathlib.Path) -> bool:
    """Read rewards.npz:success_at_step and check whether any step marked success."""
    rewards_path = ep_dir / "rewards.npz"
    if not rewards_path.exists():
        raise FileNotFoundError(f"Missing rewards.npz in {ep_dir}")
    with np.load(rewards_path) as data:
        success_at_step = data["success_at_step"]
    return bool(np.any(success_at_step))


def load_episode_hiddens(ep_dir: pathlib.Path) -> np.ndarray:
    """Concatenate all step_NNNN/suffix_residual.npz into one array.

    Returns:
        shape (total_rollout_steps, num_denoise_steps, num_collected_layers, num_tokens, hidden_dim)

    Rollout steps correspond to distinct infer() calls in the episode; denoise
    steps are the 10 flow-matching steps *within* each infer call.
    """
    step_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith("step_"))
    if not step_dirs:
        raise FileNotFoundError(f"No step_NNNN dirs in {ep_dir}")
    chunks = []
    for sd in step_dirs:
        sr_path = sd / "suffix_residual.npz"
        if not sr_path.exists():
            continue
        with np.load(sr_path) as data:
            # shape: (10, num_layers, 32, 1024)
            chunks.append(np.asarray(data["all_suffix_residual"]))
    if not chunks:
        raise FileNotFoundError(f"No suffix_residual.npz under {ep_dir}")
    return np.stack(chunks, axis=0)  # (T, 10, L, 32, 1024)


# ──────────────────────────────────────────────────────────────────────────────
# Per-task pipeline
# ──────────────────────────────────────────────────────────────────────────────


def _layer_axis_index(real_layer: int, collect_layers: tuple[int, ...]) -> int:
    """Map a real transformer layer index to its slot in all_suffix_residual axis 1."""
    try:
        return collect_layers.index(real_layer)
    except ValueError as e:
        raise ValueError(
            f"Layer {real_layer} not in collect_layers {collect_layers}; "
            f"re-run collection with --collect_layers to include it."
        ) from e


def flatten_global(ep_hiddens: list[np.ndarray], layer_axis: int) -> np.ndarray:
    """Flatten episodes into (total_samples, hidden_dim) for the 'global' strategy.

    Args:
        ep_hiddens: list of per-episode arrays, each (T, num_denoise, L, num_tokens, D)
        layer_axis: index into L dimension
    Returns:
        X: (sum_T * num_denoise * num_tokens, D) float64
    """
    slices = []
    for h in ep_hiddens:
        # (T, num_denoise, L, num_tokens, D) → slice L → flatten all but last
        s = h[:, :, layer_axis, :, :]
        slices.append(s.reshape(-1, s.shape[-1]))
    return np.concatenate(slices, axis=0).astype(np.float64, copy=False)


def flatten_per_step(ep_hiddens: list[np.ndarray], layer_axis: int, denoise_step: int) -> np.ndarray:
    """Flatten episodes into (total_samples, hidden_dim) for a per-step strategy.

    Keeps only the denoise_step-th flow-matching step; flattens across
    rollout-steps × tokens.
    """
    slices = []
    for h in ep_hiddens:
        s = h[:, denoise_step, layer_axis, :, :]  # (T, num_tokens, D)
        slices.append(s.reshape(-1, s.shape[-1]))
    return np.concatenate(slices, axis=0).astype(np.float64, copy=False)


def compute_task_conceptors(
    hiddens_success: list[np.ndarray],
    hiddens_failure: list[np.ndarray],
    layers: tuple[int, ...],
    alphas: tuple[float, ...],
    per_step_indices: tuple[int, ...],
    collect_layers: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Compute the full conceptor set for one task.

    Returns a dict mapping suffix keys (everything after ``{task}__``) to
    matrices. The caller prepends ``{task}__`` before writing to NPZ.
    """
    out: dict[str, np.ndarray] = {}

    for layer in layers:
        layer_axis = _layer_axis_index(layer, collect_layers)

        # Global per-alpha
        X_s_global = flatten_global(hiddens_success, layer_axis)
        X_f_global = flatten_global(hiddens_failure, layer_axis)
        if X_s_global.shape[0] == 0 or X_f_global.shape[0] == 0:
            logger.warning("Layer %d global: empty success or failure set; skipping", layer)
        else:
            R_s = correlation_matrix(X_s_global)
            R_f = correlation_matrix(X_f_global)
            for alpha in alphas:
                C_s = conceptor(R_s, alpha).astype(np.float32)
                C_f = conceptor(R_f, alpha).astype(np.float32)
                C_c = contrastive_conceptor(C_s, C_f).astype(np.float32)
                out[f"L{layer}__{alpha}__C_success"] = C_s
                out[f"L{layer}__{alpha}__C_failure"] = C_f
                out[f"L{layer}__{alpha}__C_contrastive"] = C_c

            v = compute_linear_direction(X_s_global, X_f_global)
            out[f"L{layer}__linear_direction"] = v

        # Per-step — conceptors at a single denoise step, fixed α=1.0.
        per_step_alpha = 1.0
        for t in per_step_indices:
            X_s = flatten_per_step(hiddens_success, layer_axis, t)
            X_f = flatten_per_step(hiddens_failure, layer_axis, t)
            if X_s.shape[0] == 0 or X_f.shape[0] == 0:
                continue
            R_s_t = correlation_matrix(X_s)
            R_f_t = correlation_matrix(X_f)
            C_s = conceptor(R_s_t, per_step_alpha).astype(np.float32)
            C_f = conceptor(R_f_t, per_step_alpha).astype(np.float32)
            C_c = contrastive_conceptor(C_s, C_f).astype(np.float32)
            out[f"L{layer}__per_step_{t}__C_success"] = C_s
            out[f"L{layer}__per_step_{t}__C_failure"] = C_f
            out[f"L{layer}__per_step_{t}__C_contrastive"] = C_c

    return out


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────


def compute_all_conceptors(
    activation_root: pathlib.Path,
    output_path: pathlib.Path,
    *,
    layers: tuple[int, ...] = (11,),
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    per_step_indices: tuple[int, ...] = DEFAULT_PER_STEP_INDICES,
    collect_layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS,
    min_episodes_per_class: int = 2,
    task_filter: tuple[str, ...] | None = None,
) -> dict:
    """Walk the activation tree, compute conceptors per task, write NPZ.

    Returns a dict summary: {num_tasks, num_keys, skipped_tasks}.

    Tasks with fewer than ``min_episodes_per_class`` successful OR failed
    episodes are skipped (with a warning) — a conceptor built from <2 episodes
    is too noisy to be useful.
    """
    activation_root = pathlib.Path(activation_root)
    output_path = pathlib.Path(output_path)

    # Group episode_dirs by task
    by_task: dict[str, list[pathlib.Path]] = {}
    for ep_dir in iter_episode_dirs(activation_root):
        task = episode_task_name(ep_dir)
        if task_filter and task not in task_filter:
            continue
        by_task.setdefault(task, []).append(ep_dir)

    all_keys: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    included: list[str] = []

    for task, ep_dirs in sorted(by_task.items()):
        success_eps = [e for e in ep_dirs if episode_is_success(e)]
        failure_eps = [e for e in ep_dirs if not episode_is_success(e)]
        if len(success_eps) < min_episodes_per_class or len(failure_eps) < min_episodes_per_class:
            logger.warning(
                "Task %s: success=%d, failure=%d (min=%d); skipping",
                task,
                len(success_eps),
                len(failure_eps),
                min_episodes_per_class,
            )
            skipped.append(task)
            continue

        logger.info("Task %s: loading %d success + %d failure episodes", task, len(success_eps), len(failure_eps))
        hiddens_success = [load_episode_hiddens(e) for e in success_eps]
        hiddens_failure = [load_episode_hiddens(e) for e in failure_eps]

        task_keys = compute_task_conceptors(
            hiddens_success,
            hiddens_failure,
            layers=layers,
            alphas=alphas,
            per_step_indices=per_step_indices,
            collect_layers=collect_layers,
        )
        for suffix, matrix in task_keys.items():
            all_keys[f"{task}__{suffix}"] = matrix

        included.append(task)
        logger.info("Task %s: wrote %d keys", task, len(task_keys))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **all_keys)
    logger.info("Wrote %d keys across %d tasks to %s", len(all_keys), len(included), output_path)

    # Sidecar metadata for reproducibility
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "activation_root": str(activation_root),
                "layers": list(layers),
                "alphas": list(alphas),
                "per_step_indices": list(per_step_indices),
                "collect_layers": list(collect_layers),
                "num_tasks_included": len(included),
                "num_tasks_skipped": len(skipped),
                "skipped_tasks": skipped,
                "included_tasks": included,
                "num_keys": len(all_keys),
            },
            f,
            indent=2,
        )

    return {
        "num_tasks": len(included),
        "num_keys": len(all_keys),
        "skipped_tasks": skipped,
        "included_tasks": included,
    }
