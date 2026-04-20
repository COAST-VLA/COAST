#!/usr/bin/env python3
"""
Conceptor Fitting + Diagnostic Analysis for pi0-fast LIBERO Activations
========================================================================
Loads per-token ``token_pre_logits`` (2048-d, final hidden state before the
LM head) from the pi0-fast activation cache and builds conceptors for each
task.  Unlike pi0.5 there is no "layer" axis (only the final pre-head hidden
state was captured) and no fixed number of "denoising steps" — each inference
step produces a variable-length token sequence.

Strategies written to the output .npz:

- ``{task}__global__{alpha}__C_success``     conceptor of successful trajectories
- ``{task}__global__{alpha}__C_failure``     conceptor of failed trajectories
- ``{task}__global__{alpha}__C_contrastive`` C_success AND NOT C_failure
- ``{task}__per_token_{pos}__{alpha}__{kind}``
      conceptors fit over activations at a specific token *position* within
      each inference step.  ``pos`` ∈ {``first``, ``mid``, ``last``}.

Activation data lives at:
  $OPENPI_DATA_HOME/activations_fast_libero/{checkpoint_step}/{task}/
      episode_NNN_env_MMM/
          metadata.json           (episode_success bool, total_inference_steps, ...)
          rewards.npz
          step_NNNN/
              hidden_states.npz   ({"token_pre_logits": (T, 2048) float16})
              tokens.npz          ({"generated_tokens": (T+1,) int32})
              token_logprobs.npz  ({"token_logprobs": (T+1,) float32})
              metadata.json       (step, inference_step, cumulative_reward, ...)

Usage (from repo root):
    uv run python experiments/pi0_fast_libero/src/conceptor_diagnostic.py \
        --checkpoint_step 1000
"""
import dataclasses
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tyro

warnings.filterwarnings("ignore")

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 10.0)
PER_TOKEN_POSITIONS = ("first", "mid", "last")  # bin names
HIDDEN_DIM = 2048
DTYPE = np.float32


# ── Helpers ──────────────────────────────────────────────────────────────


def fast_svd(X: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """SVD of mean-centred X. Returns (sigma, Vt) where sigma are eigenvalues
    of the sample covariance R = X^T X / N."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc / np.sqrt(max(1, N)), full_matrices=False)
    sigma = s ** 2
    if k is not None:
        return sigma[:k], Vt[:k]
    return sigma, Vt


def conceptor_eigenvalues(sigma: np.ndarray, alpha: float) -> np.ndarray:
    """gamma_j = sigma_j / (sigma_j + alpha^{-2})"""
    return sigma / (sigma + alpha ** -2)


def conceptor_quota(gamma: np.ndarray) -> float:
    return float(gamma.sum())


def build_C(X: np.ndarray, alpha: float) -> np.ndarray:
    """Fit a conceptor C = V diag(gamma) V^T from activations X (N, d)."""
    if X.shape[0] < 2:
        return np.zeros((X.shape[1], X.shape[1]), dtype=DTYPE) if X.shape[1] > 0 else np.zeros((HIDDEN_DIM, HIDDEN_DIM), dtype=DTYPE)
    sigma, Vt = fast_svd(X)
    gamma = conceptor_eigenvalues(sigma, alpha)
    return (Vt.T @ np.diag(gamma) @ Vt).astype(DTYPE)


def boolean_not(C: np.ndarray) -> np.ndarray:
    d = C.shape[0]
    I = np.eye(d, dtype=DTYPE)
    return (I - C).astype(DTYPE)


def boolean_and(C_a: np.ndarray, C_b: np.ndarray) -> np.ndarray:
    d = C_a.shape[0]
    eps = 1e-4 * np.eye(d, dtype=DTYPE)
    inv = np.linalg.pinv(C_a + C_b + eps)
    return (C_a @ inv @ C_b).astype(DTYPE)


def contrastive_conceptor(C_success: np.ndarray, C_failure: np.ndarray) -> np.ndarray:
    """C_success AND (NOT C_failure) — the standard contrastive target."""
    return boolean_and(C_success, boolean_not(C_failure))


# ── Data loading ─────────────────────────────────────────────────────────


def discover_tasks(activations_root: Path) -> List[str]:
    if not activations_root.exists():
        raise FileNotFoundError(f"Activation directory not found: {activations_root}")
    tasks = sorted(d.name for d in activations_root.iterdir() if d.is_dir())
    return tasks


def load_episode_metadata(ep_dir: Path) -> Optional[dict]:
    p = ep_dir / "metadata.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_episode_prelogits(
    ep_dir: Path,
) -> List[np.ndarray]:
    """Return a list of (T, 2048) arrays, one per inference step (sorted)."""
    step_dirs = sorted(d for d in ep_dir.iterdir() if d.is_dir() and d.name.startswith("step_"))
    chunks = []
    for sd in step_dirs:
        hp = sd / "hidden_states.npz"
        if not hp.exists():
            continue
        with np.load(hp) as data:
            chunks.append(data["token_pre_logits"].astype(DTYPE))
    return chunks


def collect_task_activations(task_dir: Path) -> Tuple[
    np.ndarray, np.ndarray, Dict[str, np.ndarray]
]:
    """Return (X_success, X_failure, per_token_buckets).

    ``per_token_buckets`` maps {"first", "mid", "last"} to a dict of the same
    success/failure split. Each bucket is populated with one vector per inference
    step by selecting that token position within the generated sequence.
    """
    success_chunks: List[np.ndarray] = []
    failure_chunks: List[np.ndarray] = []
    per_token: Dict[str, Dict[str, List[np.ndarray]]] = {
        pos: {"success": [], "failure": []} for pos in PER_TOKEN_POSITIONS
    }

    for ep_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        meta = load_episode_metadata(ep_dir)
        if meta is None:
            continue
        is_success = bool(meta.get("episode_success", False))
        chunks = load_episode_prelogits(ep_dir)
        if not chunks:
            continue
        # Whole-sequence activations for the global conceptor.
        all_tokens = np.concatenate(chunks, axis=0)
        (success_chunks if is_success else failure_chunks).append(all_tokens)

        # Per-token-position bucket: pick one vector per inference step.
        for c in chunks:
            T = c.shape[0]
            if T == 0:
                continue
            picks = {
                "first": c[0],
                "mid": c[T // 2],
                "last": c[-1],
            }
            for pos, vec in picks.items():
                (
                    per_token[pos]["success" if is_success else "failure"]
                ).append(vec[None, :])

    X_success = np.concatenate(success_chunks, axis=0) if success_chunks else np.zeros((0, HIDDEN_DIM), dtype=DTYPE)
    X_failure = np.concatenate(failure_chunks, axis=0) if failure_chunks else np.zeros((0, HIDDEN_DIM), dtype=DTYPE)

    per_token_flat: Dict[str, np.ndarray] = {}
    for pos in PER_TOKEN_POSITIONS:
        for kind in ("success", "failure"):
            lst = per_token[pos][kind]
            arr = np.concatenate(lst, axis=0) if lst else np.zeros((0, HIDDEN_DIM), dtype=DTYPE)
            per_token_flat[f"{pos}__{kind}"] = arr

    return X_success, X_failure, per_token_flat


# ── Main fitting ─────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    checkpoint_step: int = 1000
    activations_dir: Optional[str] = None
    """Root dir containing {checkpoint_step}/{task}/... structure. Defaults to $OPENPI_DATA_HOME/activations_fast_libero."""
    output_npz: Optional[str] = None
    out_dir: str = "experiments/pi0_fast_libero/diagnostic_results"
    """Directory for diagnostic summary JSON."""


def main(args: Args) -> None:
    if args.activations_dir is not None:
        activations_root = Path(args.activations_dir) / str(args.checkpoint_step)
    else:
        activations_root = Path(OPENPI_DATA_HOME) / "activations_fast_libero" / str(args.checkpoint_step)
    tasks = discover_tasks(activations_root)
    print(f"Found {len(tasks)} tasks under {activations_root}")

    out_npz_path = (
        Path(args.output_npz)
        if args.output_npz is not None
        else Path(OPENPI_DATA_HOME) / "pi0fast_libero_conceptors.npz"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conceptors: Dict[str, np.ndarray] = {}
    summary: Dict[str, dict] = {}

    for task in tasks:
        task_dir = activations_root / task
        print(f"\n[{task}]")
        X_success, X_failure, per_token = collect_task_activations(task_dir)
        print(f"  success activations: {X_success.shape}")
        print(f"  failure activations: {X_failure.shape}")
        task_summary = {
            "n_success_tokens": int(X_success.shape[0]),
            "n_failure_tokens": int(X_failure.shape[0]),
            "alphas": list(ALPHAS),
            "has_contrastive": X_success.shape[0] > 0 and X_failure.shape[0] > 0,
            "per_token_counts": {
                k: int(v.shape[0]) for k, v in per_token.items()
            },
        }

        for alpha in ALPHAS:
            # Global conceptors (all tokens pooled).
            if X_success.shape[0] > 0:
                C_s = build_C(X_success, alpha)
                conceptors[f"{task}__global__{alpha}__C_success"] = C_s
            if X_failure.shape[0] > 0:
                C_f = build_C(X_failure, alpha)
                conceptors[f"{task}__global__{alpha}__C_failure"] = C_f
            if X_success.shape[0] > 0 and X_failure.shape[0] > 0:
                C_c = contrastive_conceptor(C_s, C_f)
                conceptors[f"{task}__global__{alpha}__C_contrastive"] = C_c

            # Per-token-position conceptors.
            for pos in PER_TOKEN_POSITIONS:
                Xs_pos = per_token[f"{pos}__success"]
                Xf_pos = per_token[f"{pos}__failure"]
                if Xs_pos.shape[0] > 0:
                    Cs_pos = build_C(Xs_pos, alpha)
                    conceptors[f"{task}__per_token_{pos}__{alpha}__C_success"] = Cs_pos
                if Xf_pos.shape[0] > 0:
                    Cf_pos = build_C(Xf_pos, alpha)
                    conceptors[f"{task}__per_token_{pos}__{alpha}__C_failure"] = Cf_pos
                if Xs_pos.shape[0] > 0 and Xf_pos.shape[0] > 0:
                    Cc_pos = contrastive_conceptor(Cs_pos, Cf_pos)
                    conceptors[f"{task}__per_token_{pos}__{alpha}__C_contrastive"] = Cc_pos

        # Quotas at a mid alpha (1.0) for summary.
        if X_success.shape[0] > 0:
            s_s, _ = fast_svd(X_success)
            task_summary["quota_success_a1.0"] = conceptor_quota(conceptor_eigenvalues(s_s, 1.0))
        if X_failure.shape[0] > 0:
            s_f, _ = fast_svd(X_failure)
            task_summary["quota_failure_a1.0"] = conceptor_quota(conceptor_eigenvalues(s_f, 1.0))

        summary[task] = task_summary

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(conceptors)} conceptors → {out_npz_path}")
    np.savez(out_npz_path, **conceptors)

    summary_path = out_dir / f"conceptor_summary_ckpt{args.checkpoint_step}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved per-task summary → {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
