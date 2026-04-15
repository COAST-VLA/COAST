#!/usr/bin/env python3
"""
Build contrastive conceptors + linear-steering directions for pi0.5 LIBERO.

Reads per-step suffix residual activations (shape (10, 4, 10, 1024)) that were
collected by the activation-collection server, computes per-task
success/failure/contrastive conceptors on a grid of (layer, alpha), plus a
linear-steering control (mean-difference direction), and writes a single
compressed .npz compatible with `select_parameters.py` and
`conceptor_steering.py`.

Activation schema (per step_XXXX/suffix_residual.npz):
    key   = "all_suffix_residual"
    shape = (num_denoising_steps=10, num_captured_layers=4, seq_len=10, hidden=1024)
    Captured layers: [0, 5, 11, 17]  (pi0.5 suffix-model layer indices).

Output npz key naming (matches the steering code and select_parameters.KEY_RE):
    Conceptor:      {task}__L{layer}__{alpha}__C_{success|failure|contrastive}
    Per-step:       {task}__L{layer}__per_step_{ds}__C_{success|failure|contrastive}   (alpha=1.0)
    Linear:         {task}__L{layer}__linear__V_{success|failure|contrastive}          (1-D vectors)
    Per-step V:     {task}__L{layer}__linear_per_step_{ds}__V_{success|failure|contrastive}

Per-denoising-step conceptors are built at EVERY denoising step (0..9), not
just steps 0 and 9 — this is what enables the `per_step` steering strategy to
swap the conceptor at every iteration of the flow-matching loop.

Defaults write to  $OPENPI_DATA_HOME/libero_conceptors.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (pi0.5 LIBERO)
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = Path(os.environ.get(
    "OPENPI_DATA_HOME",
    str(Path.home() / ".cache" / "openpi"),
))

# Where the LIBERO activations live on disk. This is the same default that
# conceptor_diagnostic.py / the activation-collection pipeline use.
DEFAULT_ACTIVATIONS_DIR = (
    OPENPI_DATA_HOME / "activations" / "pi05_libero_2000_15env" / "openpi-libero-2000"
)
DEFAULT_OUTPUT_NPZ = OPENPI_DATA_HOME / "libero_conceptors.npz"

NUM_DENOISING_STEPS = 10
HIDDEN_DIM = 1024

# Captured layer indices as they appear in the collected tensor's layer axis.
# LAYER_MAP[layer_name] -> axis index into the (10, 4, 10, 1024) tensor.
LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}

DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
DEFAULT_LAYERS = tuple(LAYER_MAP.keys())  # (0, 5, 11, 17)
DEFAULT_DS_FOR_CONCEPTORS = 0     # which denoising step for the main (per-alpha) conceptors
DEFAULT_PER_STEP_ALPHA = 1.0
MIN_PER_CLASS = 3                  # need ≥3 success AND ≥3 failure episodes


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math
# ──────────────────────────────────────────────────────────────────────────────

def compute_conceptor_matrix(X: np.ndarray, alpha: float) -> np.ndarray:
    """C = R (R + α^-2 I)^-1, where R = X_centered^T X_centered / N."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = max(1, Xc.shape[0])
    R = (Xc.T @ Xc) / N
    d = R.shape[0]
    return R @ np.linalg.inv(R + (alpha ** -2) * np.eye(d))


def conceptor_NOT(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def contrastive_conceptor(X_s: np.ndarray, X_f: np.ndarray, alpha: float):
    """Returns (C_success, C_failure, C_contrastive = C_s · NOT C_f)."""
    C_s = compute_conceptor_matrix(X_s, alpha)
    C_f = compute_conceptor_matrix(X_f, alpha)
    C_c = C_s @ conceptor_NOT(C_f)
    return C_s, C_f, C_c


def linear_direction(X_s: np.ndarray, X_f: np.ndarray):
    """Mean-difference direction v = unit(mean_s - mean_f). Returns (mean_s, mean_f, v)."""
    v_s = X_s.mean(axis=0)
    v_f = X_f.mean(axis=0)
    v_c = v_s - v_f
    norm = np.linalg.norm(v_c)
    v_c = v_c / norm if norm > 1e-12 else v_c
    return v_s, v_f, v_c


# ──────────────────────────────────────────────────────────────────────────────
# Activation loading
# ──────────────────────────────────────────────────────────────────────────────

def discover_tasks(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_episode_metadata(task_dir: Path) -> dict[str, dict]:
    """Map episode_name -> metadata dict (only episodes with valid metadata.json)."""
    info: dict[str, dict] = {}
    for ep_dir in sorted(task_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        info[ep_dir.name] = {
            "path": ep_dir,
            "success": bool(meta.get("episode_success", False)),
            "total_inference_steps": int(meta.get("total_inference_steps", 0)),
        }
    return info


def load_task_activations(task_dir: Path, layer_idx: int, ds: int, verbose: bool = False):
    """Load per-inference-step suffix residuals for one (layer, denoising-step).

    Mean-pools over the 10 action tokens so each inference step contributes ONE
    (hidden_dim,) vector. Returns:
        X         : (n_samples, 1024) float32
        successes : list[bool]   — per-sample episode_success label
    """
    info = load_episode_metadata(task_dir)
    vectors, successes = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]   # (10, 4, 10, 1024) fp16 or fp32
                    # mean-pool over the 10 action tokens
                    vec = arr[ds, layer_idx].mean(axis=0).astype(np.float32)
            except Exception as e:
                if verbose:
                    print(f"    skip {npz_path}: {e}")
                continue
            vectors.append(vec)
            successes.append(ep["success"])
    if not vectors:
        return np.empty((0, HIDDEN_DIM), dtype=np.float32), successes
    return np.stack(vectors, axis=0), successes


# ──────────────────────────────────────────────────────────────────────────────
# Main build loop
# ──────────────────────────────────────────────────────────────────────────────

def build_conceptors(
    activations_dir: Path,
    output_npz: Path,
    alphas: tuple[float, ...],
    layers: tuple[int, ...],
    main_ds: int,
    per_step_alpha: float,
    skip_per_step: bool,
    verbose: bool,
) -> None:
    if not activations_dir.is_dir():
        sys.exit(f"[build_conceptors] activations dir not found: {activations_dir}")
    for L in layers:
        if L not in LAYER_MAP:
            sys.exit(f"[build_conceptors] layer {L} not captured (valid: {list(LAYER_MAP)})")

    tasks = discover_tasks(activations_dir)
    print(f"[build_conceptors] activations: {activations_dir}")
    print(f"[build_conceptors] tasks ({len(tasks)})")
    print(f"[build_conceptors] layers: {list(layers)}   alphas: {list(alphas)}   main ds={main_ds}")

    # ── Episode class balance (pick mixed-outcome tasks) ──
    task_class_counts = {}
    for t in tasks:
        info = load_episode_metadata(activations_dir / t)
        n_s = sum(1 for v in info.values() if v["success"])
        n_f = sum(1 for v in info.values() if not v["success"])
        task_class_counts[t] = (n_s, n_f)
    print("\n[build_conceptors] Episode-level class balance:")
    for t, (ns, nf) in task_class_counts.items():
        mark = "OK" if (ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS) else "SKIP"
        print(f"  {t[:60]:<60s} success={ns:3d}  failure={nf:3d}  [{mark}]")

    mixed_tasks = [t for t, (ns, nf) in task_class_counts.items()
                   if ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS]
    if not mixed_tasks:
        sys.exit("[build_conceptors] no mixed-outcome tasks — cannot build contrastive conceptors.")

    save_arrays: dict[str, np.ndarray] = {}
    stats = []

    for task in mixed_tasks:
        task_dir = activations_dir / task
        print(f"\n[build_conceptors] === {task[:60]} ===")

        for L in layers:
            layer_idx = LAYER_MAP[L]

            # ── main conceptors + linear direction at main_ds ──
            t0 = time.time()
            X, succ = load_task_activations(task_dir, layer_idx, main_ds, verbose=verbose)
            if X.shape[0] == 0:
                print(f"  L{L:<2d} ds={main_ds}: no samples, skipping")
                continue
            s_idx = [i for i, s in enumerate(succ) if s]
            f_idx = [i for i, s in enumerate(succ) if not s]
            if len(s_idx) < MIN_PER_CLASS or len(f_idx) < MIN_PER_CLASS:
                print(f"  L{L:<2d} ds={main_ds}: too few per class (s={len(s_idx)}, f={len(f_idx)})")
                continue

            X_s, X_f = X[s_idx], X[f_idx]

            # Linear direction (independent of alpha)
            v_s, v_f, v_c = linear_direction(X_s, X_f)
            save_arrays[f"{task}__L{L}__linear__V_success"]     = v_s.astype(np.float32)
            save_arrays[f"{task}__L{L}__linear__V_failure"]     = v_f.astype(np.float32)
            save_arrays[f"{task}__L{L}__linear__V_contrastive"] = v_c.astype(np.float32)

            # Conceptors per alpha
            for a in alphas:
                C_s, C_f, C_c = contrastive_conceptor(X_s, X_f, a)
                save_arrays[f"{task}__L{L}__{a}__C_success"]     = C_s.astype(np.float32)
                save_arrays[f"{task}__L{L}__{a}__C_failure"]     = C_f.astype(np.float32)
                save_arrays[f"{task}__L{L}__{a}__C_contrastive"] = C_c.astype(np.float32)

            # Overlap/quota summary at α=1.0
            C_s1, C_f1, _ = contrastive_conceptor(X_s, X_f, 1.0)
            q_s = float(np.trace(C_s1)) / HIDDEN_DIM
            q_f = float(np.trace(C_f1)) / HIDDEN_DIM
            num = float(np.einsum("ij,ji->", C_s1, C_f1))
            den = np.sqrt(float(np.einsum("ij,ji->", C_s1, C_s1))
                          * float(np.einsum("ij,ji->", C_f1, C_f1)))
            overlap = num / den if den > 0 else 0.0
            print(f"  L{L:<2d} ds={main_ds} "
                  f"n(s,f)=({len(s_idx):2d},{len(f_idx):2d}) "
                  f"q(Cs)={q_s:.3f} q(Cf)={q_f:.3f} overlap={overlap:.3f} "
                  f"({time.time()-t0:.1f}s)")
            stats.append((task[:40], L, len(s_idx), len(f_idx), q_s, q_f, overlap))

            # ── per-denoising-step conceptors, ALL 10 steps, at per_step_alpha ──
            if skip_per_step:
                continue
            for ds in range(NUM_DENOISING_STEPS):
                if ds == main_ds:
                    # Alias to the main-ds matrices at per_step_alpha (avoids recompute).
                    save_arrays[f"{task}__L{L}__per_step_{ds}__C_success"]     = save_arrays[f"{task}__L{L}__{per_step_alpha}__C_success"]
                    save_arrays[f"{task}__L{L}__per_step_{ds}__C_failure"]     = save_arrays[f"{task}__L{L}__{per_step_alpha}__C_failure"]
                    save_arrays[f"{task}__L{L}__per_step_{ds}__C_contrastive"] = save_arrays[f"{task}__L{L}__{per_step_alpha}__C_contrastive"]
                    save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_success"]     = save_arrays[f"{task}__L{L}__linear__V_success"]
                    save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_failure"]     = save_arrays[f"{task}__L{L}__linear__V_failure"]
                    save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_contrastive"] = save_arrays[f"{task}__L{L}__linear__V_contrastive"]
                    continue
                X_ds, succ_ds = load_task_activations(task_dir, layer_idx, ds, verbose=verbose)
                if X_ds.shape[0] == 0:
                    continue
                s = [i for i, v in enumerate(succ_ds) if v]
                f = [i for i, v in enumerate(succ_ds) if not v]
                if len(s) < MIN_PER_CLASS or len(f) < MIN_PER_CLASS:
                    continue
                C_s_ds, C_f_ds, C_c_ds = contrastive_conceptor(X_ds[s], X_ds[f], per_step_alpha)
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_success"]     = C_s_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_failure"]     = C_f_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_contrastive"] = C_c_ds.astype(np.float32)
                v_s_ds, v_f_ds, v_c_ds = linear_direction(X_ds[s], X_ds[f])
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_success"]     = v_s_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_failure"]     = v_f_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_contrastive"] = v_c_ds.astype(np.float32)

    if not save_arrays:
        sys.exit("[build_conceptors] no arrays produced — check activation dir and class balances.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[build_conceptors] writing {len(save_arrays)} arrays to {output_npz} ...")
    np.savez_compressed(output_npz, **save_arrays)
    size_mb = output_npz.stat().st_size / 1e6
    print(f"[build_conceptors] wrote {output_npz} ({size_mb:.1f} MB, {len(save_arrays)} arrays)")

    print("\n[build_conceptors] summary (alpha=1.0, ds=main):")
    print(f"  {'task':<40s} {'L':>2s} {'n_s':>3s} {'n_f':>3s} {'q(Cs)':>6s} {'q(Cf)':>6s} {'overlap':>7s}")
    for row in stats:
        print(f"  {row[0]:<40s} {row[1]:>2d} {row[2]:>3d} {row[3]:>3d} {row[4]:>6.3f} {row[5]:>6.3f} {row[6]:>7.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    p.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                   help=f"pi0.5 suffix-model layers (captured set: {list(LAYER_MAP)})")
    p.add_argument("--main-ds", type=int, default=DEFAULT_DS_FOR_CONCEPTORS,
                   help="denoising-step index (0..9) used for the main per-alpha conceptors")
    p.add_argument("--per-step-alpha", type=float, default=DEFAULT_PER_STEP_ALPHA)
    p.add_argument("--skip-per-step", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_conceptors(
        activations_dir=args.activations_dir,
        output_npz=args.output_npz,
        alphas=tuple(args.alphas),
        layers=tuple(args.layers),
        main_ds=args.main_ds,
        per_step_alpha=args.per_step_alpha,
        skip_per_step=args.skip_per_step,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
