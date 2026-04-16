#!/usr/bin/env python3
"""
Build contrastive conceptors (and linear-steering vectors) for GR00T N1.5 on RoboCasa.

Reads per-step DiT hidden-state activations collected by `groot_env/serve.py --collect-activations`,
computes per-task success/failure/contrastive conceptors at a grid of (layer, alpha),
plus a linear-steering control (mean-difference direction vector), and writes a single
compressed .npz compatible with `experiments/shared/select_parameters.py`.

Activation schema (per step_XXXX/dit_hidden_states.npz):
    key            = "all_dit_hidden_states"
    shape          = (num_denoising_steps=4, num_dit_layers=16, seq_len=49, hidden_dim=1536)
    dtype          = float16

Output npz key naming (compatible with select_parameters.KEY_RE):
    Conceptor:   {task}__L{layer}__{alpha}__C_{success|failure|contrastive}
    Per-step:    {task}__L{layer}__per_step_{ds}__C_{success|failure|contrastive}   (alpha=1.0, ds = denoising step)
    Linear:      {task}__L{layer}__linear__V_{success|failure|contrastive}          (direction vectors, 1D)
    Per-step V:  {task}__L{layer}__linear_per_step_{ds}__V_{success|failure|contrastive}

Defaults write to  $OPENPI_DATA_HOME/groot_n15_robocasa_conceptors.npz.
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
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = Path(os.environ.get(
    "OPENPI_DATA_HOME",
    str(Path.home() / ".cache" / "openpi"),
))

DEFAULT_ACTIVATIONS_DIR = (
    OPENPI_DATA_HOME
    / "huggingface/lerobot/brandonyang/groot_n15-robocasa-activations-v1-15env/checkpoint-120000"
)
DEFAULT_OUTPUT_NPZ = OPENPI_DATA_HOME / "groot_n15_robocasa_conceptors.npz"

NUM_DENOISING_STEPS = 4
NUM_DIT_LAYERS = 16
HIDDEN_DIM = 1536

DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
DEFAULT_LAYERS = tuple(range(NUM_DIT_LAYERS))
DEFAULT_DS_FOR_CONCEPTORS = 0     # which denoising step to use for main (per-alpha) conceptors
DEFAULT_PER_STEP_ALPHA = 1.0
MIN_PER_CLASS = 3                 # need ≥3 success AND ≥3 failure episodes to build a contrastive pair


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math (same math as libero_diagnostic.py, restated here to keep this script
# standalone with no cross-experiment import)
# ──────────────────────────────────────────────────────────────────────────────

def compute_conceptor_matrix(X: np.ndarray, alpha: float) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    N = max(1, Xc.shape[0])
    R = (Xc.T @ Xc) / N
    d = R.shape[0]
    return R @ np.linalg.inv(R + (alpha ** -2) * np.eye(d))


def conceptor_NOT(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def contrastive_conceptor(X_success: np.ndarray, X_failure: np.ndarray, alpha: float):
    C_s = compute_conceptor_matrix(X_success, alpha)
    C_f = compute_conceptor_matrix(X_failure, alpha)
    C_contr = C_s @ conceptor_NOT(C_f)
    return C_s, C_f, C_contr


def linear_direction(X_success: np.ndarray, X_failure: np.ndarray):
    v_s = X_success.mean(axis=0)
    v_f = X_failure.mean(axis=0)
    v_c = v_s - v_f
    norm = np.linalg.norm(v_c)
    v_c_unit = v_c / norm if norm > 1e-12 else v_c
    return v_s, v_f, v_c_unit


# ──────────────────────────────────────────────────────────────────────────────
# Activation loading
# ──────────────────────────────────────────────────────────────────────────────

def discover_tasks(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_episode_metadata(task_dir: Path) -> dict[int, dict]:
    info: dict[int, dict] = {}
    for ep_dir in sorted(task_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        ep_id = int(meta.get("episode_id", ep_dir.name.split("_")[1]))
        info[ep_id] = {
            "path": ep_dir,
            "success": bool(meta.get("episode_success", False)),
            "total_inference_steps": int(meta.get("total_inference_steps", 0)),
            "prompt": meta.get("prompt", ""),
        }
    return info


def load_task_activations(task_dir: Path, layer: int, ds: int, verbose: bool = False):
    """Load per-inference-step, mean-pooled DiT hidden states for one (layer, denoising-step).

    Returns:
        X         : (n_samples, hidden_dim) float32   — mean-pooled over 49 tokens
        successes : list[bool]                        — per-sample episode_success label
        ep_ids    : list[int]                         — per-sample source episode id
    """
    info = load_episode_metadata(task_dir)
    vectors, successes, ep_ids = [], [], []
    for ep_id, ep in info.items():
        ep_dir = ep["path"]
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "dit_hidden_states.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_dit_hidden_states"]     # (4, 16, 49, 1536) fp16
                    # mean-pool over the 49 action tokens → (hidden_dim,)
                    vec = arr[ds, layer].mean(axis=0).astype(np.float32)
            except Exception as e:
                if verbose:
                    print(f"    skip {npz_path}: {e}")
                continue
            vectors.append(vec)
            successes.append(ep["success"])
            ep_ids.append(ep_id)
    if not vectors:
        return np.empty((0, HIDDEN_DIM), dtype=np.float32), successes, ep_ids
    return np.stack(vectors, axis=0), successes, ep_ids


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

    tasks = discover_tasks(activations_dir)
    print(f"[build_conceptors] activations: {activations_dir}")
    print(f"[build_conceptors] tasks ({len(tasks)}): {tasks}")
    print(f"[build_conceptors] layers: {list(layers)}")
    print(f"[build_conceptors] alphas: {list(alphas)}    main ds={main_ds}")

    # First pass: per-task success/failure episode counts (for visibility + mixed-task filter)
    task_class_counts = {}
    for t in tasks:
        info = load_episode_metadata(activations_dir / t)
        n_s = sum(1 for v in info.values() if v["success"])
        n_f = sum(1 for v in info.values() if not v["success"])
        task_class_counts[t] = (n_s, n_f)
    print("\n[build_conceptors] Episode-level class balance:")
    for t, (ns, nf) in task_class_counts.items():
        mark = "OK" if (ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS) else "SKIP"
        print(f"  {t:<32s} success={ns:3d}  failure={nf:3d}  [{mark}]")

    mixed_tasks = [
        t for t, (ns, nf) in task_class_counts.items()
        if ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS
    ]
    if not mixed_tasks:
        sys.exit("[build_conceptors] no tasks have ≥3 success and ≥3 failure episodes — cannot build contrastive conceptors.")

    print(f"\n[build_conceptors] mixed-outcome tasks ({len(mixed_tasks)}): {mixed_tasks}")

    save_arrays: dict[str, np.ndarray] = {}
    stats = []

    for task in mixed_tasks:
        task_dir = activations_dir / task
        print(f"\n[build_conceptors] === {task} ===")

        for layer in layers:
            # ── main conceptors + linear direction at main_ds ──
            t0 = time.time()
            X, succ, _ = load_task_activations(task_dir, layer=layer, ds=main_ds, verbose=verbose)
            if X.shape[0] == 0:
                print(f"  L{layer:<2d} ds={main_ds}: no samples, skipping")
                continue
            succ_idx = [i for i, s in enumerate(succ) if s]
            fail_idx = [i for i, s in enumerate(succ) if not s]
            if len(succ_idx) < MIN_PER_CLASS or len(fail_idx) < MIN_PER_CLASS:
                print(f"  L{layer:<2d} ds={main_ds}: too few per class (s={len(succ_idx)}, f={len(fail_idx)})")
                continue

            X_s, X_f = X[succ_idx], X[fail_idx]

            # Linear direction (one set per layer — independent of alpha)
            v_s, v_f, v_c = linear_direction(X_s, X_f)
            save_arrays[f"{task}__L{layer}__linear__V_success"]     = v_s.astype(np.float32)
            save_arrays[f"{task}__L{layer}__linear__V_failure"]     = v_f.astype(np.float32)
            save_arrays[f"{task}__L{layer}__linear__V_contrastive"] = v_c.astype(np.float32)

            # Conceptors per alpha
            for alpha in alphas:
                C_s, C_f, C_c = contrastive_conceptor(X_s, X_f, alpha)
                save_arrays[f"{task}__L{layer}__{alpha}__C_success"]     = C_s.astype(np.float32)
                save_arrays[f"{task}__L{layer}__{alpha}__C_failure"]     = C_f.astype(np.float32)
                save_arrays[f"{task}__L{layer}__{alpha}__C_contrastive"] = C_c.astype(np.float32)

            # Report quota/overlap at alpha=1.0 (matches pi05 diagnostic output)
            C_s1, C_f1, C_c1 = contrastive_conceptor(X_s, X_f, 1.0)
            d = HIDDEN_DIM
            q_s = float(np.trace(C_s1)) / d
            q_f = float(np.trace(C_f1)) / d
            num = float(np.einsum("ij,ji->", C_s1, C_f1))
            denom = np.sqrt(float(np.einsum("ij,ji->", C_s1, C_s1))
                            * float(np.einsum("ij,ji->", C_f1, C_f1)))
            overlap = num / denom if denom > 0 else 0.0
            print(f"  L{layer:<2d} ds={main_ds} "
                  f"n(s,f)=({len(succ_idx):2d},{len(fail_idx):2d}) "
                  f"q(Cs)={q_s:.3f} q(Cf)={q_f:.3f} overlap={overlap:.3f} "
                  f"({time.time()-t0:.1f}s)")
            stats.append((task, layer, len(succ_idx), len(fail_idx), q_s, q_f, overlap))

            # ── per-denoising-step conceptors + linear direction (alpha=per_step_alpha) ──
            if skip_per_step:
                continue
            for ds in range(NUM_DENOISING_STEPS):
                if ds == main_ds:
                    # Reuse: write per_step alias pointing to same arrays
                    save_arrays[f"{task}__L{layer}__per_step_{ds}__C_success"]     = save_arrays[f"{task}__L{layer}__{per_step_alpha}__C_success"]
                    save_arrays[f"{task}__L{layer}__per_step_{ds}__C_failure"]     = save_arrays[f"{task}__L{layer}__{per_step_alpha}__C_failure"]
                    save_arrays[f"{task}__L{layer}__per_step_{ds}__C_contrastive"] = save_arrays[f"{task}__L{layer}__{per_step_alpha}__C_contrastive"]
                    save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_success"]     = save_arrays[f"{task}__L{layer}__linear__V_success"]
                    save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_failure"]     = save_arrays[f"{task}__L{layer}__linear__V_failure"]
                    save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_contrastive"] = save_arrays[f"{task}__L{layer}__linear__V_contrastive"]
                    continue
                X_ds, succ_ds, _ = load_task_activations(task_dir, layer=layer, ds=ds, verbose=verbose)
                if X_ds.shape[0] == 0:
                    continue
                s_idx = [i for i, s in enumerate(succ_ds) if s]
                f_idx = [i for i, s in enumerate(succ_ds) if not s]
                if len(s_idx) < MIN_PER_CLASS or len(f_idx) < MIN_PER_CLASS:
                    continue
                C_s_ds, C_f_ds, C_c_ds = contrastive_conceptor(X_ds[s_idx], X_ds[f_idx], per_step_alpha)
                save_arrays[f"{task}__L{layer}__per_step_{ds}__C_success"]     = C_s_ds.astype(np.float32)
                save_arrays[f"{task}__L{layer}__per_step_{ds}__C_failure"]     = C_f_ds.astype(np.float32)
                save_arrays[f"{task}__L{layer}__per_step_{ds}__C_contrastive"] = C_c_ds.astype(np.float32)
                v_s_ds, v_f_ds, v_c_ds = linear_direction(X_ds[s_idx], X_ds[f_idx])
                save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_success"]     = v_s_ds.astype(np.float32)
                save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_failure"]     = v_f_ds.astype(np.float32)
                save_arrays[f"{task}__L{layer}__linear_per_step_{ds}__V_contrastive"] = v_c_ds.astype(np.float32)

    if not save_arrays:
        sys.exit("[build_conceptors] no arrays produced — check activation dir and class balances.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[build_conceptors] writing {len(save_arrays)} arrays to {output_npz} ...")
    np.savez_compressed(output_npz, **save_arrays)
    size_mb = output_npz.stat().st_size / 1e6
    print(f"[build_conceptors] wrote {output_npz} ({size_mb:.1f} MB, {len(save_arrays)} arrays)")

    # Summary
    print("\n[build_conceptors] summary  (alpha=1.0, ds=main):")
    print(f"  {'task':<32s} {'L':>2s} {'n_s':>3s} {'n_f':>3s} {'q(Cs)':>6s} {'q(Cf)':>6s} {'overlap':>7s}")
    for task, layer, ns, nf, q_s, q_f, ov in stats:
        print(f"  {task:<32s} {layer:>2d} {ns:>3d} {nf:>3d} {q_s:>6.3f} {q_f:>6.3f} {ov:>7.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR,
                   help=f"directory containing per-task activation trees (default: {DEFAULT_ACTIVATIONS_DIR})")
    p.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ,
                   help=f"output .npz path (default: {DEFAULT_OUTPUT_NPZ})")
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                   help="alpha (aperture) values for the conceptor grid")
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                   help="DiT layer indices (0..15) to build conceptors at")
    p.add_argument("--main-ds", type=int, default=DEFAULT_DS_FOR_CONCEPTORS,
                   help="denoising-step index used for the main per-alpha conceptors (0..3)")
    p.add_argument("--per-step-alpha", type=float, default=DEFAULT_PER_STEP_ALPHA,
                   help="alpha used for per-denoising-step conceptors")
    p.add_argument("--skip-per-step", action="store_true",
                   help="skip per-denoising-step conceptors (faster, smaller npz)")
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
