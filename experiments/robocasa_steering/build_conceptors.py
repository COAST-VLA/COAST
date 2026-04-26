#!/usr/bin/env python3
"""Build contrastive conceptors + linear-steering directions for the
diffusion_policy DiffusionTransformerHybridImagePolicy evaluated on RoboCasa.

Input: an activation tree produced by
``diffusion_policy/collect_activations_robocasa.py``. Per-step layout:

    <activations_dir>/<task>/episode_NNN_env_NNN/step_NNNN/suffix_residual.npz
        key   = "all_suffix_residual"
        shape = (D=100, L=12, H=10, C=512)   # per the dp_v1 schema

Mean-pools over the 10 action tokens so each inference step contributes ONE
(C,)-vector per (layer, denoising_step).

Output: a single ``diffusion_policy_conceptors.npz`` with keys in the same
format as the openpi/pi0.5 LIBERO bundle so ``select_parameters.py`` and
``steering.py`` can read it unchanged:

    {task}__L{layer}__{alpha}__C_{success|failure|contrastive}          # main grid
    {task}__L{layer}__per_step_{ds}__C_{success|failure|contrastive}    # per-step grid (alpha=per_step_alpha)
    {task}__L{layer}__linear__V_{success|failure|contrastive}           # ActAdd direction
    {task}__L{layer}__linear_per_step_{ds}__V_{success|failure|contrastive}

For a 12-layer 100-denoising-step model the per-step path is costly
(3 matrices x L x K x C^2 fp32 per task), so per_step is built at only K
evenly-spaced indices (default 10) rather than all 100 steps. The steering
hook does nearest-neighbour lookup among the built indices at inference.
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
# Defaults (diffusion_policy / RoboCasa)
# ──────────────────────────────────────────────────────────────────────────────

DP_DATA_HOME = Path(os.environ.get(
    "DP_DATA_HOME",
    str(Path.home() / ".cache" / "diffusion_policy"),
))

DEFAULT_ACTIVATIONS_DIR = Path(
    os.environ.get("DP_ACTIVATIONS_DIR",
                   str(Path(__file__).resolve().parents[2] / "activations" / "latest"))
)
DEFAULT_OUTPUT_NPZ = DP_DATA_HOME / "diffusion_policy_conceptors.npz"

NUM_DENOISING_STEPS = 100
HIDDEN_DIM = 512
NUM_CAPTURED_LAYERS = 12

# Identity map: the dp_v1 schema captures all 12 decoder layers in order.
LAYER_MAP = {i: i for i in range(NUM_CAPTURED_LAYERS)}

DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
DEFAULT_LAYERS = (5, 8, 11)       # three deep decoder layers
DEFAULT_DS_FOR_CONCEPTORS = 0     # which denoising step feeds the per-alpha grid
DEFAULT_PER_STEP_ALPHA = 1.0      # legacy single-alpha default (kept for backward-compat keys)
# Matches pi05_robocasa / pi05_libero global alpha grid so per_step ablations span
# the same alpha range as global ablations (see openpi-metaworld pi05_robocasa
# conceptor_steering.py default alphas + run_comprehensive_sweep.sh).
DEFAULT_PER_STEP_ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
DEFAULT_PER_STEP_COUNT = 10       # build per-step conceptors at this many evenly-spaced ds indices
MIN_PER_CLASS = 1                 # need >=1 success AND >=1 failure episode


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math  (identical to the pi0.5 LIBERO version)
# ──────────────────────────────────────────────────────────────────────────────

def compute_conceptor_matrix(X: np.ndarray, alpha: float) -> np.ndarray:
    """C = R (R + alpha^-2 I)^-1, where R = X_centered^T X_centered / N."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = max(1, Xc.shape[0])
    R = (Xc.T @ Xc) / N
    d = R.shape[0]
    return R @ np.linalg.inv(R + (alpha ** -2) * np.eye(d))


def conceptor_NOT(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def contrastive_conceptor(X_s: np.ndarray, X_f: np.ndarray, alpha: float):
    C_s = compute_conceptor_matrix(X_s, alpha)
    C_f = compute_conceptor_matrix(X_f, alpha)
    C_c = C_s @ conceptor_NOT(C_f)
    return C_s, C_f, C_c


def linear_direction(X_s: np.ndarray, X_f: np.ndarray):
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
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))


def load_episode_metadata(task_dir: Path) -> dict[str, dict]:
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


def load_task_matrix(task_dir: Path, layer_idx: int, verbose: bool = False):
    """Load all denoising-step slices at a single layer, one row per
    per-episode-per-inference-step.

    Returns:
        X:          (N, D, C) float32, mean-pooled over the H action tokens
        successes:  list[bool] of length N, per-sample episode-success label
    """
    info = load_episode_metadata(task_dir)
    mats, successes = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]  # (D, L, H, C)
                    slab = arr[:, layer_idx].mean(axis=-2).astype(np.float32)  # (D, C)
            except Exception as e:
                if verbose:
                    print(f"    skip {npz_path}: {e}")
                continue
            mats.append(slab)
            successes.append(ep["success"])
    if not mats:
        return np.empty((0, NUM_DENOISING_STEPS, HIDDEN_DIM), dtype=np.float32), successes
    return np.stack(mats, axis=0), successes


# ──────────────────────────────────────────────────────────────────────────────
# Main build loop
# ──────────────────────────────────────────────────────────────────────────────

def _per_step_indices(count: int) -> list[int]:
    count = max(1, min(count, NUM_DENOISING_STEPS))
    return sorted(set(np.linspace(0, NUM_DENOISING_STEPS - 1, count).round().astype(int).tolist()))


def build_conceptors(
    activations_dir: Path,
    output_npz: Path,
    alphas: tuple[float, ...],
    layers: tuple[int, ...],
    main_ds: int,
    per_step_alphas: tuple[float, ...],
    per_step_count: int,
    skip_per_step: bool,
    verbose: bool,
) -> None:
    # First per_step alpha doubles as the "legacy" alpha for backward-compat
    # alpha-less keys (`{task}__L{L}__per_step_{ds}__C_*`).
    per_step_alpha_legacy = per_step_alphas[0] if per_step_alphas else DEFAULT_PER_STEP_ALPHA
    if not activations_dir.is_dir():
        sys.exit(f"[build_conceptors] activations dir not found: {activations_dir}")
    for L in layers:
        if L not in LAYER_MAP:
            sys.exit(f"[build_conceptors] layer {L} not in captured set {list(LAYER_MAP)}")
    if not (0 <= main_ds < NUM_DENOISING_STEPS):
        sys.exit(f"[build_conceptors] --main-ds must be in [0, {NUM_DENOISING_STEPS})")

    tasks = discover_tasks(activations_dir)
    ps_idx = _per_step_indices(per_step_count) if not skip_per_step else []
    print(f"[build_conceptors] activations: {activations_dir}")
    print(f"[build_conceptors] tasks ({len(tasks)}): {tasks}")
    print(f"[build_conceptors] layers: {list(layers)}  alphas: {list(alphas)}  main_ds: {main_ds}")
    print(f"[build_conceptors] per_step indices ({len(ps_idx)}): {ps_idx}")

    # Class balance per task
    task_class_counts = {}
    for t in tasks:
        info = load_episode_metadata(activations_dir / t)
        n_s = sum(1 for v in info.values() if v["success"])
        n_f = sum(1 for v in info.values() if not v["success"])
        task_class_counts[t] = (n_s, n_f)
    print(f"\n[build_conceptors] episode class balance (need >= {MIN_PER_CLASS} per class):")
    for t, (ns, nf) in task_class_counts.items():
        mark = "OK" if (ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS) else "SKIP"
        print(f"  {t[:60]:<60s} success={ns:3d}  failure={nf:3d}  [{mark}]")

    mixed_tasks = [t for t, (ns, nf) in task_class_counts.items()
                   if ns >= MIN_PER_CLASS and nf >= MIN_PER_CLASS]
    if not mixed_tasks:
        sys.exit("[build_conceptors] no mixed-outcome tasks - cannot build contrastive conceptors.")

    save_arrays: dict[str, np.ndarray] = {}
    stats = []

    for task in mixed_tasks:
        task_dir = activations_dir / task
        print(f"\n[build_conceptors] === {task[:60]} ===")

        for L in layers:
            layer_idx = LAYER_MAP[L]
            t0 = time.time()
            Xall, succ = load_task_matrix(task_dir, layer_idx, verbose=verbose)
            if Xall.shape[0] == 0:
                print(f"  L{L:<2d}: no samples, skipping")
                continue
            s_idx = [i for i, s in enumerate(succ) if s]
            f_idx = [i for i, s in enumerate(succ) if not s]
            if len(s_idx) < MIN_PER_CLASS or len(f_idx) < MIN_PER_CLASS:
                print(f"  L{L:<2d}: too few per class (s={len(s_idx)}, f={len(f_idx)})")
                continue

            # ── main (per-alpha) conceptors + linear direction at main_ds ──
            X = Xall[:, main_ds]
            X_s, X_f = X[s_idx], X[f_idx]
            v_s, v_f, v_c = linear_direction(X_s, X_f)
            save_arrays[f"{task}__L{L}__linear__V_success"]     = v_s.astype(np.float32)
            save_arrays[f"{task}__L{L}__linear__V_failure"]     = v_f.astype(np.float32)
            save_arrays[f"{task}__L{L}__linear__V_contrastive"] = v_c.astype(np.float32)

            for a in alphas:
                C_s, C_f, C_c = contrastive_conceptor(X_s, X_f, a)
                save_arrays[f"{task}__L{L}__{a}__C_success"]     = C_s.astype(np.float32)
                save_arrays[f"{task}__L{L}__{a}__C_failure"]     = C_f.astype(np.float32)
                save_arrays[f"{task}__L{L}__{a}__C_contrastive"] = C_c.astype(np.float32)

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
                  f"({time.time() - t0:.1f}s)")
            stats.append((task[:40], L, len(s_idx), len(f_idx), q_s, q_f, overlap))

            # ── per-step conceptors, sparse grid ──
            # Emit alpha-aware keys for every alpha in `per_step_alphas`:
            #   {task}__L{L}__{a}__per_step_{ds}__C_{success|failure|contrastive}
            # plus legacy alpha-less keys (alias to the first alpha) for backward-compat.
            if skip_per_step:
                continue
            v_s_main = save_arrays[f"{task}__L{L}__linear__V_success"]
            v_f_main = save_arrays[f"{task}__L{L}__linear__V_failure"]
            v_c_main = save_arrays[f"{task}__L{L}__linear__V_contrastive"]
            for ds in ps_idx:
                X_ds = Xall[:, ds]
                X_ds_s, X_ds_f = X_ds[s_idx], X_ds[f_idx]
                # Compute per-ds linear directions once (alpha-independent).
                if ds == main_ds:
                    v_s_ds, v_f_ds, v_c_ds = v_s_main, v_f_main, v_c_main
                else:
                    v_s_ds, v_f_ds, v_c_ds = linear_direction(X_ds_s, X_ds_f)
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_success"]     = v_s_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_failure"]     = v_f_ds.astype(np.float32)
                save_arrays[f"{task}__L{L}__linear_per_step_{ds}__V_contrastive"] = v_c_ds.astype(np.float32)

                for a in per_step_alphas:
                    if ds == main_ds and any(np.isclose(a, ga) for ga in alphas):
                        # Already built as a global conceptor at this (L, alpha) — alias.
                        Cs = save_arrays[f"{task}__L{L}__{a}__C_success"]
                        Cf = save_arrays[f"{task}__L{L}__{a}__C_failure"]
                        Cc = save_arrays[f"{task}__L{L}__{a}__C_contrastive"]
                    else:
                        Cs, Cf, Cc = contrastive_conceptor(X_ds_s, X_ds_f, a)
                        Cs, Cf, Cc = Cs.astype(np.float32), Cf.astype(np.float32), Cc.astype(np.float32)
                    save_arrays[f"{task}__L{L}__{a}__per_step_{ds}__C_success"]     = Cs
                    save_arrays[f"{task}__L{L}__{a}__per_step_{ds}__C_failure"]     = Cf
                    save_arrays[f"{task}__L{L}__{a}__per_step_{ds}__C_contrastive"] = Cc

                # Legacy alpha-less keys (back-compat): alias to the first per_step alpha.
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_success"]     = save_arrays[f"{task}__L{L}__{per_step_alpha_legacy}__per_step_{ds}__C_success"]
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_failure"]     = save_arrays[f"{task}__L{L}__{per_step_alpha_legacy}__per_step_{ds}__C_failure"]
                save_arrays[f"{task}__L{L}__per_step_{ds}__C_contrastive"] = save_arrays[f"{task}__L{L}__{per_step_alpha_legacy}__per_step_{ds}__C_contrastive"]

    if not save_arrays:
        sys.exit("[build_conceptors] no arrays produced - check activation dir and class balances.")

    # Stamp the list of built per-step indices so the steering hook can map
    # current_step -> nearest built index without needing to probe the npz.
    save_arrays["_per_step_indices"] = np.array(ps_idx, dtype=np.int32)
    save_arrays["_hidden_dim"]       = np.array(HIDDEN_DIM, dtype=np.int32)
    save_arrays["_num_denoising_steps"] = np.array(NUM_DENOISING_STEPS, dtype=np.int32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[build_conceptors] writing {len(save_arrays)} arrays to {output_npz} ...")
    np.savez_compressed(output_npz, **save_arrays)
    size_mb = output_npz.stat().st_size / 1e6
    print(f"[build_conceptors] wrote {output_npz} ({size_mb:.1f} MB)")

    print("\n[build_conceptors] summary (alpha=1.0, ds=main_ds):")
    print(f"  {'task':<40s} {'L':>2s} {'n_s':>3s} {'n_f':>3s} {'q(Cs)':>6s} {'q(Cf)':>6s} {'overlap':>7s}")
    for row in stats:
        print(f"  {row[0]:<40s} {row[1]:>2d} {row[2]:>3d} {row[3]:>3d} {row[4]:>6.3f} {row[5]:>6.3f} {row[6]:>7.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR,
                   help=f"<output_root>/<checkpoint_step> from collect_activations_robocasa.py "
                        f"(default: {DEFAULT_ACTIVATIONS_DIR}).")
    p.add_argument("--output-npz", type=Path, default=DEFAULT_OUTPUT_NPZ)
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                   help=f"Decoder layer indices (captured set: {list(LAYER_MAP)})")
    p.add_argument("--main-ds", type=int, default=DEFAULT_DS_FOR_CONCEPTORS,
                   help=f"Denoising step index (0..{NUM_DENOISING_STEPS - 1}) for the per-alpha grid")
    # Legacy single-alpha flag is preserved; if --per-step-alphas is also given,
    # the plural flag takes precedence.
    p.add_argument("--per-step-alpha", type=float, default=None,
                   help="Legacy. Single alpha for per_step. Overridden by --per-step-alphas.")
    p.add_argument("--per-step-alphas", type=float, nargs="+", default=list(DEFAULT_PER_STEP_ALPHAS),
                   help=f"Alphas for per_step conceptors. Keys are alpha-aware "
                        f"(`__L{{L}}__{{alpha}}__per_step_{{ds}}__C_*`). "
                        f"First alpha also feeds the legacy alpha-less keys for backward compat. "
                        f"Default: {list(DEFAULT_PER_STEP_ALPHAS)}.")
    p.add_argument("--per-step-count", type=int, default=DEFAULT_PER_STEP_COUNT,
                   help="Build per-step conceptors at K evenly-spaced ds indices (default 10).")
    p.add_argument("--skip-per-step", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Legacy --per-step-alpha (singular) -> override the plural list.
    per_step_alphas = list(args.per_step_alphas)
    if args.per_step_alpha is not None:
        per_step_alphas = [args.per_step_alpha]
    build_conceptors(
        activations_dir=args.activations_dir,
        output_npz=args.output_npz,
        alphas=tuple(args.alphas),
        layers=tuple(args.layers),
        main_ds=args.main_ds,
        per_step_alphas=tuple(per_step_alphas),
        per_step_count=args.per_step_count,
        skip_per_step=args.skip_per_step,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
