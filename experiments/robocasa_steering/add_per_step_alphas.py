#!/usr/bin/env python3
"""Add per_step conceptor matrices at additional alphas to existing conceptors.npz
files, WITHOUT re-reading the activation tree.

Math
----
A conceptor at alpha is C_a = R (R + a^-2 I)^-1, where R is the data correlation
matrix.  C_a shares eigenvectors with R, with eigenvalues e_i = lam_i / (lam_i + a^-2).

Given C_orig at known alpha_orig:
    e_i      = eigh(C_orig)
    lam_i    = e_i * alpha_orig^2 / (1 - e_i)
    e_new_i  = lam_i / (lam_i + alpha_new^-2)
    C_new    = U diag(e_new_i) U^T

Per_step contrastive C_c = C_s @ (I - C_f) is rebuilt by recomputing both C_s and
C_f at the target alpha and multiplying.

Existing per_step keys (alpha implicit at build-time `--per-step-alpha`, default 1.0):
    {task}__L{layer}__per_step_{ds}__C_{success|failure|contrastive}

New keys this script writes (alpha explicit, additive — does not delete the legacy
keys):
    {task}__L{layer}__{alpha}__per_step_{ds}__C_{success|failure|contrastive}

Usage
-----
    python experiments/robocasa_steering/add_per_step_alphas.py \\
        --conceptor-npz experiments/robocasa_steering/conceptors/<ckpt>/conceptors.npz \\
        --orig-alpha    1.0 \\
        --new-alphas    0.1 0.5 2.0
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


PER_STEP_LEGACY_RE = re.compile(
    r"^(?P<task>.+)__L(?P<layer>\d+)__per_step_(?P<ds>\d+)__C_(?P<kind>success|failure|contrastive)$"
)


def conceptor_at_alpha(C_orig: np.ndarray, alpha_orig: float, alpha_new: float,
                       eps: float = 1e-9) -> np.ndarray:
    """Recompute a symmetric conceptor at a new alpha via eigendecomposition."""
    # Symmetrize to cancel any fp asymmetry.
    C_sym = 0.5 * (C_orig.astype(np.float64) + C_orig.astype(np.float64).T)
    e, U = np.linalg.eigh(C_sym)
    e = np.clip(e, eps, 1.0 - eps)
    lam = e * (alpha_orig ** 2) / (1.0 - e)
    e_new = lam / (lam + alpha_new ** -2)
    return (U * e_new) @ U.T


def discover_per_step_legacy_keys(npz) -> dict[tuple[str, int, int], dict[str, np.ndarray]]:
    """Group existing per_step keys by (task, layer, ds) -> {kind: matrix}."""
    out: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for k in npz.files:
        m = PER_STEP_LEGACY_RE.match(k)
        if not m:
            continue
        key = (m.group("task"), int(m.group("layer")), int(m.group("ds")))
        out.setdefault(key, {})[m.group("kind")] = npz[k]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conceptor-npz", required=True, type=Path)
    ap.add_argument("--orig-alpha", type=float, default=1.0,
                    help="Alpha at which existing per_step keys were built.")
    ap.add_argument("--new-alphas", type=float, nargs="+", required=True,
                    help="Alphas to add (alpha-aware keys). May include orig-alpha "
                         "(will write alpha-aware aliases) or be entirely new.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output npz path. Default: rewrite input in place.")
    args = ap.parse_args()

    if not args.conceptor_npz.is_file():
        sys.exit(f"missing: {args.conceptor_npz}")
    out_path = args.output or args.conceptor_npz

    print(f"[add_per_step_alphas] loading {args.conceptor_npz}")
    npz = np.load(args.conceptor_npz)
    base = {k: npz[k] for k in npz.files}

    legacy = discover_per_step_legacy_keys(npz)
    print(f"  legacy per_step triples: {len(legacy)}")

    n_new_keys = 0
    for (task, L, ds), kinds in sorted(legacy.items()):
        if "success" not in kinds or "failure" not in kinds:
            print(f"  skip {task} L{L} ds{ds}: missing C_success or C_failure")
            continue
        Cs_orig = kinds["success"]
        Cf_orig = kinds["failure"]
        d = Cs_orig.shape[0]
        I_d = np.eye(d, dtype=np.float64)

        for a in args.new_alphas:
            Cs_new = conceptor_at_alpha(Cs_orig, args.orig_alpha, a)
            Cf_new = conceptor_at_alpha(Cf_orig, args.orig_alpha, a)
            Cc_new = Cs_new @ (I_d - Cf_new)
            base[f"{task}__L{L}__{a}__per_step_{ds}__C_success"]     = Cs_new.astype(np.float32)
            base[f"{task}__L{L}__{a}__per_step_{ds}__C_failure"]     = Cf_new.astype(np.float32)
            base[f"{task}__L{L}__{a}__per_step_{ds}__C_contrastive"] = Cc_new.astype(np.float32)
            n_new_keys += 3

    print(f"  added {n_new_keys} new alpha-aware per_step keys "
          f"(across {len(args.new_alphas)} alphas)")

    print(f"[add_per_step_alphas] writing {out_path}")
    np.savez(out_path, **base)
    print(f"  done.  size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
