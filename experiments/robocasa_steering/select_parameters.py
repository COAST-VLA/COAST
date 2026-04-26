#!/usr/bin/env python3
"""Parameter selection from a diffusion_policy conceptor .npz.

Identical selection rule to the pi0.5 LIBERO bundle (quota + overlap sweet
spot), but with no hard-coded hidden dim - compute_quota normalises by
C.shape[0] so 512-wide diffusion_policy conceptors and 1024-wide pi0.5
conceptors both land on the same [0, 1] quota scale.

Usage:
    python select_parameters.py \\
        --conceptor-npz ~/.cache/diffusion_policy/diffusion_policy_conceptors.npz \\
        --output-json   experiments/robocasa_steering/selected_params.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np


def compute_quota(C):
    return float(np.trace(C)) / C.shape[0]


def compute_overlap(Cs, Cf):
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    if ns * nf == 0:
        return 0.0
    return num / np.sqrt(ns * nf)


KEY_RE = re.compile(r"^(.+?)__L(\d+)__(.+?)__(C_.+)$")


def parse_npz_structure(npz):
    tasks, layers, alphas = set(), set(), set()
    for key in npz.files:
        if key.startswith("_"):
            continue
        m = KEY_RE.match(key)
        if not m:
            continue
        tasks.add(m.group(1))
        layers.add(int(m.group(2)))
        try:
            alphas.add(float(m.group(3)))
        except ValueError:
            pass
    return sorted(tasks), sorted(layers), sorted(alphas)


def select_parameters(npz, overlap_low=0.85, overlap_high=0.95,
                      candidate_betas=None, quota_alpha=10.0,
                      conceptor_type_for_quota="contrastive"):
    if candidate_betas is None:
        candidate_betas = [0.1, 0.3]

    tasks, layers, alphas = parse_npz_structure(npz)
    print(f"Found {len(tasks)} tasks, layers={layers}, alphas={alphas}")

    # Step 1: pick best layer by mean quota
    layer_quotas = {}
    for L in layers:
        quotas = []
        for t in tasks:
            key = f"{t}__L{L}__{quota_alpha}__C_{conceptor_type_for_quota}"
            if key in npz.files:
                quotas.append(compute_quota(npz[key]))
        if quotas:
            layer_quotas[L] = float(np.mean(quotas))

    if not layer_quotas:
        print("ERROR: no quota data found. Check conceptor key format.", file=sys.stderr)
        sys.exit(1)

    best_layer = max(layer_quotas, key=layer_quotas.get)
    print("\nStep 1 - Layer selection (by mean quota):")
    for L in layers:
        mark = " <- selected" if L == best_layer else ""
        print(f"  L={L:>2d}  quota={layer_quotas.get(L, float('nan')):.4f}{mark}")

    # Step 2: alphas in overlap band at best layer
    alpha_overlaps = {}
    for a in alphas:
        per_task = []
        for t in tasks:
            Cs_key = f"{t}__L{best_layer}__{a}__C_success"
            Cf_key = f"{t}__L{best_layer}__{a}__C_failure"
            if Cs_key in npz.files and Cf_key in npz.files:
                per_task.append(compute_overlap(npz[Cs_key], npz[Cf_key]))
        if per_task:
            alpha_overlaps[a] = float(np.mean(per_task))

    print(f"\nStep 2 - Alpha selection (overlap sweet spot [{overlap_low}, {overlap_high}] "
          f"at L={best_layer}):")
    selected_alphas = []
    for a in alphas:
        ov = alpha_overlaps.get(a, float("nan"))
        in_band = overlap_low <= ov <= overlap_high
        mark = " <- selected" if in_band else ""
        print(f"  alpha={a:<5g}  overlap={ov:.3f}{mark}")
        if in_band:
            selected_alphas.append(a)

    if not selected_alphas:
        band_center = (overlap_low + overlap_high) / 2
        closest = min(alphas, key=lambda a: abs(alpha_overlaps.get(a, 999) - band_center))
        selected_alphas = [closest]
        print(f"  (no alpha in band, falling back to closest: alpha={closest})")

    selected_betas = candidate_betas
    print(f"\nStep 3 - Beta selection: {selected_betas}")

    full_grid = len(layers) * len(alphas) * 3 * 3
    narrow_grid = 1 * len(selected_alphas) * len(selected_betas) * 3
    print(f"\n{'=' * 60}")
    print(f"SELECTED:  layer={best_layer}, alphas={selected_alphas}, betas={selected_betas}")
    print(f"Grid reduction: {full_grid} -> {narrow_grid} conditions/task "
          f"({full_grid / max(1, narrow_grid):.1f}x reduction)")
    print(f"{'=' * 60}")

    return {
        "best_layer": best_layer,
        "selected_alphas": selected_alphas,
        "selected_betas": selected_betas,
        "overlap_band": [overlap_low, overlap_high],
        "diagnostics": {
            "layer_quotas": {str(L): v for L, v in layer_quotas.items()},
            "alpha_overlaps_at_best_layer": {str(a): v for a, v in alpha_overlaps.items()},
            "tasks": tasks,
            "all_layers": layers,
            "all_alphas": alphas,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conceptor-npz", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--overlap-low", type=float, default=0.85)
    p.add_argument("--overlap-high", type=float, default=0.95)
    p.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--quota-alpha", type=float, default=10.0)
    args = p.parse_args()

    npz = np.load(args.conceptor_npz, allow_pickle=False)
    result = select_parameters(
        npz,
        overlap_low=args.overlap_low,
        overlap_high=args.overlap_high,
        candidate_betas=args.betas,
        quota_alpha=args.quota_alpha,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {args.output_json}")


if __name__ == "__main__":
    main()
