#!/usr/bin/env python3
"""
Parameter Selection via Conceptor Diagnostics
==============================================

Reads a pre-computed conceptor .npz file, computes quota and overlap,
and outputs a narrowed (layer, alphas, betas) parameter set for the
downstream steering sweep.

The selection rule (derived from LIBERO diagnostic analysis):
  1. Pick the layer with the highest mean quota across tasks.
  2. Keep alphas whose mean overlap at that layer falls in the
     sweet-spot band (default [0.85, 0.95]).
  3. Drop beta=0.5 (universally harmful); keep {0.1, 0.3}.

Output: a JSON file with the selected parameters, plus a human-readable
summary printed to stdout.

Usage:
    python select_parameters.py \
        --conceptor-npz /path/to/conceptors.npz \
        --output-json   /path/to/selected_params.json \
        [--overlap-low 0.85] [--overlap-high 0.95] \
        [--betas 0.1 0.3]
"""

import argparse
import json
import re
import sys

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_quota(C):
    """Quota q(C) = (1/d) tr(C)."""
    return float(np.trace(C)) / C.shape[0]


def compute_overlap(Cs, Cf):
    """Normalised similarity sim(Cs, Cf) = tr(Cs Cf) / sqrt(tr(Cs^2) tr(Cf^2))."""
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    if ns * nf == 0:
        return 0.0
    return num / np.sqrt(ns * nf)


# ──────────────────────────────────────────────────────────────────────────────
# Parse the .npz to discover tasks, layers, alphas
# ──────────────────────────────────────────────────────────────────────────────

KEY_RE = re.compile(r"^(.+?)__L(\d+)__(.+?)__(C_.+)$")


def parse_npz_structure(npz):
    """Return sorted lists of tasks, layers, and numeric alphas found in the npz."""
    tasks, layers, alphas = set(), set(), set()
    for key in npz.files:
        m = KEY_RE.match(key)
        if not m:
            continue
        tasks.add(m.group(1))
        layers.add(int(m.group(2)))
        alpha_str = m.group(3)
        # Only keep numeric alphas (skip per_step_*)
        try:
            alphas.add(float(alpha_str))
        except ValueError:
            pass
    return sorted(tasks), sorted(layers), sorted(alphas)


# ──────────────────────────────────────────────────────────────────────────────
# Main selection logic
# ──────────────────────────────────────────────────────────────────────────────

def select_parameters(npz, overlap_low=0.85, overlap_high=0.95,
                      candidate_betas=None, quota_alpha=10.0,
                      conceptor_type_for_quota="contrastive"):
    """
    Run the three-step selection rule.

    Returns a dict with:
        best_layer, selected_alphas, selected_betas,
        diagnostics (per-task/layer/alpha quota and overlap tables)
    """
    if candidate_betas is None:
        candidate_betas = [0.1, 0.3]

    tasks, layers, alphas = parse_npz_structure(npz)
    print(f"Found {len(tasks)} tasks, layers={layers}, alphas={alphas}")

    # ── Step 1: Pick best layer by mean quota ────────────────────────────────
    # Use quota_alpha for quota computation (largest alpha = sharpest conceptor).
    layer_quotas = {}
    for L in layers:
        quotas = []
        for t in tasks:
            key = f"{t}__L{L}__{quota_alpha}__C_{conceptor_type_for_quota}"
            if key in npz:
                quotas.append(compute_quota(npz[key]))
        if quotas:
            layer_quotas[L] = float(np.mean(quotas))

    if not layer_quotas:
        print("ERROR: No quota data found. Check conceptor key format.", file=sys.stderr)
        sys.exit(1)

    best_layer = max(layer_quotas, key=layer_quotas.get)
    print(f"\nStep 1 — Layer selection (by mean quota):")
    for L in layers:
        marker = " ◀ selected" if L == best_layer else ""
        q_val = layer_quotas.get(L, float("nan"))
        print(f"  L={L:>2d}  quota={q_val:.4f}{marker}")

    # ── Step 2: Pick alphas by overlap at best layer ─────────────────────────
    alpha_overlaps = {}
    overlap_detail = {}  # per-task detail for diagnostics
    for a in alphas:
        per_task = []
        for t in tasks:
            Cs_key = f"{t}__L{best_layer}__{a}__C_success"
            Cf_key = f"{t}__L{best_layer}__{a}__C_failure"
            if Cs_key in npz and Cf_key in npz:
                ov = compute_overlap(npz[Cs_key], npz[Cf_key])
                per_task.append(ov)
                overlap_detail[(t, a)] = ov
        if per_task:
            alpha_overlaps[a] = float(np.mean(per_task))

    print(f"\nStep 2 — Alpha selection (overlap sweet spot [{overlap_low}, {overlap_high}] at L={best_layer}):")
    selected_alphas = []
    for a in alphas:
        ov = alpha_overlaps.get(a, float("nan"))
        in_band = overlap_low <= ov <= overlap_high
        marker = " ◀ selected" if in_band else ""
        print(f"  α={a:<5g}  overlap={ov:.3f}{marker}")
        if in_band:
            selected_alphas.append(a)

    # Fallback: if no alpha lands in the band, pick the one closest to the band center
    if not selected_alphas:
        band_center = (overlap_low + overlap_high) / 2
        closest = min(alphas, key=lambda a: abs(alpha_overlaps.get(a, 999) - band_center))
        selected_alphas = [closest]
        print(f"  ⚠ No alpha in band — falling back to closest: α={closest}")

    # ── Step 3: Betas ────────────────────────────────────────────────────────
    selected_betas = candidate_betas
    print(f"\nStep 3 — Beta selection: {selected_betas}")

    # ── Summary ──────────────────────────────────────────────────────────────
    full_grid = len(layers) * len(alphas) * 3 * 3  # original: layers × alphas × 3 betas × 3 strategies
    narrow_grid = 1 * len(selected_alphas) * len(selected_betas) * 3  # 1 layer, selected alphas/betas, keep all strategies
    print(f"\n{'='*60}")
    print(f"SELECTED:  layer={best_layer}, alphas={selected_alphas}, betas={selected_betas}")
    print(f"Grid reduction: {full_grid} → {narrow_grid} conditions/task "
          f"({full_grid/narrow_grid:.1f}× reduction)")
    print(f"{'='*60}")

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
    parser = argparse.ArgumentParser(
        description="Select steering parameters from pre-computed conceptors."
    )
    parser.add_argument("--conceptor-npz", required=True,
                        help="Path to the pre-computed conceptors .npz file.")
    parser.add_argument("--output-json", required=True,
                        help="Where to write the selected-parameters JSON.")
    parser.add_argument("--overlap-low", type=float, default=0.85,
                        help="Lower bound of the overlap sweet-spot band.")
    parser.add_argument("--overlap-high", type=float, default=0.95,
                        help="Upper bound of the overlap sweet-spot band.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3],
                        help="Candidate beta values (default: 0.1 0.3).")
    parser.add_argument("--quota-alpha", type=float, default=10.0,
                        help="Alpha used for quota computation (default: 10.0).")
    args = parser.parse_args()

    npz = np.load(args.conceptor_npz, allow_pickle=True)
    result = select_parameters(
        npz,
        overlap_low=args.overlap_low,
        overlap_high=args.overlap_high,
        candidate_betas=args.betas,
        quota_alpha=args.quota_alpha,
    )

    # Write JSON
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {args.output_json}")


if __name__ == "__main__":
    main()
