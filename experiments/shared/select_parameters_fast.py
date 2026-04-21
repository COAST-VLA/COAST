#!/usr/bin/env python3
"""
Parameter Selection for pi0-fast Conceptor Steering
====================================================

Reads a pre-computed conceptor .npz (pi0-fast format: no layer axis),
computes quota and overlap, and outputs a narrowed (alphas, betas) parameter
set for the downstream steering sweep.

pi0-fast key format:
    {task}__{strategy}__{alpha}__C_success
    {task}__{strategy}__{alpha}__C_failure
    {task}__{strategy}__{alpha}__C_contrastive

Three steering modes:
  1. **global** (contrastive): C_success AND NOT C_failure, applied uniformly.
     Alpha selected via overlap sweet-spot band.
  2. **per_step_combined** (contrastive): first/mid/last token-position conceptors
     applied by autoregressive position (first third / mid third / last third).
     Alpha selected via overlap intersection across per_token positions.
  3. **positive_only** (C_success, global): no failure data needed.
     Alpha/beta chosen from historical pi0.5 results (0.5, 1.0, 2.0 / 0.1, 0.2, 0.3).

Baseline success rate is read from activation metadata (no GPU re-run needed).

Usage:
    python select_parameters_fast.py \
        --conceptor-npz /path/to/conceptors.npz \
        --output-json   /path/to/selected_params.json \
        --activations-dir /path/to/activations/{ckpt_step}/ \
        [--overlap-low 0.85] [--overlap-high 0.95]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_quota(C):
    return float(np.trace(C)) / C.shape[0]


def compute_overlap(Cs, Cf):
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    if ns * nf == 0:
        return 0.0
    return num / np.sqrt(ns * nf)


# ──────────────────────────────────────────────────────────────────────────────
# Parse the .npz to discover tasks, strategies, alphas
# ──────────────────────────────────────────────────────────────────────────────

KEY_RE = re.compile(r"^(.+?)__(.+?)__([0-9.]+)__(C_.+)$")


def parse_npz_structure(npz):
    tasks, strategies, alphas = set(), set(), set()
    for key in npz.files:
        m = KEY_RE.match(key)
        if not m:
            continue
        tasks.add(m.group(1))
        strategies.add(m.group(2))
        try:
            alphas.add(float(m.group(3)))
        except ValueError:
            pass
    return sorted(tasks), sorted(strategies), sorted(alphas)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline from activation metadata
# ──────────────────────────────────────────────────────────────────────────────

def compute_baseline_from_activations(activations_dir: Path):
    """Read episode metadata to compute per-task baseline success rates."""
    baselines = {}
    if not activations_dir.exists():
        return baselines
    for task_dir in sorted(d for d in activations_dir.iterdir() if d.is_dir()):
        n_success, n_total = 0, 0
        for ep_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
            meta_path = ep_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            n_total += 1
            if meta.get("episode_success", False):
                n_success += 1
        if n_total > 0:
            baselines[task_dir.name] = {
                "success_rate": n_success / n_total,
                "n_success": n_success,
                "n_total": n_total,
            }
    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# Overlap-based alpha selection helper
# ──────────────────────────────────────────────────────────────────────────────

def select_alphas_by_overlap(npz, tasks, strategy, alphas, overlap_low, overlap_high):
    alpha_overlaps = {}
    for a in alphas:
        per_task = []
        for t in tasks:
            Cs_key = f"{t}__{strategy}__{a}__C_success"
            Cf_key = f"{t}__{strategy}__{a}__C_failure"
            if Cs_key in npz and Cf_key in npz:
                per_task.append(compute_overlap(npz[Cs_key], npz[Cf_key]))
        if per_task:
            alpha_overlaps[a] = float(np.mean(per_task))

    selected = []
    for a in alphas:
        ov = alpha_overlaps.get(a, float("nan"))
        in_band = overlap_low <= ov <= overlap_high
        marker = " ◀ selected" if in_band else ""
        print(f"    α={a:<5g}  overlap={ov:.3f}{marker}")
        if in_band:
            selected.append(a)

    if not selected:
        band_center = (overlap_low + overlap_high) / 2
        closest = min(alphas, key=lambda a: abs(alpha_overlaps.get(a, 999) - band_center))
        selected = [closest]
        print(f"    ⚠ No alpha in band — falling back to closest: α={closest}")

    return selected, alpha_overlaps


# ──────────────────────────────────────────────────────────────────────────────
# Main selection logic
# ──────────────────────────────────────────────────────────────────────────────

BETAS = [0.1, 0.2, 0.3]

POSITIVE_ONLY_ALPHAS = [0.5, 1.0, 2.0]

PER_STEP_COMBINED_CANDIDATES = [0.1, 0.5, 1.0]


def select_parameters(npz, overlap_low=0.85, overlap_high=0.95,
                      activations_dir=None):
    tasks, strategies, alphas = parse_npz_structure(npz)
    print(f"Found {len(tasks)} tasks, strategies={strategies}, alphas={alphas}")

    result = {"selected_betas": BETAS, "overlap_band": [overlap_low, overlap_high]}

    # ── Baseline from activation data ────────────────────────────────────
    if activations_dir:
        baselines = compute_baseline_from_activations(Path(activations_dir))
        print(f"\nBaseline (from activation metadata, {len(baselines)} tasks):")
        for t in sorted(baselines):
            b = baselines[t]
            print(f"    {t:70s}  SR={b['success_rate']:.3f}  ({b['n_success']}/{b['n_total']})")
        result["baselines"] = baselines
    else:
        print("\n  No activations dir provided — baseline will be computed at runtime.")
        result["baselines"] = {}

    # ── 1. Global (contrastive) ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Strategy: global (contrastive)")
    print(f"{'='*60}")
    global_alphas, global_overlaps = select_alphas_by_overlap(
        npz, tasks, "global", alphas, overlap_low, overlap_high,
    )
    result["global"] = {
        "selected_alphas": global_alphas,
        "alpha_overlaps": {str(a): v for a, v in global_overlaps.items()},
    }

    # ── 2. Per-step combined (contrastive) ───────────────────────────────
    print(f"\n{'='*60}")
    print("Strategy: per_step_combined (contrastive)")
    print(f"{'='*60}")
    print(f"  Candidate alphas: {PER_STEP_COMBINED_CANDIDATES}")
    print(f"  (Intersection of sweet-spot across first/mid/last positions)")
    per_step_overlaps = {}
    for pos in ["per_token_first", "per_token_mid", "per_token_last"]:
        print(f"\n  {pos}:")
        for a in PER_STEP_COMBINED_CANDIDATES:
            per_task = []
            for t in tasks:
                Cs_key = f"{t}__{pos}__{a}__C_success"
                Cf_key = f"{t}__{pos}__{a}__C_failure"
                if Cs_key in npz and Cf_key in npz:
                    per_task.append(compute_overlap(npz[Cs_key], npz[Cf_key]))
            if per_task:
                ov = float(np.mean(per_task))
                per_step_overlaps.setdefault(a, {})[pos] = ov
                print(f"    α={a:<5g}  overlap={ov:.3f}")

    result["per_step_combined"] = {
        "selected_alphas": PER_STEP_COMBINED_CANDIDATES,
        "per_position_overlaps": {str(a): v for a, v in per_step_overlaps.items()},
    }

    # ── 3. Positive-only (C_success, global) ─────────────────────────��───
    print(f"\n{'='*60}")
    print("Strategy: positive_only (C_success, global)")
    print(f"{'='*60}")
    print(f"  Alphas from historical pi0.5 results: {POSITIVE_ONLY_ALPHAS}")
    print(f"  Quota (C_success) at each alpha:")
    for a in POSITIVE_ONLY_ALPHAS:
        quotas = []
        for t in tasks:
            key = f"{t}__global__{a}__C_success"
            if key in npz:
                quotas.append(compute_quota(npz[key]))
        if quotas:
            print(f"    α={a:<5g}  mean_quota={np.mean(quotas):.4f}  (n={len(quotas)} tasks)")

    result["positive_only"] = {
        "selected_alphas": POSITIVE_ONLY_ALPHAS,
    }

    # ── Summary ──────────────────────────────────────────────────────────
    n_global = len(global_alphas) * len(BETAS)
    n_combined = len(PER_STEP_COMBINED_CANDIDATES) * len(BETAS)
    n_pos = len(POSITIVE_ONLY_ALPHAS) * len(BETAS)
    total = n_global + n_combined + n_pos + 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  global (contrastive):      alphas={global_alphas}, betas={BETAS}  ({n_global} conditions)")
    print(f"  per_step_combined (contr): alphas={PER_STEP_COMBINED_CANDIDATES}, betas={BETAS}  ({n_combined} conditions)")
    print(f"  positive_only (C_success): alphas={POSITIVE_ONLY_ALPHAS}, betas={BETAS}  ({n_pos} conditions)")
    print(f"  + 1 baseline (from activation data)")
    print(f"  TOTAL: {total} conditions/task")

    result["diagnostics"] = {
        "tasks": tasks,
        "all_strategies": strategies,
        "all_alphas": alphas,
        "conditions_per_task": total,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Select steering parameters from pre-computed pi0-fast conceptors."
    )
    parser.add_argument("--conceptor-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--activations-dir", default=None,
                        help="Path to activations/{ckpt_step}/ for baseline SR computation.")
    parser.add_argument("--overlap-low", type=float, default=0.85)
    parser.add_argument("--overlap-high", type=float, default=0.95)
    args = parser.parse_args()

    npz = np.load(args.conceptor_npz, allow_pickle=True)
    result = select_parameters(
        npz,
        overlap_low=args.overlap_low,
        overlap_high=args.overlap_high,
        activations_dir=args.activations_dir,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {args.output_json}")


if __name__ == "__main__":
    main()
