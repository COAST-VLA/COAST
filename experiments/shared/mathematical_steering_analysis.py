#!/usr/bin/env python3
"""
Mathematical Analysis of Conceptor Steering Parameters
=======================================================

Given the steering formula:
    h' = h @ [(1-β)I + β·C_contrastive]^T

We can mathematically derive optimal (α, β) by analyzing how the
steering transformation moves failure activations toward the success
subspace, without running any rollouts.

Key metrics computed:
  1. Alignment shift:  tr(C_s · M · C_f · M^T) / tr(C_s · C_f)
     How much does steering increase failure→success projection?
  2. Distortion:       ||M - I||_F
     How much does steering perturb activations overall?
  3. Efficiency:       alignment_shift / distortion
     Useful signal per unit of perturbation
  4. Steered overlap:  sim(C_s, M·C_f·M^T)
     After steering, how similar are the subspaces?

Then we correlate these metrics with actual rollout success rates
to validate which metric best predicts performance.

Usage:
    python mathematical_steering_analysis.py
"""

import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = os.environ.get(
    "OPENPI_DATA_HOME",
    "/vast/projects/ungar/stellar/miaom/.cache/openpi"
)
LIBERO_NPZ = os.path.join(OPENPI_DATA_HOME, "libero_conceptors.npz")
LIBERO_RESULTS = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results"

METAWORLD_RESULTS = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_metaworld/steering_results/assembly-v3/results_assembly-v3.csv"

OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/shared/analysis_output"

LAYERS = [5, 11, 17]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
BETAS = [0.1, 0.3, 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math
# ──────────────────────────────────────────────────────────────────────────────

def quota(C):
    return float(np.trace(C)) / C.shape[0]


def overlap(Cs, Cf):
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    if ns * nf == 0:
        return 0.0
    return num / np.sqrt(ns * nf)


def contrastive_conceptor(Cs, Cf):
    d = Cs.shape[0]
    C_not_f = np.eye(d) - Cf
    inner = Cs + C_not_f - Cs @ C_not_f + 1e-8 * np.eye(d)
    return Cs @ np.linalg.inv(inner) @ C_not_f


def steering_matrix(Cc, beta):
    """M = (1-β)I + β·C_c"""
    d = Cc.shape[0]
    return (1 - beta) * np.eye(d) + beta * Cc


# ──────────────────────────────────────────────────────────────────────────────
# Mathematical steering metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_steering_metrics(Cs, Cf, Cc, beta):
    """
    Compute mathematical metrics for a given steering configuration.

    Args:
        Cs: success conceptor
        Cf: failure conceptor
        Cc: contrastive conceptor (C_s AND NOT C_f)
        beta: steering strength

    Returns: dict of metrics
    """
    d = Cs.shape[0]
    M = steering_matrix(Cc, beta)

    # Original overlap
    orig_overlap = overlap(Cs, Cf)

    # Steered failure conceptor: M @ Cf @ M^T (approximate)
    # This represents where failure activations land after steering
    Cf_steered = M @ Cf @ M.T

    # 1. Alignment shift: how much do steered failure acts project onto success subspace?
    #    tr(Cs · Cf_steered) vs tr(Cs · Cf)
    align_before = float(np.einsum("ij,ji->", Cs, Cf))
    align_after = float(np.einsum("ij,ji->", Cs, Cf_steered))
    alignment_shift = align_after - align_before

    # 2. Steered overlap: sim(Cs, M·Cf·M^T)
    steered_ovl = overlap(Cs, Cf_steered)

    # 3. Distortion: ||M - I||_F measures total perturbation
    distortion = float(np.linalg.norm(M - np.eye(d), "fro"))

    # 4. Efficiency: alignment gain per unit distortion
    efficiency = alignment_shift / (distortion + 1e-10)

    # 5. Contrastive energy: tr(Cc) — how much contrastive subspace exists
    contrastive_energy = float(np.trace(Cc))

    # 6. Projected steering magnitude: ||β · Cc||_F
    steering_magnitude = beta * float(np.linalg.norm(Cc, "fro"))

    # 7. Success preservation: tr(M @ Cs @ M^T) / tr(Cs)
    #    Does steering preserve the success subspace?
    Cs_steered = M @ Cs @ M.T
    success_preservation = float(np.trace(Cs_steered)) / (float(np.trace(Cs)) + 1e-10)

    # 8. Contrastive boost: how much does steering increase the
    #    "success-unique" energy in steered failure activations?
    #    tr(Cc @ Cf_steered) vs tr(Cc @ Cf)
    contrast_before = float(np.einsum("ij,ji->", Cc, Cf))
    contrast_after = float(np.einsum("ij,ji->", Cc, Cf_steered))
    contrastive_boost = contrast_after - contrast_before

    return {
        "orig_overlap": orig_overlap,
        "steered_overlap": steered_ovl,
        "overlap_delta": steered_ovl - orig_overlap,
        "alignment_shift": alignment_shift,
        "distortion": distortion,
        "efficiency": efficiency,
        "contrastive_energy": contrastive_energy,
        "steering_magnitude": steering_magnitude,
        "success_preservation": success_preservation,
        "contrastive_boost": contrastive_boost,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load LIBERO data
# ──────────────────────────────────────────────────────────────────────────────

def load_libero_data():
    """Load conceptors and sweep success rates for LIBERO."""
    print("Loading LIBERO conceptors...")
    npz = np.load(LIBERO_NPZ, allow_pickle=True)

    # Discover tasks
    tasks = set()
    for k in npz.files:
        m = re.match(r"^(.+?)__L\d+__", k)
        if m:
            tasks.add(m.group(1))
    tasks = sorted(tasks)
    print(f"  {len(tasks)} tasks")

    # Load success rates
    print("Loading LIBERO sweep results...")
    cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
    success_rates = {}  # (task, L, alpha, beta) → SR

    result_dirs = [d for d in os.listdir(LIBERO_RESULTS)
                   if os.path.isdir(os.path.join(LIBERO_RESULTS, d))]

    # Map short dir names back to full task names
    for rd in result_dirs:
        summary_path = os.path.join(LIBERO_RESULTS, rd, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)

        # Find matching task name
        task_name = None
        for t in tasks:
            if t.startswith(rd) or rd.startswith(t[:60]):
                task_name = t
                break
        if task_name is None:
            continue

        for entry in data["conditions"]:
            m = cond_re.match(entry["condition"])
            if m:
                L = int(m.group(1))
                a = float(m.group(2))
                b = float(m.group(3))
                success_rates[(task_name, L, a, b)] = float(entry["success_rate"])

    print(f"  {len(success_rates)} (task,L,α,β) conditions loaded")
    return npz, tasks, success_rates


def load_metaworld_data():
    """Load MetaWorld sweep results from CSV."""
    print("Loading MetaWorld results...")
    import csv
    results = {}
    cond_re = re.compile(r"^strategy(\d+)_a([\d.]+)_b([\d.]+)$")
    with open(METAWORLD_RESULTS) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = cond_re.match(row["condition"])
            if m and m.group(1) == "3":  # strategy3 = global
                a = float(m.group(2))
                b = float(m.group(3))
                results[(a, b)] = float(row["success_rate"])
    print(f"  {len(results)} global conditions loaded")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz, tasks, sr = load_libero_data()
    mw_sr = load_metaworld_data()

    # ── Compute metrics for all LIBERO (task, L, α, β) ───────────────────
    print("\nComputing mathematical steering metrics for LIBERO...")
    rows = []  # Each row: {task, L, alpha, beta, success_rate, ...metrics}

    for task in tasks:
        for L in LAYERS:
            for a in ALPHAS:
                # Load conceptors
                Cs_key = f"{task}__L{L}__{a}__C_success"
                Cf_key = f"{task}__L{L}__{a}__C_failure"
                Cc_key = f"{task}__L{L}__{a}__C_contrastive"
                if Cs_key not in npz or Cf_key not in npz or Cc_key not in npz:
                    continue

                Cs = npz[Cs_key]
                Cf = npz[Cf_key]
                Cc = npz[Cc_key]

                for b in BETAS:
                    key = (task, L, a, b)
                    if key not in sr:
                        continue

                    metrics = compute_steering_metrics(Cs, Cf, Cc, b)
                    row = {
                        "task": task,
                        "layer": L,
                        "alpha": a,
                        "beta": b,
                        "success_rate": sr[key],
                        **metrics,
                    }
                    rows.append(row)

    print(f"  {len(rows)} data points")

    # ── Correlation analysis ─────────────────────────────────────────────
    metric_names = [
        "orig_overlap", "steered_overlap", "overlap_delta",
        "alignment_shift", "distortion", "efficiency",
        "contrastive_energy", "steering_magnitude",
        "success_preservation", "contrastive_boost",
    ]

    sr_vec = np.array([r["success_rate"] for r in rows])

    print(f"\n{'='*70}")
    print(f"Correlation of mathematical metrics with rollout success rate")
    print(f"{'='*70}")
    print(f"{'Metric':<25s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman ρ':>10s} {'p-value':>10s}")
    print(f"{'-'*70}")

    correlations = {}
    for mn in metric_names:
        vals = np.array([r[mn] for r in rows])
        # Remove NaN/Inf
        mask = np.isfinite(vals) & np.isfinite(sr_vec)
        if mask.sum() < 10:
            continue
        pr, pp = stats.pearsonr(vals[mask], sr_vec[mask])
        sr_corr, sp = stats.spearmanr(vals[mask], sr_vec[mask])
        correlations[mn] = {"pearson_r": pr, "pearson_p": pp,
                           "spearman_rho": sr_corr, "spearman_p": sp}
        print(f"{mn:<25s} {pr:>10.3f} {pp:>10.2e} {sr_corr:>10.3f} {sp:>10.2e}")

    # ── Per-layer correlation ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Per-layer correlation (top metrics)")
    print(f"{'='*70}")
    best_metrics = sorted(correlations.keys(),
                         key=lambda m: abs(correlations[m]["spearman_rho"]),
                         reverse=True)[:5]

    for L in LAYERS:
        layer_rows = [r for r in rows if r["layer"] == L]
        if len(layer_rows) < 5:
            continue
        sr_l = np.array([r["success_rate"] for r in layer_rows])
        print(f"\n  Layer {L} ({len(layer_rows)} points):")
        for mn in best_metrics:
            vals = np.array([r[mn] for r in layer_rows])
            mask = np.isfinite(vals) & np.isfinite(sr_l)
            if mask.sum() < 5:
                continue
            rho, p = stats.spearmanr(vals[mask], sr_l[mask])
            print(f"    {mn:<25s}  ρ={rho:>7.3f}  p={p:.2e}")

    # ── Mathematical optimal vs actual optimal ───────────────────────────
    print(f"\n{'='*70}")
    print("Mathematical optimal vs actual optimal per task")
    print(f"{'='*70}")

    # Find the metric with highest |Spearman ρ|
    best_metric = best_metrics[0]
    print(f"Using best predictor: {best_metric} "
          f"(ρ={correlations[best_metric]['spearman_rho']:.3f})")

    # Higher or lower is better?
    sign = 1 if correlations[best_metric]["spearman_rho"] > 0 else -1

    n_match_layer = 0
    n_match_alpha = 0
    n_match_beta = 0
    n_match_exact = 0
    n_tasks = 0

    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        n_tasks += 1

        # Actual best
        actual_best = max(task_rows, key=lambda r: r["success_rate"])
        # Math best
        math_best = max(task_rows, key=lambda r: sign * r[best_metric])

        match_L = actual_best["layer"] == math_best["layer"]
        match_a = actual_best["alpha"] == math_best["alpha"]
        match_b = actual_best["beta"] == math_best["beta"]
        match_all = match_L and match_a and match_b

        n_match_layer += match_L
        n_match_alpha += match_a
        n_match_beta += match_b
        n_match_exact += match_all

        task_short = task[:50]
        print(f"\n  {task_short}...")
        print(f"    Actual best:  L={actual_best['layer']:>2d}  α={actual_best['alpha']:<5g}  "
              f"β={actual_best['beta']:<4g}  SR={actual_best['success_rate']:.3f}")
        print(f"    Math best:    L={math_best['layer']:>2d}  α={math_best['alpha']:<5g}  "
              f"β={math_best['beta']:<4g}  SR={math_best['success_rate']:.3f}  "
              f"{best_metric}={sign*math_best[best_metric]:.4f}")
        print(f"    Match: layer={'✓' if match_L else '✗'}  "
              f"alpha={'✓' if match_a else '✗'}  "
              f"beta={'✓' if match_b else '✗'}")

    print(f"\n  Summary ({n_tasks} tasks):")
    print(f"    Layer match:  {n_match_layer}/{n_tasks} ({100*n_match_layer/n_tasks:.0f}%)")
    print(f"    Alpha match:  {n_match_alpha}/{n_tasks} ({100*n_match_alpha/n_tasks:.0f}%)")
    print(f"    Beta match:   {n_match_beta}/{n_tasks} ({100*n_match_beta/n_tasks:.0f}%)")
    print(f"    Exact match:  {n_match_exact}/{n_tasks} ({100*n_match_exact/n_tasks:.0f}%)")

    # ── Also check: what SR does the math-best achieve? ──────────────────
    math_sr_vals = [max([r for r in rows if r["task"] == t],
                        key=lambda r: sign * r[best_metric])["success_rate"]
                    for t in tasks if any(r["task"] == t for r in rows)]
    actual_sr_vals = [max([r for r in rows if r["task"] == t],
                          key=lambda r: r["success_rate"])["success_rate"]
                      for t in tasks if any(r["task"] == t for r in rows)]

    print(f"\n    Mean SR (actual best):  {np.mean(actual_sr_vals):.3f}")
    print(f"    Mean SR (math best):    {np.mean(math_sr_vals):.3f}")
    print(f"    Gap:                    {np.mean(actual_sr_vals) - np.mean(math_sr_vals):.3f}")

    # ── Generate figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Top row: scatter plots of top 3 metrics vs success rate
    for idx, mn in enumerate(best_metrics[:3]):
        ax = axes[0, idx]
        vals = np.array([r[mn] for r in rows])
        colors = np.array([{5: "#1f77b4", 11: "#2ca02c", 17: "#ff7f0e"}[r["layer"]] for r in rows])

        for L, c in [(5, "#1f77b4"), (11, "#2ca02c"), (17, "#ff7f0e")]:
            mask = np.array([r["layer"] == L for r in rows])
            ax.scatter(vals[mask], sr_vec[mask], c=c, alpha=0.4, s=15, label=f"L={L}")

        rho = correlations[mn]["spearman_rho"]
        ax.set_xlabel(mn.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Success Rate", fontsize=10)
        ax.set_title(f"ρ = {rho:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)

    # Bottom-left: per-task comparison (actual vs math-predicted SR)
    ax = axes[1, 0]
    ax.bar(np.arange(len(actual_sr_vals)) - 0.15, actual_sr_vals, 0.3,
           color="#2ca02c", alpha=0.8, label="Actual best")
    ax.bar(np.arange(len(math_sr_vals)) + 0.15, math_sr_vals, 0.3,
           color="#ff7f0e", alpha=0.8, label=f"Math best ({best_metric})")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"T{i}" for i in range(len(tasks))], fontsize=8)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title("Per-Task: Actual vs Math-Optimal SR", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # Bottom-middle: distortion vs success rate, colored by beta
    ax = axes[1, 1]
    for b, c, mk in [(0.1, "#1f77b4", "o"), (0.3, "#2ca02c", "s"), (0.5, "#ff7f0e", "^")]:
        mask = np.array([r["beta"] == b for r in rows])
        dist = np.array([r["distortion"] for r in rows])
        ax.scatter(dist[mask], sr_vec[mask], c=c, alpha=0.4, s=15, marker=mk, label=f"β={b}")
    ax.set_xlabel("Distortion ||M - I||", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title("Distortion vs Success", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # Bottom-right: efficiency vs success rate
    ax = axes[1, 2]
    eff_vals = np.array([r["efficiency"] for r in rows])
    for L, c in [(5, "#1f77b4"), (11, "#2ca02c"), (17, "#ff7f0e")]:
        mask = np.array([r["layer"] == L for r in rows])
        ax.scatter(eff_vals[mask], sr_vec[mask], c=c, alpha=0.4, s=15, label=f"L={L}")
    ax.set_xlabel("Efficiency (alignment / distortion)", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title("Steering Efficiency vs Success", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Mathematical Steering Analysis — LIBERO",
                 fontsize=14, fontweight="bold")

    out_pdf = os.path.join(OUTPUT_DIR, "math_steering_analysis.pdf")
    out_png = os.path.join(OUTPUT_DIR, "math_steering_analysis.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")

    # ── Save full results ────────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "math_steering_analysis.json")
    summary = {
        "correlations": correlations,
        "best_metric": best_metric,
        "best_metric_rho": correlations[best_metric]["spearman_rho"],
        "mean_sr_actual_best": float(np.mean(actual_sr_vals)),
        "mean_sr_math_best": float(np.mean(math_sr_vals)),
        "n_tasks": n_tasks,
        "layer_match_rate": n_match_layer / n_tasks,
        "alpha_match_rate": n_match_alpha / n_tasks,
        "beta_match_rate": n_match_beta / n_tasks,
        "exact_match_rate": n_match_exact / n_tasks,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
