#!/usr/bin/env python3
"""
Analysis 2: Success and failure subspaces partially overlap but are not identical.

Panel B: Scatter plot — overlap (x) vs (conceptor SR − linear SR) (y), one dot per task.
         Color by baseline success rate. Spearman ρ annotation.

Also: overlap vs (conceptor SR − baseline SR) supplementary plot.

Saves:
  - panel_B_overlap.pdf / .png
  - analysis_2_data.json
"""

import sys
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
from shared_utils import (
    apply_neurips_style, COLORS, MIXED_TASKS,
    find_steerable_tasks, collect_outcome_activations,
    compute_conceptor, conceptor_overlap,
    load_steering_csv, get_baseline_sr, get_best_conceptor_sr, get_best_linear_sr,
    ensure_output_dir,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ALPHA = 0.5
LAYER_IDX = 2
DENOISE_STEP = 0


def run_analysis_2():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("Finding steerable tasks...")
    steerable = find_steerable_tasks(MIXED_TASKS)

    results = {}
    for i, task in enumerate(MIXED_TASKS):
        if task not in steerable or not steerable[task]["has_failures"]:
            continue

        splits = steerable[task]

        # Load steering results
        csv_rows = load_steering_csv(task)
        if csv_rows is None:
            logger.info(f"Skipping {task} (no steering results CSV)")
            continue

        baseline_sr = get_baseline_sr(csv_rows)
        best_conceptor_sr = get_best_conceptor_sr(csv_rows, strategy="strategy3")
        best_perstep_sr = get_best_conceptor_sr(csv_rows, strategy="strategy5")
        best_linear_sr = get_best_linear_sr(csv_rows)

        if baseline_sr is None or best_conceptor_sr is None or best_linear_sr is None:
            logger.info(f"Skipping {task} (missing SR data)")
            continue

        # Use best of global/per-step
        best_any_conceptor = max(best_conceptor_sr, best_perstep_sr or -1)

        logger.info(f"[{i+1}] {task}: loading activations for overlap...")
        success_acts, failure_acts = collect_outcome_activations(
            task, splits, layer_idx=LAYER_IDX, mean_pool=True
        )

        X_pos = success_acts[DENOISE_STEP]
        X_neg = failure_acts[DENOISE_STEP]

        C_pos, _ = compute_conceptor(X_pos, alpha=ALPHA)
        C_neg, _ = compute_conceptor(X_neg, alpha=ALPHA)

        overlap = conceptor_overlap(C_pos, C_neg)

        results[task] = {
            "overlap": float(overlap),
            "baseline_sr": float(baseline_sr),
            "best_conceptor_sr": float(best_any_conceptor),
            "best_linear_sr": float(best_linear_sr),
            "conceptor_minus_linear": float(best_any_conceptor - best_linear_sr),
            "conceptor_minus_baseline": float(best_any_conceptor - baseline_sr),
        }
        logger.info(f"  overlap={overlap:.3f}, C_SR={best_any_conceptor:.3f}, "
                     f"L_SR={best_linear_sr:.3f}, gap={best_any_conceptor - best_linear_sr:+.3f}")

    # Save raw data
    with open(out_dir / "analysis_2_data.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis_2_data.json with {len(results)} tasks")

    if len(results) < 3:
        logger.warning("Not enough tasks with complete data for scatter plot")
        return results

    # ── Panel B: Overlap vs (Conceptor SR - Linear SR) ───────────────────────
    tasks_list = sorted(results.keys())
    overlaps = np.array([results[t]["overlap"] for t in tasks_list])
    gaps = np.array([results[t]["conceptor_minus_linear"] for t in tasks_list])
    baseline_srs = np.array([results[t]["baseline_sr"] for t in tasks_list])

    fig, ax = plt.subplots(figsize=(2.2, 1.6))

    # Colormap by baseline SR
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.cm.RdYlGn

    sc = ax.scatter(overlaps, gaps, c=baseline_srs, cmap=cmap, norm=norm,
                    s=30, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=3)

    # Regression line
    if len(overlaps) >= 3:
        rho, p_val = stats.spearmanr(overlaps, gaps)
        slope, intercept = np.polyfit(overlaps, gaps, 1)
        x_fit = np.linspace(overlaps.min() - 0.02, overlaps.max() + 0.02, 50)
        ax.plot(x_fit, slope * x_fit + intercept, color='gray', linestyle='--',
                linewidth=0.8, alpha=0.6, zorder=2)
        ax.text(0.03, 0.97, f'$\\rho={rho:.2f}$\n$p={p_val:.3f}$',
                transform=ax.transAxes, fontsize=6, verticalalignment='top',
                color=COLORS['dark'])

    ax.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('Overlap $\\mathrm{tr}(\\mathbf{C}^+\\mathbf{C}^-) / \\mathrm{tr}(\\mathbf{C}^+)$')
    ax.set_ylabel('$\\Delta$SR (conceptor $-$ linear)')
    ax.set_title('(B)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # Compact colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=15, pad=0.02)
    cbar.set_label('Baseline SR', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(out_dir / "panel_B_overlap.pdf")
    fig.savefig(out_dir / "panel_B_overlap.png")
    plt.close(fig)
    logger.info("Saved panel_B_overlap.pdf/png")

    # ── Supplementary: Overlap vs (Conceptor SR - Baseline SR) ───────────────
    gaps_vs_base = np.array([results[t]["conceptor_minus_baseline"] for t in tasks_list])

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    sc = ax.scatter(overlaps, gaps_vs_base, c=baseline_srs, cmap=cmap, norm=norm,
                    s=30, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=3)

    if len(overlaps) >= 3:
        rho2, p2 = stats.spearmanr(overlaps, gaps_vs_base)
        slope2, intercept2 = np.polyfit(overlaps, gaps_vs_base, 1)
        x_fit = np.linspace(overlaps.min() - 0.02, overlaps.max() + 0.02, 50)
        ax.plot(x_fit, slope2 * x_fit + intercept2, color='gray', linestyle='--',
                linewidth=0.8, alpha=0.6, zorder=2)
        ax.text(0.03, 0.97, f'$\\rho={rho2:.2f}$, $p={p2:.3f}$',
                transform=ax.transAxes, fontsize=7, verticalalignment='top')

    ax.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('Overlap')
    ax.set_ylabel('$\\Delta$SR (conceptor $-$ baseline)')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Baseline SR', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "analysis_2_overlap_vs_baseline_gap.pdf")
    fig.savefig(out_dir / "analysis_2_overlap_vs_baseline_gap.png")
    plt.close(fig)
    logger.info("Saved analysis_2_overlap_vs_baseline_gap.pdf/png")

    logger.info("Analysis 2 complete.")
    return results


if __name__ == "__main__":
    run_analysis_2()
