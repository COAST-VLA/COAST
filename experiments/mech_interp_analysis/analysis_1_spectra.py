#!/usr/bin/env python3
"""
Analysis 1: Subspace geometry is low-rank and separable.

Panel A: Eigenvalue spectrum plot (log-scale y) showing C+ and C- curves per task.
Also: quota vs baseline success rate (task difficulty), effective rank comparison.

Saves:
  - panel_A_spectra.pdf / .png
  - analysis_1_quota_vs_difficulty.pdf / .png
  - analysis_1_data.json  (all computed quantities for master figure)
"""

import sys
import json
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
from shared_utils import (
    apply_neurips_style, COLORS, TASK_PALETTE, MIXED_TASKS,
    find_steerable_tasks, collect_outcome_activations,
    compute_conceptor, contrastive_conceptor, conceptor_quota, effective_rank,
    load_steering_csv, get_baseline_sr, ensure_output_dir,
    LAYER_MAP,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.5  # aperture for conceptor computation
LAYER_IDX = 2  # layer 11
DENOISE_STEP = 0  # representative step for spectra


def run_analysis_1():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("Finding steerable tasks...")
    steerable = find_steerable_tasks(MIXED_TASKS)
    logger.info(f"Found {len(steerable)} steerable tasks")

    # Collect per-task data
    results = {}
    for i, task in enumerate(MIXED_TASKS):
        if task not in steerable:
            logger.info(f"Skipping {task} (not steerable)")
            continue
        splits = steerable[task]
        if not splits["has_failures"]:
            logger.info(f"Skipping {task} (no failures for contrastive)")
            continue

        logger.info(f"[{i+1}/{len(MIXED_TASKS)}] Processing {task}...")
        success_acts, failure_acts = collect_outcome_activations(
            task, splits, layer_idx=LAYER_IDX, mean_pool=True
        )

        # Use denoise step 0 for representative spectra
        X_pos = success_acts[DENOISE_STEP]
        X_neg = failure_acts[DENOISE_STEP]

        C_pos, eigs_pos = compute_conceptor(X_pos, alpha=ALPHA)
        C_neg, eigs_neg = compute_conceptor(X_neg, alpha=ALPHA)
        C_steer = contrastive_conceptor(C_pos, C_neg)
        eigs_steer = np.linalg.eigvalsh(C_steer)[::-1]

        q_pos = conceptor_quota(C_pos)
        q_neg = conceptor_quota(C_neg)
        q_steer = conceptor_quota(C_steer)
        er_pos = effective_rank(np.clip(eigs_pos, 1e-12, None))
        er_neg = effective_rank(np.clip(eigs_neg, 1e-12, None))

        # Baseline success rate from steering results
        csv_rows = load_steering_csv(task)
        baseline_sr = get_baseline_sr(csv_rows) if csv_rows else splits["success"].__len__() / 15.0

        results[task] = {
            "eigs_pos": eigs_pos.tolist(),
            "eigs_neg": eigs_neg.tolist(),
            "eigs_steer": eigs_steer.tolist(),
            "quota_pos": float(q_pos),
            "quota_neg": float(q_neg),
            "quota_steer": float(q_steer),
            "effective_rank_pos": float(er_pos),
            "effective_rank_neg": float(er_neg),
            "baseline_sr": float(baseline_sr),
            "n_success_envs": len(splits["success"]),
            "n_failure_envs": len(splits["failure"]),
        }
        logger.info(f"  q+={q_pos:.1f}, q-={q_neg:.1f}, q_steer={q_steer:.1f}, "
                     f"ER+={er_pos:.1f}, baseline_SR={baseline_sr:.2f}")

    # Save raw data
    with open(out_dir / "analysis_1_data.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis_1_data.json with {len(results)} tasks")

    # ── Panel A: Eigenvalue spectra ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(2.2, 1.6))

    tasks_sorted = sorted(results.keys(), key=lambda t: results[t]["baseline_sr"])
    n_tasks = len(tasks_sorted)

    for idx, task in enumerate(tasks_sorted):
        color = TASK_PALETTE[idx % len(TASK_PALETTE)]
        eigs_pos = np.array(results[task]["eigs_pos"])
        eigs_neg = np.array(results[task]["eigs_neg"])
        dims = np.arange(1, len(eigs_pos) + 1)

        ax.plot(dims, np.clip(eigs_pos, 1e-6, None), color=color,
                alpha=0.5, linewidth=0.8, linestyle='-')
        ax.plot(dims, np.clip(eigs_neg, 1e-6, None), color=color,
                alpha=0.3, linewidth=0.6, linestyle='--')

    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue $\\gamma_j$')
    ax.set_title('(A)', fontsize=8, fontweight='bold', loc='left', pad=4)
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-4, 1.05)
    ax.axhline(y=0.5, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)

    # Simple legend: solid = C+, dashed = C-
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['dark'], linewidth=1.0, linestyle='-', label='$\\mathbf{C}^+$'),
        Line2D([0], [0], color=COLORS['dark'], linewidth=0.8, linestyle='--', label='$\\mathbf{C}^-$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

    fig.tight_layout()
    fig.savefig(out_dir / "panel_A_spectra.pdf")
    fig.savefig(out_dir / "panel_A_spectra.png")
    plt.close(fig)
    logger.info("Saved panel_A_spectra.pdf/png")

    # ── Supplementary: Quota vs Baseline SR ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.0, 2.2))

    baseline_srs = [results[t]["baseline_sr"] for t in tasks_sorted]
    quotas_pos = [results[t]["quota_pos"] for t in tasks_sorted]
    quotas_neg = [results[t]["quota_neg"] for t in tasks_sorted]
    quotas_steer = [results[t]["quota_steer"] for t in tasks_sorted]

    ax.scatter(baseline_srs, quotas_pos, c=COLORS['teal'], s=25, alpha=0.75,
               edgecolors='white', linewidth=0.5, label='$q(\\mathbf{C}^+)$', zorder=3)
    ax.scatter(baseline_srs, quotas_neg, c=COLORS['coral'], s=25, alpha=0.75,
               edgecolors='white', linewidth=0.5, label='$q(\\mathbf{C}^-)$', zorder=3)
    ax.scatter(baseline_srs, quotas_steer, c=COLORS['gold'], s=25, alpha=0.75,
               edgecolors='white', linewidth=0.5, marker='D',
               label='$q(\\mathbf{C}_{\\mathrm{steer}})$', zorder=3)

    ax.set_xlabel('Baseline success rate')
    ax.set_ylabel('Quota $q(\\mathbf{C})$')
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out_dir / "analysis_1_quota_vs_difficulty.pdf")
    fig.savefig(out_dir / "analysis_1_quota_vs_difficulty.png")
    plt.close(fig)
    logger.info("Saved analysis_1_quota_vs_difficulty.pdf/png")

    logger.info("Analysis 1 complete.")
    return results


if __name__ == "__main__":
    run_analysis_1()
