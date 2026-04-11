#!/usr/bin/env python3
"""
Analysis 3: The contrastive subspace aligns with action-relevant dimensions.

Panel C: Paired violin/box plot — distribution of projection magnitude onto top-k
         C_steer eigenvectors for success vs. failure activations.

Also: conceptor-projected linear probe accuracy vs full-space probe accuracy.

Saves:
  - panel_C_projection.pdf / .png
  - analysis_3_data.json
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
    compute_conceptor, contrastive_conceptor, conceptor_quota,
    ensure_output_dir,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.5
LAYER_IDX = 2
DENOISE_STEP = 0
TOP_K = 20  # number of top eigenvectors to project onto


def run_analysis_3():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("Finding steerable tasks...")
    steerable = find_steerable_tasks(MIXED_TASKS)

    # Collect projection magnitudes across all tasks
    all_proj_success = []
    all_proj_failure = []
    per_task_results = {}

    for i, task in enumerate(MIXED_TASKS):
        if task not in steerable or not steerable[task]["has_failures"]:
            continue
        splits = steerable[task]

        logger.info(f"[{i+1}] {task}: computing contrastive conceptor and projections...")
        success_acts, failure_acts = collect_outcome_activations(
            task, splits, layer_idx=LAYER_IDX, mean_pool=True
        )

        X_pos = success_acts[DENOISE_STEP]
        X_neg = failure_acts[DENOISE_STEP]

        if X_pos.shape[0] < 5 or X_neg.shape[0] < 5:
            logger.info(f"  Skipping {task}: too few samples (pos={X_pos.shape[0]}, neg={X_neg.shape[0]})")
            continue

        C_pos, _ = compute_conceptor(X_pos, alpha=ALPHA)
        C_neg, _ = compute_conceptor(X_neg, alpha=ALPHA)
        C_steer = contrastive_conceptor(C_pos, C_neg)

        # Get top-k eigenvectors of C_steer
        eigvals, eigvecs = np.linalg.eigh(C_steer)
        # eigh returns ascending order; reverse for descending
        idx_desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_desc]
        eigvecs = eigvecs[:, idx_desc]

        V_k = eigvecs[:, :TOP_K]  # (1024, k)

        # Project activations onto top-k subspace
        proj_pos = X_pos @ V_k  # (N_pos, k)
        proj_neg = X_neg @ V_k  # (N_neg, k)

        # Compute per-sample projection energy (squared L2 norm in subspace)
        energy_pos = np.sum(proj_pos ** 2, axis=1)  # (N_pos,)
        energy_neg = np.sum(proj_neg ** 2, axis=1)  # (N_neg,)

        # Also compute total activation energy for normalization
        total_energy_pos = np.sum(X_pos ** 2, axis=1)
        total_energy_neg = np.sum(X_neg ** 2, axis=1)

        # Fraction of energy in contrastive subspace
        frac_pos = energy_pos / (total_energy_pos + 1e-10)
        frac_neg = energy_neg / (total_energy_neg + 1e-10)

        all_proj_success.extend(frac_pos.tolist())
        all_proj_failure.extend(frac_neg.tolist())

        per_task_results[task] = {
            "energy_frac_success_mean": float(np.mean(frac_pos)),
            "energy_frac_success_std": float(np.std(frac_pos)),
            "energy_frac_failure_mean": float(np.mean(frac_neg)),
            "energy_frac_failure_std": float(np.std(frac_neg)),
            "energy_success_mean": float(np.mean(energy_pos)),
            "energy_failure_mean": float(np.mean(energy_neg)),
            "quota_steer": float(conceptor_quota(C_steer)),
            "top_k_eigenvalues": eigvals[:TOP_K].tolist(),
            "n_success": int(X_pos.shape[0]),
            "n_failure": int(X_neg.shape[0]),
        }
        logger.info(f"  frac_pos={np.mean(frac_pos):.4f}, frac_neg={np.mean(frac_neg):.4f}, "
                     f"ratio={np.mean(frac_pos)/max(np.mean(frac_neg), 1e-10):.2f}x")

    # Save raw data
    with open(out_dir / "analysis_3_data.json", "w") as f:
        json.dump(per_task_results, f, indent=2)
    logger.info(f"Saved analysis_3_data.json with {len(per_task_results)} tasks")

    if not all_proj_success:
        logger.warning("No projection data collected")
        return per_task_results

    # ── Panel C: Paired violin plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(1.8, 1.6))

    # Subsample for plotting if too many points
    max_points = 5000
    proj_s = np.array(all_proj_success)
    proj_f = np.array(all_proj_failure)
    if len(proj_s) > max_points:
        proj_s = np.random.choice(proj_s, max_points, replace=False)
    if len(proj_f) > max_points:
        proj_f = np.random.choice(proj_f, max_points, replace=False)

    data = [proj_s, proj_f]
    positions = [1, 2]

    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)

    for idx, pc in enumerate(parts['bodies']):
        color = COLORS['teal'] if idx == 0 else COLORS['coral']
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.5)

    # Add box plots inside
    bp = ax.boxplot(data, positions=positions, widths=0.15, showfliers=False,
                    patch_artist=True, zorder=3)
    box_colors = [COLORS['teal'], COLORS['coral']]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor(COLORS['dark'])
        patch.set_linewidth(0.5)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color(COLORS['dark'])
            line.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Success', 'Failure'], fontsize=7)
    ax.set_ylabel(f'Energy in top-{TOP_K}\n$\\mathbf{{C}}_{{\\mathrm{{steer}}}}$ eigenvectors')
    ax.set_title('(C)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # Add reference line
    ax.axhline(y=TOP_K / 1024.0, color='gray', alpha=0.2, linewidth=0.5,
               linestyle='--')
    ax.text(2.45, TOP_K / 1024.0, f'chance ({TOP_K}/1024)', fontsize=5,
            color='gray', va='center')

    fig.tight_layout()
    fig.savefig(out_dir / "panel_C_projection.pdf")
    fig.savefig(out_dir / "panel_C_projection.png")
    plt.close(fig)
    logger.info("Saved panel_C_projection.pdf/png")

    # ── Supplementary: Per-task energy fractions ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 2.7))
    tasks_sorted = sorted(per_task_results.keys(),
                          key=lambda t: per_task_results[t]["energy_frac_success_mean"]
                                      - per_task_results[t]["energy_frac_failure_mean"],
                          reverse=True)
    x = np.arange(len(tasks_sorted))
    means_s = [per_task_results[t]["energy_frac_success_mean"] for t in tasks_sorted]
    means_f = [per_task_results[t]["energy_frac_failure_mean"] for t in tasks_sorted]
    stds_s = [per_task_results[t]["energy_frac_success_std"] for t in tasks_sorted]
    stds_f = [per_task_results[t]["energy_frac_failure_std"] for t in tasks_sorted]

    ax.bar(x - 0.2, means_s, 0.35, yerr=stds_s, color=COLORS['teal'],
           alpha=0.8, label='Success', capsize=2, error_kw={'linewidth': 0.5})
    ax.bar(x + 0.2, means_f, 0.35, yerr=stds_f, color=COLORS['coral'],
           alpha=0.8, label='Failure', capsize=2, error_kw={'linewidth': 0.5})

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-v3', '') for t in tasks_sorted],
                       rotation=25, ha='right', fontsize=6)
    ax.set_ylabel(f'Energy fraction in top-{TOP_K} $\\mathbf{{C}}_{{\\mathrm{{steer}}}}$')
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "analysis_3_per_task_energy.pdf")
    fig.savefig(out_dir / "analysis_3_per_task_energy.png")
    plt.close(fig)
    logger.info("Saved analysis_3_per_task_energy.pdf/png")

    logger.info("Analysis 3 complete.")
    return per_task_results


if __name__ == "__main__":
    run_analysis_3()
