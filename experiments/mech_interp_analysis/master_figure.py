#!/usr/bin/env python3
"""
Master Figure: Combine all 5 analysis panels into one NeurIPS-ready figure.

Reads pre-computed JSON data from individual analyses and assembles
a single 1×5 panel figure at full NeurIPS text width.

Run this AFTER all individual analyses have completed.

Saves:
  - master_figure.pdf / .png
"""

import sys
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
from shared_utils import (
    apply_neurips_style, COLORS, TASK_PALETTE, ensure_output_dir,
    NUM_DENOISE_STEPS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


def build_master_figure():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    # Load all pre-computed data
    with open(out_dir / "analysis_1_data.json") as f:
        data_1 = json.load(f)
    with open(out_dir / "analysis_2_data.json") as f:
        data_2 = json.load(f)
    with open(out_dir / "analysis_3_data.json") as f:
        data_3 = json.load(f)
    with open(out_dir / "analysis_4_data.json") as f:
        data_4 = json.load(f)
    with open(out_dir / "analysis_5_data.json") as f:
        data_5 = json.load(f)

    # ── Create figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 9, hspace=0.55, wspace=0.9,
                          height_ratios=[1, 1],
                          width_ratios=[1, 1, 0.3, 1, 1, 0.3, 1, 1, 0.1])
    # Row 1: A(0:2), B(3:5), C(6:8) with gap columns
    axes = [
        fig.add_subplot(gs[0, 0:2]),  # Panel A
        fig.add_subplot(gs[0, 3:5]),  # Panel B
        fig.add_subplot(gs[0, 6:9]),  # Panel C
        fig.add_subplot(gs[1, 0:4]),  # Panel D (wider)
        fig.add_subplot(gs[1, 5:9]),  # Panel E
    ]

    # ── Panel A: Eigenvalue spectra ──────────────────────────────────────────
    ax = axes[0]
    tasks_sorted = sorted(data_1.keys(), key=lambda t: data_1[t]["baseline_sr"])

    for idx, task in enumerate(tasks_sorted):
        color = TASK_PALETTE[idx % len(TASK_PALETTE)]
        eigs_pos = np.array(data_1[task]["eigs_pos"])
        eigs_neg = np.array(data_1[task]["eigs_neg"])
        dims = np.arange(1, len(eigs_pos) + 1)

        ax.plot(dims, np.clip(eigs_pos, 1e-6, None), color=color,
                alpha=0.5, linewidth=0.6, linestyle='-')
        ax.plot(dims, np.clip(eigs_neg, 1e-6, None), color=color,
                alpha=0.3, linewidth=0.4, linestyle='--')

    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('$\\gamma_j$')
    ax.set_title('(A)', fontsize=8, fontweight='bold', loc='left', pad=4)
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-4, 1.05)
    ax.axhline(y=0.5, color='gray', alpha=0.2, linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], color=COLORS['dark'], linewidth=0.8, linestyle='-', label='$\\mathbf{C}^+$'),
        Line2D([0], [0], color=COLORS['dark'], linewidth=0.6, linestyle='--', label='$\\mathbf{C}^-$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=5,
              handlelength=1.2, handletextpad=0.3)

    # ── Panel B: Overlap vs performance gap ──────────────────────────────────
    ax = axes[1]
    tasks_b = sorted(data_2.keys())

    if tasks_b:
        overlaps = np.array([data_2[t]["overlap"] for t in tasks_b])
        gaps = np.array([data_2[t]["conceptor_minus_baseline"] for t in tasks_b])
        baseline_srs = np.array([data_2[t]["baseline_sr"] for t in tasks_b])

        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.cm.RdYlGn

        sc = ax.scatter(overlaps, gaps, c=baseline_srs, cmap=cmap, norm=norm,
                        s=20, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=3)

        if len(overlaps) >= 3:
            rho, p_val = stats.spearmanr(overlaps, gaps)
            slope, intercept = np.polyfit(overlaps, gaps, 1)
            x_fit = np.linspace(overlaps.min() - 0.02, overlaps.max() + 0.02, 50)
            ax.plot(x_fit, slope * x_fit + intercept, color='gray', linestyle='--',
                    linewidth=0.6, alpha=0.6, zorder=2)
            ax.text(0.03, 0.97, f'$\\rho$={rho:.2f}\n$p$={p_val:.3f}',
                    transform=ax.transAxes, fontsize=5, verticalalignment='top',
                    color=COLORS['dark'])

        ax.axhline(y=0, color='gray', alpha=0.2, linewidth=0.5)


    ax.set_xlabel('Overlap')
    ax.set_ylabel('$\\Delta$SR (C $-$ baseline)')
    ax.set_title('(B)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # ── Panel C: Projection violin ───────────────────────────────────────────
    ax = axes[2]
    TOP_K = 20

    # Aggregate projection fractions across all tasks
    all_frac_s, all_frac_f = [], []
    for task, td in data_3.items():
        all_frac_s.append(td["energy_frac_success_mean"])
        all_frac_f.append(td["energy_frac_failure_mean"])

    if all_frac_s:
        all_frac_s = np.array(all_frac_s)
        all_frac_f = np.array(all_frac_f)

        data_violin = [all_frac_s, all_frac_f]
        positions = [1, 2]

        parts = ax.violinplot(data_violin, positions=positions, showmeans=False,
                              showmedians=False, showextrema=False)
        for idx_v, pc in enumerate(parts['bodies']):
            color = COLORS['teal'] if idx_v == 0 else COLORS['coral']
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('white')
            pc.set_linewidth(0.5)

        bp = ax.boxplot(data_violin, positions=positions, widths=0.2,
                        showfliers=False, patch_artist=True, zorder=3)
        box_colors = [COLORS['teal'], COLORS['coral']]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor(COLORS['dark'])
            patch.set_linewidth(0.4)
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color(COLORS['dark'])
                line.set_linewidth(0.4)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(0.8)

        ax.axhline(y=TOP_K / 1024.0, color='gray', alpha=0.3, linewidth=0.4,
                   linestyle='--')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Succ.', 'Fail.'], fontsize=6)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylabel(f'Energy in top-{TOP_K}\n$\\mathbf{{C}}_{{\\mathrm{{steer}}}}$ eigenvecs')
    ax.set_title('(C)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # ── Panel D: Quota heatmap ───────────────────────────────────────────────
    ax = axes[3]

    tasks_sorted_d = sorted(data_4.keys(), key=lambda t: data_4[t]["baseline_sr"])
    if tasks_sorted_d:
        quota_matrix = np.array([data_4[t]["quotas"] for t in tasks_sorted_d])
        n_tasks_d = len(tasks_sorted_d)

        im = ax.imshow(quota_matrix, aspect='auto', cmap='cividis',
                       interpolation='nearest')

        ax.set_xticks(range(NUM_DENOISE_STEPS))
        ax.set_xticklabels(range(NUM_DENOISE_STEPS), fontsize=4)
        ax.set_xlabel('Denoise step $t$')

        # y-axis: show baseline SR instead of task names (more compact)
        tick_labels = [f'{data_4[t]["baseline_sr"]:.0%}' for t in tasks_sorted_d]
        ax.set_yticks(range(n_tasks_d))
        ax.set_yticklabels(tick_labels, fontsize=3.5)
        ax.set_ylabel('Task (sorted by baseline SR)')

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=12, pad=0.03)
        cbar.set_label('$q(\\mathbf{C}^{(t)}_{\\mathrm{steer}})$', fontsize=5)
        cbar.ax.tick_params(labelsize=4)

        for edge in range(n_tasks_d + 1):
            ax.axhline(y=edge - 0.5, color='white', linewidth=0.2)
        for edge in range(NUM_DENOISE_STEPS + 1):
            ax.axvline(x=edge - 0.5, color='white', linewidth=0.2)

    ax.set_title('(D)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # ── Panel E: Cosine similarity of C_steer across denoising steps ─────────
    ax = axes[4]

    # Collect consecutive cosine similarities across all tasks
    all_cosines = []
    for task, td in data_4.items():
        cosines = td.get("consecutive_cosines", [])
        if cosines:
            all_cosines.append(cosines)

    if all_cosines:
        cosine_matrix = np.array(all_cosines)  # (n_tasks, 9)
        mean_cos = cosine_matrix.mean(axis=0)
        std_cos = cosine_matrix.std(axis=0)
        transitions = np.arange(len(mean_cos))
        labels = [f'{t}→{t+1}' for t in range(len(mean_cos))]

        ax.plot(transitions, mean_cos, color=COLORS['slate'], linewidth=1.5,
                marker='o', markersize=3, zorder=3)
        ax.fill_between(transitions, mean_cos - std_cos, np.minimum(mean_cos + std_cos, 1.0),
                        color=COLORS['slate'], alpha=0.15)

        ax.axhline(y=1.0, color='gray', alpha=0.2, linewidth=0.5)
        ax.axhline(y=0.95, color='gray', alpha=0.15, linewidth=0.4, linestyle='--')

        ax.set_xticks(transitions)
        ax.set_xticklabels(labels, fontsize=5, rotation=45, ha='right')
        ax.set_ylim(0.90, 1.005)

    ax.set_xlabel('Transition $t \\to t{+}1$')
    ax.set_ylabel('cos($\\mathbf{C}^{(t)}_{\\mathrm{steer}}$, $\\mathbf{C}^{(t+1)}_{\\mathrm{steer}}$)')
    ax.set_title('(E)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.savefig(out_dir / "master_figure.pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_dir / "master_figure.png", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved master_figure.pdf/png")


if __name__ == "__main__":
    build_master_figure()
