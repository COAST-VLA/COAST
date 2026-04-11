#!/usr/bin/env python3
"""
Analysis 4: Denoising step dynamics reveal when steering matters most.

Panel D: Heatmap — denoising step (x, 0-9) vs task (y, sorted by difficulty),
         colored by quota of per-step contrastive conceptors.

Also: cosine similarity between C_steer^(t) and C_steer^(t+1),
      per-step linear probe accuracy for success prediction.

Saves:
  - panel_D_dynamics.pdf / .png
  - analysis_4_data.json
"""

import sys
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
from shared_utils import (
    apply_neurips_style, COLORS, MIXED_TASKS, NUM_DENOISE_STEPS,
    find_steerable_tasks, collect_outcome_activations,
    compute_conceptor, contrastive_conceptor, conceptor_quota,
    load_steering_csv, get_baseline_sr, ensure_output_dir,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ALPHA = 0.5
LAYER_IDX = 2


def cosine_sim_matrices(C_A, C_B):
    """Cosine similarity between two conceptor matrices (flattened)."""
    a = C_A.flatten()
    b = C_B.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def run_analysis_4():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("Finding steerable tasks...")
    steerable = find_steerable_tasks(MIXED_TASKS)

    results = {}
    for i, task in enumerate(MIXED_TASKS):
        if task not in steerable or not steerable[task]["has_failures"]:
            continue
        splits = steerable[task]

        logger.info(f"[{i+1}] {task}: computing per-step conceptors...")
        success_acts, failure_acts = collect_outcome_activations(
            task, splits, layer_idx=LAYER_IDX, mean_pool=True
        )

        step_quotas = []
        step_probe_accs = []
        step_conceptors = []
        step_cosines = []

        for t in range(NUM_DENOISE_STEPS):
            X_pos = success_acts[t]
            X_neg = failure_acts[t]

            if X_pos.shape[0] < 5 or X_neg.shape[0] < 5:
                step_quotas.append(0.0)
                step_probe_accs.append(0.5)
                step_conceptors.append(None)
                continue

            C_pos, _ = compute_conceptor(X_pos, alpha=ALPHA)
            C_neg, _ = compute_conceptor(X_neg, alpha=ALPHA)
            C_steer = contrastive_conceptor(C_pos, C_neg)
            step_conceptors.append(C_steer)

            q = conceptor_quota(C_steer)
            step_quotas.append(float(q))

            # Per-step linear probe for success prediction
            X_all = np.concatenate([X_pos, X_neg], axis=0)
            y_all = np.concatenate([np.ones(X_pos.shape[0]),
                                    np.zeros(X_neg.shape[0])])
            # Subsample if too large for speed
            max_n = 2000
            if len(y_all) > max_n:
                idx = np.random.choice(len(y_all), max_n, replace=False)
                X_all = X_all[idx]
                y_all = y_all[idx]

            try:
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                scores = cross_val_score(clf, X_all, y_all, cv=3, scoring='accuracy')
                step_probe_accs.append(float(np.mean(scores)))
            except Exception as e:
                logger.warning(f"  Probe failed for step {t}: {e}")
                step_probe_accs.append(0.5)

        # Cosine similarity between consecutive step conceptors
        for t in range(NUM_DENOISE_STEPS - 1):
            if step_conceptors[t] is not None and step_conceptors[t + 1] is not None:
                cs = cosine_sim_matrices(step_conceptors[t], step_conceptors[t + 1])
                step_cosines.append(float(cs))
            else:
                step_cosines.append(float('nan'))

        # Get baseline SR for sorting
        csv_rows = load_steering_csv(task)
        baseline_sr = get_baseline_sr(csv_rows) if csv_rows else len(splits["success"]) / 15.0

        results[task] = {
            "quotas": step_quotas,
            "probe_accs": step_probe_accs,
            "consecutive_cosines": step_cosines,
            "baseline_sr": float(baseline_sr),
        }
        logger.info(f"  quotas: [{', '.join(f'{q:.1f}' for q in step_quotas)}]")
        logger.info(f"  probe:  [{', '.join(f'{a:.3f}' for a in step_probe_accs)}]")

    # Save raw data
    with open(out_dir / "analysis_4_data.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis_4_data.json with {len(results)} tasks")

    if len(results) < 2:
        logger.warning("Not enough tasks for heatmap")
        return results

    # ── Panel D: Quota heatmap ───────────────────────────────────────────────
    # Sort tasks by baseline SR (difficulty)
    tasks_sorted = sorted(results.keys(), key=lambda t: results[t]["baseline_sr"])
    n_tasks = len(tasks_sorted)

    quota_matrix = np.array([results[t]["quotas"] for t in tasks_sorted])

    fig, ax = plt.subplots(figsize=(2.2, 1.6))

    # Use cividis for a fresh look
    im = ax.imshow(quota_matrix, aspect='auto', cmap='cividis',
                   interpolation='nearest')

    ax.set_xticks(range(NUM_DENOISE_STEPS))
    ax.set_xticklabels(range(NUM_DENOISE_STEPS), fontsize=5)
    ax.set_xlabel('Denoising step $t$')

    # Abbreviate task names for y-axis
    short_names = [t.replace('-v3', '').replace('-', '\n') if n_tasks <= 15
                   else '' for t in tasks_sorted]
    ax.set_yticks(range(n_tasks))
    if n_tasks <= 15:
        ax.set_yticklabels(short_names, fontsize=4)
    else:
        # Show every other task, or just show baseline SR
        tick_labels = []
        for j, t in enumerate(tasks_sorted):
            sr = results[t]["baseline_sr"]
            tick_labels.append(f'{sr:.0%}')
        ax.set_yticklabels(tick_labels, fontsize=4)
    ax.set_ylabel('Task (sorted by difficulty)')

    ax.set_title('(D)', fontsize=8, fontweight='bold', loc='left', pad=4)

    # Compact colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=15, pad=0.03)
    cbar.set_label('$q(\\mathbf{C}_{\\mathrm{steer}}^{(t)})$', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # White grid lines
    for edge in range(n_tasks + 1):
        ax.axhline(y=edge - 0.5, color='white', linewidth=0.3)
    for edge in range(NUM_DENOISE_STEPS + 1):
        ax.axvline(x=edge - 0.5, color='white', linewidth=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "panel_D_dynamics.pdf")
    fig.savefig(out_dir / "panel_D_dynamics.png")
    plt.close(fig)
    logger.info("Saved panel_D_dynamics.pdf/png")

    # ── Supplementary: Probe accuracy heatmap ────────────────────────────────
    probe_matrix = np.array([results[t]["probe_accs"] for t in tasks_sorted])

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(probe_matrix, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(NUM_DENOISE_STEPS))
    ax.set_xticklabels(range(NUM_DENOISE_STEPS))
    ax.set_xlabel('Denoising step $t$')
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([t.replace('-v3', '') for t in tasks_sorted], fontsize=6)
    ax.set_ylabel('Task')
    ax.set_title('Per-step success probe accuracy')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('3-fold CV accuracy')
    for edge in range(n_tasks + 1):
        ax.axhline(y=edge - 0.5, color='white', linewidth=0.3)
    for edge in range(NUM_DENOISE_STEPS + 1):
        ax.axvline(x=edge - 0.5, color='white', linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "analysis_4_probe_heatmap.pdf")
    fig.savefig(out_dir / "analysis_4_probe_heatmap.png")
    plt.close(fig)
    logger.info("Saved analysis_4_probe_heatmap.pdf/png")

    # ── Supplementary: Consecutive cosine similarity ─────────────────────────
    fig, ax = plt.subplots(figsize=(4, 2.5))
    cosine_matrix = np.array([results[t]["consecutive_cosines"] for t in tasks_sorted])
    # This is (n_tasks, 9)
    mean_cos = np.nanmean(cosine_matrix, axis=0)
    std_cos = np.nanstd(cosine_matrix, axis=0)
    steps = np.arange(NUM_DENOISE_STEPS - 1)
    ax.plot(steps, mean_cos, color=COLORS['slate'], linewidth=1.5, zorder=3)
    ax.fill_between(steps, mean_cos - std_cos, mean_cos + std_cos,
                    color=COLORS['slate'], alpha=0.15)
    ax.set_xlabel('Transition $t \\to t+1$')
    ax.set_ylabel('Cosine sim $\\cos(\\mathbf{C}_{\\mathrm{steer}}^{(t)}, \\mathbf{C}_{\\mathrm{steer}}^{(t+1)})$')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'{t}→{t+1}' for t in steps], fontsize=6)
    ax.axhline(y=1.0, color='gray', alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "analysis_4_cosine_similarity.pdf")
    fig.savefig(out_dir / "analysis_4_cosine_similarity.png")
    plt.close(fig)
    logger.info("Saved analysis_4_cosine_similarity.pdf/png")

    logger.info("Analysis 4 complete.")
    return results


if __name__ == "__main__":
    run_analysis_4()
