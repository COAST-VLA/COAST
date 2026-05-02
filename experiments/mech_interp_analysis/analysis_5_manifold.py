#!/usr/bin/env python3
"""
Analysis 5: Conceptor steering preserves on-manifold behavior.

Panel E: Line plot — PCA reconstruction error vs steering strength β,
         with separate lines for conceptor and linear steering (at matched norms).

Also: nearest-neighbor distance to success activation set.

Saves:
  - panel_E_manifold.pdf / .png
  - analysis_5_data.json
"""

import sys
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/nlpgpu/data/miaom/openpi-metaworld/experiments/mech_interp_analysis")
from shared_utils import (
    apply_neurips_style, COLORS, MIXED_TASKS, HIDDEN_DIM,
    find_steerable_tasks, collect_outcome_activations,
    compute_conceptor, contrastive_conceptor,
    ensure_output_dir,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ALPHA = 0.5
LAYER_IDX = 2
DENOISE_STEP = 0
PCA_COMPONENTS = 50  # fit PCA on success activations

# Steering strengths to sweep
BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def compute_pca_recon_error(X, pca_model):
    """Mean per-sample reconstruction error under a fitted PCA model."""
    X_proj = pca_model.inverse_transform(pca_model.transform(X))
    errors = np.mean((X - X_proj) ** 2, axis=1)  # per-sample MSE
    return float(np.mean(errors)), float(np.std(errors))


def compute_nn_distance(X_query, X_reference, k=5):
    """Mean distance from query points to k-nearest neighbors in reference set."""
    # Subsample reference if too large
    max_ref = 3000
    if X_reference.shape[0] > max_ref:
        idx = np.random.choice(X_reference.shape[0], max_ref, replace=False)
        X_reference = X_reference[idx]

    nn = NearestNeighbors(n_neighbors=min(k, X_reference.shape[0]), algorithm='auto')
    nn.fit(X_reference)
    distances, _ = nn.kneighbors(X_query)
    return float(np.mean(distances)), float(np.std(np.mean(distances, axis=1)))


def run_analysis_5():
    apply_neurips_style()
    out_dir = ensure_output_dir()

    logger.info("Finding steerable tasks...")
    steerable = find_steerable_tasks(MIXED_TASKS)

    # Aggregate across tasks
    all_conceptor_pca_errors = {beta: [] for beta in BETAS}
    all_linear_pca_errors = {beta: [] for beta in BETAS}
    all_conceptor_nn_dists = {beta: [] for beta in BETAS}
    all_linear_nn_dists = {beta: [] for beta in BETAS}
    all_conceptor_norms = {beta: [] for beta in BETAS}
    all_linear_norms = {beta: [] for beta in BETAS}

    per_task_results = {}

    for i, task in enumerate(MIXED_TASKS):
        if task not in steerable or not steerable[task]["has_failures"]:
            continue
        splits = steerable[task]

        logger.info(f"[{i+1}] {task}: computing manifold metrics...")
        success_acts, failure_acts = collect_outcome_activations(
            task, splits, layer_idx=LAYER_IDX, mean_pool=True
        )

        X_pos = success_acts[DENOISE_STEP]
        X_neg = failure_acts[DENOISE_STEP]

        if X_pos.shape[0] < 10 or X_neg.shape[0] < 10:
            logger.info(f"  Skipping {task}: too few samples")
            continue

        # Compute conceptor and linear steering vectors
        C_pos, _ = compute_conceptor(X_pos, alpha=ALPHA)
        C_neg, _ = compute_conceptor(X_neg, alpha=ALPHA)
        C_steer = contrastive_conceptor(C_pos, C_neg)

        mu_pos = np.mean(X_pos, axis=0)
        mu_neg = np.mean(X_neg, axis=0)
        steer_vec = mu_pos - mu_neg
        steer_dir = steer_vec / (np.linalg.norm(steer_vec) + 1e-10)

        # Fit PCA on success activations
        n_comp = min(PCA_COMPONENTS, X_pos.shape[0] - 1, HIDDEN_DIM)
        pca = PCA(n_components=n_comp)
        pca.fit(X_pos)

        task_results = {"betas": [], "conceptor_pca_error": [], "linear_pca_error": [],
                        "conceptor_nn_dist": [], "linear_nn_dist": [],
                        "conceptor_norm": [], "linear_norm": []}

        for beta in BETAS:
            # Conceptor steering: h' = h @ M^T where M = (1-β)I + βC
            I = np.eye(HIDDEN_DIM)
            M = (1 - beta) * I + beta * C_steer

            X_neg_conceptor = X_neg @ M.T
            conceptor_delta = X_neg_conceptor - X_neg
            conceptor_norm = float(np.mean(np.linalg.norm(conceptor_delta, axis=1)))

            # Linear steering at matched norm: find alpha_lin such that
            # ||alpha_lin * d|| ≈ conceptor_norm for a single sample
            if conceptor_norm > 1e-6:
                alpha_lin = conceptor_norm / (np.linalg.norm(steer_dir) + 1e-10)
            else:
                alpha_lin = 0.0

            X_neg_linear = X_neg + alpha_lin * steer_dir[np.newaxis, :]
            linear_delta = X_neg_linear - X_neg
            linear_norm = float(np.mean(np.linalg.norm(linear_delta, axis=1)))

            # PCA reconstruction errors
            pca_err_conceptor, _ = compute_pca_recon_error(X_neg_conceptor, pca)
            pca_err_linear, _ = compute_pca_recon_error(X_neg_linear, pca)

            # NN distances to success set
            nn_dist_conceptor, _ = compute_nn_distance(X_neg_conceptor, X_pos, k=5)
            nn_dist_linear, _ = compute_nn_distance(X_neg_linear, X_pos, k=5)

            task_results["betas"].append(beta)
            task_results["conceptor_pca_error"].append(pca_err_conceptor)
            task_results["linear_pca_error"].append(pca_err_linear)
            task_results["conceptor_nn_dist"].append(nn_dist_conceptor)
            task_results["linear_nn_dist"].append(nn_dist_linear)
            task_results["conceptor_norm"].append(conceptor_norm)
            task_results["linear_norm"].append(linear_norm)

            all_conceptor_pca_errors[beta].append(pca_err_conceptor)
            all_linear_pca_errors[beta].append(pca_err_linear)
            all_conceptor_nn_dists[beta].append(nn_dist_conceptor)
            all_linear_nn_dists[beta].append(nn_dist_linear)
            all_conceptor_norms[beta].append(conceptor_norm)
            all_linear_norms[beta].append(linear_norm)

        per_task_results[task] = task_results
        logger.info(f"  β=0.3: conceptor_PCA_err={task_results['conceptor_pca_error'][BETAS.index(0.3)]:.4f}, "
                     f"linear_PCA_err={task_results['linear_pca_error'][BETAS.index(0.3)]:.4f}")

    # Save raw data
    serializable_results = {
        "per_task": per_task_results,
        "aggregate": {
            "betas": BETAS,
            "conceptor_pca_mean": [float(np.mean(all_conceptor_pca_errors[b])) for b in BETAS],
            "conceptor_pca_std": [float(np.std(all_conceptor_pca_errors[b])) for b in BETAS],
            "linear_pca_mean": [float(np.mean(all_linear_pca_errors[b])) for b in BETAS],
            "linear_pca_std": [float(np.std(all_linear_pca_errors[b])) for b in BETAS],
            "conceptor_nn_mean": [float(np.mean(all_conceptor_nn_dists[b])) for b in BETAS],
            "conceptor_nn_std": [float(np.std(all_conceptor_nn_dists[b])) for b in BETAS],
            "linear_nn_mean": [float(np.mean(all_linear_nn_dists[b])) for b in BETAS],
            "linear_nn_std": [float(np.std(all_linear_nn_dists[b])) for b in BETAS],
            "conceptor_norm_mean": [float(np.mean(all_conceptor_norms[b])) for b in BETAS],
            "linear_norm_mean": [float(np.mean(all_linear_norms[b])) for b in BETAS],
        }
    }
    with open(out_dir / "analysis_5_data.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved analysis_5_data.json with {len(per_task_results)} tasks")

    if not per_task_results:
        logger.warning("No task data collected")
        return

    agg = serializable_results["aggregate"]

    # ── Panel E: PCA recon error vs β ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(2.2, 1.6))

    betas_arr = np.array(BETAS)
    c_mean = np.array(agg["conceptor_pca_mean"])
    c_std = np.array(agg["conceptor_pca_std"])
    l_mean = np.array(agg["linear_pca_mean"])
    l_std = np.array(agg["linear_pca_std"])

    ax.plot(betas_arr, c_mean, color=COLORS['teal'], linewidth=1.5,
            label='Conceptor', zorder=3)
    ax.fill_between(betas_arr, c_mean - c_std, c_mean + c_std,
                    color=COLORS['teal'], alpha=0.15)

    ax.plot(betas_arr, l_mean, color=COLORS['coral'], linewidth=1.5,
            label='Linear (matched $\\|\\delta\\|$)', zorder=3)
    ax.fill_between(betas_arr, l_mean - l_std, l_mean + l_std,
                    color=COLORS['coral'], alpha=0.15)

    ax.set_xlabel('Steering strength $\\beta$')
    ax.set_ylabel('PCA recon. error')
    ax.set_title('(E)', fontsize=8, fontweight='bold', loc='left', pad=4)
    ax.legend(fontsize=6, loc='upper left')

    # Reference line at β=0
    ax.axhline(y=c_mean[0], color='gray', alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "panel_E_manifold.pdf")
    fig.savefig(out_dir / "panel_E_manifold.png")
    plt.close(fig)
    logger.info("Saved panel_E_manifold.pdf/png")

    # ── Supplementary: NN distance vs β ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    nn_c_mean = np.array(agg["conceptor_nn_mean"])
    nn_c_std = np.array(agg["conceptor_nn_std"])
    nn_l_mean = np.array(agg["linear_nn_mean"])
    nn_l_std = np.array(agg["linear_nn_std"])

    ax.plot(betas_arr, nn_c_mean, color=COLORS['teal'], linewidth=1.5,
            label='Conceptor', zorder=3)
    ax.fill_between(betas_arr, nn_c_mean - nn_c_std, nn_c_mean + nn_c_std,
                    color=COLORS['teal'], alpha=0.15)

    ax.plot(betas_arr, nn_l_mean, color=COLORS['coral'], linewidth=1.5,
            label='Linear (matched norm)', zorder=3)
    ax.fill_between(betas_arr, nn_l_mean - nn_l_std, nn_l_mean + nn_l_std,
                    color=COLORS['coral'], alpha=0.15)

    ax.set_xlabel('Steering strength $\\beta$')
    ax.set_ylabel('Mean 5-NN distance to success set')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "analysis_5_nn_distance.pdf")
    fig.savefig(out_dir / "analysis_5_nn_distance.png")
    plt.close(fig)
    logger.info("Saved analysis_5_nn_distance.pdf/png")

    # ── Supplementary: Intervention norm comparison ──────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    c_norms = np.array(agg["conceptor_norm_mean"])
    l_norms = np.array(agg["linear_norm_mean"])
    ax.plot(betas_arr, c_norms, color=COLORS['teal'], linewidth=1.5,
            label='Conceptor $\\|\\delta\\|$', marker='o', markersize=3)
    ax.plot(betas_arr, l_norms, color=COLORS['coral'], linewidth=1.5,
            label='Linear $\\|\\delta\\|$ (matched)', marker='s', markersize=3)
    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('Mean intervention norm')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "analysis_5_norm_comparison.pdf")
    fig.savefig(out_dir / "analysis_5_norm_comparison.png")
    plt.close(fig)
    logger.info("Saved analysis_5_norm_comparison.pdf/png")

    logger.info("Analysis 5 complete.")
    return serializable_results


if __name__ == "__main__":
    run_analysis_5()
