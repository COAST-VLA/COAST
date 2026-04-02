#!/usr/bin/env python3
"""
Conceptor Diagnostic Analysis for VLA Activations
===================================================
Loads residual stream activations from the brandonyang/ml45-activations
HuggingFace dataset and runs four diagnostic checks:

  (a) Singular value spectra of per-task conceptors
  (b) Conceptor similarity / overlap matrix across tasks
  (c) Boolean operations (AND, NOT) between conceptors
  (d) Linear probe validation on conceptor subspaces

Tasks analysed (behaviorally distinct):
  - reach-v3          (simple, ~4 inference steps, usually succeeds)
  - button-press-v3   (contact-based, ~7 steps)
  - drawer-open-v3    (articulated manipulation, ~11 steps)
  - assembly-v3       (complex, usually fails, 30 steps)

Reference conceptor implementation:
  /nlpgpu/data/miaom/conceptor/src/e2e.py           — train_Conceptor()
  /nlpgpu/data/miaom/conceptor/analyze_conceptor_spectrum.py — spectral helpers
"""

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────

REPO = "brandonyang/ml45-activations"
CHECKPOINT = "5000"

TASKS = ["reach-v3", "button-press-v3", "drawer-open-v3", "assembly-v3"]

# Which layer index (within the 4 captured: [0, 5, 11, 17]) to analyse.
# Index 2 → layer 11 (second-to-last captured, mid-to-late).
LAYER_IDX_IN_FILE = 2   # 0-indexed into the 4-layer axis → layer 11

# Which denoising step to use (0 = noisiest / decision point, 9 = final refined)
DENOISE_STEPS = [0, 9]  # analyse both, primary plots use step 0

ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]

OUT_DIR = Path("experiments/conceptor_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colours
TASK_COLORS = {
    "reach-v3": "#1f77b4",
    "button-press-v3": "#ff7f0e",
    "drawer-open-v3": "#2ca02c",
    "assembly-v3": "#d62728",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ── Helpers (adapted from conceptor codebase) ───────────────────────────

def fast_svd(X, k=None):
    """SVD of mean-centred X.  Returns eigenvalues of R = X^T X / N."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc / np.sqrt(max(1, N)), full_matrices=False)
    sigma = s ** 2
    if k is not None:
        return sigma[:k], Vt[:k]
    return sigma, Vt


def conceptor_eigenvalues(sigma, alpha):
    """γ_j = σ_j / (σ_j + α⁻²)"""
    return sigma / (sigma + alpha ** -2)


def conceptor_quota(gamma):
    """q(C) = trace(C) = Σ γ_j"""
    return float(gamma.sum())


def conceptor_entropy(gamma):
    """H(C) = -Σ [γ log₂ γ + (1-γ) log₂(1-γ)]"""
    g = np.clip(gamma, 1e-12, 1 - 1e-12)
    return -float(np.sum(g * np.log2(g) + (1 - g) * np.log2(1 - g)))


def compute_conceptor_matrix(X, alpha):
    """Compute full d×d conceptor C = R (R + α⁻² I)⁻¹ from data matrix X (n, d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    R = (Xc.T @ Xc) / max(1, N)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + alpha ** (-2) * np.eye(d))
    return C, R


def conceptor_AND(C_A, C_B):
    """Soft AND: C_A AND C_B  (Jaeger 2014, Eq. 19).
    C_A AND C_B = (C_A^{-1} + C_B^{-1} - I)^{-1}
    Using regularised pseudo-inverse for numerical stability."""
    d = C_A.shape[0]
    I = np.eye(d)
    eps = 1e-6
    C_A_inv = np.linalg.inv(C_A + eps * I)
    C_B_inv = np.linalg.inv(C_B + eps * I)
    inner = C_A_inv + C_B_inv - I
    return np.linalg.inv(inner + eps * I)


def conceptor_NOT(C):
    """NOT C = I - C"""
    return np.eye(C.shape[0]) - C


def conceptor_overlap(C_A, C_B):
    """Fraction of C_A's subspace that overlaps with C_B:
       overlap = trace(C_A @ C_B) / trace(C_A)"""
    tr_A = np.trace(C_A)
    if tr_A < 1e-10:
        return 0.0
    return float(np.trace(C_A @ C_B) / tr_A)


# ── Data Loading ─────────────────────────────────────────────────────────

def list_episode_steps(task_name):
    """Return list of (env_id, step_number) pairs for a task in the small dataset."""
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=True))

    episodes = {}
    for f_obj in all_files:
        if not hasattr(f_obj, 'rfilename'):
            continue
        parts = f_obj.rfilename.split('/')
        if len(parts) >= 4 and parts[0] == CHECKPOINT and parts[1] == task_name:
            ep = parts[2]  # episode_000_env_XXX
            if parts[3].startswith('step_'):
                step = parts[3]
                if ep not in episodes:
                    episodes[ep] = set()
                episodes[ep].add(step)
    return episodes


def download_task_activations(task_name, denoise_step=0, layer_idx=LAYER_IDX_IN_FILE):
    """Download suffix_residual.npz for all envs/steps and return aggregated vectors.

    Returns:
        X: np.ndarray of shape (n_samples, 1024) — residual stream vectors
        metadata_list: list of step metadata dicts
    """
    print(f"  Loading {task_name}...")
    from huggingface_hub import HfApi
    api = HfApi()
    all_files = list(api.list_repo_tree(REPO, repo_type="dataset", recursive=True))

    # Find all step directories for this task
    step_paths = set()
    for f_obj in all_files:
        if not hasattr(f_obj, 'rfilename'):
            continue
        r = f_obj.rfilename
        if r.startswith(f"{CHECKPOINT}/{task_name}/") and r.endswith("suffix_residual.npz"):
            # e.g. 5000/reach-v3/episode_000_env_000/step_0000/suffix_residual.npz
            step_paths.add(r)

    vectors = []
    metas = []

    for sp in sorted(step_paths):
        # Download residual
        f = hf_hub_download(REPO, sp, repo_type="dataset")
        data = np.load(f)
        # shape: (10, 4, 32, 1024) — (denoise_steps, layers, action_tokens, hidden_dim)
        all_residual = data["all_suffix_residual"]

        # Extract specified denoising step and layer
        # Mean-pool over action tokens (32 tokens → 1 vector of dim 1024)
        residual = all_residual[denoise_step, layer_idx]  # (32, 1024)
        mean_vec = residual.mean(axis=0)  # (1024,)
        vectors.append(mean_vec)

        # Also keep the full 32-token matrix for richer analysis
        # We'll use mean-pooled for conceptors, full for probes

        # Try to load step metadata
        meta_path = sp.replace("suffix_residual.npz", "metadata.json")
        try:
            mf = hf_hub_download(REPO, meta_path, repo_type="dataset")
            with open(mf) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
        metas.append(meta)

    X = np.array(vectors, dtype=np.float32)
    print(f"    → {task_name}: {X.shape[0]} samples, dim={X.shape[1]}")
    return X, metas


def download_episode_metadata(task_name):
    """Download episode-level metadata for success/failure labels."""
    results = {}
    for env_id in range(2):  # small dataset has 2 envs
        path = f"{CHECKPOINT}/{task_name}/episode_000_env_{env_id:03d}/metadata.json"
        try:
            f = hf_hub_download(REPO, path, repo_type="dataset")
            with open(f) as fh:
                results[env_id] = json.load(fh)
        except Exception:
            pass
    return results


# ── Step 1 & 2: Load data and compute conceptors ────────────────────────

def run_analysis():
    print("=" * 70)
    print("CONCEPTOR DIAGNOSTIC ANALYSIS FOR VLA ACTIVATIONS")
    print("=" * 70)

    # Load activations for each denoising step
    task_data = {}      # {(task, denoise_step): (X, metas)}
    task_episode_meta = {}  # {task: {env_id: meta}}

    for task in TASKS:
        task_episode_meta[task] = download_episode_metadata(task)
        for ds in DENOISE_STEPS:
            X, metas = download_task_activations(task, denoise_step=ds)
            task_data[(task, ds)] = (X, metas)

    # ── Step 2: Compute conceptors for each task and alpha ──
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING CONCEPTORS")
    print("=" * 70)

    # Store results: {(task, denoise_step, alpha): (C, R, sigma, gamma)}
    conceptors = {}
    K_MAX = 100  # max eigenvalues to keep

    for task in TASKS:
        for ds in DENOISE_STEPS:
            X, _ = task_data[(task, ds)]
            sigma, Vt = fast_svd(X, k=min(K_MAX, X.shape[0], X.shape[1]))

            for alpha in ALPHAS:
                gamma = conceptor_eigenvalues(sigma, alpha)
                q = conceptor_quota(gamma)
                h = conceptor_entropy(gamma)

                # Also compute full conceptor matrix for boolean ops (only at default alpha)
                if alpha == 1.0:
                    C_full, R_full = compute_conceptor_matrix(X, alpha)
                else:
                    C_full, R_full = None, None

                conceptors[(task, ds, alpha)] = {
                    "sigma": sigma, "gamma": gamma, "Vt": Vt,
                    "quota": q, "entropy": h,
                    "C": C_full, "R": R_full,
                }
                if ds == 0:
                    print(f"  {task} | denoise={ds} | α={alpha:5.1f} | "
                          f"quota={q:6.1f} | entropy={h:6.1f} | "
                          f"n(γ>0.5)={int(np.sum(gamma > 0.5)):3d} | "
                          f"n(γ>0.9)={int(np.sum(gamma > 0.9)):3d}")

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (a): Singular Value Spectra
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (a): CONCEPTOR EIGENVALUE SPECTRA")
    print("=" * 70)

    for ds in DENOISE_STEPS:
        fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4 * len(ALPHAS), 3.5), sharey=True)
        fig.suptitle(f"Conceptor Eigenvalue Spectra — Denoising Step {ds} (Layer 11)", fontsize=12)

        for ai, alpha in enumerate(ALPHAS):
            ax = axes[ai]
            for task in TASKS:
                gamma = conceptors[(task, ds, alpha)]["gamma"]
                n_plot = min(50, len(gamma))
                ax.plot(range(1, n_plot + 1), gamma[:n_plot],
                        color=TASK_COLORS[task], lw=1.5, label=task)
            ax.axhline(0.5, color="gray", ls=":", lw=0.7, alpha=0.5)
            ax.set_title(f"α = {alpha}")
            ax.set_xlabel("Eigenvalue index")
            if ai == 0:
                ax.set_ylabel("γ_j (conceptor eigenvalue)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0, 52)
            if ai == len(ALPHAS) - 1:
                ax.legend(fontsize=6, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT_DIR / f"spectra_denoise_{ds}.png")
        plt.close(fig)
        print(f"  Saved spectra_denoise_{ds}.png")

    # Combined plot: fixed alpha=1.0, both denoising steps side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle("Conceptor Eigenvalue Spectra (α=1.0, Layer 11)", fontsize=12)
    for di, ds in enumerate(DENOISE_STEPS):
        ax = axes[di]
        for task in TASKS:
            gamma = conceptors[(task, ds, 1.0)]["gamma"]
            n_plot = min(50, len(gamma))
            ax.plot(range(1, n_plot + 1), gamma[:n_plot],
                    color=TASK_COLORS[task], lw=2, label=task)
        ax.axhline(0.5, color="gray", ls=":", lw=0.7)
        ax.set_title(f"Denoising Step {ds}")
        ax.set_xlabel("Eigenvalue index j")
        if di == 0:
            ax.set_ylabel("γ_j")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 52)
        ax.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "spectra_combined.png")
    plt.close(fig)
    print("  Saved spectra_combined.png")

    # Alpha sweep: quota vs alpha for each task
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Conceptor Quota q(C) = trace(C) vs Aperture α", fontsize=12)
    for di, ds in enumerate(DENOISE_STEPS):
        ax = axes[di]
        for task in TASKS:
            quotas = [conceptors[(task, ds, a)]["quota"] for a in ALPHAS]
            ax.plot(ALPHAS, quotas, 'o-', color=TASK_COLORS[task], lw=1.5, label=task)
        ax.set_xlabel("α (aperture)")
        ax.set_ylabel("Quota q(C)")
        ax.set_title(f"Denoising Step {ds}")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "quota_vs_alpha.png")
    plt.close(fig)
    print("  Saved quota_vs_alpha.png")

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (b): Conceptor Similarity / Overlap Matrix
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (b): CONCEPTOR SIMILARITY MATRIX")
    print("=" * 70)

    for ds in DENOISE_STEPS:
        # Compute full conceptor matrices at alpha=1.0
        Cs = {}
        for task in TASKS:
            entry = conceptors[(task, ds, 1.0)]
            if entry["C"] is not None:
                Cs[task] = entry["C"]
            else:
                X, _ = task_data[(task, ds)]
                C_full, _ = compute_conceptor_matrix(X, 1.0)
                Cs[task] = C_full

        # Overlap matrix
        n_tasks = len(TASKS)
        overlap_mat = np.zeros((n_tasks, n_tasks))
        frob_dist_mat = np.zeros((n_tasks, n_tasks))

        for i, t_i in enumerate(TASKS):
            for j, t_j in enumerate(TASKS):
                overlap_mat[i, j] = conceptor_overlap(Cs[t_i], Cs[t_j])
                frob_dist_mat[i, j] = np.linalg.norm(Cs[t_i] - Cs[t_j], 'fro')

        # Print
        print(f"\n  Overlap matrix (denoise step {ds}, α=1.0):")
        print(f"  {'':>20s}", end="")
        for t in TASKS:
            print(f"  {t:>16s}", end="")
        print()
        for i, t_i in enumerate(TASKS):
            print(f"  {t_i:>20s}", end="")
            for j in range(n_tasks):
                print(f"  {overlap_mat[i, j]:>16.3f}", end="")
            print()

        print(f"\n  Frobenius distance matrix (denoise step {ds}, α=1.0):")
        print(f"  {'':>20s}", end="")
        for t in TASKS:
            print(f"  {t:>16s}", end="")
        print()
        for i, t_i in enumerate(TASKS):
            print(f"  {t_i:>20s}", end="")
            for j in range(n_tasks):
                print(f"  {frob_dist_mat[i, j]:>16.1f}", end="")
            print()

        # Plot heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Conceptor Similarity (Denoising Step {ds}, α=1.0, Layer 11)", fontsize=12)

        short_names = [t.replace("-v3", "") for t in TASKS]

        im1 = ax1.imshow(overlap_mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax1.set_xticks(range(n_tasks))
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.set_yticks(range(n_tasks))
        ax1.set_yticklabels(short_names)
        ax1.set_title("Overlap: tr(C_A C_B) / tr(C_A)")
        for i in range(n_tasks):
            for j in range(n_tasks):
                ax1.text(j, i, f"{overlap_mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(frob_dist_mat, cmap="YlOrRd")
        ax2.set_xticks(range(n_tasks))
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.set_yticks(range(n_tasks))
        ax2.set_yticklabels(short_names)
        ax2.set_title("Frobenius Distance ‖C_A − C_B‖_F")
        for i in range(n_tasks):
            for j in range(n_tasks):
                ax2.text(j, i, f"{frob_dist_mat[i,j]:.0f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT_DIR / f"similarity_denoise_{ds}.png")
        plt.close(fig)
        print(f"  Saved similarity_denoise_{ds}.png")

    # Also sweep alpha for overlap analysis
    print("\n  Overlap (reach vs assembly) across α values:")
    for alpha in ALPHAS:
        for ds in DENOISE_STEPS:
            X_r, _ = task_data[("reach-v3", ds)]
            X_a, _ = task_data[("assembly-v3", ds)]
            C_r, _ = compute_conceptor_matrix(X_r, alpha)
            C_a, _ = compute_conceptor_matrix(X_a, alpha)
            ov = conceptor_overlap(C_r, C_a)
            print(f"    ds={ds} α={alpha:5.1f} → overlap={ov:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (c): Boolean Operations (AND, NOT)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (c): BOOLEAN OPERATIONS")
    print("=" * 70)

    ds = 0  # use denoising step 0
    Cs = {}
    for task in TASKS:
        entry = conceptors[(task, ds, 1.0)]
        if entry["C"] is not None:
            Cs[task] = entry["C"]
        else:
            X, _ = task_data[(task, ds)]
            C_full, _ = compute_conceptor_matrix(X, 1.0)
            Cs[task] = C_full

    # AND: shared subspace between reach and button-press
    C_reach_AND_button = conceptor_AND(Cs["reach-v3"], Cs["button-press-v3"])
    # NOT: what's unique to reaching (not in button-press)
    C_reach_NOT_button = Cs["reach-v3"] @ conceptor_NOT(Cs["button-press-v3"])
    # NOT: what's unique to assembly
    C_assembly_NOT_reach = Cs["assembly-v3"] @ conceptor_NOT(Cs["reach-v3"])

    # Analyse spectra of boolean results
    boolean_results = {
        "reach AND button": C_reach_AND_button,
        "reach NOT button": C_reach_NOT_button,
        "assembly NOT reach": C_assembly_NOT_reach,
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Boolean Conceptor Spectra (Denoising Step 0, α=1.0, Layer 11)", fontsize=12)

    colors_bool = ["#9467bd", "#8c564b", "#e377c2"]
    for bi, (name, C_bool) in enumerate(boolean_results.items()):
        ax = axes[bi]
        eigs = np.sort(np.real(np.linalg.eigvals(C_bool)))[::-1]
        n_plot = min(50, len(eigs))
        ax.plot(range(1, n_plot + 1), eigs[:n_plot], color=colors_bool[bi], lw=2)
        ax.axhline(0.5, color="gray", ls=":", lw=0.7)
        ax.axhline(0.0, color="gray", ls="-", lw=0.5, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("eigenvalue")
        ax.set_xlim(0, 52)

        q = float(np.sum(np.clip(eigs, 0, None)))
        n_above_half = int(np.sum(eigs > 0.5))
        ax.text(0.95, 0.95, f"quota≈{q:.1f}\nn(>0.5)={n_above_half}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        print(f"  {name}: quota={q:.1f}, n(eig>0.5)={n_above_half}, "
              f"top-5 eigs={eigs[:5].round(3)}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "boolean_operations.png")
    plt.close(fig)
    print("  Saved boolean_operations.png")

    # Success vs Failure analysis for assembly-v3
    print("\n  Success vs Failure conceptor (assembly-v3):")
    ep_meta = task_episode_meta["assembly-v3"]
    X_assembly, step_metas = task_data[("assembly-v3", 0)]

    # Split by success_so_far at each step
    success_vecs = []
    failure_vecs = []
    for i, m in enumerate(step_metas):
        if m.get("success_so_far", False):
            success_vecs.append(X_assembly[i])
        else:
            failure_vecs.append(X_assembly[i])

    print(f"    assembly-v3: {len(success_vecs)} success steps, {len(failure_vecs)} failure steps")

    if len(success_vecs) >= 3 and len(failure_vecs) >= 3:
        X_succ = np.array(success_vecs, dtype=np.float32)
        X_fail = np.array(failure_vecs, dtype=np.float32)
        C_succ, _ = compute_conceptor_matrix(X_succ, 1.0)
        C_fail, _ = compute_conceptor_matrix(X_fail, 1.0)
        C_succ_NOT_fail = C_succ @ conceptor_NOT(C_fail)

        eigs_s = np.sort(np.real(np.linalg.eigvals(C_succ)))[::-1]
        eigs_f = np.sort(np.real(np.linalg.eigvals(C_fail)))[::-1]
        eigs_snf = np.sort(np.real(np.linalg.eigvals(C_succ_NOT_fail)))[::-1]

        print(f"    C_success: quota={float(eigs_s.clip(0).sum()):.1f}")
        print(f"    C_failure: quota={float(eigs_f.clip(0).sum()):.1f}")
        print(f"    C_success NOT C_failure: quota={float(eigs_snf.clip(0).sum()):.1f}")
        print(f"    overlap(success, failure) = {conceptor_overlap(C_succ, C_fail):.4f}")
    else:
        print("    Not enough success/failure data for assembly-v3 (both envs may fail)")
        # Try a different task that has mixed outcomes
        for alt_task in ["drawer-open-v3", "button-press-v3"]:
            X_alt, metas_alt = task_data[(alt_task, 0)]
            succ_v, fail_v = [], []
            for i, m in enumerate(metas_alt):
                if m.get("success_so_far", False):
                    succ_v.append(X_alt[i])
                else:
                    fail_v.append(X_alt[i])
            if len(succ_v) >= 3 and len(fail_v) >= 3:
                print(f"\n    Using {alt_task} instead: {len(succ_v)} success, {len(fail_v)} failure")
                X_s = np.array(succ_v, dtype=np.float32)
                X_f = np.array(fail_v, dtype=np.float32)
                C_s, _ = compute_conceptor_matrix(X_s, 1.0)
                C_f, _ = compute_conceptor_matrix(X_f, 1.0)
                print(f"    overlap(success, failure) = {conceptor_overlap(C_s, C_f):.4f}")
                break

    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (d): Linear Probe Validation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DIAGNOSTIC (d): LINEAR PROBE ON CONCEPTOR SUBSPACES")
    print("=" * 70)

    ds = 0  # use step 0

    # --- (d.1) Task identity classification ---
    print("\n  (d.1) Task identity classification:")

    # Build dataset: X_all, y_all
    all_X = []
    all_y = []
    for ti, task in enumerate(TASKS):
        X, _ = task_data[(task, ds)]
        all_X.append(X)
        all_y.extend([ti] * X.shape[0])
    all_X = np.vstack(all_X)
    all_y = np.array(all_y)

    print(f"    Total samples: {all_X.shape[0]}, classes: {len(TASKS)}")

    # (A) Full-space probe (baseline)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs_full = []
    for train_idx, test_idx in skf.split(all_X, all_y):
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(all_X[train_idx], all_y[train_idx])
        accs_full.append(accuracy_score(all_y[test_idx], clf.predict(all_X[test_idx])))
    print(f"    Full-space (1024d) probe: {np.mean(accs_full):.3f} ± {np.std(accs_full):.3f}")

    # (B) Conceptor subspace probes at various ranks
    for k in [5, 10, 20, 50]:
        # Use top-k eigenvectors of the combined conceptor (all tasks)
        X_all_centered = all_X - all_X.mean(axis=0, keepdims=True)
        sigma_all, Vt_all = fast_svd(all_X, k=k)
        X_proj = all_X @ Vt_all.T  # (n, k)

        accs_proj = []
        for train_idx, test_idx in skf.split(X_proj, all_y):
            clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            clf.fit(X_proj[train_idx], all_y[train_idx])
            accs_proj.append(accuracy_score(all_y[test_idx], clf.predict(X_proj[test_idx])))
        print(f"    Top-{k:2d} PCA subspace probe:   {np.mean(accs_proj):.3f} ± {np.std(accs_proj):.3f}")

    # (C) Per-task conceptor subspace probes
    for k in [5, 10, 20]:
        # For each task, get top-k eigenvectors of its conceptor
        # Concatenate them as features
        all_projs = []
        for task in TASKS:
            entry = conceptors[(task, ds, 1.0)]
            Vt_task = entry["Vt"][:k]  # (k, 1024)
            proj = all_X @ Vt_task.T  # (n, k) — projection onto task's top-k directions
            all_projs.append(proj)
        X_concat = np.hstack(all_projs)  # (n, k * n_tasks)

        accs_conc = []
        for train_idx, test_idx in skf.split(X_concat, all_y):
            clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
            clf.fit(X_concat[train_idx], all_y[train_idx])
            accs_conc.append(accuracy_score(all_y[test_idx], clf.predict(X_concat[test_idx])))
        print(f"    Per-task top-{k:2d} conceptor proj: {np.mean(accs_conc):.3f} ± {np.std(accs_conc):.3f}")

    # --- (d.2) Episode phase classification (early vs late) ---
    print("\n  (d.2) Episode phase classification (early vs late half):")

    phase_X = []
    phase_y = []
    for task in TASKS:
        X, metas = task_data[(task, ds)]
        n = X.shape[0]
        for i in range(n):
            phase_X.append(X[i])
            inf_step = metas[i].get("inference_step", i)
            total_steps = n // 2  # 2 envs
            phase_y.append(0 if inf_step < total_steps // 2 else 1)  # 0=early, 1=late

    phase_X = np.array(phase_X, dtype=np.float32)
    phase_y = np.array(phase_y)

    if len(np.unique(phase_y)) > 1:
        accs_phase = []
        for train_idx, test_idx in skf.split(phase_X, phase_y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(phase_X[train_idx], phase_y[train_idx])
            accs_phase.append(accuracy_score(phase_y[test_idx], clf.predict(phase_X[test_idx])))
        print(f"    Full-space phase probe: {np.mean(accs_phase):.3f} ± {np.std(accs_phase):.3f}")

        # Conceptor subspace
        for k in [5, 10, 20]:
            _, Vt_phase = fast_svd(phase_X, k=k)
            X_proj_phase = phase_X @ Vt_phase.T
            accs_pp = []
            for train_idx, test_idx in skf.split(X_proj_phase, phase_y):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X_proj_phase[train_idx], phase_y[train_idx])
                accs_pp.append(accuracy_score(phase_y[test_idx], clf.predict(X_proj_phase[test_idx])))
            print(f"    Top-{k:2d} subspace phase probe: {np.mean(accs_pp):.3f} ± {np.std(accs_pp):.3f}")

    # --- (d.3) Success/failure classification ---
    print("\n  (d.3) Success/failure probe (across all tasks):")

    sf_X = []
    sf_y = []
    for task in TASKS:
        ep_meta = task_episode_meta[task]
        X, metas = task_data[(task, ds)]
        for i in range(X.shape[0]):
            sf_X.append(X[i])
            # Use episode-level success
            env_id = metas[i].get("env_id", 0)
            ep_success = ep_meta.get(env_id, {}).get("episode_success", False)
            sf_y.append(1 if ep_success else 0)

    sf_X = np.array(sf_X, dtype=np.float32)
    sf_y = np.array(sf_y)

    if len(np.unique(sf_y)) > 1:
        accs_sf = []
        for train_idx, test_idx in skf.split(sf_X, sf_y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sf_X[train_idx], sf_y[train_idx])
            accs_sf.append(accuracy_score(sf_y[test_idx], clf.predict(sf_X[test_idx])))
        print(f"    Full-space success probe: {np.mean(accs_sf):.3f} ± {np.std(accs_sf):.3f}")
        print(f"    (baseline: {max(sf_y.mean(), 1-sf_y.mean()):.3f} majority class)")

        for k in [5, 10, 20]:
            _, Vt_sf = fast_svd(sf_X, k=k)
            X_proj_sf = sf_X @ Vt_sf.T
            accs_sf_proj = []
            for train_idx, test_idx in skf.split(X_proj_sf, sf_y):
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(X_proj_sf[train_idx], sf_y[train_idx])
                accs_sf_proj.append(accuracy_score(sf_y[test_idx], clf.predict(X_proj_sf[test_idx])))
            print(f"    Top-{k:2d} subspace success probe: {np.mean(accs_sf_proj):.3f} ± {np.std(accs_sf_proj):.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Checklist:")
    # Check 1: Sharp spectra?
    for task in TASKS:
        gamma = conceptors[(task, 0, 1.0)]["gamma"]
        n_above_half = int(np.sum(gamma > 0.5))
        total = len(gamma)
        ratio = n_above_half / total if total > 0 else 0
        sharp = "YES" if ratio < 0.3 else "MARGINAL" if ratio < 0.5 else "NO (flat)"
        print(f"    Sharp spectrum for {task:>20s}? {sharp:>12s} "
              f"(n(γ>0.5)={n_above_half}/{total}, ratio={ratio:.2f})")

    # Check 2: Task separation?
    for ds in [0]:
        Cs_check = {}
        for task in TASKS:
            e = conceptors[(task, ds, 1.0)]
            if e["C"] is not None:
                Cs_check[task] = e["C"]
        if len(Cs_check) == len(TASKS):
            off_diag = []
            for i, t_i in enumerate(TASKS):
                for j, t_j in enumerate(TASKS):
                    if i != j:
                        off_diag.append(conceptor_overlap(Cs_check[t_i], Cs_check[t_j]))
            mean_offdiag = np.mean(off_diag)
            max_offdiag = np.max(off_diag)
            print(f"\n    Task separation (ds={ds}): mean off-diag overlap = {mean_offdiag:.3f}, "
                  f"max = {max_offdiag:.3f}")
            separated = "YES" if mean_offdiag < 0.7 else "MARGINAL" if mean_offdiag < 0.85 else "NO"
            print(f"    Clear task separation? {separated}")

    print(f"\n  All plots saved to: {OUT_DIR.resolve()}")
    print("  Done!")


if __name__ == "__main__":
    run_analysis()
