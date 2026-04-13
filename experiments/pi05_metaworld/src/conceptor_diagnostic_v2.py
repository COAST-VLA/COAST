#!/usr/bin/env python3
"""
Conceptor Diagnostic Analysis v2 — 15-env dataset, all layers, linear probe baselines
=======================================================================================
Extended analysis using brandonyang/ml45-activations-15 (15 envs per task).

Improvements over v1:
  1. 15 envs per task → more statistical power, success/failure splits within tasks
  2. Sweep all 4 captured layers (0, 5, 11, 17)
  3. Linear probe baselines for paper comparison:
     - Full-space logistic regression
     - PCA-projected probe
     - Random subspace probe
     - Conceptor-projected probe
     - Comparison: accuracy vs. dimensionality curves

Reference: /nlpgpu/data/miaom/conceptor/src/e2e.py, analyze_conceptor_spectrum.py
"""

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────

REPO = "brandonyang/ml45-activations-15"
CHECKPOINT = "5000"

TASKS = ["reach-v3", "button-press-v3", "drawer-open-v3", "assembly-v3"]

# All 4 captured layers in the V1 dataset
LAYER_INDICES = {0: "Layer 0 (input)", 1: "Layer 5 (early-mid)",
                 2: "Layer 11 (mid-late)", 3: "Layer 17 (output)"}
LAYER_NAMES = {0: "L0", 1: "L5", 2: "L11", 3: "L17"}

DENOISE_STEPS = [0, 9]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
NUM_ENVS = 15

OUT_DIR = Path("experiments/conceptor_results_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COLORS = {
    "reach-v3": "#1f77b4",
    "button-press-v3": "#ff7f0e",
    "drawer-open-v3": "#2ca02c",
    "assembly-v3": "#d62728",
}
LAYER_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


# ── Conceptor Helpers ────────────────────────────────────────────────────

def fast_svd(X, k=None):
    """SVD of mean-centred X. Returns eigenvalues of R = X^T X / N and right singular vectors."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc / np.sqrt(max(1, N)), full_matrices=False)
    sigma = s ** 2
    if k is not None:
        return sigma[:k], Vt[:k]
    return sigma, Vt


def conceptor_eigenvalues(sigma, alpha):
    return sigma / (sigma + alpha ** -2)


def conceptor_quota(gamma):
    return float(gamma.sum())


def conceptor_entropy(gamma):
    g = np.clip(gamma, 1e-12, 1 - 1e-12)
    return -float(np.sum(g * np.log2(g) + (1 - g) * np.log2(1 - g)))


def compute_conceptor_matrix(X, alpha):
    """Full d×d conceptor C = R (R + α⁻² I)⁻¹."""
    Xc = X - X.mean(axis=0, keepdims=True)
    N = Xc.shape[0]
    R = (Xc.T @ Xc) / max(1, N)
    d = R.shape[0]
    C = R @ np.linalg.inv(R + alpha ** (-2) * np.eye(d))
    return C, R


def conceptor_AND(C_A, C_B):
    d = C_A.shape[0]
    I = np.eye(d)
    eps = 1e-6
    C_A_inv = np.linalg.inv(C_A + eps * I)
    C_B_inv = np.linalg.inv(C_B + eps * I)
    return np.linalg.inv(C_A_inv + C_B_inv - I + eps * I)


def conceptor_NOT(C):
    return np.eye(C.shape[0]) - C


def conceptor_overlap(C_A, C_B):
    tr_A = np.trace(C_A)
    if tr_A < 1e-10:
        return 0.0
    return float(np.trace(C_A @ C_B) / tr_A)


# ── Data Loading ─────────────────────────────────────────────────────────

def download_task_activations_15env(task_name, denoise_step=0, layer_idx=2):
    """Download suffix_residual.npz for all 15 envs and return aggregated data.

    Returns:
        X: (n_samples, 1024)
        metas: list of dicts with step/episode info
        env_ids: list of env_id per sample
        episode_success: list of bool per sample
    """
    print(f"  Loading {task_name} (layer_idx={layer_idx}, denoise_step={denoise_step})...")

    vectors = []
    metas = []
    env_ids = []
    episode_successes = []

    for env_id in range(NUM_ENVS):
        # Get episode metadata
        ep_path = f"{CHECKPOINT}/{task_name}/episode_000_env_{env_id:03d}/metadata.json"
        try:
            f = hf_hub_download(REPO, ep_path, repo_type="dataset")
            with open(f) as fh:
                ep_meta = json.load(fh)
            ep_success = ep_meta.get("episode_success", False)
            n_inf_steps = ep_meta.get("total_inference_steps", 30)
        except Exception:
            continue

        # Download each inference step
        for step_idx in range(n_inf_steps):
            step_num = step_idx * 10  # env steps: 0, 10, 20, ...
            res_path = f"{CHECKPOINT}/{task_name}/episode_000_env_{env_id:03d}/step_{step_num:04d}/suffix_residual.npz"
            meta_path = f"{CHECKPOINT}/{task_name}/episode_000_env_{env_id:03d}/step_{step_num:04d}/metadata.json"

            try:
                f = hf_hub_download(REPO, res_path, repo_type="dataset")
                data = np.load(f)
                all_residual = data["all_suffix_residual"]
                # shape: (10, 4, 32, 1024)
                residual = all_residual[denoise_step, layer_idx]  # (32, 1024)
                mean_vec = residual.mean(axis=0)  # (1024,)
                vectors.append(mean_vec)

                # Step metadata
                try:
                    mf = hf_hub_download(REPO, meta_path, repo_type="dataset")
                    with open(mf) as fh:
                        step_meta = json.load(fh)
                except Exception:
                    step_meta = {"inference_step": step_idx, "env_id": env_id}

                step_meta["env_id"] = env_id
                step_meta["episode_success"] = ep_success
                metas.append(step_meta)
                env_ids.append(env_id)
                episode_successes.append(ep_success)
            except Exception:
                break  # no more steps for this env

    X = np.array(vectors, dtype=np.float32)
    print(f"    → {task_name}: {X.shape[0]} samples, dim={X.shape[1]}, "
          f"success_rate={sum(episode_successes)/max(1,len(episode_successes)):.2f}")
    return X, metas, env_ids, episode_successes


def download_all_data():
    """Download activations for all tasks × layers × denoising steps."""
    data = {}  # {(task, layer_idx, denoise_step): (X, metas, env_ids, ep_successes)}

    for task in TASKS:
        for layer_idx in LAYER_INDICES:
            for ds in DENOISE_STEPS:
                key = (task, layer_idx, ds)
                X, metas, env_ids, ep_succ = download_task_activations_15env(
                    task, denoise_step=ds, layer_idx=layer_idx)
                data[key] = (X, metas, env_ids, ep_succ)
    return data


# ── Analysis Functions ───────────────────────────────────────────────────

def run_conceptor_analysis(data):
    """Compute conceptors for all task × layer × denoise_step × alpha combos."""
    print("\n" + "=" * 70)
    print("CONCEPTOR COMPUTATION")
    print("=" * 70)

    results = {}
    K_MAX = 200

    for task in TASKS:
        for layer_idx in LAYER_INDICES:
            for ds in DENOISE_STEPS:
                X, _, _, _ = data[(task, layer_idx, ds)]
                sigma, Vt = fast_svd(X, k=min(K_MAX, X.shape[0], X.shape[1]))

                for alpha in ALPHAS:
                    gamma = conceptor_eigenvalues(sigma, alpha)
                    q = conceptor_quota(gamma)
                    h = conceptor_entropy(gamma)

                    C_full, R_full = None, None
                    if alpha == 1.0:
                        C_full, R_full = compute_conceptor_matrix(X, alpha)

                    results[(task, layer_idx, ds, alpha)] = {
                        "sigma": sigma, "gamma": gamma, "Vt": Vt,
                        "quota": q, "entropy": h,
                        "C": C_full, "R": R_full, "n_samples": X.shape[0],
                    }

    # Print summary table
    print(f"\n{'Task':>20s} | {'Layer':>5s} | {'DS':>2s} | {'α':>5s} | "
          f"{'Quota':>6s} | {'n(γ>.5)':>7s} | {'n(γ>.9)':>7s} | {'N':>5s}")
    print("-" * 80)
    for task in TASKS:
        for layer_idx in LAYER_INDICES:
            for alpha in [0.1, 1.0, 10.0]:
                r = results[(task, layer_idx, 0, alpha)]
                gamma = r["gamma"]
                print(f"{task:>20s} | {LAYER_NAMES[layer_idx]:>5s} | {0:>2d} | {alpha:>5.1f} | "
                      f"{r['quota']:>6.1f} | {int(np.sum(gamma>0.5)):>7d} | "
                      f"{int(np.sum(gamma>0.9)):>7d} | {r['n_samples']:>5d}")

    return results


def plot_spectra(conceptors):
    """Diagnostic (a): Eigenvalue spectra across layers and tasks."""
    print("\n" + "=" * 70)
    print("PLOTTING SPECTRA")
    print("=" * 70)

    # Main figure: 4 layers × 2 denoising steps, tasks overlaid
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    fig.suptitle("Conceptor Eigenvalue Spectra (α=1.0) — 15 envs per task", fontsize=13)

    for di, ds in enumerate(DENOISE_STEPS):
        for li, layer_idx in enumerate(LAYER_INDICES):
            ax = axes[di, li]
            for task in TASKS:
                gamma = conceptors[(task, layer_idx, ds, 1.0)]["gamma"]
                n_plot = min(80, len(gamma))
                ax.plot(range(1, n_plot + 1), gamma[:n_plot],
                        color=TASK_COLORS[task], lw=1.5, label=task if di == 0 and li == 0 else None)
            ax.axhline(0.5, color="gray", ls=":", lw=0.7)
            ax.set_xlim(0, 82)
            ax.set_ylim(-0.02, 1.02)
            if di == 0:
                ax.set_title(LAYER_NAMES[layer_idx])
            if di == 1:
                ax.set_xlabel("Eigenvalue index j")
            if li == 0:
                ax.set_ylabel(f"Denoise {ds}\nγ_j")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=1,
               bbox_to_anchor=(0.99, 0.95))
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(OUT_DIR / "spectra_all_layers.png")
    plt.close(fig)
    print("  Saved spectra_all_layers.png")

    # Quota vs alpha per layer
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
    fig.suptitle("Conceptor Quota q(C) vs Aperture α — 15 envs per task", fontsize=13)
    for di, ds in enumerate(DENOISE_STEPS):
        for li, layer_idx in enumerate(LAYER_INDICES):
            ax = axes[di, li]
            for task in TASKS:
                quotas = [conceptors[(task, layer_idx, ds, a)]["quota"] for a in ALPHAS]
                ax.plot(ALPHAS, quotas, 'o-', color=TASK_COLORS[task], lw=1.5,
                        label=task if di == 0 and li == 0 else None)
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if di == 0:
                ax.set_title(LAYER_NAMES[layer_idx])
            if di == 1:
                ax.set_xlabel("α")
            if li == 0:
                ax.set_ylabel(f"Denoise {ds}\nQuota q(C)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=1,
               bbox_to_anchor=(0.99, 0.95))
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(OUT_DIR / "quota_vs_alpha_all_layers.png")
    plt.close(fig)
    print("  Saved quota_vs_alpha_all_layers.png")

    # Quota across layers (fixed alpha=1.0) — shows which layer is best
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Conceptor Quota Across Layers (α=1.0)", fontsize=12)
    layer_xs = [0, 5, 11, 17]
    for di, ds in enumerate(DENOISE_STEPS):
        ax = axes[di]
        for task in TASKS:
            quotas = [conceptors[(task, li, ds, 1.0)]["quota"] for li in range(4)]
            ax.plot(layer_xs, quotas, 'o-', color=TASK_COLORS[task], lw=2, label=task)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Quota q(C)")
        ax.set_title(f"Denoising Step {ds}")
        ax.set_xticks(layer_xs)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "quota_across_layers.png")
    plt.close(fig)
    print("  Saved quota_across_layers.png")


def plot_similarity(data, conceptors):
    """Diagnostic (b): Similarity matrices across layers."""
    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS")
    print("=" * 70)

    short_names = [t.replace("-v3", "") for t in TASKS]
    n_tasks = len(TASKS)

    # One figure per denoising step, showing all 4 layers
    for ds in DENOISE_STEPS:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle(f"Conceptor Similarity (Denoising Step {ds}, α=1.0) — 15 envs", fontsize=13)

        for li, layer_idx in enumerate(LAYER_INDICES):
            Cs = {}
            for task in TASKS:
                entry = conceptors[(task, layer_idx, ds, 1.0)]
                if entry["C"] is not None:
                    Cs[task] = entry["C"]
                else:
                    X, _, _, _ = data[(task, layer_idx, ds)]
                    Cs[task], _ = compute_conceptor_matrix(X, 1.0)

            overlap_mat = np.zeros((n_tasks, n_tasks))
            frob_mat = np.zeros((n_tasks, n_tasks))
            for i, t_i in enumerate(TASKS):
                for j, t_j in enumerate(TASKS):
                    overlap_mat[i, j] = conceptor_overlap(Cs[t_i], Cs[t_j])
                    frob_mat[i, j] = np.linalg.norm(Cs[t_i] - Cs[t_j], 'fro')

            # Overlap
            ax = axes[0, li]
            im = ax.imshow(overlap_mat, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(n_tasks))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n_tasks))
            ax.set_yticklabels(short_names, fontsize=7)
            for i in range(n_tasks):
                for j in range(n_tasks):
                    ax.text(j, i, f"{overlap_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"{LAYER_NAMES[layer_idx]} — Overlap")

            # Frobenius
            ax2 = axes[1, li]
            im2 = ax2.imshow(frob_mat, cmap="YlOrRd")
            ax2.set_xticks(range(n_tasks))
            ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
            ax2.set_yticks(range(n_tasks))
            ax2.set_yticklabels(short_names, fontsize=7)
            for i in range(n_tasks):
                for j in range(n_tasks):
                    ax2.text(j, i, f"{frob_mat[i,j]:.0f}", ha="center", va="center", fontsize=7)
            ax2.set_title(f"{LAYER_NAMES[layer_idx]} — Frobenius")

            # Print
            print(f"\n  Overlap matrix (ds={ds}, {LAYER_NAMES[layer_idx]}, α=1.0):")
            for i, t_i in enumerate(TASKS):
                row = " ".join(f"{overlap_mat[i,j]:.3f}" for j in range(n_tasks))
                print(f"    {t_i:>20s}: {row}")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUT_DIR / f"similarity_all_layers_ds{ds}.png")
        plt.close(fig)
        print(f"  Saved similarity_all_layers_ds{ds}.png")

    # Mean off-diagonal overlap per layer (summary plot)
    fig, ax = plt.subplots(figsize=(8, 4))
    layer_xs = [0, 5, 11, 17]
    for di, ds in enumerate(DENOISE_STEPS):
        mean_offdiags = []
        for li in range(4):
            Cs = {}
            for task in TASKS:
                entry = conceptors[(task, li, ds, 1.0)]
                if entry["C"] is not None:
                    Cs[task] = entry["C"]
                else:
                    X, _, _, _ = data[(task, li, ds)]
                    Cs[task], _ = compute_conceptor_matrix(X, 1.0)
            off = []
            for i, t_i in enumerate(TASKS):
                for j, t_j in enumerate(TASKS):
                    if i != j:
                        off.append(conceptor_overlap(Cs[t_i], Cs[t_j]))
            mean_offdiags.append(np.mean(off))
        style = '-o' if ds == 0 else '--s'
        ax.plot(layer_xs, mean_offdiags, style, lw=2, label=f"Denoise step {ds}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean off-diagonal overlap")
    ax.set_title("Task Separation Across Layers (α=1.0)")
    ax.set_xticks(layer_xs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "separation_across_layers.png")
    plt.close(fig)
    print("  Saved separation_across_layers.png")


def plot_boolean_ops(data, conceptors):
    """Diagnostic (c): Boolean operations + success/failure within assembly."""
    print("\n" + "=" * 70)
    print("BOOLEAN OPERATIONS")
    print("=" * 70)

    ds = 0
    alpha = 1.0

    # For each layer, compute boolean ops
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    fig.suptitle("Boolean Conceptor Spectra Across Layers (Denoise 0, α=1.0)", fontsize=13)

    bool_ops = [
        ("reach AND button", lambda Cs: conceptor_AND(Cs["reach-v3"], Cs["button-press-v3"])),
        ("reach NOT button", lambda Cs: Cs["reach-v3"] @ conceptor_NOT(Cs["button-press-v3"])),
        ("assembly NOT reach", lambda Cs: Cs["assembly-v3"] @ conceptor_NOT(Cs["reach-v3"])),
        ("success NOT failure", None),  # handled specially
    ]

    for li in range(4):
        Cs = {}
        for task in TASKS:
            entry = conceptors[(task, li, ds, alpha)]
            if entry["C"] is not None:
                Cs[task] = entry["C"]
            else:
                X, _, _, _ = data[(task, li, ds)]
                Cs[task], _ = compute_conceptor_matrix(X, alpha)

        # Success/failure split for assembly (3/15 success)
        X_asm, metas_asm, _, ep_succ_asm = data[("assembly-v3", li, ds)]
        succ_idx = [i for i, s in enumerate(ep_succ_asm) if s]
        fail_idx = [i for i, s in enumerate(ep_succ_asm) if not s]

        C_succ, C_fail = None, None
        if len(succ_idx) >= 3 and len(fail_idx) >= 3:
            X_s = X_asm[succ_idx]
            X_f = X_asm[fail_idx]
            C_succ, _ = compute_conceptor_matrix(X_s, alpha)
            C_fail, _ = compute_conceptor_matrix(X_f, alpha)

        bool_ops_concrete = [
            ("reach ∧ button", conceptor_AND(Cs["reach-v3"], Cs["button-press-v3"])),
            ("reach ¬ button", Cs["reach-v3"] @ conceptor_NOT(Cs["button-press-v3"])),
            ("assembly ¬ reach", Cs["assembly-v3"] @ conceptor_NOT(Cs["reach-v3"])),
        ]
        if C_succ is not None and C_fail is not None:
            bool_ops_concrete.append(("success ¬ failure", C_succ @ conceptor_NOT(C_fail)))
        else:
            bool_ops_concrete.append(("success ¬ failure", None))

        colors_bool = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for bi, (name, C_bool) in enumerate(bool_ops_concrete):
            ax = axes[bi, li]
            if C_bool is not None:
                eigs = np.sort(np.real(np.linalg.eigvals(C_bool)))[::-1]
                n_plot = min(80, len(eigs))
                ax.plot(range(1, n_plot + 1), eigs[:n_plot], color=colors_bool[bi], lw=1.5)
                q = float(np.sum(np.clip(eigs, 0, None)))
                n_half = int(np.sum(eigs > 0.5))
                ax.text(0.95, 0.95, f"q≈{q:.0f}\nn>.5={n_half}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
                print(f"  {LAYER_NAMES[li]} | {name}: quota={q:.1f}, n(>0.5)={n_half}")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", fontsize=10)
            ax.axhline(0.5, color="gray", ls=":", lw=0.7)
            ax.axhline(0.0, color="gray", ls="-", lw=0.5, alpha=0.3)
            ax.set_xlim(0, 82)
            if li == 0:
                ax.set_ylabel(name)
            if bi == 0:
                ax.set_title(LAYER_NAMES[li])
            if bi == 3:
                ax.set_xlabel("Eigenvalue index")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "boolean_all_layers.png")
    plt.close(fig)
    print("  Saved boolean_all_layers.png")

    # Success vs failure detailed for assembly at best layer
    print("\n  Assembly success/failure overlap across layers:")
    for li in range(4):
        X_asm, _, _, ep_succ_asm = data[("assembly-v3", li, 0)]
        succ_idx = [i for i, s in enumerate(ep_succ_asm) if s]
        fail_idx = [i for i, s in enumerate(ep_succ_asm) if not s]
        if len(succ_idx) >= 3 and len(fail_idx) >= 3:
            C_s, _ = compute_conceptor_matrix(X_asm[succ_idx], 1.0)
            C_f, _ = compute_conceptor_matrix(X_asm[fail_idx], 1.0)
            ov = conceptor_overlap(C_s, C_f)
            ov_r = conceptor_overlap(C_f, C_s)
            print(f"    {LAYER_NAMES[li]}: overlap(succ→fail)={ov:.3f}, "
                  f"overlap(fail→succ)={ov_r:.3f}, "
                  f"n_succ={len(succ_idx)}, n_fail={len(fail_idx)}")


def run_probe_comparison(data, conceptors):
    """Diagnostic (d): Linear probe baselines vs. conceptor projections."""
    print("\n" + "=" * 70)
    print("LINEAR PROBE COMPARISON (Paper Baselines)")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    K_VALUES = [5, 10, 20, 50, 100]

    results_table = []  # for final summary table

    for ds in DENOISE_STEPS:
        for li in range(4):
            print(f"\n  --- Denoise Step {ds}, {LAYER_NAMES[li]} ---")

            # ── Build datasets ──
            # (A) Task classification
            task_X, task_y = [], []
            for ti, task in enumerate(TASKS):
                X, _, _, _ = data[(task, li, ds)]
                task_X.append(X)
                task_y.extend([ti] * X.shape[0])
            task_X = np.vstack(task_X)
            task_y = np.array(task_y)

            # (B) Success/failure classification
            sf_X, sf_y = [], []
            for task in TASKS:
                X, _, _, ep_succ = data[(task, li, ds)]
                for i in range(X.shape[0]):
                    sf_X.append(X[i])
                    sf_y.append(1 if ep_succ[i] else 0)
            sf_X = np.array(sf_X, dtype=np.float32)
            sf_y = np.array(sf_y)

            # (C) Within-assembly success/failure
            asm_X, _, _, asm_succ = data[("assembly-v3", li, ds)]
            asm_y = np.array([1 if s else 0 for s in asm_succ])

            datasets = [
                ("Task ID (4-way)", task_X, task_y, "multi"),
                ("Success/Fail (all)", sf_X, sf_y, "binary"),
                ("Asm Succ/Fail", asm_X, asm_y, "binary"),
            ]

            for dname, X_full, y, dtype in datasets:
                if len(np.unique(y)) < 2:
                    print(f"    {dname}: SKIPPED (only one class)")
                    continue

                n_samples = X_full.shape[0]
                d = X_full.shape[1]
                majority = max(np.mean(y == c) for c in np.unique(y))

                # ── (1) Full-space probes ──
                # Logistic regression
                accs_lr = []
                f1s_lr = []
                for tr, te in cv.split(X_full, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_full[tr])
                    X_te = scaler.transform(X_full[te])
                    clf = LogisticRegression(max_iter=3000, C=1.0,
                                             multi_class="multinomial" if dtype == "multi" else "auto")
                    clf.fit(X_tr, y[tr])
                    pred = clf.predict(X_te)
                    accs_lr.append(accuracy_score(y[te], pred))
                    f1s_lr.append(f1_score(y[te], pred, average="macro"))

                # Ridge classifier
                accs_ridge = []
                for tr, te in cv.split(X_full, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_full[tr])
                    X_te = scaler.transform(X_full[te])
                    clf = RidgeClassifier(alpha=1.0)
                    clf.fit(X_tr, y[tr])
                    accs_ridge.append(accuracy_score(y[te], clf.predict(X_te)))

                print(f"\n    {dname} (n={n_samples}, majority={majority:.3f}):")
                print(f"      Full-1024d LogReg:  acc={np.mean(accs_lr):.3f}±{np.std(accs_lr):.3f}, "
                      f"F1={np.mean(f1s_lr):.3f}")
                print(f"      Full-1024d Ridge:   acc={np.mean(accs_ridge):.3f}±{np.std(accs_ridge):.3f}")

                results_table.append({
                    "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                    "method": "Full LogReg", "k": d,
                    "acc": np.mean(accs_lr), "std": np.std(accs_lr),
                    "f1": np.mean(f1s_lr), "n": n_samples,
                })
                results_table.append({
                    "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                    "method": "Full Ridge", "k": d,
                    "acc": np.mean(accs_ridge), "std": np.std(accs_ridge),
                    "f1": np.nan, "n": n_samples,
                })

                # ── (2) PCA subspace probes ──
                _, Vt_full = fast_svd(X_full, k=min(max(K_VALUES), X_full.shape[0]))
                for k in K_VALUES:
                    if k > Vt_full.shape[0]:
                        continue
                    X_pca = X_full @ Vt_full[:k].T

                    accs = []
                    f1s = []
                    for tr, te in cv.split(X_pca, y):
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_pca[tr])
                        X_te = scaler.transform(X_pca[te])
                        clf = LogisticRegression(max_iter=3000, C=1.0,
                                                 multi_class="multinomial" if dtype == "multi" else "auto")
                        clf.fit(X_tr, y[tr])
                        pred = clf.predict(X_te)
                        accs.append(accuracy_score(y[te], pred))
                        f1s.append(f1_score(y[te], pred, average="macro"))

                    print(f"      PCA-{k:3d}d LogReg:   acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
                    results_table.append({
                        "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                        "method": f"PCA", "k": k,
                        "acc": np.mean(accs), "std": np.std(accs),
                        "f1": np.mean(f1s), "n": n_samples,
                    })

                # ── (3) Random subspace probes ──
                rng = np.random.RandomState(42)
                for k in K_VALUES:
                    if k > d:
                        continue
                    accs_rand_runs = []
                    for _ in range(5):  # 5 random draws
                        V_rand = rng.randn(d, k).astype(np.float32)
                        V_rand, _ = np.linalg.qr(V_rand)  # orthonormalize
                        X_rand = X_full @ V_rand

                        accs_r = []
                        for tr, te in cv.split(X_rand, y):
                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X_rand[tr])
                            X_te = scaler.transform(X_rand[te])
                            clf = LogisticRegression(max_iter=3000, C=1.0,
                                                     multi_class="multinomial" if dtype == "multi" else "auto")
                            clf.fit(X_tr, y[tr])
                            accs_r.append(accuracy_score(y[te], clf.predict(X_te)))
                        accs_rand_runs.append(np.mean(accs_r))

                    print(f"      Random-{k:3d}d LogReg: acc={np.mean(accs_rand_runs):.3f}±{np.std(accs_rand_runs):.3f}")
                    results_table.append({
                        "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                        "method": f"Random", "k": k,
                        "acc": np.mean(accs_rand_runs), "std": np.std(accs_rand_runs),
                        "f1": np.nan, "n": n_samples,
                    })

                # ── (4) Conceptor subspace probes ──
                for k in K_VALUES:
                    # Per-task conceptor eigenvectors, concatenated
                    projs = []
                    for task in TASKS:
                        entry = conceptors[(task, li, ds, 1.0)]
                        Vt_task = entry["Vt"]
                        k_use = min(k, Vt_task.shape[0])
                        projs.append(X_full @ Vt_task[:k_use].T)
                    X_conc = np.hstack(projs)

                    accs_c = []
                    f1s_c = []
                    for tr, te in cv.split(X_conc, y):
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_conc[tr])
                        X_te = scaler.transform(X_conc[te])
                        clf = LogisticRegression(max_iter=3000, C=1.0,
                                                 multi_class="multinomial" if dtype == "multi" else "auto")
                        clf.fit(X_tr, y[tr])
                        pred = clf.predict(X_te)
                        accs_c.append(accuracy_score(y[te], pred))
                        f1s_c.append(f1_score(y[te], pred, average="macro"))

                    total_dims = X_conc.shape[1]
                    print(f"      Conceptor-{k:3d}d×4 ({total_dims}d): "
                          f"acc={np.mean(accs_c):.3f}±{np.std(accs_c):.3f}")
                    results_table.append({
                        "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                        "method": f"Conceptor", "k": k,
                        "acc": np.mean(accs_c), "std": np.std(accs_c),
                        "f1": np.mean(f1s_c), "n": n_samples,
                    })

                # ── (5) Single conceptor soft classification ──
                # Project each sample through each task's conceptor, use energy as feature
                for alpha_c in [0.1, 1.0]:
                    energies = []
                    for task in TASKS:
                        entry = conceptors[(task, li, ds, alpha_c)]
                        sigma_t = entry["sigma"]
                        gamma_t = conceptor_eigenvalues(sigma_t, alpha_c)
                        Vt_t = entry["Vt"]
                        k_use = min(len(gamma_t), Vt_t.shape[0])
                        proj = X_full @ Vt_t[:k_use].T  # (n, k_use)
                        energy = (proj ** 2 * gamma_t[:k_use]).sum(axis=1)  # weighted energy
                        energies.append(energy[:, None])
                    X_energy = np.hstack(energies)  # (n, 4)

                    accs_e = []
                    for tr, te in cv.split(X_energy, y):
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_energy[tr])
                        X_te = scaler.transform(X_energy[te])
                        clf = LogisticRegression(max_iter=3000, C=1.0,
                                                 multi_class="multinomial" if dtype == "multi" else "auto")
                        clf.fit(X_tr, y[tr])
                        accs_e.append(accuracy_score(y[te], clf.predict(X_te)))

                    print(f"      Conceptor energy (α={alpha_c}, 4d): "
                          f"acc={np.mean(accs_e):.3f}±{np.std(accs_e):.3f}")
                    results_table.append({
                        "ds": ds, "layer": LAYER_NAMES[li], "dataset": dname,
                        "method": f"ConceptorEnergy(α={alpha_c})", "k": 4,
                        "acc": np.mean(accs_e), "std": np.std(accs_e),
                        "f1": np.nan, "n": n_samples,
                    })

    return results_table


def plot_probe_comparison(results_table):
    """Generate publication-quality accuracy-vs-dimensionality plots."""
    print("\n" + "=" * 70)
    print("PLOTTING PROBE COMPARISON FIGURES")
    print("=" * 70)

    import pandas as pd
    df = pd.DataFrame(results_table)
    df.to_csv(OUT_DIR / "probe_results.csv", index=False)
    print(f"  Saved probe_results.csv ({len(df)} rows)")

    # Plot: accuracy vs k for each (dataset, layer, ds) combo
    # Focus on denoise step 0, all layers, task-ID and success/fail datasets
    for ds in DENOISE_STEPS:
        for dname in df["dataset"].unique():
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
            fig.suptitle(f"{dname} — Denoise Step {ds}", fontsize=12)

            for li_idx, lname in enumerate(["L0", "L5", "L11", "L17"]):
                ax = axes[li_idx]
                sub = df[(df.ds == ds) & (df.layer == lname) & (df.dataset == dname)]

                methods = {"PCA": ("#1f77b4", "-o"),
                           "Random": ("#7f7f7f", "--x"),
                           "Conceptor": ("#d62728", "-s")}

                for method, (color, style) in methods.items():
                    msub = sub[sub.method == method].sort_values("k")
                    if not msub.empty:
                        ax.errorbar(msub.k, msub.acc, yerr=msub["std"],
                                    color=color, fmt=style, lw=1.5, capsize=3,
                                    label=method if li_idx == 0 else None, markersize=5)

                # Full-space baseline
                full_lr = sub[sub.method == "Full LogReg"]
                if not full_lr.empty:
                    ax.axhline(full_lr.iloc[0]["acc"], color="green", ls="--", lw=1, alpha=0.7,
                               label="Full 1024d" if li_idx == 0 else None)

                # Majority baseline
                majority = max(sub["n"].iloc[0] if len(sub) > 0 else 1, 1)
                # Can't compute from table easily, skip

                ax.set_title(lname)
                ax.set_xlabel("Dimensions (k)")
                if li_idx == 0:
                    ax.set_ylabel("Accuracy")
                ax.set_xscale("log")
                ax.grid(True, alpha=0.3)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", fontsize=8,
                       bbox_to_anchor=(0.99, 0.95))
            plt.tight_layout(rect=[0, 0, 0.92, 0.93])
            safe_dname = dname.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(OUT_DIR / f"probe_comparison_{safe_dname}_ds{ds}.png")
            plt.close(fig)
            print(f"  Saved probe_comparison_{safe_dname}_ds{ds}.png")

    # Summary figure: best layer per method for task ID
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Best Probe Accuracy Across Layers — Task ID (4-way)", fontsize=12)

    for di, ds in enumerate(DENOISE_STEPS):
        ax = axes[di]
        sub = df[(df.ds == ds) & (df.dataset == "Task ID (4-way)")]

        layer_xs = [0, 5, 11, 17]
        lname_map = {"L0": 0, "L5": 5, "L11": 11, "L17": 17}

        for method, color in [("PCA", "#1f77b4"), ("Random", "#7f7f7f"), ("Conceptor", "#d62728")]:
            msub = sub[(sub.method == method) & (sub.k == 20)]
            if not msub.empty:
                accs = [msub[msub.layer == ln].iloc[0]["acc"] if len(msub[msub.layer == ln]) > 0 else np.nan
                        for ln in ["L0", "L5", "L11", "L17"]]
                ax.plot(layer_xs, accs, 'o-', color=color, lw=2, label=f"{method} (k=20)")

        # Full baseline
        full_sub = sub[sub.method == "Full LogReg"]
        if not full_sub.empty:
            accs_full = [full_sub[full_sub.layer == ln].iloc[0]["acc"] if len(full_sub[full_sub.layer == ln]) > 0 else np.nan
                         for ln in ["L0", "L5", "L11", "L17"]]
            ax.plot(layer_xs, accs_full, 's--', color="green", lw=2, label="Full LogReg (1024d)")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Denoise Step {ds}")
        ax.set_xticks(layer_xs)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "probe_across_layers_taskid.png")
    plt.close(fig)
    print("  Saved probe_across_layers_taskid.png")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CONCEPTOR DIAGNOSTIC v2 — 15 envs, all layers, probe baselines")
    print("=" * 70)

    # Step 1: Download all data
    print("\nSTEP 1: DOWNLOADING ACTIVATIONS")
    data = download_all_data()

    # Step 2: Compute conceptors
    conceptors = run_conceptor_analysis(data)

    # Step 3: Spectra plots
    plot_spectra(conceptors)

    # Step 4: Similarity analysis
    plot_similarity(data, conceptors)

    # Step 5: Boolean operations
    plot_boolean_ops(data, conceptors)

    # Step 6: Probe comparison
    results_table = run_probe_comparison(data, conceptors)

    # Step 7: Probe comparison plots
    plot_probe_comparison(results_table)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"Results saved to: {OUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
