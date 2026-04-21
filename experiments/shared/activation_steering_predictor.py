#!/usr/bin/env python3
"""
Direct Activation-Space Steering Predictor
==========================================

Instead of using conceptor overlap as a proxy, directly compute:
  h'_f = h_f @ [(1-β)I + β·C_c]^T
then measure how close h'_f is to h_s using multiple metrics.

This should be a stronger predictor because it captures centroid
shifts and distributional properties, not just subspace overlap.

Metrics:
  1. Centroid cosine:     cos(mean(h'_f), mean(h_s))
  2. Centroid L2 shift:   ||mean(h'_f) - mean(h_s)||  /  ||mean(h_f) - mean(h_s)||
  3. Mean pairwise cos:   mean_i mean_j cos(h'_f_i, h_s_j)
  4. MMD (approx):        ||mean(h'_f) - mean(h_s)||² - ||mean(h_f) - mean(h_s)||²
  5. Silhouette improvement: how much better separated after steering

Validated against actual LIBERO sweep success rates.
"""

import json
import os
import glob
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

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

ACTIVATIONS_DIR = "/vast/projects/ungar/stellar/miaom/.cache/openpi/activations/pi05_libero_2000_15env/openpi-libero-2000"
LIBERO_RESULTS = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results"
OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/shared/analysis_output"

LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}  # model_layer → npz index
LAYERS = [5, 11, 17]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
BETAS = [0.1, 0.3, 0.5]
DENOISE_STEP = 0  # first denoising step


# ──────────────────────────────────────────────────────────────────────────────
# Load raw activations
# ──────────────────────────────────────────────────────────────────────────────

def load_task_activations(task_dir, layer):
    """Load success/failure activations for one task at one layer.

    Returns: (X_success, X_failure) each shape (N, 1024)
    """
    lidx = LAYER_MAP[layer]
    episodes = sorted([e for e in os.listdir(task_dir) if e.startswith("episode_")])

    X_success, X_failure = [], []

    for ep in episodes:
        ep_dir = os.path.join(task_dir, ep)
        meta_path = os.path.join(ep_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        is_success = meta.get("episode_success", False)

        # Load all steps
        steps = sorted([s for s in os.listdir(ep_dir) if s.startswith("step_")])
        for step in steps:
            res_path = os.path.join(ep_dir, step, "suffix_residual.npz")
            if not os.path.exists(res_path):
                continue
            data = np.load(res_path)
            residual = data["all_suffix_residual"]  # (10, 4, N_tokens, 1024)
            # Mean-pool across action tokens at denoising step 0
            act = residual[DENOISE_STEP, lidx, :, :].mean(axis=0)  # (1024,)
            if is_success:
                X_success.append(act)
            else:
                X_failure.append(act)

    return np.stack(X_success), np.stack(X_failure)


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor math
# ──────────────────────────────────────────────────────────────────────────────

def compute_conceptor(X, alpha=1.0):
    d = X.shape[1]
    R = (X.T @ X) / X.shape[0]
    reg = (alpha ** -2) * np.eye(d)
    return R @ np.linalg.inv(R + reg)


def contrastive_conceptor(Cs, Cf):
    d = Cs.shape[0]
    C_not_f = np.eye(d) - Cf
    inner = Cs + C_not_f - Cs @ C_not_f + 1e-8 * np.eye(d)
    return Cs @ np.linalg.inv(inner) @ C_not_f


def steering_matrix(Cc, beta):
    d = Cc.shape[0]
    return (1 - beta) * np.eye(d) + beta * Cc


# ──────────────────────────────────────────────────────────────────────────────
# Activation-space similarity metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_activation_metrics(X_success, X_failure, Cc, beta):
    """
    Steer failure activations and measure closeness to success.
    Also steer success activations and measure self-consistency.
    """
    M = steering_matrix(Cc, beta)
    d = Cc.shape[0]

    # Steer failure → success direction
    X_f_steered = X_failure @ M.T  # (N_f, 1024)

    # Centroids
    mu_s = X_success.mean(axis=0)
    mu_f = X_failure.mean(axis=0)
    mu_fs = X_f_steered.mean(axis=0)

    # 1. Centroid cosine similarity (steered failure vs success)
    cos_orig = 1.0 - cosine_dist(mu_f, mu_s)
    cos_steered = 1.0 - cosine_dist(mu_fs, mu_s)

    # 2. Centroid L2 distance ratio (closer to 0 = steered failure is at success)
    dist_orig = float(np.linalg.norm(mu_f - mu_s))
    dist_steered = float(np.linalg.norm(mu_fs - mu_s))
    dist_ratio = dist_steered / (dist_orig + 1e-10)  # <1 means improvement

    # 3. Mean pairwise cosine (sample-level)
    # Subsample to keep computation tractable
    n_sub = min(200, X_f_steered.shape[0], X_success.shape[0])
    idx_f = np.random.choice(X_f_steered.shape[0], n_sub, replace=False)
    idx_s = np.random.choice(X_success.shape[0], n_sub, replace=False)
    Xf_sub = X_f_steered[idx_f]
    Xs_sub = X_success[idx_s]
    # Normalize
    Xf_norm = Xf_sub / (np.linalg.norm(Xf_sub, axis=1, keepdims=True) + 1e-10)
    Xs_norm = Xs_sub / (np.linalg.norm(Xs_sub, axis=1, keepdims=True) + 1e-10)
    pairwise_cos = float(np.mean(Xf_norm @ Xs_norm.T))

    # 4. Projection onto contrastive direction
    # How much does steering move failure towards the success-unique subspace?
    # Use top eigenvector of Cc as contrastive direction
    eigvals, eigvecs = np.linalg.eigh(Cc)
    top_dir = eigvecs[:, -1]  # top eigenvector
    proj_f_orig = float(np.mean(X_failure @ top_dir))
    proj_f_steered = float(np.mean(X_f_steered @ top_dir))
    proj_s = float(np.mean(X_success @ top_dir))
    proj_shift_toward_success = abs(proj_f_steered - proj_s) - abs(proj_f_orig - proj_s)
    # Negative = improvement (steered failure is closer to success)

    # 5. Mahalanobis-like: use success covariance
    cov_s = np.cov(X_success.T) + 1e-6 * np.eye(d)
    # Use PCA-reduced version for stability
    U, S, Vt = np.linalg.svd(cov_s, full_matrices=False)
    k = min(50, d)  # top 50 PCs
    proj_matrix = U[:, :k] / np.sqrt(S[:k] + 1e-8)  # whitening
    mu_s_w = mu_s @ proj_matrix
    mu_f_w = mu_f @ proj_matrix
    mu_fs_w = mu_fs @ proj_matrix
    mahal_orig = float(np.linalg.norm(mu_f_w - mu_s_w))
    mahal_steered = float(np.linalg.norm(mu_fs_w - mu_s_w))
    mahal_ratio = mahal_steered / (mahal_orig + 1e-10)

    # 6. Success self-consistency: steer success and check it doesn't drift
    X_s_steered = X_success @ M.T
    mu_ss = X_s_steered.mean(axis=0)
    success_drift = float(np.linalg.norm(mu_ss - mu_s)) / (float(np.linalg.norm(mu_s)) + 1e-10)

    return {
        "cos_steered": cos_steered,
        "cos_gain": cos_steered - cos_orig,
        "dist_ratio": dist_ratio,
        "pairwise_cos": pairwise_cos,
        "proj_shift": proj_shift_toward_success,
        "mahal_ratio": mahal_ratio,
        "success_drift": success_drift,
        # Composite: alignment gain minus disruption
        "net_benefit": (1 - dist_ratio) - success_drift,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load sweep results
# ──────────────────────────────────────────────────────────────────────────────

def load_sweep_results():
    cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
    sr = {}
    for rd in os.listdir(LIBERO_RESULTS):
        summary_path = os.path.join(LIBERO_RESULTS, rd, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)
        for entry in data["conditions"]:
            m = cond_re.match(entry["condition"])
            if m:
                task = rd
                L = int(m.group(1))
                a = float(m.group(2))
                b = float(m.group(3))
                sr[(task, L, a, b)] = float(entry["success_rate"])
    return sr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    sr = load_sweep_results()
    tasks = sorted(os.listdir(ACTIVATIONS_DIR))

    # Map between activation dir names and result dir names
    task_to_result_dir = {}
    result_dirs = [rd for rd in os.listdir(LIBERO_RESULTS)
                   if os.path.isdir(os.path.join(LIBERO_RESULTS, rd))]
    for task in tasks:
        for rd in result_dirs:
            if task.startswith(rd) or rd.startswith(task[:55]):
                task_to_result_dir[task] = rd
                break

    print(f"Found {len(tasks)} tasks, {len(task_to_result_dir)} matched to results")

    # ── Compute metrics for all (task, L, α, β) ─────────────────────────
    print("\nLoading activations and computing steering metrics...")
    rows = []

    for task in tasks:
        rd = task_to_result_dir.get(task)
        if not rd:
            continue

        task_dir = os.path.join(ACTIVATIONS_DIR, task)
        task_short = task[:50]

        for L in LAYERS:
            print(f"  {task_short}  L={L}...", end="", flush=True)
            X_s, X_f = load_task_activations(task_dir, L)
            print(f"  {X_s.shape[0]}S/{X_f.shape[0]}F", end="")

            for a in ALPHAS:
                Cs = compute_conceptor(X_s, a)
                Cf = compute_conceptor(X_f, a)
                Cc = contrastive_conceptor(Cs, Cf)

                for b in BETAS:
                    key = (rd, L, a, b)
                    if key not in sr:
                        continue

                    metrics = compute_activation_metrics(X_s, X_f, Cc, b)
                    rows.append({
                        "task": task, "layer": L, "alpha": a, "beta": b,
                        "success_rate": sr[key],
                        **metrics,
                    })
            print()

    print(f"\n{len(rows)} data points")

    # ── Correlation analysis ─────────────────────────────────────────────
    sr_vec = np.array([r["success_rate"] for r in rows])
    metric_names = ["cos_steered", "cos_gain", "dist_ratio", "pairwise_cos",
                    "proj_shift", "mahal_ratio", "success_drift", "net_benefit"]

    print(f"\n{'='*80}")
    print(f"{'Metric':<20s} {'Spearman ρ':>10s} {'p-value':>10s} {'Pearson r':>10s} {'Direction':>12s}")
    print(f"{'='*80}")

    correlations = {}
    for mn in metric_names:
        vals = np.array([r[mn] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(sr_vec)
        if mask.sum() < 10:
            continue
        rho, p = stats.spearmanr(vals[mask], sr_vec[mask])
        pr, pp = stats.pearsonr(vals[mask], sr_vec[mask])
        direction = "↑ higher better" if rho > 0 else "↓ lower better"
        correlations[mn] = {"spearman_rho": rho, "spearman_p": p,
                           "pearson_r": pr, "pearson_p": pp}
        print(f"  {mn:<20s} {rho:>10.3f} {p:>10.1e} {pr:>10.3f} {direction:>12s}")

    # ── Fixed-beta analysis (β=0.1): focus on layer+alpha selection ──────
    print(f"\n{'='*80}")
    print("Fixed β=0.1: correlation for LAYER + ALPHA selection only")
    print(f"{'='*80}")
    for beta_fix in [0.1, 0.3]:
        sub = [r for r in rows if r["beta"] == beta_fix]
        sr_sub = np.array([r["success_rate"] for r in sub])
        print(f"\n  β={beta_fix} ({len(sub)} points):")
        for mn in metric_names:
            vals = np.array([r[mn] for r in sub])
            mask = np.isfinite(vals) & np.isfinite(sr_sub)
            if mask.sum() < 5:
                continue
            rho, p = stats.spearmanr(vals[mask], sr_sub[mask])
            if abs(rho) > 0.2:
                print(f"    {mn:<20s} ρ={rho:>7.3f}  (p={p:.1e})")

    # ── Per-task: math-optimal vs actual-optimal ─────────────────────────
    best_metric = max(correlations.keys(),
                     key=lambda m: abs(correlations[m]["spearman_rho"]))
    sign = 1 if correlations[best_metric]["spearman_rho"] > 0 else -1

    print(f"\n{'='*80}")
    print(f"Per-task parameter selection using best metric: {best_metric}")
    print(f"(ρ = {correlations[best_metric]['spearman_rho']:.3f})")
    print(f"{'='*80}")

    n_match_L, n_match_a, n_match_b = 0, 0, 0
    n_match_La = 0
    math_sr_list, actual_sr_list = [], []

    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue

        actual_best = max(task_rows, key=lambda r: r["success_rate"])
        math_best = max(task_rows, key=lambda r: sign * r[best_metric])

        mL = actual_best["layer"] == math_best["layer"]
        mA = actual_best["alpha"] == math_best["alpha"]
        mB = actual_best["beta"] == math_best["beta"]
        n_match_L += mL
        n_match_a += mA
        n_match_b += mB
        n_match_La += (mL and mA)
        math_sr_list.append(math_best["success_rate"])
        actual_sr_list.append(actual_best["success_rate"])

        task_short = task[:45]
        print(f"\n  {task_short}...")
        print(f"    Actual: L={actual_best['layer']:>2d}  α={actual_best['alpha']:<5g}  "
              f"β={actual_best['beta']:<4g}  SR={actual_best['success_rate']:.3f}")
        print(f"    Math:   L={math_best['layer']:>2d}  α={math_best['alpha']:<5g}  "
              f"β={math_best['beta']:<4g}  SR={math_best['success_rate']:.3f}  "
              f"{best_metric}={math_best[best_metric]:.4f}")
        print(f"    Match: L={'✓' if mL else '✗'}  α={'✓' if mA else '✗'}  β={'✓' if mB else '✗'}")

    n = len([t for t in tasks if any(r["task"] == t for r in rows)])
    print(f"\n  Summary ({n} tasks):")
    print(f"    Layer match:    {n_match_L}/{n} ({100*n_match_L/n:.0f}%)")
    print(f"    Alpha match:    {n_match_a}/{n} ({100*n_match_a/n:.0f}%)")
    print(f"    L+A match:      {n_match_La}/{n} ({100*n_match_La/n:.0f}%)")
    print(f"    Beta match:     {n_match_b}/{n} ({100*n_match_b/n:.0f}%)")
    print(f"    Mean SR actual: {np.mean(actual_sr_list):.3f}")
    print(f"    Mean SR math:   {np.mean(math_sr_list):.3f}")
    print(f"    Gap:            {np.mean(actual_sr_list) - np.mean(math_sr_list):.3f}")

    # ── Generate figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.30)

    # Top row: scatter of top 4 metrics vs SR
    top4 = sorted(correlations.keys(),
                  key=lambda m: abs(correlations[m]["spearman_rho"]),
                  reverse=True)[:4]

    for idx, mn in enumerate(top4):
        ax = fig.add_subplot(gs[0, idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        vals = np.array([r[mn] for r in rows])
        for L, c in [(5, "#1f77b4"), (11, "#2ca02c"), (17, "#ff7f0e")]:
            mask = np.array([r["layer"] == L for r in rows])
            ax.scatter(vals[mask], sr_vec[mask], c=c, alpha=0.35, s=12, label=f"L={L}")
        rho = correlations[mn]["spearman_rho"]
        ax.set_xlabel(mn.replace("_", " ").title(), fontsize=9)
        ax.set_ylabel("Success Rate", fontsize=9)
        ax.set_title(f"ρ = {rho:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, frameon=False)

    # Bottom-left: actual vs math SR per task
    ax = fig.add_subplot(gs[1, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x = np.arange(n)
    ax.bar(x - 0.15, actual_sr_list, 0.3, color="#2ca02c", alpha=0.8, label="Actual best")
    ax.bar(x + 0.15, math_sr_list, 0.3, color="#ff7f0e", alpha=0.8,
           label=f"Math best ({best_metric})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=8)
    ax.set_ylabel("Success Rate", fontsize=9)
    ax.set_title(f"Actual vs Math-Optimal (gap={np.mean(actual_sr_list)-np.mean(math_sr_list):.3f})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)

    # Bottom-middle: heatmap of metric correlations by layer
    ax = fig.add_subplot(gs[1, 1:3])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    heat_data = []
    for L in LAYERS:
        layer_rows = [r for r in rows if r["layer"] == L]
        sr_l = np.array([r["success_rate"] for r in layer_rows])
        row_data = []
        for mn in metric_names:
            vals = np.array([r[mn] for r in layer_rows])
            mask = np.isfinite(vals) & np.isfinite(sr_l)
            if mask.sum() > 5:
                rho, _ = stats.spearmanr(vals[mask], sr_l[mask])
                row_data.append(rho)
            else:
                row_data.append(0)
        heat_data.append(row_data)
    heat_arr = np.array(heat_data)
    im = ax.imshow(heat_arr, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f"L={L}" for L in LAYERS])
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=7)
    for i in range(len(LAYERS)):
        for j in range(len(metric_names)):
            ax.text(j, i, f"{heat_arr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(heat_arr[i,j]) > 0.3 else "black")
    ax.set_title("Spearman ρ by Layer × Metric", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    # Bottom-right: correlation comparison bar chart
    ax = fig.add_subplot(gs[1, 3])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    rhos = [correlations[m]["spearman_rho"] for m in metric_names if m in correlations]
    names = [m for m in metric_names if m in correlations]
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in rhos]
    ax.barh(range(len(names)), rhos, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman ρ with SR", fontsize=9)
    ax.set_title("Metric Ranking", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

    fig.suptitle("Direct Activation-Space Steering Predictor — LIBERO",
                 fontsize=14, fontweight="bold")

    out_pdf = os.path.join(OUTPUT_DIR, "activation_steering_predictor.pdf")
    out_png = os.path.join(OUTPUT_DIR, "activation_steering_predictor.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")

    # ── Save JSON ────────────────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "activation_steering_predictor.json")
    summary = {
        "correlations": {k: {sk: float(sv) for sk, sv in v.items()}
                        for k, v in correlations.items()},
        "best_metric": best_metric,
        "best_rho": float(correlations[best_metric]["spearman_rho"]),
        "mean_sr_actual": float(np.mean(actual_sr_list)),
        "mean_sr_math": float(np.mean(math_sr_list)),
        "gap": float(np.mean(actual_sr_list) - np.mean(math_sr_list)),
        "layer_match": f"{n_match_L}/{n}",
        "alpha_match": f"{n_match_a}/{n}",
        "layer_alpha_match": f"{n_match_La}/{n}",
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
