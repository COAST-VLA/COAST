#!/usr/bin/env python3
"""
Optimize Steering Parameters by Maximizing Steered-Failure→Success Overlap
==========================================================================

Principle: apply h'_f = h_f @ [(1-β)I + β·C_c]^T to failure activations,
then find (α, β) that maximizes overlap(C_s, C'_f) — i.e., steered failure
activations look most like success.

Two approaches:
  (A) Approximate:  C'_f ≈ M · C_f · M^T  (fast, conceptor-only)
  (B) Exact:        Recover R_f from C_f, compute R'_f = M·R_f·M^T,
                    then C'_f = R'_f · (R'_f + α⁻²I)⁻¹  (correct)

Validate against actual rollout success rates from the LIBERO sweep.
"""

import json
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

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
OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/shared/analysis_output"

LAYERS = [5, 11, 17]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
BETAS_COARSE = [0.1, 0.3, 0.5]
BETAS_FINE = np.arange(0.01, 0.81, 0.01).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────────────

def overlap(A, B):
    """Normalised similarity sim(A, B) = tr(AB) / sqrt(tr(A²)tr(B²))."""
    num = float(np.einsum("ij,ji->", A, B))
    na = float(np.einsum("ij,ji->", A, A))
    nb = float(np.einsum("ij,ji->", B, B))
    if na * nb == 0:
        return 0.0
    return num / np.sqrt(na * nb)


def steering_matrix(Cc, beta):
    d = Cc.shape[0]
    return (1 - beta) * np.eye(d) + beta * Cc


def recover_correlation(C, alpha):
    """Recover R from C and α:  R = α⁻² · C · (I - C)⁻¹."""
    d = C.shape[0]
    I = np.eye(d)
    # Regularize (I - C) to avoid singularity
    ImC = I - C + 1e-10 * I
    return (alpha ** -2) * C @ np.linalg.inv(ImC)


def build_conceptor_from_R(R, alpha):
    """C = R · (R + α⁻²I)⁻¹."""
    d = R.shape[0]
    return R @ np.linalg.inv(R + (alpha ** -2) * np.eye(d))


def contrastive_conceptor(Cs, Cf):
    d = Cs.shape[0]
    C_not_f = np.eye(d) - Cf
    inner = Cs + C_not_f - Cs @ C_not_f + 1e-8 * np.eye(d)
    return Cs @ np.linalg.inv(inner) @ C_not_f


# ──────────────────────────────────────────────────────────────────────────────
# Steered overlap computation
# ──────────────────────────────────────────────────────────────────────────────

def steered_overlap_approx(Cs, Cf, Cc, beta):
    """Approximate: overlap(C_s, M·C_f·M^T)."""
    M = steering_matrix(Cc, beta)
    Cf_steered = M @ Cf @ M.T
    return overlap(Cs, Cf_steered)


def steered_overlap_exact(Cs, Cf, Cc, alpha, beta):
    """Exact: recover R_f, compute R'_f = M·R_f·M^T, rebuild C'_f, measure overlap."""
    M = steering_matrix(Cc, beta)
    R_f = recover_correlation(Cf, alpha)
    R_f_steered = M @ R_f @ M.T
    # Symmetrize (numerical safety)
    R_f_steered = 0.5 * (R_f_steered + R_f_steered.T)
    C_f_steered = build_conceptor_from_R(R_f_steered, alpha)
    return overlap(Cs, C_f_steered)


def optimal_beta_for_overlap(Cs, Cf, Cc, alpha, method="exact", beta_range=(0.01, 0.8)):
    """Find beta that maximizes steered-failure→success overlap."""
    def neg_overlap(b):
        if method == "exact":
            return -steered_overlap_exact(Cs, Cf, Cc, alpha, b)
        else:
            return -steered_overlap_approx(Cs, Cf, Cc, b)

    result = minimize_scalar(neg_overlap, bounds=beta_range, method="bounded")
    return result.x, -result.fun


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

def load_libero():
    print("Loading LIBERO conceptors...")
    npz = np.load(LIBERO_NPZ, allow_pickle=True)

    tasks = sorted(set(re.match(r"^(.+?)__L\d+__", k).group(1)
                       for k in npz.files if re.match(r"^(.+?)__L\d+__", k)))
    print(f"  {len(tasks)} tasks")

    print("Loading LIBERO sweep results...")
    cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
    sr = {}

    for rd in os.listdir(LIBERO_RESULTS):
        summary_path = os.path.join(LIBERO_RESULTS, rd, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            data = json.load(f)

        task_name = None
        for t in tasks:
            if t.startswith(rd) or rd.startswith(t[:60]):
                task_name = t
                break
        if not task_name:
            continue

        for entry in data["conditions"]:
            m = cond_re.match(entry["condition"])
            if m:
                sr[(task_name, int(m.group(1)), float(m.group(2)), float(m.group(3)))] = \
                    float(entry["success_rate"])

    print(f"  {len(sr)} conditions loaded")
    return npz, tasks, sr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz, tasks, sr = load_libero()

    # ── 1. Compute steered overlap for all (task, L, α, β) ───────────────
    print("\nComputing steered-failure→success overlap...")
    rows = []

    for task in tasks:
        for L in LAYERS:
            for a in ALPHAS:
                Cs_key = f"{task}__L{L}__{a}__C_success"
                Cf_key = f"{task}__L{L}__{a}__C_failure"
                Cc_key = f"{task}__L{L}__{a}__C_contrastive"
                if any(k not in npz for k in [Cs_key, Cf_key, Cc_key]):
                    continue

                Cs, Cf, Cc = npz[Cs_key], npz[Cf_key], npz[Cc_key]
                orig_ovl = overlap(Cs, Cf)

                for b in BETAS_COARSE:
                    key = (task, L, a, b)
                    if key not in sr:
                        continue

                    ovl_approx = steered_overlap_approx(Cs, Cf, Cc, b)
                    ovl_exact = steered_overlap_exact(Cs, Cf, Cc, a, b)

                    rows.append({
                        "task": task, "layer": L, "alpha": a, "beta": b,
                        "success_rate": sr[key],
                        "orig_overlap": orig_ovl,
                        "steered_ovl_approx": ovl_approx,
                        "steered_ovl_exact": ovl_exact,
                        "ovl_gain_approx": ovl_approx - orig_ovl,
                        "ovl_gain_exact": ovl_exact - orig_ovl,
                    })

    print(f"  {len(rows)} data points")

    # ── 2. Correlation analysis ──────────────────────────────────────────
    sr_vec = np.array([r["success_rate"] for r in rows])

    print(f"\n{'='*70}")
    print("Correlation of steered overlap metrics with rollout SR")
    print(f"{'='*70}")
    metrics = ["orig_overlap", "steered_ovl_approx", "steered_ovl_exact",
               "ovl_gain_approx", "ovl_gain_exact"]

    correlations = {}
    for mn in metrics:
        vals = np.array([r[mn] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(sr_vec)
        rho, p = stats.spearmanr(vals[mask], sr_vec[mask])
        pr, pp = stats.pearsonr(vals[mask], sr_vec[mask])
        correlations[mn] = {"spearman_rho": rho, "spearman_p": p,
                           "pearson_r": pr, "pearson_p": pp}
        print(f"  {mn:<25s}  Spearman ρ={rho:>7.3f}  (p={p:.1e})  "
              f"Pearson r={pr:>7.3f}")

    # ── 3. Per-task: find math-optimal (α, β) by max steered overlap ────
    print(f"\n{'='*70}")
    print("Per-task: maximize steered overlap (exact) to select parameters")
    print(f"{'='*70}")

    n_match_L, n_match_a, n_match_b, n_match_exact = 0, 0, 0, 0
    math_sr_list, actual_sr_list = [], []
    task_results = []

    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue

        actual_best = max(task_rows, key=lambda r: r["success_rate"])
        math_best = max(task_rows, key=lambda r: r["steered_ovl_exact"])

        mL = actual_best["layer"] == math_best["layer"]
        mA = actual_best["alpha"] == math_best["alpha"]
        mB = actual_best["beta"] == math_best["beta"]
        n_match_L += mL
        n_match_a += mA
        n_match_b += mB
        n_match_exact += (mL and mA and mB)

        math_sr_list.append(math_best["success_rate"])
        actual_sr_list.append(actual_best["success_rate"])

        task_short = task[:45]
        print(f"\n  {task_short}...")
        print(f"    Actual best:  L={actual_best['layer']:>2d}  α={actual_best['alpha']:<5g}  "
              f"β={actual_best['beta']:<4g}  SR={actual_best['success_rate']:.3f}")
        print(f"    Math best:    L={math_best['layer']:>2d}  α={math_best['alpha']:<5g}  "
              f"β={math_best['beta']:<4g}  SR={math_best['success_rate']:.3f}  "
              f"steered_ovl={math_best['steered_ovl_exact']:.4f}")
        print(f"    Match: L={'✓' if mL else '✗'}  α={'✓' if mA else '✗'}  β={'✓' if mB else '✗'}")

        task_results.append({
            "task": task_short,
            "actual": actual_best,
            "math": math_best,
        })

    n = len(tasks)
    print(f"\n  Summary ({n} tasks):")
    print(f"    Layer match:  {n_match_L}/{n} ({100*n_match_L/n:.0f}%)")
    print(f"    Alpha match:  {n_match_a}/{n} ({100*n_match_a/n:.0f}%)")
    print(f"    Beta match:   {n_match_b}/{n} ({100*n_match_b/n:.0f}%)")
    print(f"    Exact match:  {n_match_exact}/{n} ({100*n_match_exact/n:.0f}%)")
    print(f"    Mean SR (actual best): {np.mean(actual_sr_list):.3f}")
    print(f"    Mean SR (math best):   {np.mean(math_sr_list):.3f}")
    print(f"    Gap:                   {np.mean(actual_sr_list) - np.mean(math_sr_list):.3f}")

    # ── 4. Fine β sweep: for each (task, L, α), find optimal β ──────────
    print(f"\n{'='*70}")
    print("Fine β optimization (exact overlap, per task at best layer+alpha)")
    print(f"{'='*70}")

    fine_results = []
    for task in tasks:
        # Use the math-best (L, α) from above
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        math_best = max(task_rows, key=lambda r: r["steered_ovl_exact"])
        L, a = math_best["layer"], math_best["alpha"]

        Cs = npz[f"{task}__L{L}__{a}__C_success"]
        Cf = npz[f"{task}__L{L}__{a}__C_failure"]
        Cc = npz[f"{task}__L{L}__{a}__C_contrastive"]

        opt_beta, opt_ovl = optimal_beta_for_overlap(Cs, Cf, Cc, a, method="exact")

        # Also sweep fine betas and record
        beta_curve = []
        for b in BETAS_FINE:
            ov = steered_overlap_exact(Cs, Cf, Cc, a, b)
            beta_curve.append((b, ov))

        task_short = task[:45]
        print(f"  {task_short}  L={L} α={a:g}  β*={opt_beta:.3f}  ovl*={opt_ovl:.4f}")
        fine_results.append({
            "task": task, "layer": L, "alpha": a,
            "opt_beta": opt_beta, "opt_overlap": opt_ovl,
            "beta_curve": beta_curve,
        })

    # ── 5. Generate figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.30})
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # (a) Steered overlap (exact) vs actual SR
    ax = axes[0, 0]
    vals = np.array([r["steered_ovl_exact"] for r in rows])
    for L, c in [(5, "#1f77b4"), (11, "#2ca02c"), (17, "#ff7f0e")]:
        mask = np.array([r["layer"] == L for r in rows])
        ax.scatter(vals[mask], sr_vec[mask], c=c, alpha=0.4, s=15, label=f"L={L}")
    rho = correlations["steered_ovl_exact"]["spearman_rho"]
    ax.set_xlabel("Steered Overlap (exact)", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title(f"(a) Steered Overlap vs SR  (ρ={rho:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # (b) Overlap gain vs actual SR
    ax = axes[0, 1]
    vals = np.array([r["ovl_gain_exact"] for r in rows])
    for L, c in [(5, "#1f77b4"), (11, "#2ca02c"), (17, "#ff7f0e")]:
        mask = np.array([r["layer"] == L for r in rows])
        ax.scatter(vals[mask], sr_vec[mask], c=c, alpha=0.4, s=15, label=f"L={L}")
    rho = correlations["ovl_gain_exact"]["spearman_rho"]
    ax.set_xlabel("Overlap Gain (steered − orig)", fontsize=10)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title(f"(b) Overlap Gain vs SR  (ρ={rho:.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # (c) Approx vs exact comparison
    ax = axes[0, 2]
    approx = np.array([r["steered_ovl_approx"] for r in rows])
    exact = np.array([r["steered_ovl_exact"] for r in rows])
    ax.scatter(approx, exact, alpha=0.3, s=10, c="#555")
    ax.plot([0.6, 1.05], [0.6, 1.05], "r--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Steered Overlap (approx)", fontsize=10)
    ax.set_ylabel("Steered Overlap (exact)", fontsize=10)
    ax.set_title("(c) Approx vs Exact", fontsize=11, fontweight="bold")

    # (d) Per-task actual vs math-optimal SR
    ax = axes[1, 0]
    x = np.arange(n)
    ax.bar(x - 0.15, actual_sr_list, 0.3, color="#2ca02c", alpha=0.8, label="Actual best")
    ax.bar(x + 0.15, math_sr_list, 0.3, color="#ff7f0e", alpha=0.8, label="Math best (max steered ovl)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=8)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title(f"(d) Actual vs Math-Optimal SR (gap={np.mean(actual_sr_list)-np.mean(math_sr_list):.3f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # (e) Fine β curves (overlap vs β) for a few tasks
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(fine_results))))
    for i, fr in enumerate(fine_results[:6]):
        betas, ovls = zip(*fr["beta_curve"])
        ax.plot(betas, ovls, color=colors[i], lw=1.2, alpha=0.8,
                label=f"T{i} α={fr['alpha']:g}")
        ax.axvline(fr["opt_beta"], color=colors[i], ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel(r"$\beta$", fontsize=11)
    ax.set_ylabel("Steered Overlap (exact)", fontsize=10)
    ax.set_title("(e) Overlap vs β (fine sweep)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # (f) Optimal β distribution
    ax = axes[1, 2]
    opt_betas = [fr["opt_beta"] for fr in fine_results]
    ax.hist(opt_betas, bins=20, range=(0, 0.8), color="#4e79a7", edgecolor="black", linewidth=0.6)
    ax.axvline(np.mean(opt_betas), color="red", ls="--", lw=1.5,
               label=f"mean={np.mean(opt_betas):.3f}")
    ax.axvline(np.median(opt_betas), color="orange", ls="--", lw=1.5,
               label=f"median={np.median(opt_betas):.3f}")
    ax.set_xlabel(r"Optimal $\beta$", fontsize=11)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("(f) Distribution of Math-Optimal β", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle("Steering Parameter Optimization via Steered-Failure→Success Overlap",
                 fontsize=14, fontweight="bold")

    out_pdf = os.path.join(OUTPUT_DIR, "steered_overlap_optimization.pdf")
    out_png = os.path.join(OUTPUT_DIR, "steered_overlap_optimization.png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")

    # ── Save JSON ────────────────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "steered_overlap_optimization.json")
    summary = {
        "correlations": {k: {sk: float(sv) for sk, sv in v.items()}
                        for k, v in correlations.items()},
        "per_task": [{
            "task": tr["task"][:50],
            "actual_best": {"L": tr["actual"]["layer"], "a": tr["actual"]["alpha"],
                           "b": tr["actual"]["beta"], "SR": tr["actual"]["success_rate"]},
            "math_best": {"L": tr["math"]["layer"], "a": tr["math"]["alpha"],
                         "b": tr["math"]["beta"], "SR": tr["math"]["success_rate"],
                         "steered_ovl": tr["math"]["steered_ovl_exact"]},
        } for tr in task_results],
        "fine_beta": [{
            "task": fr["task"][:50], "L": fr["layer"], "a": fr["alpha"],
            "opt_beta": round(fr["opt_beta"], 4), "opt_overlap": round(fr["opt_overlap"], 4),
        } for fr in fine_results],
        "summary": {
            "mean_sr_actual": float(np.mean(actual_sr_list)),
            "mean_sr_math": float(np.mean(math_sr_list)),
            "gap": float(np.mean(actual_sr_list) - np.mean(math_sr_list)),
            "layer_match": f"{n_match_L}/{n}",
            "alpha_match": f"{n_match_a}/{n}",
            "beta_match": f"{n_match_b}/{n}",
            "exact_match": f"{n_match_exact}/{n}",
        },
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
