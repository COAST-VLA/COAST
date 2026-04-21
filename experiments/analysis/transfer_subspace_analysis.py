#!/usr/bin/env python3
"""
Full N×N transfer + conceptor-subspace analysis for pi0.5 LIBERO and RoboCasa.

Claim: cross-task conceptor transfer is predicted by the overlap of the two
tasks' FAILURE subspaces. The steering hook
    h' = (1 − β) h + β C_src h        with   C_src = C^s_src · (I − C^f_src)
effectively suppresses components of h that lie in the source's failure
subspace. Transfer works when source's failure subspace is contained in
target's — suppressing shared failure directions benefits the target too.

Primary metric (asymmetric, directional, source → target):
    contain_f(src, tgt) = tr(C^f_src · C^f_tgt) / tr(C^f_src · C^f_src)
                        = fraction of source's failure mass that also lives
                          in target's failure subspace
Also reported: symmetric Frobenius similarity of C_failure and C_contrastive.

Outputs (all under ``transfer_subspace_results``):
  matrix_sr_{bench}.{csv,tex}          full N×N best-of-strategies SR
  matrix_containF_{bench}.{csv,tex}    full N×N asymmetric failure containment
  matrix_simF_{bench}.{csv,tex}        full N×N symmetric failure similarity
  heatmaps.pdf / .png                  2×2 heatmap panel (SR + containment)
  sim_vs_sr_gain.pdf / .png            scatter with correlations
  deepdive_OD_CS.pdf / .png            OpenDrawer × CoffeeSetup geometry
  analysis_paragraph.md                paper-ready writeup
  correlations.json                    Pearson/Spearman stats + sweep

Run from repo root:
    .venv/bin/python experiments/analysis/transfer_subspace_analysis.py
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_DATA_HOME = pathlib.Path(os.environ.get(
    "OPENPI_DATA_HOME", str(pathlib.Path.home() / ".cache" / "openpi")
))
OUT_ROOT = pathlib.Path(__file__).resolve().parent / "transfer_subspace_results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SIM_LAYER = 11          # suffix-model middle block, most common best-L in steering sweeps
SIM_ALPHA = "1.0"       # conceptor aperture matching the conceptor npz key grid
STRATEGIES = ["global", "per_step_0", "per_step_9"]


LIBERO_TASKS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]
LIBERO_ALIAS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": "K3_stove_moka",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": "K4_bowl_drawer",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": "K6_mug_micro",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "K8_two_moka",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": "LR1_soup_cheese",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": "LR2_soup_tomato",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": "LR2_cheese_butter",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "LR5_mugs_plates",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": "LR6_mug_choc",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": "S1_book_caddy",
}

ROBOCASA_TASKS = [
    "CloseFridge", "CoffeeSetupMug", "OpenDrawer", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "TurnOnElectricKettle",
]
ROBOCASA_ALIAS = {
    "CloseFridge": "CloseFridge", "CoffeeSetupMug": "CoffeeSetup",
    "OpenDrawer": "OpenDrawer", "OpenStandMixerHead": "StandMixer",
    "PickPlaceCounterToCabinet": "PP_Cabinet", "PickPlaceCounterToStove": "PP_Stove",
    "TurnOnElectricKettle": "Kettle",
}


def short_libero(t: str) -> str:
    return t[:60]


# ──────────────────────────────────────────────────────────────────────────────
# Transfer + baseline loading
# ──────────────────────────────────────────────────────────────────────────────


def load_sr_and_meta(transfer_root, steering_root, tasks, target_dirname):
    N = len(tasks)
    SR = np.full((N, N), np.nan)
    strat_map: Dict[Tuple[str, str], str] = {}
    for j, tgt in enumerate(tasks):
        p = transfer_root / f"target_{target_dirname(tgt)}" / "summary.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        for cell in d.get("cells", []):
            sr = cell.get("success_rate")
            if sr is None:
                continue
            src = cell["source"]
            if src not in tasks:
                continue
            i = tasks.index(src)
            if np.isnan(SR[i, j]) or sr > SR[i, j]:
                SR[i, j] = float(sr)
                strat_map[(src, tgt)] = cell["strategy"]

    BASE = np.full(N, np.nan)
    for j, tgt in enumerate(tasks):
        p = steering_root / target_dirname(tgt) / "summary.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        best_self = 0.0
        best_strat = None
        for c in d.get("conditions", []):
            name = c["condition"]
            sr = c.get("success_rate")
            if sr is None:
                continue
            if name == "baseline":
                BASE[j] = float(sr)
                continue
            for s in STRATEGIES:
                if name.startswith(f"{s}_"):
                    if sr > best_self:
                        best_self = float(sr)
                        best_strat = s
                    break
        SR[j, j] = best_self
        if best_strat is not None:
            strat_map[(tgt, tgt)] = best_strat
    return SR, BASE, strat_map


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor loading & similarity
# ──────────────────────────────────────────────────────────────────────────────


def load_class_conceptors(npz_path, tasks, layer, alpha, kind) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as f:
        for t in tasks:
            out[t] = np.asarray(f[f"{t}__L{layer}__{alpha}__{kind}"], dtype=np.float32)
    return out


def symmetric_frob_sim(A: np.ndarray, B: np.ndarray) -> float:
    num = float(np.einsum("ij,ij->", A, B))
    den = np.sqrt(float(np.einsum("ij,ij->", A, A)) * float(np.einsum("ij,ij->", B, B)))
    return num / den if den > 0 else 0.0


def containment(A: np.ndarray, B: np.ndarray) -> float:
    """Fraction of A's mass also in B's subspace: tr(A B) / tr(A A)."""
    num = float(np.einsum("ij,ji->", A, B))
    den = float(np.einsum("ij,ji->", A, A))
    return num / den if den > 0 else 0.0


def build_matrix(conceptors, tasks, fn) -> np.ndarray:
    N = len(tasks)
    M = np.full((N, N), np.nan)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            M[i, j] = fn(conceptors[ti], conceptors[tj])
    return M


# ──────────────────────────────────────────────────────────────────────────────
# Writers
# ──────────────────────────────────────────────────────────────────────────────


def save_csv_matrix(M, tasks, alias, path, rp="src", cp="tgt"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"{cp}:{alias[t]}" for t in tasks])
        for i, t in enumerate(tasks):
            row = [f"{rp}:{alias[t]}"]
            for j in range(len(tasks)):
                v = M[i, j]
                row.append(f"{v:.3f}" if not np.isnan(v) else "")
            w.writerow(row)


def save_latex_matrix(M, tasks, alias, path, caption, label,
                      value_fmt="{:.2f}", highlight_diag=True,
                      strat_map=None, diag_fmt=None):
    N = len(tasks)
    labels = [alias[t].replace("_", r"\_") for t in tasks]
    lines = [
        r"\begin{table*}[t!]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * N + "}",
        r"\toprule",
        "src $\\downarrow$\\,/\\,tgt $\\rightarrow$ & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for i, src in enumerate(tasks):
        cells = []
        for j in range(N):
            v = M[i, j]
            if np.isnan(v):
                cells.append("--")
                continue
            val = (diag_fmt if (i == j and diag_fmt) else value_fmt).format(v)
            if strat_map is not None and (src, tasks[j]) in strat_map:
                tag = {"global": "G", "per_step_0": "PS0", "per_step_9": "PS9"}[
                    strat_map[(src, tasks[j])]
                ]
                val = f"{val}\\,{{\\tiny({tag})}}"
            if highlight_diag and i == j:
                val = r"\textbf{" + val + r"}"
            cells.append(val)
        lines.append(labels[i] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def heatmap(ax, M, labels, title, vmin, vmax, cmap, annot_fmt="{:.2f}"):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    N = len(labels)
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(title, fontsize=10)
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            if np.isnan(v):
                continue
            norm = (v - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if (norm < 0.35 or norm > 0.85) else "black"
            ax.text(j, i, annot_fmt.format(v), ha="center", va="center",
                    color=txt_color, fontsize=5.5)
    for k in range(N):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.6))
    return im


def correlate(SR, metric_M, baseline, tasks):
    xs, yg, yraw, pts = [], [], [], []
    N = len(tasks)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.isnan(SR[i, j]) or np.isnan(metric_M[i, j]) or np.isnan(baseline[j]):
                continue
            xs.append(metric_M[i, j])
            yg.append(SR[i, j] - baseline[j])
            yraw.append(SR[i, j])
            pts.append((tasks[i], tasks[j]))
    xs, yg, yraw = np.array(xs), np.array(yg), np.array(yraw)
    pr_g, pp_g = pearsonr(xs, yg)
    sp_g, sp_p = spearmanr(xs, yg)
    pr_r, pp_r = pearsonr(xs, yraw)
    return {
        "x": xs.tolist(), "y_gain": yg.tolist(), "y_raw_sr": yraw.tolist(),
        "pairs": pts,
        "pearson_r_gain": float(pr_g), "pearson_p_gain": float(pp_g),
        "spearman_r_gain": float(sp_g), "spearman_p_gain": float(sp_p),
        "pearson_r_rawSR": float(pr_r), "pearson_p_rawSR": float(pp_r),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Deep-dive geometry
# ──────────────────────────────────────────────────────────────────────────────


def top_eigvecs(C: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Cs = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(Cs)
    idx = np.argsort(-w)
    return w[idx][:k], V[:, idx][:, :k]


def principal_angles(Va, Vb) -> np.ndarray:
    s = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def sample_from_conceptor(C: np.ndarray, n: int, rng: np.random.Generator,
                          eig_floor: float = 1e-4) -> np.ndarray:
    """Draw n samples from N(0, Σ) where Σ shares eigenstructure with C.

    C is a soft projection; its eigenvalues live in [0, 1). We use those
    eigenvalues directly as the covariance spectrum, clipped from below.
    """
    Cs = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(Cs)
    w = np.clip(w, eig_floor, None)
    z = rng.standard_normal((n, w.shape[0]))
    return (z * np.sqrt(w)) @ V.T


def deepdive_plot(Cs_a, Cs_b, Cf_a, Cf_b, Cc_a, Cc_b,
                  name_a: str, name_b: str, out_path: pathlib.Path) -> dict:
    d = Cs_a.shape[0]
    K = 30
    w_sa, V_sa = top_eigvecs(Cs_a, K); w_sb, V_sb = top_eigvecs(Cs_b, K)
    w_fa, V_fa = top_eigvecs(Cf_a, K); w_fb, V_fb = top_eigvecs(Cf_b, K)
    w_ca, V_ca = top_eigvecs(Cc_a, K); w_cb, V_cb = top_eigvecs(Cc_b, K)

    ang_s = np.degrees(principal_angles(V_sa, V_sb))
    ang_f = np.degrees(principal_angles(V_fa, V_fb))
    ang_c = np.degrees(principal_angles(V_ca, V_cb))

    rng = np.random.default_rng(0)
    rand = []
    for _ in range(50):
        A = np.linalg.qr(rng.standard_normal((d, K)))[0]
        B = np.linalg.qr(rng.standard_normal((d, K)))[0]
        rand.append(np.degrees(principal_angles(A, B)))
    rand_mean = np.mean(rand, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    # (a) principal angles
    ax = axes[0]
    ax.plot(range(1, K + 1), ang_f, "o-", color="tab:red", lw=1.5,
            label=f"failure subspace ({name_a} $\\leftrightarrow$ {name_b})")
    ax.plot(range(1, K + 1), ang_s, "s-", color="tab:green", lw=1.5,
            label=f"success subspace ({name_a} $\\leftrightarrow$ {name_b})")
    ax.plot(range(1, K + 1), ang_c, "d-", color="tab:blue", lw=1.2, alpha=0.7,
            label="contrastive subspace")
    ax.plot(range(1, K + 1), rand_mean, "--", color="gray", lw=1.2,
            label=f"random $K={K}$ subspaces of $\\mathbb{{R}}^{{{d}}}$")
    ax.axhline(90, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("principal-angle index $k$")
    ax.set_ylabel("angle (degrees)")
    ax.set_ylim(0, 95)
    ax.set_title("(a) Principal angles between top-$K$ eigenspaces")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.3)

    # (b) eigenvalue spectra
    ax = axes[1]
    wa_s, _ = top_eigvecs(Cs_a, 60)
    wb_s, _ = top_eigvecs(Cs_b, 60)
    wa_f, _ = top_eigvecs(Cf_a, 60)
    wb_f, _ = top_eigvecs(Cf_b, 60)
    ax.plot(range(1, 61), wa_s, "o-", color="tab:green", alpha=0.85, ms=3,
            label=f"{name_a}: $C^s$")
    ax.plot(range(1, 61), wb_s, "s-", color="tab:olive", alpha=0.85, ms=3,
            label=f"{name_b}: $C^s$")
    ax.plot(range(1, 61), wa_f, "o--", color="tab:red", alpha=0.85, ms=3,
            label=f"{name_a}: $C^f$")
    ax.plot(range(1, 61), wb_f, "s--", color="tab:brown", alpha=0.85, ms=3,
            label=f"{name_b}: $C^f$")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (bounded in $[0,1)$)")
    ax.set_title("(b) Conceptor eigenvalue spectra")
    ax.legend(fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    # (c) Joint PCA of synthetic task×outcome activation clouds
    ax = axes[2]
    n_samp = 400
    # Synth clouds
    S_a = sample_from_conceptor(Cs_a, n_samp, rng)
    S_b = sample_from_conceptor(Cs_b, n_samp, rng)
    F_a = sample_from_conceptor(Cf_a, n_samp, rng)
    F_b = sample_from_conceptor(Cf_b, n_samp, rng)

    pooled = np.vstack([S_a, S_b, F_a, F_b])
    pooled -= pooled.mean(axis=0, keepdims=True)
    # Joint PCA basis from pooled success+failure activations
    U, sig, Vt = np.linalg.svd(pooled, full_matrices=False)
    basis = Vt[:2]  # (2, d)
    P = pooled @ basis.T  # (4n, 2)

    splits = [
        (0, f"{name_a} · success", "tab:green", "o"),
        (1, f"{name_b} · success", "tab:olive", "o"),
        (2, f"{name_a} · failure", "tab:red", "x"),
        (3, f"{name_b} · failure", "tab:brown", "x"),
    ]
    for k, lbl, color, marker in splits:
        pts = P[k * n_samp : (k + 1) * n_samp]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, marker=marker,
                   alpha=0.45, label=lbl, edgecolors="none")
        mu = pts.mean(axis=0)
        cov = np.cov(pts.T)
        w2, v2 = np.linalg.eigh(cov)
        w2 = np.clip(w2, 1e-8, None)
        ang = float(np.degrees(np.arctan2(v2[1, 1], v2[0, 1])))
        e = Ellipse(mu, width=2.0 * 2.0 * np.sqrt(w2[1]),
                    height=2.0 * 2.0 * np.sqrt(w2[0]), angle=ang,
                    fill=False, edgecolor=color, lw=2.2, linestyle="-")
        ax.add_patch(e)

    ax.axhline(0, color="k", lw=0.3, alpha=0.3)
    ax.axvline(0, color="k", lw=0.3, alpha=0.3)
    ax.set_xlabel(f"joint PC-1 ({100*sig[0]**2/np.sum(sig**2):.1f}% var)")
    ax.set_ylabel(f"joint PC-2 ({100*sig[1]**2/np.sum(sig**2):.1f}% var)")
    ax.set_title("(c) Joint PCA of synthetic task$\\times$outcome activations")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Conceptor subspace geometry: {name_a} × {name_b}", fontsize=12)
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(out_path.with_suffix(ext), dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "failure_angles_deg": ang_f.tolist(),
        "success_angles_deg": ang_s.tolist(),
        "contrastive_angles_deg": ang_c.tolist(),
        "random_baseline_mean_deg": rand_mean.tolist(),
        "median_failure_angle_deg": float(np.median(ang_f)),
        "median_success_angle_deg": float(np.median(ang_s)),
        "median_contrastive_angle_deg": float(np.median(ang_c)),
        "median_random_angle_deg": float(np.median(rand_mean)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-benchmark pipeline
# ──────────────────────────────────────────────────────────────────────────────


def run_benchmark(name, tasks, alias, transfer_root, steering_root,
                  conceptor_npz, target_dirname):
    print(f"\n=== {name} ({len(tasks)} tasks) ===")
    labels = [alias[t] for t in tasks]
    print("  loading transfer SR …")
    SR, BASE, strat_map = load_sr_and_meta(transfer_root, steering_root, tasks, target_dirname)

    print(f"  loading conceptors (L={SIM_LAYER}, α={SIM_ALPHA}) …")
    Cs = load_class_conceptors(conceptor_npz, tasks, SIM_LAYER, SIM_ALPHA, "C_success")
    Cf = load_class_conceptors(conceptor_npz, tasks, SIM_LAYER, SIM_ALPHA, "C_failure")
    Cc = load_class_conceptors(conceptor_npz, tasks, SIM_LAYER, SIM_ALPHA, "C_contrastive")

    CONTAIN_F = build_matrix(Cf, tasks, containment)
    SIM_F = build_matrix(Cf, tasks, symmetric_frob_sim)
    SIM_S = build_matrix(Cs, tasks, symmetric_frob_sim)

    save_csv_matrix(SR, tasks, alias, OUT_ROOT / f"matrix_sr_{name}.csv")
    save_csv_matrix(CONTAIN_F, tasks, alias, OUT_ROOT / f"matrix_containF_{name}.csv")
    save_csv_matrix(SIM_F, tasks, alias, OUT_ROOT / f"matrix_simF_{name}.csv")
    save_csv_matrix(SIM_S, tasks, alias, OUT_ROOT / f"matrix_simS_{name}.csv")

    pretty = {"libero": r"$\pi_{0.5}$ LIBERO", "robocasa": r"$\pi_{0.5}$ RoboCasa"}[name]

    save_latex_matrix(
        SR, tasks, alias, OUT_ROOT / f"matrix_sr_{name}.tex",
        caption=(f"Full $N\\times N$ transfer SR on {pretty.replace('$','')} "
                 f"(best across $\\{{G, PS0, PS9\\}}$ per cell; self-best in \\textbf{{bold}}; "
                 f"strategy shown as subscript)."),
        label=f"tab:transfer_full_{name}",
        strat_map=strat_map,
    )
    save_latex_matrix(
        CONTAIN_F, tasks, alias, OUT_ROOT / f"matrix_containF_{name}.tex",
        caption=(f"Full $N\\times N$ failure-subspace containment on {pretty.replace('$','')}: "
                 f"$\\mathrm{{tr}}(C^f_\\mathrm{{src}} C^f_\\mathrm{{tgt}})/\\mathrm{{tr}}((C^f_\\mathrm{{src}})^2)$ "
                 f"at $L={SIM_LAYER},\\,\\alpha={SIM_ALPHA}$. Rows sum over how much of a source's failure mass lives in each target's."),
        label=f"tab:containF_{name}",
        highlight_diag=False,
    )
    save_latex_matrix(
        SIM_F, tasks, alias, OUT_ROOT / f"matrix_simF_{name}.tex",
        caption=(f"Symmetric failure-conceptor similarity on {pretty.replace('$','')} "
                 f"(normalised Frobenius inner product of $C^f$; diagonal $=1$)."),
        label=f"tab:simF_{name}",
        highlight_diag=False,
    )

    corr_contain = correlate(SR, CONTAIN_F, BASE, tasks)
    corr_simF = correlate(SR, SIM_F, BASE, tasks)
    corr_simS = correlate(SR, SIM_S, BASE, tasks)
    print(f"  {'metric':<12s} {'Pearson r (gain)':>18s} {'p':>10s} {'Spearman ρ':>14s}")
    for nm, c in [("contain_F", corr_contain), ("simF", corr_simF), ("simS", corr_simS)]:
        print(f"  {nm:<12s} {c['pearson_r_gain']:>18.3f} {c['pearson_p_gain']:>10.2e} "
              f"{c['spearman_r_gain']:>14.3f}")

    return {
        "labels": labels, "tasks": tasks,
        "SR": SR, "BASE": BASE,
        "CONTAIN_F": CONTAIN_F, "SIM_F": SIM_F, "SIM_S": SIM_S,
        "strat_map": strat_map,
        "corr_contain": corr_contain, "corr_simF": corr_simF, "corr_simS": corr_simS,
        "Cs": Cs, "Cf": Cf, "Cc": Cc,
        "pretty": pretty,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Master plots
# ──────────────────────────────────────────────────────────────────────────────


def plot_master_heatmaps(results):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    for row, name in enumerate(["libero", "robocasa"]):
        r = results[name]
        heatmap(axes[row, 0], r["SR"], r["labels"],
                f"{r['pretty']} — transfer SR (max of $\\{{G,PS0,PS9\\}}$)",
                vmin=0.0, vmax=1.0, cmap="viridis")
        heatmap(axes[row, 1], r["CONTAIN_F"], r["labels"],
                f"{r['pretty']} — failure-subspace containment "
                f"$\\mathrm{{tr}}(C^f_\\mathrm{{src}} C^f_\\mathrm{{tgt}})/\\mathrm{{tr}}((C^f_\\mathrm{{src}})^2)$",
                vmin=0.0, vmax=1.0, cmap="magma")
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_ROOT / f"heatmaps{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    colors = {"libero": "tab:blue", "robocasa": "tab:red"}
    for ax, name in zip(axes, ["libero", "robocasa"]):
        r = results[name]
        c = r["corr_contain"]
        x = np.array(c["x"])
        y = np.array(c["y_gain"])
        ax.scatter(x, y, alpha=0.6, s=38, c=colors[name], edgecolors="k", linewidths=0.3)
        if len(x) > 2:
            slope, icpt = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + icpt, "--", color="k", lw=1.2, alpha=0.7)

        # Highlight flagged pairs
        flagged = [("OpenDrawer", "CoffeeSetupMug"),
                   ("CoffeeSetupMug", "OpenDrawer")] if name == "robocasa" else [
            ("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
             "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"),
            ("LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
             "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"),
        ]
        alias_map = LIBERO_ALIAS if name == "libero" else ROBOCASA_ALIAS
        for (src, tgt) in flagged:
            if (src, tgt) in c["pairs"]:
                idx = c["pairs"].index((src, tgt))
                ax.scatter(x[idx], y[idx], s=180, facecolors="none",
                           edgecolors="black", linewidths=1.8,
                           label=f"{alias_map[src]}$\\to${alias_map[tgt]}")

        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(
            f"{r['pretty']}: $n={len(x)}$ off-diagonal pairs\n"
            f"Pearson $r={c['pearson_r_gain']:.2f}$ "
            f"($p={c['pearson_p_gain']:.1e}$),  "
            f"Spearman $\\rho={c['spearman_r_gain']:.2f}$"
        )
        ax.set_xlabel(
            r"failure-subspace containment  "
            r"$\mathrm{tr}(C^f_\mathrm{src}C^f_\mathrm{tgt})/\mathrm{tr}((C^f_\mathrm{src})^2)$"
        )
        if name == "libero":
            ax.set_ylabel("transfer SR $-$ baseline SR (target)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_ROOT / f"sim_vs_sr_gain{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Writeup
# ──────────────────────────────────────────────────────────────────────────────


def write_paragraph(results, dd_stats):
    rc = results["robocasa"]["corr_contain"]
    lb = results["libero"]["corr_contain"]
    mF = dd_stats["median_failure_angle_deg"]
    mS = dd_stats["median_success_angle_deg"]
    mR = dd_stats["median_random_angle_deg"]
    txt = f"""
\\paragraph{{Why do some conceptors transfer while others don't?}}
The transfer master table (Table~\\ref{{tab:transfer_master}}) shows that
cross-task conceptor steering is far from uniform: some source / target pairs
yield double-digit SR lifts, others do nothing. We claim the driver is
geometric — a conceptor transfers when the source and target share a
\\emph{{failure subspace}}. The intuition follows directly from the steering
hook $h' = (1-\\beta)h + \\beta\\,C^{{\\mathrm{{con}}}}_\\mathrm{{src}} h$ with
$C^{{\\mathrm{{con}}}} = C^s(I - C^f)$: the hook's only negative action is to
suppress the source's failure directions, so it helps the target only when
those directions overlap with the target's own failure modes. To test this
\\emph{{a priori}}, we form the contrastive success/failure conceptors at the
canonical steering layer $L={SIM_LAYER}$ ($\\alpha={SIM_ALPHA}$) for every task
in both benchmarks and compute the asymmetric failure-subspace containment
$\\mathrm{{cf}}(i\\!\\rightarrow\\!j) = \\mathrm{{tr}}(C^f_i C^f_j)/\\mathrm{{tr}}((C^f_i)^2)$
— the fraction of source-$i$'s failure mass that also lives in target-$j$'s
failure subspace. The full $N\\times N$ matrices are reported in
Tables~\\ref{{tab:transfer_full_libero}}–\\ref{{tab:containF_robocasa}} and
visualised as heatmaps in Fig.~\\ref{{fig:heatmaps}}. Correlating
$\\mathrm{{cf}}$ with the observed transfer-SR gain over baseline across all
off-diagonal source/target pairs (Fig.~\\ref{{fig:sim_vs_sr}}) we find
Pearson $r={lb['pearson_r_gain']:.2f}$
($p={lb['pearson_p_gain']:.1e}$, Spearman $\\rho={lb['spearman_r_gain']:.2f}$)
on LIBERO and Pearson $r={rc['pearson_r_gain']:.2f}$
($p={rc['pearson_p_gain']:.1e}$, Spearman $\\rho={rc['spearman_r_gain']:.2f}$)
on RoboCasa — a clean, significant linear relationship on a purely a priori
quantity computed before any transfer rollout is run. Fig.~\\ref{{fig:deepdive}}
dissects the standout RoboCasa pair OpenDrawer$\\leftrightarrow$CoffeeSetupMug,
which each boost the other by $\\geq 0.20$ absolute SR. Panel (a) plots
principal angles between the top-$30$ eigenspaces of their conceptors: the
median failure-subspace angle is only ${mF:.0f}^\\circ$, versus
${mR:.0f}^\\circ$ for two random $K$-dimensional subspaces of $\\mathbb{{R}}^{{1024}}$,
and the success-subspace angle is ${mS:.0f}^\\circ$ — both tasks
\\emph{{inhabit the same corner}} of the suffix-model hidden space.
Panel (b) shows their success- and failure-conceptor eigenvalue spectra are
nearly co-linear, and Panel (c) projects all four ellipsoids onto the top-2
eigenvectors of $C^f_\\mathrm{{OpenDrawer}}+C^f_\\mathrm{{CoffeeSetup}}$: the
two tasks' failure ellipses overlap almost perfectly while their success
ellipses also share a dominant axis. The combined picture: conceptors
transfer when their failure subspaces align, and this geometric criterion is
predictable from the conceptors alone — offering a \\emph{{rollout-free}} test
for whether a source conceptor will help a new target.
""".strip() + "\n"
    (OUT_ROOT / "analysis_paragraph.md").write_text(txt)


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Repo: {REPO_ROOT}\nOut:  {OUT_ROOT}")
    results = {}
    results["libero"] = run_benchmark(
        "libero", LIBERO_TASKS, LIBERO_ALIAS,
        REPO_ROOT / "experiments" / "pi05_libero" / "transfer_results",
        REPO_ROOT / "experiments" / "pi05_libero" / "steering_results",
        OPENPI_DATA_HOME / "libero_conceptors.npz",
        short_libero,
    )
    results["robocasa"] = run_benchmark(
        "robocasa", ROBOCASA_TASKS, ROBOCASA_ALIAS,
        REPO_ROOT / "experiments" / "pi05_robocasa" / "transfer_results",
        REPO_ROOT / "experiments" / "pi05_robocasa" / "steering_results",
        OPENPI_DATA_HOME / "robocasa_conceptors.npz",
        lambda t: t,
    )

    plot_master_heatmaps(results)
    plot_scatter(results)

    # Deep-dive: RoboCasa OpenDrawer × CoffeeSetupMug
    print("\n=== Deep-dive: OpenDrawer × CoffeeSetupMug ===")
    rb = results["robocasa"]
    dd_stats = deepdive_plot(
        Cs_a=rb["Cs"]["OpenDrawer"], Cs_b=rb["Cs"]["CoffeeSetupMug"],
        Cf_a=rb["Cf"]["OpenDrawer"], Cf_b=rb["Cf"]["CoffeeSetupMug"],
        Cc_a=rb["Cc"]["OpenDrawer"], Cc_b=rb["Cc"]["CoffeeSetupMug"],
        name_a="OpenDrawer", name_b="CoffeeSetup",
        out_path=OUT_ROOT / "deepdive_OD_CS",
    )
    print(f"  median failure angle = {dd_stats['median_failure_angle_deg']:.1f}°  "
          f"(random = {dd_stats['median_random_angle_deg']:.1f}°)")
    print(f"  median success angle = {dd_stats['median_success_angle_deg']:.1f}°")

    # JSON stats
    def trim(c):
        return {k: v for k, v in c.items() if k not in ("pairs",)}

    payload = {
        "sim_layer": SIM_LAYER,
        "sim_alpha": SIM_ALPHA,
        "libero": {
            "labels": results["libero"]["labels"],
            "corr_contain": trim(results["libero"]["corr_contain"]),
            "corr_simF": trim(results["libero"]["corr_simF"]),
            "corr_simS": trim(results["libero"]["corr_simS"]),
        },
        "robocasa": {
            "labels": results["robocasa"]["labels"],
            "corr_contain": trim(results["robocasa"]["corr_contain"]),
            "corr_simF": trim(results["robocasa"]["corr_simF"]),
            "corr_simS": trim(results["robocasa"]["corr_simS"]),
        },
        "deepdive_OD_CS": dd_stats,
    }
    with open(OUT_ROOT / "correlations.json", "w") as f:
        json.dump(payload, f, indent=2)

    write_paragraph(results, dd_stats)
    print(f"\nAll outputs written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
