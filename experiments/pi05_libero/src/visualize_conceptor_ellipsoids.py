#!/usr/bin/env python3
"""
Conceptor Ellipsoid Visualization for NeurIPS Paper
====================================================

Produces publication-quality figures showing how conceptor steering reshapes
activations in representation space.

Key insight: conceptors are SPD matrices with eigenvalues in [0,1], defining
ellipsoids in hidden space. The contrastive conceptor C_s · NOT(C_f) captures
directions unique to successful behaviour. Steering projects activations onto
this discriminative subspace, increasing "success subspace energy" while
decreasing "failure subspace energy".

Figures produced:
  1. Per-task PCA scatter + confidence ellipses (all available tasks)
  2. Subspace energy bar chart: how steering redistributes energy
  3. Eigenspectrum comparison of C_success, C_failure, C_contrastive
  4. Combined paper panel (3 representative tasks, compact)
  5. Shift-vs-improvement scatter
  6. Conceptor geometry (ellipsoid overlays centered at centroids)

Usage:
    cd /vast/projects/ungar/stellar/miaom/openpi-new
    uv run experiments/pi05_libero/src/visualize_conceptor_ellipsoids.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Style (NeurIPS-friendly)
# ──────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
})

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = Path(os.environ.get(
    "OPENPI_DATA_HOME",
    str(Path.home() / ".cache" / "openpi"),
))
ACTIVATIONS_DIR = (
    OPENPI_DATA_HOME / "activations" / "pi05_libero_2000_15env" / "openpi-libero-2000"
)
STEERED_ACTIVATIONS_DIR = (
    OPENPI_DATA_HOME / "activations" / "pi05_steered_activations" / "pi05_libero"
)
CONCEPTORS_PATH = OPENPI_DATA_HOME / "libero_conceptors.npz"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "steering_results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "diagnostic_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}
HIDDEN_DIM = 1024
MIN_PER_CLASS = 3

TASKS = [
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

TASK_SHORT = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": "Stove + Moka",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": "Bowl in Drawer",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": "Mug in Microwave",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "Two Mokas on Stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": "Soup+Cheese in Basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": "Soup+Tomato in Basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": "Cheese+Butter in Basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "Two Mugs on Plates",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": "Mug+Pudding on Plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": "Book in Caddy",
}

# ──────────────────────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────────────────────

C_SUCCESS = "#2ca02c"
C_FAILURE = "#d62728"
C_STEERED = "#1f77b4"
C_STEERED_FAIL = "#9467bd"   # purple — steered failure (distinct from steered success)
C_CONTRASTIVE = "#ff7f0e"
C_BASELINE_AVG = "#7f7f7f"   # gray — baseline average
C_STEERED_AVG = "#17becf"    # cyan — steered average

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_episode_metadata(task_dir: Path) -> dict[str, dict]:
    info = {}
    for ep_dir in sorted(task_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
            continue
        meta_path = ep_dir / "metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        info[ep_dir.name] = {
            "path": ep_dir,
            "success": bool(meta.get("episode_success", False)),
        }
    return info


def load_task_activations(task_dir: Path, layer_idx: int, ds: int = 0):
    """Load and return (X_success, X_failure), each (n, 1024) mean-pooled over action tokens."""
    info = load_episode_metadata(task_dir)
    succ_vecs, fail_vecs = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        is_success = ep["success"]
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]  # (10, 4, 10, 1024)
                    vec = arr[ds, layer_idx].mean(axis=0).astype(np.float32)
            except Exception:
                continue
            if is_success:
                succ_vecs.append(vec)
            else:
                fail_vecs.append(vec)
    X_s = np.stack(succ_vecs) if succ_vecs else np.empty((0, HIDDEN_DIM), dtype=np.float32)
    X_f = np.stack(fail_vecs) if fail_vecs else np.empty((0, HIDDEN_DIM), dtype=np.float32)
    return X_s, X_f


def load_steered_task_activations(task_dir: Path, layer_idx: int, ds: int = 0):
    """Load steered activations and split by episode outcome.

    Returns (X_steered_success, X_steered_failure), each (n, 1024).
    These are REAL post-steering activations captured during steered rollouts,
    not mathematically approximated ones.
    """
    info = load_episode_metadata(task_dir)
    succ_vecs, fail_vecs = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        is_success = ep["success"]
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]  # (10, 4, 10, 1024)
                    vec = arr[ds, layer_idx].mean(axis=0).astype(np.float32)
            except Exception:
                continue
            if is_success:
                succ_vecs.append(vec)
            else:
                fail_vecs.append(vec)
    X_ss = np.stack(succ_vecs) if succ_vecs else np.empty((0, HIDDEN_DIM), dtype=np.float32)
    X_sf = np.stack(fail_vecs) if fail_vecs else np.empty((0, HIDDEN_DIM), dtype=np.float32)
    return X_ss, X_sf


def apply_steering(X: np.ndarray, C: np.ndarray, beta: float) -> np.ndarray:
    """h' = h @ M^T where M = (1-beta)I + beta*C."""
    d = C.shape[0]
    M = (1 - beta) * np.eye(d) + beta * C
    return X @ M.T


def subspace_energy(X: np.ndarray, C: np.ndarray) -> float:
    """Mean per-sample energy in the subspace defined by C: E[h^T C h] / E[||h||^2]."""
    # h^T C h for each row h
    energies = np.einsum("ni,ij,nj->n", X, C, X)
    norms_sq = np.sum(X ** 2, axis=1)
    return float(np.mean(energies) / (np.mean(norms_sq) + 1e-12))


def pca_project(X_list: list[np.ndarray], n_components: int = 2):
    """Joint PCA. Returns (projected_list, V, mean)."""
    X_all = np.concatenate(X_list, axis=0)
    mean = X_all.mean(axis=0)
    X_centered = X_all - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt[:n_components].T
    results = []
    for X in X_list:
        results.append((X - mean) @ V)
    return results, V, mean


def discriminative_pca(X_list: list[np.ndarray], C_contrastive: np.ndarray,
                       n_components: int = 2):
    """Project onto top eigenvectors of C_contrastive (the discriminative subspace)."""
    evals, evecs = np.linalg.eigh(C_contrastive)
    # Top eigenvectors (largest eigenvalues)
    idx = evals.argsort()[::-1][:n_components]
    V = evecs[:, idx]  # (d, n_components)

    X_all = np.concatenate(X_list, axis=0)
    mean = X_all.mean(axis=0)
    results = []
    for X in X_list:
        results.append((X - mean) @ V)
    return results, V, mean


def draw_ellipse(ax, X_2d, color, n_std=2.0, alpha_fill=0.12,
                 alpha_edge=0.7, ls="-", lw=1.5, zorder=2):
    if X_2d.shape[0] < 2:
        return
    mean = X_2d.mean(axis=0)
    cov = np.cov(X_2d, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2 * n_std * np.sqrt(max(vals[0], 0))
    h = 2 * n_std * np.sqrt(max(vals[1], 0))
    ax.add_patch(matplotlib.patches.Ellipse(
        mean, w, h, angle=angle,
        facecolor=color, alpha=alpha_fill, linewidth=0, zorder=zorder))
    ax.add_patch(matplotlib.patches.Ellipse(
        mean, w, h, angle=angle,
        facecolor="none", edgecolor=color,
        alpha=alpha_edge, linewidth=lw, linestyle=ls, zorder=zorder + 1))


def project_conceptor_ellipse(C: np.ndarray, V: np.ndarray):
    """Project conceptor into PCA basis → (width, height, angle)."""
    C_2d = V.T @ C @ V
    vals, vecs = np.linalg.eigh(C_2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2 * np.sqrt(max(vals[0], 1e-12))
    h = 2 * np.sqrt(max(vals[1], 1e-12))
    return w, h, angle


def load_steering_results():
    """Load per-task baseline and best global-strategy success rates."""
    results = {}
    if not RESULTS_DIR.is_dir():
        return results
    cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
    for task in TASKS:
        found = None
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (task.startswith(d.name) or d.name.startswith(task[:40])):
                found = d
                break
        if found is None:
            continue
        summary_path = found / "summary.json"
        if not summary_path.is_file():
            continue
        with open(summary_path) as fh:
            data = json.load(fh)
        baseline = 0.0
        best_rate, best_cond = 0.0, ""
        best_global_rate, best_global_L, best_global_a, best_global_b = 0.0, 11, 1.0, 0.3
        for entry in data["conditions"]:
            cname = entry["condition"]
            rate = float(entry["success_rate"])
            if cname == "baseline":
                baseline = rate
            if rate > best_rate:
                best_rate = rate
                best_cond = cname
            m = cond_re.match(cname)
            if m and rate > best_global_rate:
                best_global_rate = rate
                best_global_L = int(m.group(1))
                best_global_a = float(m.group(2))
                best_global_b = float(m.group(3))
        results[task] = {
            "baseline": baseline,
            "best_rate": best_rate,
            "best_condition": best_cond,
            "best_global_rate": best_global_rate,
            "best_global_L": best_global_L,
            "best_global_a": best_global_a,
            "best_global_b": best_global_b,
        }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Per-task PCA scatter + ellipses
# ──────────────────────────────────────────────────────────────────────────────

def fig1_pca_ellipsoids(conceptors_npz, layer=11, alpha=1.0, beta=0.3, ds=0):
    """PCA scatter with confidence ellipses — one subplot per task."""
    layer_idx = LAYER_MAP[layer]
    available = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig1] No activation directories found. Skipping.")
        return

    n = len(available)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows),
                             squeeze=False)

    for idx, task in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        task_dir = ACTIVATIONS_DIR / task
        short = TASK_SHORT.get(task, task[:25])
        print(f"  [fig1] Loading {short}...")

        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        key = f"{task}__L{layer}__{alpha}__C_contrastive"
        if key not in conceptors_npz:
            ax.text(0.5, 0.5, "No conceptor", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        C_cont = conceptors_npz[key]
        X_f_steered = apply_steering(X_f, C_cont, beta)

        # Use discriminative PCA (top eigenvectors of C_contrastive)
        projs, V, gmean = discriminative_pca([X_s, X_f, X_f_steered], C_cont)
        X_s_2d, X_f_2d, X_fs_2d = projs

        # Scatter
        ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=8, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=8, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_fs_2d[:, 0], X_fs_2d[:, 1], c=C_STEERED, s=8, alpha=0.35,
                   edgecolors="none", marker="^", zorder=3)

        # Ellipses
        draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0)
        draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0)
        draw_ellipse(ax, X_fs_2d, C_STEERED, n_std=2.0, ls="--")

        # Arrow: failure centroid → steered centroid
        cf = X_f_2d.mean(0)
        cs = X_fs_2d.mean(0)
        ax.annotate("", xy=cs, xytext=cf,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=2.0,
                                     mutation_scale=15), zorder=10)

        ax.set_title(short, fontsize=10, fontweight="bold")
        ax.set_xlabel("Discriminative PC 1", fontsize=8)
        ax.set_ylabel("Discriminative PC 2", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Activation Distributions in Discriminative Subspace\n"
        rf"(Projected onto top eigenvectors of $C_s \cdot \neg C_f$, L{layer}, "
        rf"$\alpha$={alpha}, $\beta$={beta})",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "conceptor_ellipsoid_pca")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Subspace Energy Redistribution
# ──────────────────────────────────────────────────────────────────────────────

def fig2_subspace_energy(conceptors_npz, layer=11, alpha=1.0, beta=0.3, ds=0):
    """Grouped bar chart: contrastive energy + energy ratio before/after steering.

    Uses per-task best global parameters for the most accurate picture.
    """
    results = load_steering_results()

    task_names = []
    E_c_before, E_c_after = [], []     # contrastive energy
    ratio_before, ratio_after = [], []  # success/failure energy ratio
    sr_deltas = []

    for task in TASKS:
        task_dir = ACTIVATIONS_DIR / task
        if not task_dir.is_dir():
            continue
        r = results.get(task, {})
        L = r.get("best_global_L", layer)
        a = r.get("best_global_a", alpha)
        b = r.get("best_global_b", beta)
        layer_idx = LAYER_MAP[L]

        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        key_s = f"{task}__L{L}__{a}__C_success"
        key_f = f"{task}__L{L}__{a}__C_failure"
        if any(k not in conceptors_npz for k in [key_c, key_s, key_f]):
            continue

        C_cont = conceptors_npz[key_c]
        C_succ = conceptors_npz[key_s]
        C_fail = conceptors_npz[key_f]
        X_f_steered = apply_steering(X_f, C_cont, b)

        E_c_before.append(subspace_energy(X_f, C_cont))
        E_c_after.append(subspace_energy(X_f_steered, C_cont))

        es_b = subspace_energy(X_f, C_succ)
        ef_b = subspace_energy(X_f, C_fail)
        es_a = subspace_energy(X_f_steered, C_succ)
        ef_a = subspace_energy(X_f_steered, C_fail)
        ratio_before.append(es_b / (ef_b + 1e-12))
        ratio_after.append(es_a / (ef_a + 1e-12))

        short = TASK_SHORT.get(task, task[:20])
        task_names.append(f"{short}\n(L{L},$\\alpha$={a},$\\beta$={b})")
        sr_deltas.append(r.get("best_global_rate", 0) - r.get("baseline", 0))

    if not task_names:
        print("[fig2] No data. Skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, 2 * len(task_names)), 7.5),
                                    gridspec_kw={"hspace": 0.45})
    x = np.arange(len(task_names))
    w = 0.35

    # Panel A: Contrastive subspace energy
    ax1.bar(x - w / 2, E_c_before, w, color=C_FAILURE, alpha=0.7, edgecolor="black",
            linewidth=0.4, label="Before steering")
    ax1.bar(x + w / 2, E_c_after, w, color=C_STEERED, alpha=0.7, edgecolor="black",
            linewidth=0.4, label="After steering")
    for i in range(len(task_names)):
        delta = E_c_after[i] - E_c_before[i]
        y = max(E_c_before[i], E_c_after[i])
        ax1.text(i, y + 0.001, f"{delta:+.4f}", ha="center", va="bottom",
                 fontsize=7, fontweight="bold",
                 color=C_STEERED if delta > 0 else C_FAILURE)
    ax1.set_ylabel(r"$\mathbb{E}[\mathbf{h}^\top C_{\mathrm{contrastive}}\, \mathbf{h}]"
                   r"\;/\;\mathbb{E}[\|\mathbf{h}\|^2]$", fontsize=10)
    ax1.set_title("Contrastive Subspace Energy of Failure Activations\n"
                  "(per-task best global parameters)",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, fontsize=7)
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Success/failure energy ratio
    ax2.bar(x - w / 2, ratio_before, w, color=C_FAILURE, alpha=0.7, edgecolor="black",
            linewidth=0.4, label="Before steering")
    ax2.bar(x + w / 2, ratio_after, w, color=C_STEERED, alpha=0.7, edgecolor="black",
            linewidth=0.4, label="After steering")
    for i in range(len(task_names)):
        delta = ratio_after[i] - ratio_before[i]
        y = max(ratio_before[i], ratio_after[i])
        ax2.text(i, y + 0.0005, f"{delta:+.4f}", ha="center", va="bottom",
                 fontsize=7, fontweight="bold",
                 color=C_STEERED if delta > 0 else C_FAILURE)
    ax2.set_ylabel(r"$\frac{\mathbb{E}[h^\top C_s h]}{\mathbb{E}[h^\top C_f h]}$",
                   fontsize=12)
    ax2.set_title("Success / Failure Energy Ratio of Failure Activations",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(task_names, fontsize=7)
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, "conceptor_subspace_energy")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Eigenspectrum Comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig3_eigenspectrum(conceptors_npz, layer=11, alpha=1.0):
    """Eigenvalue spectra of C_success, C_failure, C_contrastive per task."""
    n_show = 50
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), squeeze=False)

    for idx, task in enumerate(TASKS[:10]):
        ax = axes[idx // 5, idx % 5]
        for ctype, color, label in [
            ("success", C_SUCCESS, r"$C_{\mathrm{succ}}$"),
            ("failure", C_FAILURE, r"$C_{\mathrm{fail}}$"),
            ("contrastive", C_CONTRASTIVE, r"$C_s \cdot \neg C_f$"),
        ]:
            key = f"{task}__L{layer}__{alpha}__C_{ctype}"
            if key not in conceptors_npz:
                continue
            C = conceptors_npz[key]
            eigs = np.linalg.eigvalsh(C)[::-1][:n_show]
            ax.plot(range(len(eigs)), eigs, color=color, lw=1.5, alpha=0.85,
                    label=label)
            # Shade area under curve
            ax.fill_between(range(len(eigs)), 0, eigs, color=color, alpha=0.06)

        ax.set_title(TASK_SHORT.get(task, task[:20]), fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // 5 == 1:
            ax.set_xlabel("Eigenvalue rank", fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel("Eigenvalue", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.suptitle(
        f"Conceptor Eigenspectra (Layer {layer}, "
        rf"$\alpha$={alpha})"
        "\nConceptors define soft subspaces; contrastive conceptor isolates success-specific dimensions",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "conceptor_eigenspectrum")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Paper Panel — 3 representative tasks, discriminative PCA + conceptor ellipses
# ──────────────────────────────────────────────────────────────────────────────

def fig4_paper_panel(conceptors_npz, ds=0):
    """Compact figure for the paper: 3 tasks with discriminative-PCA scatter,
    conceptor ellipses, and success-rate annotations."""
    results = load_steering_results()
    available = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig4] No tasks available. Skipping.")
        return

    # Pick 3 diverse tasks by baseline success rate
    if results:
        available_with_results = [t for t in available if t in results]
        if len(available_with_results) >= 3:
            available_with_results.sort(key=lambda t: results[t]["baseline"])
            pick = [available_with_results[0],
                    available_with_results[len(available_with_results) // 2],
                    available_with_results[-1]]
        else:
            pick = available_with_results or available[:3]
    else:
        pick = available[:3]

    n = len(pick)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0))
    if n == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, task in enumerate(pick):
        ax = axes[i]
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        b = r.get("best_global_b", 0.3)
        layer_idx = LAYER_MAP[L]

        task_dir = ACTIVATIONS_DIR / task
        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        key_s = f"{task}__L{L}__{a}__C_success"
        key_f = f"{task}__L{L}__{a}__C_failure"
        if any(k not in conceptors_npz for k in [key_c, key_s, key_f]):
            continue

        C_cont = conceptors_npz[key_c]
        C_succ = conceptors_npz[key_s]
        C_fail = conceptors_npz[key_f]
        X_f_steered = apply_steering(X_f, C_cont, b)

        # Discriminative PCA
        projs, V, gmean = discriminative_pca([X_s, X_f, X_f_steered], C_cont)
        X_s_2d, X_f_2d, X_fs_2d = projs

        # Scatter
        ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=10, alpha=0.3,
                   edgecolors="none", zorder=3)
        ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=10, alpha=0.3,
                   edgecolors="none", zorder=3)
        ax.scatter(X_fs_2d[:, 0], X_fs_2d[:, 1], c=C_STEERED, s=10, alpha=0.3,
                   edgecolors="none", marker="^", zorder=3)

        # Confidence ellipses (data distribution)
        draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0, alpha_fill=0.08, lw=1.2)
        draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0, alpha_fill=0.08, lw=1.2)
        draw_ellipse(ax, X_fs_2d, C_STEERED, n_std=2.0, alpha_fill=0.08, ls="--", lw=1.2)

        # Conceptor ellipses (centered at respective centroids)
        data_range = max(
            np.ptp(np.concatenate([X_s_2d, X_f_2d, X_fs_2d])[:, 0]),
            np.ptp(np.concatenate([X_s_2d, X_f_2d, X_fs_2d])[:, 1]),
        )
        for C_mat, center, color, lstyle, lw_c in [
            (C_succ, X_s_2d.mean(0), C_SUCCESS, "-", 2.5),
            (C_fail, X_f_2d.mean(0), C_FAILURE, "-", 2.5),
            (C_cont, X_f_2d.mean(0), C_CONTRASTIVE, "--", 2.5),
        ]:
            w, h, angle = project_conceptor_ellipse(C_mat, V)
            scale = data_range * 0.30 / max(w, h, 1e-6)
            ell = matplotlib.patches.Ellipse(
                center, w * scale, h * scale, angle=angle,
                facecolor=color, edgecolor=color,
                alpha=0.06, linewidth=lw_c, linestyle=lstyle, zorder=4)
            ax.add_patch(ell)
            ell_b = matplotlib.patches.Ellipse(
                center, w * scale, h * scale, angle=angle,
                facecolor="none", edgecolor=color,
                alpha=0.8, linewidth=lw_c, linestyle=lstyle, zorder=5)
            ax.add_patch(ell_b)

        # Centroids
        c_s = X_s_2d.mean(0)
        c_f = X_f_2d.mean(0)
        c_fs = X_fs_2d.mean(0)
        ax.plot(*c_s, "o", color=C_SUCCESS, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_f, "o", color=C_FAILURE, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_fs, "^", color=C_STEERED, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)

        # Arrow
        ax.annotate("", xy=c_fs, xytext=c_f,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=2.5,
                                     mutation_scale=18), zorder=10)

        short = TASK_SHORT.get(task, task[:25])
        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} baseline "
                     rf"$\rightarrow$ {best:.0%} steered, "
                     rf"L{L}, $\alpha$={a}, $\beta$={b})",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Discriminative PC 1", fontsize=9)
        if i == 0:
            ax.set_ylabel("Discriminative PC 2", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        # Panel label
        ax.text(-0.08, 1.12, panel_labels[i], transform=ax.transAxes,
                fontsize=13, fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.4, label="Success activations"),
        mpatches.Patch(color=C_FAILURE, alpha=0.4, label="Failure activations"),
        mpatches.Patch(color=C_STEERED, alpha=0.4, label="Steered failure activations"),
        mlines.Line2D([0], [0], color=C_SUCCESS, lw=2.5, label=r"$C_{\mathrm{success}}$ ellipsoid"),
        mlines.Line2D([0], [0], color=C_FAILURE, lw=2.5, label=r"$C_{\mathrm{failure}}$ ellipsoid"),
        mlines.Line2D([0], [0], color=C_CONTRASTIVE, lw=2.5, ls="--",
                      label=r"$C_s \cdot \neg C_f$ ellipsoid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, fancybox=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        "Conceptor Steering in Activation Space: Ellipsoidal Geometry",
        fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    _save(fig, "conceptor_ellipsoid_paper")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Shift-vs-Improvement Scatter
# ──────────────────────────────────────────────────────────────────────────────

def fig5_shift_scatter(conceptors_npz, layer=11, alpha=1.0, beta=0.3, ds=0):
    """x = increase in contrastive energy (per-task best params), y = success-rate improvement."""
    results = load_steering_results()

    delta_energy, delta_sr, names = [], [], []

    for task in TASKS:
        task_dir = ACTIVATIONS_DIR / task
        if not task_dir.is_dir() or task not in results:
            continue
        r = results[task]
        L = r.get("best_global_L", layer)
        a = r.get("best_global_a", alpha)
        b = r.get("best_global_b", beta)
        layer_idx = LAYER_MAP[L]

        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            continue
        C_cont = conceptors_npz[key_c]
        X_f_steered = apply_steering(X_f, C_cont, b)

        de = subspace_energy(X_f_steered, C_cont) - subspace_energy(X_f, C_cont)
        delta_energy.append(de)
        delta_sr.append(r["best_global_rate"] - r["baseline"])
        names.append(TASK_SHORT.get(task, task[:20]))

    if len(delta_energy) < 2:
        print("[fig5] Insufficient data. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(delta_energy, delta_sr, s=80, c=C_STEERED, edgecolors="black",
               linewidth=0.5, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (delta_energy[i], delta_sr[i]), fontsize=7.5,
                    xytext=(6, 6), textcoords="offset points")

    if len(delta_energy) >= 3:
        from scipy.stats import pearsonr
        z = np.polyfit(delta_energy, delta_sr, 1)
        p_fn = np.poly1d(z)
        x_fit = np.linspace(min(delta_energy) - 0.005, max(delta_energy) + 0.005, 100)
        ax.plot(x_fit, p_fn(x_fit), "--", color="gray", lw=1.5, alpha=0.6)
        r, pval = pearsonr(delta_energy, delta_sr)
        ax.text(0.02, 0.98, f"Pearson r = {r:.2f} (p = {pval:.3f})",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(r"$\Delta$ Contrastive Energy "
                  r"$(\mathbb{E}[h^\top C_{\mathrm{contrastive}} h] / \mathbb{E}[\|h\|^2])$",
                  fontsize=11)
    ax.set_ylabel(r"$\Delta$ Success Rate (steered $-$ baseline)", fontsize=11)
    ax.set_title("Contrastive Energy Gain Predicts Task Improvement (per-task best params)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "conceptor_shift_scatter")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Conceptor Geometry
# ──────────────────────────────────────────────────────────────────────────────

def fig6_conceptor_geometry(conceptors_npz, layer=11, alpha=1.0, beta=0.3, ds=0):
    """Conceptor ellipsoids in discriminative PCA space with equal aspect ratio.
    This is the 'circle' figure showing how one ellipsoid morphs to another."""
    layer_idx = LAYER_MAP[layer]
    available = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig6] No data. Skipping.")
        return

    n = min(len(available), 5)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0))
    if n == 1:
        axes = [axes]

    for i, task in enumerate(available[:n]):
        ax = axes[i]
        task_dir = ACTIVATIONS_DIR / task
        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            continue

        keys = {t: f"{task}__L{layer}__{alpha}__C_{t}" for t in
                ["success", "failure", "contrastive"]}
        if any(k not in conceptors_npz for k in keys.values()):
            continue
        C_s = conceptors_npz[keys["success"]]
        C_f = conceptors_npz[keys["failure"]]
        C_c = conceptors_npz[keys["contrastive"]]

        X_f_steered = apply_steering(X_f, C_c, beta)
        projs, V, gmean = discriminative_pca([X_s, X_f, X_f_steered], C_c)
        X_s_2d, X_f_2d, X_fs_2d = projs

        # Light scatter background
        for X_2d, c, m in [(X_s_2d, C_SUCCESS, "o"), (X_f_2d, C_FAILURE, "o"),
                           (X_fs_2d, C_STEERED, "^")]:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=c, s=5, alpha=0.15,
                       edgecolors="none", marker=m, zorder=1)

        c_s = X_s_2d.mean(0)
        c_f = X_f_2d.mean(0)
        c_fs = X_fs_2d.mean(0)

        # Conceptor ellipses with consistent scaling
        data_range = max(
            np.ptp(np.concatenate([X_s_2d, X_f_2d, X_fs_2d])[:, 0]),
            np.ptp(np.concatenate([X_s_2d, X_f_2d, X_fs_2d])[:, 1]),
        )

        for C_mat, center, color, lstyle, lw_c, alpha_f in [
            (C_s, c_s, C_SUCCESS, "-", 2.8, 0.08),
            (C_f, c_f, C_FAILURE, "-", 2.8, 0.08),
            (C_c, c_f, C_CONTRASTIVE, "--", 2.8, 0.05),
        ]:
            w, h, angle = project_conceptor_ellipse(C_mat, V)
            scale = data_range * 0.40 / max(w, h, 1e-6)
            # Fill
            ax.add_patch(matplotlib.patches.Ellipse(
                center, w * scale, h * scale, angle=angle,
                facecolor=color, edgecolor="none", alpha=alpha_f,
                linewidth=0, zorder=4))
            # Border
            ax.add_patch(matplotlib.patches.Ellipse(
                center, w * scale, h * scale, angle=angle,
                facecolor="none", edgecolor=color, alpha=0.85,
                linewidth=lw_c, linestyle=lstyle, zorder=5))

        # Centroids
        ax.plot(*c_s, "o", color=C_SUCCESS, markersize=11, markeredgecolor="white",
                markeredgewidth=2, zorder=11)
        ax.plot(*c_f, "o", color=C_FAILURE, markersize=11, markeredgecolor="white",
                markeredgewidth=2, zorder=11)
        ax.plot(*c_fs, "^", color=C_STEERED, markersize=11, markeredgecolor="white",
                markeredgewidth=2, zorder=11)

        # Arrow
        ax.annotate("", xy=c_fs, xytext=c_f,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=3,
                                     mutation_scale=20), zorder=10)

        ax.set_title(TASK_SHORT.get(task, task[:25]), fontsize=11, fontweight="bold")
        ax.set_xlabel("Discrim. PC 1", fontsize=10)
        if i == 0:
            ax.set_ylabel("Discrim. PC 2", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")

    handles = [
        mlines.Line2D([0], [0], color=C_SUCCESS, lw=2.8,
                      label=r"$C_{\mathrm{success}}$"),
        mlines.Line2D([0], [0], color=C_FAILURE, lw=2.8,
                      label=r"$C_{\mathrm{failure}}$"),
        mlines.Line2D([0], [0], color=C_CONTRASTIVE, lw=2.8, ls="--",
                      label=r"$C_s \cdot \neg C_f$"),
        mlines.Line2D([0], [0], color=C_STEERED, lw=3, marker=">",
                      markersize=8, label="Steering shift"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=True, fancybox=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Conceptor Ellipsoids: Steering Morphs Failure Toward Success Subspace\n"
        rf"$h^\prime = h \cdot [(1{{-}}\beta)\,I + \beta\,C_s \cdot \neg C_f]^\top$"
        rf"  (L{layer}, $\alpha$={alpha}, $\beta$={beta})",
        fontsize=13, fontweight="bold", y=1.07)
    fig.tight_layout()
    _save(fig, "conceptor_geometry")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7: Success vs Failure vs Steered-Success (per-task best params)
# ──────────────────────────────────────────────────────────────────────────────

def fig7_steered_success(conceptors_npz, ds=0):
    """PCA scatter showing original success, original failure, and steered success.

    'Steered success' = the best conceptor transformation applied to the original
    success activations.  This shows how steering reshapes even the successful
    representations — moving them further into the discriminative subspace and
    away from the failure region.

    Uses per-task best global parameters from the steering sweep.
    """
    results = load_steering_results()
    available = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig7] No tasks. Skipping.")
        return

    n = len(available)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows),
                             squeeze=False)

    for idx, task in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        b = r.get("best_global_b", 0.3)
        layer_idx = LAYER_MAP[L]

        task_dir = ACTIVATIONS_DIR / task
        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax.text(0.5, 0.5, "No conceptor", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        C_cont = conceptors_npz[key_c]

        # Steered SUCCESS activations (not failure!)
        X_s_steered = apply_steering(X_s, C_cont, b)

        # Discriminative PCA on all three groups
        projs, V, gmean = discriminative_pca([X_s, X_f, X_s_steered], C_cont)
        X_s_2d, X_f_2d, X_ss_2d = projs

        # Scatter
        ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=10, alpha=0.3,
                   edgecolors="none", zorder=3)
        ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=10, alpha=0.3,
                   edgecolors="none", zorder=3)
        ax.scatter(X_ss_2d[:, 0], X_ss_2d[:, 1], c=C_STEERED, s=10, alpha=0.3,
                   edgecolors="none", marker="D", zorder=3)

        # Confidence ellipses
        draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0, alpha_fill=0.08)
        draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0, alpha_fill=0.08)
        draw_ellipse(ax, X_ss_2d, C_STEERED, n_std=2.0, alpha_fill=0.08, ls="--")

        # Centroids
        c_s = X_s_2d.mean(0)
        c_f = X_f_2d.mean(0)
        c_ss = X_ss_2d.mean(0)
        ax.plot(*c_s, "o", color=C_SUCCESS, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_f, "o", color=C_FAILURE, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_ss, "D", color=C_STEERED, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)

        # Arrow: success centroid → steered-success centroid
        ax.annotate("", xy=c_ss, xytext=c_s,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=2.5,
                                     mutation_scale=18), zorder=10)

        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} $\\rightarrow$ {best:.0%},  "
                     f"L{L}, $\\alpha$={a}, $\\beta$={b})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Discrim. PC 1", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Discrim. PC 2", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Original success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Original failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success"),
        mlines.Line2D([0], [0], color=C_STEERED, lw=2.5, marker=">",
                      markersize=7, label="Steering direction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, fancybox=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Success, Failure, and Steered-Success Activations in Discriminative Subspace\n"
        r"Steering reshapes success activations: $h^\prime_{\mathrm{succ}} = "
        r"h_{\mathrm{succ}} \cdot [(1{-}\beta)\,I + \beta\,C_s \cdot \neg C_f]^\top$"
        "  (per-task best params)",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "steered_success_pca")


# ──────────────────────────────────────────────────────────────────────────────
# Figures 8 & 9: Joint t-SNE — consistent embedding for steered-success and
#                steered-failure views (one t-SNE per task with all 4 groups)
# ──────────────────────────────────────────────────────────────────────────────

def fig8_9_joint_tsne(conceptors_npz, ds=0):
    """Run a single joint t-SNE per task with all 4 groups (success, failure,
    steered-success, steered-failure), then produce two figures from the same
    embedding so the success/failure ellipses are consistent across charts.
    """
    from sklearn.manifold import TSNE

    results = load_steering_results()
    available = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig8/9] No tasks. Skipping.")
        return

    # Pre-compute joint t-SNE embeddings for all tasks
    task_data = {}  # task -> dict of 2D arrays + metadata
    for task in available:
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        b = r.get("best_global_b", 0.3)
        layer_idx = LAYER_MAP[L]

        task_dir = ACTIVATIONS_DIR / task
        print(f"  [fig8/9] {short} (L{L}, a={a}, b={b})...")
        X_s, X_f = load_task_activations(task_dir, layer_idx, ds)
        if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
            task_data[task] = {"skip": "Insufficient data", "short": short, "r": r}
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            task_data[task] = {"skip": "No conceptor", "short": short, "r": r}
            continue
        C_cont = conceptors_npz[key_c]

        # Steered versions
        X_s_steered = apply_steering(X_s, C_cont, b)
        X_f_steered = apply_steering(X_f, C_cont, b)

        # Subsample (t-SNE is O(n^2))
        max_pts = 400  # 4 groups, keep total manageable
        rng = np.random.RandomState(42)
        if X_s.shape[0] > max_pts:
            sel_s = rng.choice(X_s.shape[0], max_pts, replace=False)
            X_s_sub, X_ss_sub = X_s[sel_s], X_s_steered[sel_s]
        else:
            X_s_sub, X_ss_sub = X_s, X_s_steered
        if X_f.shape[0] > max_pts:
            sel_f = rng.choice(X_f.shape[0], max_pts, replace=False)
            X_f_sub, X_fs_sub = X_f[sel_f], X_f_steered[sel_f]
        else:
            X_f_sub, X_fs_sub = X_f, X_f_steered

        # Joint t-SNE on all 4 groups
        n_s = X_s_sub.shape[0]
        n_f = X_f_sub.shape[0]
        n_ss = X_ss_sub.shape[0]
        n_fs = X_fs_sub.shape[0]
        X_all = np.concatenate([X_s_sub, X_f_sub, X_ss_sub, X_fs_sub], axis=0)
        perplexity = min(30, X_all.shape[0] // 5)
        tsne = TSNE(n_components=2, perplexity=max(5, perplexity),
                     random_state=42, init="pca", learning_rate="auto")
        X_2d = tsne.fit_transform(X_all)

        off = 0
        X_s_2d = X_2d[off:off + n_s]; off += n_s
        X_f_2d = X_2d[off:off + n_f]; off += n_f
        X_ss_2d = X_2d[off:off + n_ss]; off += n_ss
        X_fs_2d = X_2d[off:off + n_fs]

        task_data[task] = {
            "short": short, "r": r, "L": L, "a": a, "b": b,
            "X_s_2d": X_s_2d, "X_f_2d": X_f_2d,
            "X_ss_2d": X_ss_2d, "X_fs_2d": X_fs_2d,
        }

    # ---- Figure 8: Success + Failure + Steered-Success ----
    n = len(available)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig8, axes8 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows),
                                squeeze=False)
    for idx, task in enumerate(available):
        ax = axes8[idx // ncols, idx % ncols]
        d = task_data[task]
        short = d["short"]
        if "skip" in d:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, d["skip"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        X_s_2d, X_f_2d, X_ss_2d = d["X_s_2d"], d["X_f_2d"], d["X_ss_2d"]
        r = d["r"]

        ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=10, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=10, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_ss_2d[:, 0], X_ss_2d[:, 1], c=C_STEERED, s=10, alpha=0.35,
                   edgecolors="none", marker="D", zorder=3)

        draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0, alpha_fill=0.06)
        draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0, alpha_fill=0.06)
        draw_ellipse(ax, X_ss_2d, C_STEERED, n_std=2.0, alpha_fill=0.06, ls="--")

        c_s, c_f, c_ss = X_s_2d.mean(0), X_f_2d.mean(0), X_ss_2d.mean(0)
        ax.plot(*c_s, "o", color=C_SUCCESS, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_f, "o", color=C_FAILURE, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_ss, "D", color=C_STEERED, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.annotate("", xy=c_ss, xytext=c_s,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=2.5,
                                     mutation_scale=18), zorder=10)

        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} $\\rightarrow$ {best:.0%},  "
                     f"L{d['L']}, $\\alpha$={d['a']}, $\\beta$={d['b']})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes8[idx // ncols, idx % ncols].set_visible(False)

    handles8 = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Original success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Original failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success"),
        mlines.Line2D([0], [0], color=C_STEERED, lw=2.5, marker=">",
                      markersize=7, label="Steering direction"),
    ]
    fig8.legend(handles=handles8, loc="lower center", ncol=4, fontsize=9,
                frameon=True, fancybox=True, edgecolor="#cccccc",
                bbox_to_anchor=(0.5, -0.03))
    fig8.suptitle(
        "Success, Failure, and Steered-Success Activations (t-SNE)\n"
        "Per-task best global parameters",
        fontsize=13, fontweight="bold", y=1.03)
    fig8.tight_layout()
    _save(fig8, "steered_success_tsne")

    # ---- Figure 9: Success + Failure + Steered-Failure ----
    fig9, axes9 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows),
                                squeeze=False)
    for idx, task in enumerate(available):
        ax = axes9[idx // ncols, idx % ncols]
        d = task_data[task]
        short = d["short"]
        if "skip" in d:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, d["skip"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        X_s_2d, X_f_2d, X_fs_2d = d["X_s_2d"], d["X_f_2d"], d["X_fs_2d"]
        r = d["r"]

        ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=10, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=10, alpha=0.35,
                   edgecolors="none", zorder=3)
        ax.scatter(X_fs_2d[:, 0], X_fs_2d[:, 1], c=C_STEERED, s=10, alpha=0.35,
                   edgecolors="none", marker="^", zorder=3)

        draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0, alpha_fill=0.06)
        draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0, alpha_fill=0.06)
        draw_ellipse(ax, X_fs_2d, C_STEERED, n_std=2.0, alpha_fill=0.06, ls="--")

        c_s, c_f, c_fs = X_s_2d.mean(0), X_f_2d.mean(0), X_fs_2d.mean(0)
        ax.plot(*c_s, "o", color=C_SUCCESS, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_f, "o", color=C_FAILURE, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.plot(*c_fs, "^", color=C_STEERED, markersize=9, markeredgecolor="white",
                markeredgewidth=1.5, zorder=11)
        ax.annotate("", xy=c_fs, xytext=c_f,
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED, lw=2.5,
                                     mutation_scale=18), zorder=10)

        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} $\\rightarrow$ {best:.0%},  "
                     f"L{d['L']}, $\\alpha$={d['a']}, $\\beta$={d['b']})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes9[idx // ncols, idx % ncols].set_visible(False)

    handles9 = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Original success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Original failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered failure"),
        mlines.Line2D([0], [0], color=C_STEERED, lw=2.5, marker=">",
                      markersize=7, label="Steering direction"),
    ]
    fig9.legend(handles=handles9, loc="lower center", ncol=4, fontsize=9,
                frameon=True, fancybox=True, edgecolor="#cccccc",
                bbox_to_anchor=(0.5, -0.03))
    fig9.suptitle(
        "Conceptor Steering Moves Failure Activations Toward Success (t-SNE)\n"
        r"$h^\prime_{\mathrm{fail}} = h_{\mathrm{fail}} \cdot "
        r"[(1{-}\beta)\,I + \beta\,C_s \cdot \neg C_f]^\top$"
        "  (per-task best params)",
        fontsize=12, fontweight="bold", y=1.03)
    fig9.tight_layout()
    _save(fig9, "steered_failure_tsne")


# ══════════════════════════════════════════════════════════════════════════════
# NEW FIGURES: Real steered activations (post-collection)
# These require steered activations to have been collected and saved to
# STEERED_ACTIVATIONS_DIR.  They gracefully skip if the data isn't there yet.
# ══════════════════════════════════════════════════════════════════════════════

def _check_steered_data():
    """Return True if real steered activation data exists."""
    if not STEERED_ACTIVATIONS_DIR.is_dir():
        print(f"  [skip] Steered activations not found: {STEERED_ACTIVATIONS_DIR}")
        return False
    return True


def _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results):
    """Load baseline (success, failure) and real steered (success, failure) activations.

    Returns dict with keys: X_s, X_f, X_ss, X_sf, or None if data is insufficient.
    """
    task_dir_base = ACTIVATIONS_DIR / task
    task_dir_steered = STEERED_ACTIVATIONS_DIR / task

    if not task_dir_base.is_dir():
        return None
    X_s, X_f = load_task_activations(task_dir_base, layer_idx, ds)

    if not task_dir_steered.is_dir():
        return None
    X_ss, X_sf = load_steered_task_activations(task_dir_steered, layer_idx, ds)

    # Need at least some data in each group (allow 0 for steered-failure since
    # high steering success rate may leave few/no failed episodes)
    if X_s.shape[0] < MIN_PER_CLASS or X_f.shape[0] < MIN_PER_CLASS:
        return None
    if X_ss.shape[0] < 1 and X_sf.shape[0] < 1:
        return None

    return {"X_s": X_s, "X_f": X_f, "X_ss": X_ss, "X_sf": X_sf}


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10: 4-group t-SNE (Success + Failure + Steered-Success + Steered-Failure)
# ──────────────────────────────────────────────────────────────────────────────

def fig10_four_group_tsne(conceptors_npz, ds=0):
    """Hero figure: joint t-SNE of all 4 activation groups using real steered data."""
    if not _check_steered_data():
        return
    from sklearn.manifold import TSNE

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig10] No tasks with both baseline and steered data. Skipping.")
        return

    n = len(available)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows),
                             squeeze=False)

    for idx, task in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        b = r.get("best_global_b", 0.3)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        X_s, X_f = data["X_s"], data["X_f"]
        X_ss, X_sf = data["X_ss"], data["X_sf"]

        print(f"  [fig10] {short}: baseline s={X_s.shape[0]} f={X_f.shape[0]}, "
              f"steered s={X_ss.shape[0]} f={X_sf.shape[0]}")

        # Subsample for t-SNE
        max_pts = 300
        rng = np.random.RandomState(42)
        def _sub(X, n=max_pts):
            return X[rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)] if X.shape[0] > 0 else X

        X_s_sub = _sub(X_s)
        X_f_sub = _sub(X_f)
        X_ss_sub = _sub(X_ss)
        X_sf_sub = _sub(X_sf)

        groups = [X_s_sub, X_f_sub, X_ss_sub, X_sf_sub]
        sizes = [g.shape[0] for g in groups]
        X_all = np.concatenate([g for g in groups if g.shape[0] > 0], axis=0)
        perplexity = min(30, X_all.shape[0] // 5)
        tsne = TSNE(n_components=2, perplexity=max(5, perplexity),
                     random_state=42, init="pca", learning_rate="auto")
        X_2d = tsne.fit_transform(X_all)

        # Split back
        off = 0
        X_2d_groups = []
        for sz in sizes:
            X_2d_groups.append(X_2d[off:off + sz] if sz > 0 else np.empty((0, 2)))
            off += sz
        X_s_2d, X_f_2d, X_ss_2d, X_sf_2d = X_2d_groups

        # Scatter: baseline behind, steered in front
        if X_f_2d.shape[0] > 0:
            ax.scatter(X_f_2d[:, 0], X_f_2d[:, 1], c=C_FAILURE, s=8, alpha=0.3,
                       edgecolors="none", zorder=2)
        if X_s_2d.shape[0] > 0:
            ax.scatter(X_s_2d[:, 0], X_s_2d[:, 1], c=C_SUCCESS, s=8, alpha=0.3,
                       edgecolors="none", zorder=2)
        if X_sf_2d.shape[0] > 0:
            ax.scatter(X_sf_2d[:, 0], X_sf_2d[:, 1], c=C_STEERED_FAIL, s=10, alpha=0.4,
                       edgecolors="none", marker="^", zorder=4)
        if X_ss_2d.shape[0] > 0:
            ax.scatter(X_ss_2d[:, 0], X_ss_2d[:, 1], c=C_STEERED, s=10, alpha=0.4,
                       edgecolors="none", marker="D", zorder=4)

        # Confidence ellipses
        if X_f_2d.shape[0] >= 2:
            draw_ellipse(ax, X_f_2d, C_FAILURE, n_std=2.0, alpha_fill=0.05)
        if X_s_2d.shape[0] >= 2:
            draw_ellipse(ax, X_s_2d, C_SUCCESS, n_std=2.0, alpha_fill=0.05)
        if X_ss_2d.shape[0] >= 2:
            draw_ellipse(ax, X_ss_2d, C_STEERED, n_std=2.0, alpha_fill=0.05, ls="--")
        if X_sf_2d.shape[0] >= 2:
            draw_ellipse(ax, X_sf_2d, C_STEERED_FAIL, n_std=2.0, alpha_fill=0.05, ls="--")

        # Centroids + arrows
        centroids = {}
        for label, X2d, color, marker in [
            ("s", X_s_2d, C_SUCCESS, "o"), ("f", X_f_2d, C_FAILURE, "o"),
            ("ss", X_ss_2d, C_STEERED, "D"), ("sf", X_sf_2d, C_STEERED_FAIL, "^"),
        ]:
            if X2d.shape[0] > 0:
                c = X2d.mean(0)
                centroids[label] = c
                ax.plot(*c, marker, color=color, markersize=9,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=11)

        # Arrow: failure → steered-failure (shows steering effect on failures)
        if "f" in centroids and "sf" in centroids:
            ax.annotate("", xy=centroids["sf"], xytext=centroids["f"],
                         arrowprops=dict(arrowstyle="-|>", color=C_STEERED_FAIL,
                                         lw=2.5, mutation_scale=18), zorder=10)

        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} $\\rightarrow$ {best:.0%},  "
                     f"L{L}, $\\alpha$={a}, $\\beta$={b})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Baseline success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Baseline failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success"),
        mpatches.Patch(color=C_STEERED_FAIL, alpha=0.5, label="Steered failure"),
        mlines.Line2D([0], [0], color=C_STEERED_FAIL, lw=2.5, marker=">",
                      markersize=7, label="Steering shift"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               frameon=True, fancybox=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "All Four Activation Groups Under Conceptor Steering (t-SNE)\n"
        "Real post-steering activations, per-task best global params",
        fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "four_group_tsne")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 11: 4-group subspace energy bar chart
# ──────────────────────────────────────────────────────────────────────────────

def fig11_four_group_energy(conceptors_npz, ds=0):
    """Bar chart: contrastive subspace energy for all 4 groups per task."""
    if not _check_steered_data():
        return

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig11] No tasks. Skipping.")
        return

    task_names, energies = [], {"s": [], "f": [], "ss": [], "sf": []}
    for task in available:
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            continue
        C_cont = conceptors_npz[key_c]

        short = TASK_SHORT.get(task, task[:20])
        task_names.append(short)
        for grp_key, X in [("s", data["X_s"]), ("f", data["X_f"]),
                           ("ss", data["X_ss"]), ("sf", data["X_sf"])]:
            if X.shape[0] >= 1:
                energies[grp_key].append(subspace_energy(X, C_cont))
            else:
                energies[grp_key].append(0.0)

    if not task_names:
        print("[fig11] No valid tasks. Skipping.")
        return

    n = len(task_names)
    x = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n), 5))
    bars_config = [
        (-1.5, "s", C_SUCCESS, "Baseline success"),
        (-0.5, "f", C_FAILURE, "Baseline failure"),
        (0.5, "ss", C_STEERED, "Steered success"),
        (1.5, "sf", C_STEERED_FAIL, "Steered failure"),
    ]
    for offset, key, color, label in bars_config:
        vals = energies[key]
        ax.bar(x + offset * width, vals, width, color=color, alpha=0.8, label=label,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(r"Contrastive subspace energy $\frac{E[\mathbf{h}^\top C \mathbf{h}]}"
                  r"{E[\|\mathbf{h}\|^2]}$", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Contrastive Subspace Energy: Steering Aligns Failure Activations\n"
                 "with the Success-Discriminative Subspace",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    _save(fig, "four_group_energy_bars")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 12: Shift magnitude vs. performance gain scatter
# ──────────────────────────────────────────────────────────────────────────────

def fig12_shift_vs_gain(conceptors_npz, ds=0):
    """Scatter: per-task delta subspace energy vs. delta success rate."""
    if not _check_steered_data():
        return

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    tasks_plotted, delta_e, delta_sr = [], [], []
    for task in available:
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]
        baseline_sr = r.get("baseline", 0)
        steered_sr = r.get("best_global_rate", 0)

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            continue
        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            continue
        C_cont = conceptors_npz[key_c]

        # Energy shift: how much did steering move failure activations?
        if data["X_f"].shape[0] < 1:
            continue
        e_f = subspace_energy(data["X_f"], C_cont)
        # Use steered-failure if available, else all steered
        if data["X_sf"].shape[0] >= 1:
            e_sf = subspace_energy(data["X_sf"], C_cont)
        elif data["X_ss"].shape[0] >= 1:
            # If all steered episodes succeeded, use steered-success as proxy
            e_sf = subspace_energy(data["X_ss"], C_cont)
        else:
            continue

        tasks_plotted.append(TASK_SHORT.get(task, task[:20]))
        delta_e.append(e_sf - e_f)
        delta_sr.append(steered_sr - baseline_sr)

    if len(tasks_plotted) < 2:
        print("[fig12] Need >= 2 tasks. Skipping.")
        return

    delta_e = np.array(delta_e)
    delta_sr = np.array(delta_sr)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(delta_e, delta_sr, c=C_STEERED, s=100, edgecolors="white",
               linewidths=1.5, zorder=5)
    for i, name in enumerate(tasks_plotted):
        ax.annotate(name, (delta_e[i], delta_sr[i]), fontsize=7.5,
                    textcoords="offset points", xytext=(6, 6),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Trend line
    if len(delta_e) >= 3:
        coeffs = np.polyfit(delta_e, delta_sr, 1)
        x_fit = np.linspace(delta_e.min() - 0.01, delta_e.max() + 0.01, 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="gray", alpha=0.6, lw=1.5)
        corr = np.corrcoef(delta_e, delta_sr)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=11, va="top", color="gray")

    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(r"$\Delta$ Contrastive Subspace Energy (steered failure $-$ baseline failure)",
                  fontsize=10)
    ax.set_ylabel(r"$\Delta$ Success Rate (steered $-$ baseline)", fontsize=10)
    ax.set_title("Activation Shift Predicts Performance Gain\n"
                 "Tasks where steering moves activations more also improve more",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "shift_vs_gain")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 13: Centroid trajectory plot (clean version of fig10)
# ──────────────────────────────────────────────────────────────────────────────

def fig13_centroid_trajectories(conceptors_npz, ds=0):
    """Minimal plot: only centroids + arrows, no scatter clutter.
    Shows the direction and magnitude of steering for each group.
    """
    if not _check_steered_data():
        return
    from sklearn.manifold import TSNE

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]
    if not available:
        print("[fig13] No tasks. Skipping.")
        return

    n = len(available)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows),
                             squeeze=False)

    for idx, task in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        b = r.get("best_global_b", 0.3)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax.set_title(short, fontsize=9)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        # Subsample + joint t-SNE to get centroids in consistent space
        max_pts = 300
        rng = np.random.RandomState(42)
        def _sub(X, n=max_pts):
            return X[rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)] if X.shape[0] > 0 else X

        groups = [_sub(data["X_s"]), _sub(data["X_f"]),
                  _sub(data["X_ss"]), _sub(data["X_sf"])]
        sizes = [g.shape[0] for g in groups]
        X_all = np.concatenate([g for g in groups if g.shape[0] > 0])
        perplexity = min(30, X_all.shape[0] // 5)
        tsne = TSNE(n_components=2, perplexity=max(5, perplexity),
                     random_state=42, init="pca", learning_rate="auto")
        X_2d = tsne.fit_transform(X_all)

        off = 0
        centroids = {}
        for label, sz, color in [("s", sizes[0], C_SUCCESS), ("f", sizes[1], C_FAILURE),
                                  ("ss", sizes[2], C_STEERED), ("sf", sizes[3], C_STEERED_FAIL)]:
            if sz > 0:
                pts = X_2d[off:off + sz]
                centroids[label] = pts.mean(0)
                # Light scatter (very faint, just for context)
                ax.scatter(pts[:, 0], pts[:, 1], c=color, s=3, alpha=0.1,
                           edgecolors="none", zorder=1)
            off += sz

        # Large centroid markers
        markers = {"s": "o", "f": "o", "ss": "D", "sf": "^"}
        colors = {"s": C_SUCCESS, "f": C_FAILURE, "ss": C_STEERED, "sf": C_STEERED_FAIL}
        for label, c in centroids.items():
            ax.plot(*c, markers[label], color=colors[label], markersize=14,
                    markeredgecolor="white", markeredgewidth=2.0, zorder=11)

        # Arrows: success→steered-success, failure→steered-failure
        arrow_pairs = [("s", "ss", C_STEERED), ("f", "sf", C_STEERED_FAIL)]
        for src, dst, color in arrow_pairs:
            if src in centroids and dst in centroids:
                ax.annotate("", xy=centroids[dst], xytext=centroids[src],
                             arrowprops=dict(arrowstyle="-|>", color=color,
                                             lw=3.0, mutation_scale=22), zorder=10)

        bl = r.get("baseline", 0)
        best = r.get("best_global_rate", 0)
        ax.set_title(f"{short}\n({bl:.0%} $\\rightarrow$ {best:.0%})",
                     fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SUCCESS,
                      markersize=12, label="Baseline success"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_FAILURE,
                      markersize=12, label="Baseline failure"),
        mlines.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_STEERED,
                      markersize=10, label="Steered success"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_STEERED_FAIL,
                      markersize=10, label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, fancybox=True, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Centroid Trajectories Under Conceptor Steering\n"
                 "Arrows show how steering moves each group's centroid",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "centroid_trajectories")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 14: Population-level view (aggregate across all tasks)
# ──────────────────────────────────────────────────────────────────────────────

def fig14_population_view(conceptors_npz, ds=0):
    """Aggregate view: average centroids across all tasks, showing the
    global distributional effect of steering.
    Uses discriminative PCA (top eigenvectors of average C_contrastive).
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    # Collect centroids per task (in original 1024-d space)
    all_centroids = {"s": [], "f": [], "ss": [], "sf": []}
    # Also collect raw points for scatter
    all_points = {"s": [], "f": [], "ss": [], "sf": []}
    C_contrastives = []

    for task in available:
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            continue
        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            continue
        C_contrastives.append(conceptors_npz[key_c])

        rng = np.random.RandomState(42)
        for key, X in [("s", data["X_s"]), ("f", data["X_f"]),
                       ("ss", data["X_ss"]), ("sf", data["X_sf"])]:
            if X.shape[0] > 0:
                all_centroids[key].append(X.mean(0))
                # Subsample per task for scatter
                n_sub = min(50, X.shape[0])
                idx = rng.choice(X.shape[0], n_sub, replace=False)
                all_points[key].append(X[idx])

    if not C_contrastives:
        print("[fig14] No valid tasks. Skipping.")
        return

    # Average contrastive conceptor for projection
    C_avg = np.mean(C_contrastives, axis=0)

    # Pool all points
    all_X = {}
    for key in ["s", "f", "ss", "sf"]:
        if all_points[key]:
            all_X[key] = np.concatenate(all_points[key], axis=0)
        else:
            all_X[key] = np.empty((0, HIDDEN_DIM), dtype=np.float32)

    # Discriminative PCA using average C_contrastive
    combined = [all_X[k] for k in ["s", "f", "ss", "sf"] if all_X[k].shape[0] > 0]
    projected, V, mean = discriminative_pca(combined, C_avg, n_components=2)

    # Re-split projected
    proj = {}
    off = 0
    for key in ["s", "f", "ss", "sf"]:
        n = all_X[key].shape[0]
        if n > 0:
            proj[key] = projected[0][off:off + n] if off == 0 else None
            off += n

    # Simpler: project each group directly
    proj = {}
    for key in ["s", "f", "ss", "sf"]:
        if all_X[key].shape[0] > 0:
            proj[key] = (all_X[key] - mean) @ V

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_config = [
        ("f", C_FAILURE, "o", "Baseline failure", 0.15, 8),
        ("s", C_SUCCESS, "o", "Baseline success", 0.15, 8),
        ("sf", C_STEERED_FAIL, "^", "Steered failure", 0.25, 12),
        ("ss", C_STEERED, "D", "Steered success", 0.25, 12),
    ]
    for key, color, marker, label, alpha, size in plot_config:
        if key in proj and proj[key].shape[0] > 0:
            ax.scatter(proj[key][:, 0], proj[key][:, 1], c=color, s=size,
                       alpha=alpha, edgecolors="none", marker=marker, label=label, zorder=3)

    # Centroids (large markers)
    centroid_proj = {}
    for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
        if key in proj and proj[key].shape[0] > 0:
            c = proj[key].mean(0)
            centroid_proj[key] = c
            ax.plot(*c, marker, color=color, markersize=16,
                    markeredgecolor="white", markeredgewidth=2.5, zorder=11)

    # Confidence ellipses
    for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                           ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
        if key in proj and proj[key].shape[0] >= 2:
            draw_ellipse(ax, proj[key], color, n_std=2.0, alpha_fill=0.06, ls=ls)

    # Arrows
    if "f" in centroid_proj and "sf" in centroid_proj:
        ax.annotate("", xy=centroid_proj["sf"], xytext=centroid_proj["f"],
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED_FAIL,
                                     lw=3, mutation_scale=20), zorder=10)
    if "s" in centroid_proj and "ss" in centroid_proj:
        ax.annotate("", xy=centroid_proj["ss"], xytext=centroid_proj["s"],
                     arrowprops=dict(arrowstyle="-|>", color=C_STEERED,
                                     lw=3, mutation_scale=20), zorder=10)

    ax.set_xlabel("Discriminative PC 1", fontsize=11)
    ax.set_ylabel("Discriminative PC 2", fontsize=11)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Population-Level Effect of Conceptor Steering\n"
                 "All tasks pooled, projected onto average contrastive subspace",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "population_view")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 16: Per-task population view (discriminative PCA per task)
# ──────────────────────────────────────────────────────────────────────────────

def fig16_per_task_population_view(conceptors_npz, ds=0, task_filter=None, single_row=False,
                                    out_name="per_task_population_view"):
    """Per-task version of the population view: each task gets its own panel
    projected onto that task's C_contrastive top-2 eigenvectors.

    task_filter: optional list of task names (full or short) to include, in order.
    single_row: if True, arrange all panels in one row.
    out_name: filename stem for saving.
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    all_available = [t for t in TASKS
                     if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    if task_filter is not None:
        # Allow short or full names; preserve user-provided order
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        available = []
        for t in task_filter:
            if t in all_available:
                available.append(t)
            elif t in short_to_full and short_to_full[t] in all_available:
                available.append(short_to_full[t])
            else:
                print(f"[fig16] Task not found or missing data: {t}")
    else:
        available = all_available

    if not available:
        print("[fig16] No tasks with both baseline and steered data. Skipping.")
        return

    n = len(available)
    if single_row:
        ncols, nrows = n, 1
    else:
        ncols = min(5, n)
        nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, task in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax.set_title(f"{short}\n(insufficient data)", fontsize=9)
            ax.axis("off")
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax.set_title(f"{short}\n(no conceptor)", fontsize=9)
            ax.axis("off")
            continue
        C = conceptors_npz[key_c]

        # Collect all points for this task
        groups = {"s": data["X_s"], "f": data["X_f"],
                  "ss": data["X_ss"], "sf": data["X_sf"]}

        # Discriminative PCA using this task's C_contrastive
        combined = [groups[k] for k in ["s", "f", "ss", "sf"] if groups[k].shape[0] > 0]
        if not combined:
            ax.set_title(f"{short}\n(no data)", fontsize=9)
            ax.axis("off")
            continue
        _, V, mean = discriminative_pca(combined, C, n_components=2)

        # Project each group
        proj = {}
        for key in ["s", "f", "ss", "sf"]:
            if groups[key].shape[0] > 0:
                proj[key] = (groups[key] - mean) @ V

        # Scatter
        plot_config = [
            ("f", C_FAILURE, "o", "Baseline fail", 0.2, 8),
            ("s", C_SUCCESS, "o", "Baseline succ", 0.2, 8),
            ("sf", C_STEERED_FAIL, "^", "Steered fail", 0.3, 12),
            ("ss", C_STEERED, "D", "Steered succ", 0.3, 12),
        ]
        for key, color, marker, label, alpha, size in plot_config:
            if key in proj and proj[key].shape[0] > 0:
                ax.scatter(proj[key][:, 0], proj[key][:, 1], c=color, s=size,
                           alpha=alpha, edgecolors="none", marker=marker,
                           label=label if idx == 0 else None, zorder=3)

        # Confidence ellipses
        for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                               ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
            if key in proj and proj[key].shape[0] >= 2:
                draw_ellipse(ax, proj[key], color, n_std=2.0, alpha_fill=0.06, ls=ls)

        # Centroids
        centroid_proj = {}
        for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                    ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
            if key in proj and proj[key].shape[0] > 0:
                c = proj[key].mean(0)
                centroid_proj[key] = c
                ax.plot(*c, marker, color=color, markersize=12,
                        markeredgecolor="white", markeredgewidth=2, zorder=11)

        # Arrows from baseline to steered centroids
        if "f" in centroid_proj and "sf" in centroid_proj:
            ax.annotate("", xy=centroid_proj["sf"], xytext=centroid_proj["f"],
                         arrowprops=dict(arrowstyle="-|>", color=C_STEERED_FAIL,
                                         lw=2, mutation_scale=15), zorder=10)
        if "s" in centroid_proj and "ss" in centroid_proj:
            ax.annotate("", xy=centroid_proj["ss"], xytext=centroid_proj["s"],
                         arrowprops=dict(arrowstyle="-|>", color=C_STEERED,
                                         lw=2, mutation_scale=15), zorder=10)

        # Steered success rate from collection
        sr_steered = r.get("best_global_sr")
        sr_base = r.get("baseline_sr")
        subtitle_parts = [f"L{L}"]
        if sr_base is not None:
            subtitle_parts.append(f"base={sr_base:.0%}")
        if sr_steered is not None:
            subtitle_parts.append(f"steered={sr_steered:.0%}")
        ax.set_title(f"{short}\n({', '.join(subtitle_parts)})", fontsize=9, fontweight="bold")
        ax.set_xlabel("Disc. PC 1", fontsize=8)
        ax.set_ylabel("Disc. PC 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Turn off unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    # Shared legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Baseline success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Baseline failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success"),
        mpatches.Patch(color=C_STEERED_FAIL, alpha=0.5, label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle("Per-Task Discriminative PCA: Baseline vs Steered Activations",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_name)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 17: Trajectory-mode clustering (per-episode embeddings)
# ──────────────────────────────────────────────────────────────────────────────

def load_episode_trajectory_embeddings(task_dir: Path, layer_idx: int, ds: int = 0,
                                        mode: str = "mean",
                                        projection_V: np.ndarray | None = None,
                                        projection_mean: np.ndarray | None = None):
    """Per-episode trajectory embedding.

    mode:
      - "mean": mean of per-step activations (1024-d)
      - "phases": [first_third_mean, mid_third_mean, last_third_mean] concat (3072-d)
      - "mean_std": [mean, std] concat (2048-d)
      - "projected_phases": phase-means projected to 2D first (requires V, mean), then concat (6-d)

    Returns (emb_success, emb_failure). Embedding dim varies with mode.
    """
    info = load_episode_metadata(task_dir)
    succ_embs, fail_embs = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        is_success = ep["success"]
        step_vecs = []
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]
                    vec = arr[ds, layer_idx].mean(axis=0).astype(np.float32)
            except Exception:
                continue
            step_vecs.append(vec)
        if not step_vecs:
            continue
        steps = np.stack(step_vecs)
        n_steps = steps.shape[0]
        if mode == "mean":
            ep_emb = steps.mean(axis=0)
        elif mode == "phases":
            t1, t2 = n_steps // 3, 2 * n_steps // 3
            p1 = steps[:max(1, t1)].mean(axis=0)
            p2 = steps[t1:max(t1 + 1, t2)].mean(axis=0)
            p3 = steps[t2:].mean(axis=0) if n_steps > t2 else steps[-1]
            ep_emb = np.concatenate([p1, p2, p3])
        elif mode == "mean_std":
            ep_emb = np.concatenate([steps.mean(axis=0), steps.std(axis=0)])
        elif mode == "projected_phases":
            assert projection_V is not None and projection_mean is not None
            t1, t2 = n_steps // 3, 2 * n_steps // 3
            p1 = steps[:max(1, t1)].mean(axis=0)
            p2 = steps[t1:max(t1 + 1, t2)].mean(axis=0)
            p3 = steps[t2:].mean(axis=0) if n_steps > t2 else steps[-1]
            ep_emb = np.concatenate([
                (p1 - projection_mean) @ projection_V,
                (p2 - projection_mean) @ projection_V,
                (p3 - projection_mean) @ projection_V,
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if is_success:
            succ_embs.append(ep_emb)
        else:
            fail_embs.append(ep_emb)
    d = len(succ_embs[0]) if succ_embs else (len(fail_embs[0]) if fail_embs else HIDDEN_DIM)
    E_s = np.stack(succ_embs) if succ_embs else np.empty((0, d), dtype=np.float32)
    E_f = np.stack(fail_embs) if fail_embs else np.empty((0, d), dtype=np.float32)
    return E_s, E_f


def _cluster_within_group(E: np.ndarray, k_max: int = 2, min_silhouette: float = 0.05,
                           force_k: int | None = None, seed: int = 42):
    """k-means with silhouette-based k selection (or forced k). Returns (labels, k, score)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = E.shape[0]
    if n < 4:
        return np.zeros(n, dtype=int), 1, -1.0
    if force_k is not None and force_k > 1 and n > force_k:
        km = KMeans(n_clusters=force_k, n_init=10, random_state=seed).fit(E)
        try:
            score = silhouette_score(E, km.labels_)
        except Exception:
            score = -1.0
        return km.labels_, force_k, score
    best_k, best_labels, best_score = 1, np.zeros(n, dtype=int), -1.0
    for k in range(2, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(E)
        try:
            score = silhouette_score(E, km.labels_)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_labels, best_score = k, km.labels_, score
    if best_score < min_silhouette:
        return np.zeros(n, dtype=int), 1, best_score
    return best_labels, best_k, best_score


def fig17_trajectory_modes_stacked(conceptors_npz, ds=0, task_filter=None,
                                    out_name="trajectory_modes_stacked",
                                    embedding_mode: str = "mean",
                                    min_silhouette: float = 0.05,
                                    force_k: int | None = None,
                                    share_axes: bool = True):
    """Two-row figure:
      Top row: per-step activations in discriminative PCA (same as fig16).
      Bottom row: per-episode trajectory embeddings, clustered within success/failure.

    Uses the same discriminative PCA (top-2 eigenvectors of C_contrastive) for both rows.
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    all_available = [t for t in TASKS
                     if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    if task_filter is not None:
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        available = []
        for t in task_filter:
            if t in all_available:
                available.append(t)
            elif t in short_to_full and short_to_full[t] in all_available:
                available.append(short_to_full[t])
            else:
                print(f"[fig17] Task not found or missing data: {t}")
    else:
        available = all_available

    if not available:
        print("[fig17] No tasks. Skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 9.5), squeeze=False)

    # Shades for success/failure sub-clusters
    succ_shades = ["#2ca02c", "#7fd17f"]  # dark + light green
    fail_shades = ["#d62728", "#f08f90"]  # dark + light red

    for idx, task in enumerate(available):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax_top.set_title(f"{short}\n(insufficient data)", fontsize=9)
            ax_top.axis("off"); ax_bot.axis("off")
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax_top.axis("off"); ax_bot.axis("off")
            continue
        C = conceptors_npz[key_c]

        # --- Top row: per-step activations (same as fig16) ---
        groups = {"s": data["X_s"], "f": data["X_f"],
                  "ss": data["X_ss"], "sf": data["X_sf"]}
        combined = [groups[k] for k in ["s", "f", "ss", "sf"] if groups[k].shape[0] > 0]
        _, V, mean = discriminative_pca(combined, C, n_components=2)

        proj = {k: (groups[k] - mean) @ V for k in groups if groups[k].shape[0] > 0}

        plot_config = [
            ("f", C_FAILURE, "o", 0.2, 8),
            ("s", C_SUCCESS, "o", 0.2, 8),
            ("sf", C_STEERED_FAIL, "^", 0.3, 12),
            ("ss", C_STEERED, "D", 0.3, 12),
        ]
        for key, color, marker, alpha, size in plot_config:
            if key in proj:
                ax_top.scatter(proj[key][:, 0], proj[key][:, 1], c=color, s=size,
                               alpha=alpha, edgecolors="none", marker=marker, zorder=3)
        for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                               ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
            if key in proj and proj[key].shape[0] >= 2:
                draw_ellipse(ax_top, proj[key], color, n_std=2.0, alpha_fill=0.06, ls=ls)
        for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                    ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
            if key in proj:
                c = proj[key].mean(0)
                ax_top.plot(*c, marker, color=color, markersize=12,
                            markeredgecolor="white", markeredgewidth=2, zorder=11)

        ax_top.set_title(f"{short}\n(L{L})", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("Disc. PC 1", fontsize=8)
        if idx == 0:
            ax_top.set_ylabel("Per-step activations\nDisc. PC 2", fontsize=9)
        ax_top.tick_params(labelsize=7)

        # --- Bottom row: per-episode trajectory embeddings + within-group clustering ---
        task_dir_base = ACTIVATIONS_DIR / task
        E_s, E_f = load_episode_trajectory_embeddings(task_dir_base, layer_idx, ds,
                                                       mode=embedding_mode,
                                                       projection_V=V, projection_mean=mean)

        if E_s.shape[0] == 0 and E_f.shape[0] == 0:
            ax_bot.axis("off")
            continue

        # For "mean" mode with same dim as per-step, reuse top-row V for shared axes.
        # For other modes, do a separate 2D PCA on (E_s, E_f) combined.
        emb_dim = E_s.shape[1] if E_s.shape[0] > 0 else E_f.shape[1]
        if embedding_mode == "mean" and emb_dim == HIDDEN_DIM and share_axes:
            proj_es = (E_s - mean) @ V if E_s.shape[0] > 0 else np.empty((0, 2))
            proj_ef = (E_f - mean) @ V if E_f.shape[0] > 0 else np.empty((0, 2))
        else:
            # Own 2D discriminative PCA on episode embeddings using C projected to embedding space
            # Simpler fallback: plain PCA on the combined embeddings.
            combined_E = np.concatenate(
                [X for X in (E_s, E_f) if X.shape[0] > 0], axis=0)
            e_mean = combined_E.mean(0)
            _, _, Vt_e = np.linalg.svd(combined_E - e_mean, full_matrices=False)
            Ve = Vt_e[:2].T
            proj_es = (E_s - e_mean) @ Ve if E_s.shape[0] > 0 else np.empty((0, 2))
            proj_ef = (E_f - e_mean) @ Ve if E_f.shape[0] > 0 else np.empty((0, 2))

        # Cluster within each group
        if E_s.shape[0] > 0:
            labels_s, ks, sil_s = _cluster_within_group(E_s, k_max=2,
                                                        min_silhouette=min_silhouette,
                                                        force_k=force_k)
        else:
            labels_s, ks, sil_s = np.array([]), 1, -1.0
        if E_f.shape[0] > 0:
            labels_f, kf, sil_f = _cluster_within_group(E_f, k_max=2,
                                                        min_silhouette=min_silhouette,
                                                        force_k=force_k)
        else:
            labels_f, kf, sil_f = np.array([]), 1, -1.0

        # Plot success sub-clusters (circles, green shades)
        for ci in range(ks):
            mask = labels_s == ci
            if mask.sum() == 0:
                continue
            pts = proj_es[mask]
            ax_bot.scatter(pts[:, 0], pts[:, 1], c=succ_shades[ci], s=90,
                           alpha=0.85, marker="o", edgecolors="black", linewidths=0.6,
                           label=f"Succ mode {ci+1} (n={mask.sum()})" if idx == 0 else None,
                           zorder=4)
            if pts.shape[0] >= 2:
                draw_ellipse(ax_bot, pts, succ_shades[ci], n_std=1.5, alpha_fill=0.10, ls="-")

        # Plot failure sub-clusters (squares, red shades)
        for ci in range(kf):
            mask = labels_f == ci
            if mask.sum() == 0:
                continue
            pts = proj_ef[mask]
            ax_bot.scatter(pts[:, 0], pts[:, 1], c=fail_shades[ci], s=90,
                           alpha=0.85, marker="s", edgecolors="black", linewidths=0.6,
                           label=f"Fail mode {ci+1} (n={mask.sum()})" if idx == 0 else None,
                           zorder=4)
            if pts.shape[0] >= 2:
                draw_ellipse(ax_bot, pts, fail_shades[ci], n_std=1.5, alpha_fill=0.10, ls="-")

        if embedding_mode == "mean" and share_axes:
            ax_bot.set_xlim(ax_top.get_xlim())
            ax_bot.set_ylim(ax_top.get_ylim())

        subtitle = (f"succ: k={ks} (sil={sil_s:.2f}), "
                    f"fail: k={kf} (sil={sil_f:.2f})")
        ax_bot.set_title(subtitle, fontsize=8)
        ax_bot.set_xlabel("Disc. PC 1", fontsize=8)
        if idx == 0:
            ax_bot.set_ylabel("Per-episode trajectories\nDisc. PC 2", fontsize=9)
        ax_bot.tick_params(labelsize=7)

    # Shared legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    top_handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Baseline success (per-step)"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Baseline failure (per-step)"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success (per-step)"),
        mpatches.Patch(color=C_STEERED_FAIL, alpha=0.5, label="Steered failure (per-step)"),
    ]
    bot_handles = [
        mlines.Line2D([], [], color=succ_shades[0], marker="o", linestyle="",
                      markersize=10, markeredgecolor="black", label="Success mode 1"),
        mlines.Line2D([], [], color=succ_shades[1], marker="o", linestyle="",
                      markersize=10, markeredgecolor="black", label="Success mode 2"),
        mlines.Line2D([], [], color=fail_shades[0], marker="s", linestyle="",
                      markersize=10, markeredgecolor="black", label="Failure mode 1"),
        mlines.Line2D([], [], color=fail_shades[1], marker="s", linestyle="",
                      markersize=10, markeredgecolor="black", label="Failure mode 2"),
    ]
    fig.legend(handles=top_handles + bot_handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    mode_desc = {
        "mean": "mean-pooled trajectory embedding",
        "phases": "3-phase (early/mid/late) embedding",
        "mean_std": "[mean, std] embedding",
        "projected_phases": "3-phase projected embedding (6-D)",
    }.get(embedding_mode, embedding_mode)
    force_desc = f", forced k={force_k}" if force_k else f", sil-min={min_silhouette}"
    fig.suptitle(f"Trajectory Modes — bottom row: {mode_desc}{force_desc}",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, out_name)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 20: Combined PCA (A) + temporal discriminative projection (B)
# ──────────────────────────────────────────────────────────────────────────────

def fig20_combined_pca_and_temporal(conceptors_npz, ds=0, task_filter=None,
                                     out_name="combined_pca_temporal", n_bins=50):
    """Two-row hero figure for NeurIPS:
      (A) per-task discriminative PCA of per-step activations (4 groups).
      (B) per-task projection onto top eigenvector of C_contrastive over
          normalized trajectory time, mean ±1σ band.
    Shares the same 6 tasks and identical color/marker scheme in both rows.
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    all_available = [t for t in TASKS
                     if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    if task_filter is not None:
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        available = []
        for t in task_filter:
            if t in all_available:
                available.append(t)
            elif t in short_to_full and short_to_full[t] in all_available:
                available.append(short_to_full[t])
            else:
                print(f"[fig20] Task not found or missing data: {t}")
    else:
        available = all_available

    if not available:
        print("[fig20] No tasks. Skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.0), squeeze=False)
    t_axis = np.linspace(0, 1, n_bins)

    for idx, task in enumerate(available):
        ax_pca = axes[0, idx]
        ax_t = axes[1, idx]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax_pca.axis("off"); ax_t.axis("off")
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax_pca.axis("off"); ax_t.axis("off")
            continue
        C = conceptors_npz[key_c]

        # ── (A) Per-task discriminative PCA scatter ──
        groups = {"s": data["X_s"], "f": data["X_f"],
                  "ss": data["X_ss"], "sf": data["X_sf"]}
        combined = [groups[k] for k in ["s", "f", "ss", "sf"] if groups[k].shape[0] > 0]
        _, V, mean = discriminative_pca(combined, C, n_components=2)
        proj = {k: (groups[k] - mean) @ V for k in groups if groups[k].shape[0] > 0}

        plot_cfg = [
            ("f", C_FAILURE, "o", 0.2, 6),
            ("s", C_SUCCESS, "o", 0.2, 6),
            ("sf", C_STEERED_FAIL, "^", 0.3, 10),
            ("ss", C_STEERED, "D", 0.3, 10),
        ]
        for key, color, marker, alpha, size in plot_cfg:
            if key in proj:
                ax_pca.scatter(proj[key][:, 0], proj[key][:, 1], c=color, s=size,
                               alpha=alpha, edgecolors="none", marker=marker, zorder=3)
        for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                               ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
            if key in proj and proj[key].shape[0] >= 2:
                draw_ellipse(ax_pca, proj[key], color, n_std=2.0, alpha_fill=0.06, ls=ls)
        for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                    ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
            if key in proj:
                c = proj[key].mean(0)
                ax_pca.plot(*c, marker, color=color, markersize=9,
                            markeredgecolor="white", markeredgewidth=1.5, zorder=11)

        ax_pca.set_title(f"{short}", fontsize=10, fontweight="bold")
        ax_pca.set_xticks([]); ax_pca.set_yticks([])
        for spine in ax_pca.spines.values():
            spine.set_linewidth(0.6)
        if idx == 0:
            ax_pca.set_ylabel("Disc. PC 2", fontsize=9)
        ax_pca.set_xlabel("Disc. PC 1", fontsize=9)

        # ── (B) Temporal projection onto top eigenvector of C_contrastive ──
        eigvals, eigvecs = np.linalg.eigh(C)
        disc_axis = eigvecs[:, -1]

        s_trajs, f_trajs = load_episode_trajectories(ACTIVATIONS_DIR / task, layer_idx, ds)
        ss_trajs, sf_trajs = load_episode_trajectories(STEERED_ACTIVATIONS_DIR / task, layer_idx, ds)

        disc_by_group = {"s": [], "f": [], "ss": [], "sf": []}

        def _process(trajs, key):
            for traj in trajs:
                disc_by_group[key].append(traj @ disc_axis)

        _process(s_trajs, "s"); _process(f_trajs, "f")
        _process(ss_trajs, "ss"); _process(sf_trajs, "sf")

        line_spec = [
            ("f", C_FAILURE, "-"),
            ("s", C_SUCCESS, "-"),
            ("sf", C_STEERED_FAIL, "--"),
            ("ss", C_STEERED, "--"),
        ]
        for key, color, ls in line_spec:
            m, s = _group_mean_std(disc_by_group[key], n_bins)
            if m is None:
                continue
            ax_t.plot(t_axis, m, color=color, ls=ls, lw=2.0, zorder=5)
            ax_t.fill_between(t_axis, m - s, m + s, color=color, alpha=0.12, zorder=3)

        ax_t.set_xlabel("Normalized trajectory time", fontsize=9)
        if idx == 0:
            ax_t.set_ylabel(r"Projection onto $\mathbf{v}_1(C)$", fontsize=9)
        ax_t.tick_params(labelsize=7)
        ax_t.grid(alpha=0.2, lw=0.4)
        for spine in ["top", "right"]:
            ax_t.spines[spine].set_visible(False)

    # Panel labels (A) and (B)
    fig.text(0.005, 0.95, "(A)", fontsize=16, fontweight="bold", va="top")
    fig.text(0.005, 0.46, "(B)", fontsize=16, fontweight="bold", va="top")

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.6, label="Baseline success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.6, label="Baseline failure"),
        mlines.Line2D([], [], color=C_STEERED, ls="--", lw=2.2,
                      marker="D", markersize=8, markerfacecolor=C_STEERED,
                      markeredgecolor="white", label="Steered success"),
        mlines.Line2D([], [], color=C_STEERED_FAIL, ls="--", lw=2.2,
                      marker="^", markersize=8, markerfacecolor=C_STEERED_FAIL,
                      markeredgecolor="white", label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, frameon=False)

    fig.tight_layout(rect=(0.015, 0.02, 1.0, 1.0))
    _save(fig, out_name)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 19: Temporal line charts of trajectories (per-step over normalized time)
# ──────────────────────────────────────────────────────────────────────────────

def load_episode_trajectories(task_dir: Path, layer_idx: int, ds: int = 0):
    """Load per-episode per-step activations. Returns (succ_trajs, fail_trajs)
    where each is a list of (n_steps, 1024) arrays — one per episode.
    """
    info = load_episode_metadata(task_dir)
    succ_trajs, fail_trajs = [], []
    for _, ep in info.items():
        ep_dir = ep["path"]
        is_success = ep["success"]
        step_vecs = []
        for step_dir in sorted(ep_dir.glob("step_*")):
            npz_path = step_dir / "suffix_residual.npz"
            if not npz_path.is_file():
                continue
            try:
                with np.load(npz_path) as f:
                    arr = f["all_suffix_residual"]
                    vec = arr[ds, layer_idx].mean(axis=0).astype(np.float32)
            except Exception:
                continue
            step_vecs.append(vec)
        if not step_vecs:
            continue
        traj = np.stack(step_vecs)
        if is_success:
            succ_trajs.append(traj)
        else:
            fail_trajs.append(traj)
    return succ_trajs, fail_trajs


def _resample_scalar_series(series: np.ndarray, n_bins: int) -> np.ndarray:
    """Linear interpolation of (n_steps,) scalar series to n_bins."""
    n_steps = len(series)
    if n_steps == 1:
        return np.full(n_bins, series[0])
    return np.interp(np.linspace(0, 1, n_bins),
                     np.linspace(0, 1, n_steps), series)


def _group_mean_std(scalar_series_list: list[np.ndarray], n_bins: int = 50):
    """Resample each episode's scalar series to n_bins and compute mean/std across episodes.

    Returns (mean (n_bins,), std (n_bins,)) or (None, None) if empty.
    """
    if not scalar_series_list:
        return None, None
    resampled = np.stack([_resample_scalar_series(s, n_bins) for s in scalar_series_list])
    return resampled.mean(axis=0), resampled.std(axis=0)


def fig19_temporal_disc_projection(conceptors_npz, ds=0, task_filter=None,
                                    out_name="temporal_disc_projection",
                                    n_bins: int = 50):
    """Two-row figure: temporal line charts per task.
      Top row: projection onto top eigenvector of C_contrastive over normalized time.
      Bottom row: cosine similarity to baseline-success centroid over normalized time.
      4 lines per panel (baseline succ/fail, steered succ/fail) with ±1σ bands.
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    all_available = [t for t in TASKS
                     if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    if task_filter is not None:
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        available = []
        for t in task_filter:
            if t in all_available:
                available.append(t)
            elif t in short_to_full and short_to_full[t] in all_available:
                available.append(short_to_full[t])
            else:
                print(f"[fig19] Task not found or missing data: {t}")
    else:
        available = all_available

    if not available:
        print("[fig19] No tasks. Skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8), squeeze=False)

    t_axis = np.linspace(0, 1, n_bins)

    for idx, task in enumerate(available):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax_top.axis("off"); ax_bot.axis("off")
            continue
        C = conceptors_npz[key_c]

        # Discriminative axis: top eigenvector of C_contrastive
        eigvals, eigvecs = np.linalg.eigh(C)
        disc_axis = eigvecs[:, -1]  # largest eigenvalue

        # Load trajectories
        s_trajs, f_trajs = load_episode_trajectories(ACTIVATIONS_DIR / task, layer_idx, ds)
        ss_trajs, sf_trajs = load_episode_trajectories(STEERED_ACTIVATIONS_DIR / task, layer_idx, ds)

        if not s_trajs and not f_trajs and not ss_trajs and not sf_trajs:
            ax_top.axis("off"); ax_bot.axis("off")
            continue

        # Baseline-success centroid (pool per-step) for cosine reference
        if s_trajs:
            mu_s = np.concatenate(s_trajs, axis=0).mean(axis=0)
            mu_s_norm = mu_s / (np.linalg.norm(mu_s) + 1e-12)
        else:
            mu_s_norm = None

        # For each group: list of per-episode scalar series (disc projection, cosine to mu_s)
        disc_by_group = {"s": [], "f": [], "ss": [], "sf": []}
        cos_by_group = {"s": [], "f": [], "ss": [], "sf": []}

        def _process(trajs, key):
            for traj in trajs:
                proj = traj @ disc_axis  # (n_steps,)
                disc_by_group[key].append(proj)
                if mu_s_norm is not None:
                    norms = np.linalg.norm(traj, axis=1) + 1e-12
                    cos = (traj @ mu_s_norm) / norms
                    cos_by_group[key].append(cos)

        _process(s_trajs, "s")
        _process(f_trajs, "f")
        _process(ss_trajs, "ss")
        _process(sf_trajs, "sf")

        # Plot top row: discriminative projection over time
        plot_spec = [
            ("f", C_FAILURE, "-", "Baseline failure"),
            ("s", C_SUCCESS, "-", "Baseline success"),
            ("sf", C_STEERED_FAIL, "--", "Steered failure"),
            ("ss", C_STEERED, "--", "Steered success"),
        ]
        for key, color, ls, _ in plot_spec:
            m, s = _group_mean_std(disc_by_group[key], n_bins)
            if m is None:
                continue
            ax_top.plot(t_axis, m, color=color, ls=ls, lw=2.2, zorder=5)
            ax_top.fill_between(t_axis, m - s, m + s, color=color, alpha=0.15, zorder=3)

        ax_top.set_title(f"{short}\n(L{L})", fontsize=10, fontweight="bold")
        if idx == 0:
            ax_top.set_ylabel(r"Projection onto $\mathbf{v}_1(C_{\mathrm{contrast}})$",
                              fontsize=9)
        ax_top.tick_params(labelsize=7)
        ax_top.grid(alpha=0.25, lw=0.5)

        # Plot bottom row: cosine similarity to baseline-success centroid over time
        if mu_s_norm is not None:
            for key, color, ls, _ in plot_spec:
                m, s = _group_mean_std(cos_by_group[key], n_bins)
                if m is None:
                    continue
                ax_bot.plot(t_axis, m, color=color, ls=ls, lw=2.2, zorder=5)
                ax_bot.fill_between(t_axis, m - s, m + s, color=color, alpha=0.15, zorder=3)
        ax_bot.set_xlabel("Normalized trajectory time", fontsize=9)
        if idx == 0:
            ax_bot.set_ylabel(r"$\cos(\mathbf{h}_t, \boldsymbol{\mu}_{\mathrm{succ}})$",
                              fontsize=9)
        ax_bot.tick_params(labelsize=7)
        ax_bot.grid(alpha=0.25, lw=0.5)

    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=C_SUCCESS,       ls="-",  lw=2.2, label="Baseline success"),
        mlines.Line2D([], [], color=C_FAILURE,       ls="-",  lw=2.2, label="Baseline failure"),
        mlines.Line2D([], [], color=C_STEERED,       ls="--", lw=2.2, label="Steered success"),
        mlines.Line2D([], [], color=C_STEERED_FAIL,  ls="--", lw=2.2, label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    fig.suptitle("Temporal trajectory analysis: projection onto discriminative axis (top) "
                 "and cosine similarity to baseline-success centroid (bottom)",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, out_name)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 18: Per-episode 4-group stacked (temporal-phase embeddings)
# ──────────────────────────────────────────────────────────────────────────────

def fig18_trajectory_4groups_stacked(conceptors_npz, ds=0, task_filter=None,
                                      out_name="trajectory_4groups_stacked",
                                      embedding_mode: str = "phases"):
    """Two-row figure:
      Top row: per-step activations in discriminative PCA (same as fig16).
      Bottom row: per-episode trajectory embeddings, colored by same 4 groups
                  (baseline succ/fail + steered succ/fail) as the top row.
                  Bottom row uses its OWN 2D PCA on the trajectory embeddings
                  so each panel is zoomed to the data.
    """
    if not _check_steered_data():
        return

    results = load_steering_results()
    all_available = [t for t in TASKS
                     if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    if task_filter is not None:
        short_to_full = {v: k for k, v in TASK_SHORT.items()}
        available = []
        for t in task_filter:
            if t in all_available:
                available.append(t)
            elif t in short_to_full and short_to_full[t] in all_available:
                available.append(short_to_full[t])
            else:
                print(f"[fig18] Task not found or missing data: {t}")
    else:
        available = all_available

    if not available:
        print("[fig18] No tasks. Skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 9.5), squeeze=False)

    for idx, task in enumerate(available):
        ax_top = axes[0, idx]
        ax_bot = axes[1, idx]
        short = TASK_SHORT.get(task, task[:25])
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        a = r.get("best_global_a", 1.0)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            ax_top.set_title(f"{short}\n(insufficient data)", fontsize=9)
            ax_top.axis("off"); ax_bot.axis("off")
            continue

        key_c = f"{task}__L{L}__{a}__C_contrastive"
        if key_c not in conceptors_npz:
            ax_top.axis("off"); ax_bot.axis("off")
            continue
        C = conceptors_npz[key_c]

        # --- Top row: per-step activations (same as fig16) ---
        groups = {"s": data["X_s"], "f": data["X_f"],
                  "ss": data["X_ss"], "sf": data["X_sf"]}
        combined = [groups[k] for k in ["s", "f", "ss", "sf"] if groups[k].shape[0] > 0]
        _, V_top, mean_top = discriminative_pca(combined, C, n_components=2)
        proj_top = {k: (groups[k] - mean_top) @ V_top for k in groups if groups[k].shape[0] > 0}

        plot_cfg = [
            ("f", C_FAILURE, "o", 0.2, 8),
            ("s", C_SUCCESS, "o", 0.2, 8),
            ("sf", C_STEERED_FAIL, "^", 0.3, 12),
            ("ss", C_STEERED, "D", 0.3, 12),
        ]
        for key, color, marker, alpha, size in plot_cfg:
            if key in proj_top:
                ax_top.scatter(proj_top[key][:, 0], proj_top[key][:, 1], c=color, s=size,
                               alpha=alpha, edgecolors="none", marker=marker, zorder=3)
        for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                               ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
            if key in proj_top and proj_top[key].shape[0] >= 2:
                draw_ellipse(ax_top, proj_top[key], color, n_std=2.0, alpha_fill=0.06, ls=ls)
        for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                    ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
            if key in proj_top:
                c = proj_top[key].mean(0)
                ax_top.plot(*c, marker, color=color, markersize=12,
                            markeredgecolor="white", markeredgewidth=2, zorder=11)

        ax_top.set_title(f"{short}\n(L{L})", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("Disc. PC 1", fontsize=8)
        if idx == 0:
            ax_top.set_ylabel("Per-step activations\nDisc. PC 2", fontsize=9)
        ax_top.tick_params(labelsize=7)

        # --- Bottom row: per-episode trajectory embeddings, 4 groups ---
        E_s, E_f = load_episode_trajectory_embeddings(
            ACTIVATIONS_DIR / task, layer_idx, ds, mode=embedding_mode)
        E_ss, E_sf = load_episode_trajectory_embeddings(
            STEERED_ACTIVATIONS_DIR / task, layer_idx, ds, mode=embedding_mode)

        ep_groups = {"s": E_s, "f": E_f, "ss": E_ss, "sf": E_sf}
        combined_E = [E for E in ep_groups.values() if E.shape[0] > 0]
        if not combined_E:
            ax_bot.axis("off")
            continue

        # Plain PCA on combined episode embeddings for a per-panel 2D view
        all_E = np.concatenate(combined_E, axis=0)
        e_mean = all_E.mean(0)
        _, _, Vt_e = np.linalg.svd(all_E - e_mean, full_matrices=False)
        V_bot = Vt_e[:2].T

        proj_bot = {k: (ep_groups[k] - e_mean) @ V_bot
                    for k in ep_groups if ep_groups[k].shape[0] > 0}

        # Match top-row styling: same colors and marker shapes, bigger markers for episodes
        ep_plot_cfg = [
            ("f", C_FAILURE, "o", 0.75, 70),
            ("s", C_SUCCESS, "o", 0.75, 70),
            ("sf", C_STEERED_FAIL, "^", 0.85, 90),
            ("ss", C_STEERED, "D", 0.85, 90),
        ]
        for key, color, marker, alpha, size in ep_plot_cfg:
            if key in proj_bot:
                ax_bot.scatter(proj_bot[key][:, 0], proj_bot[key][:, 1], c=color, s=size,
                               alpha=alpha, marker=marker,
                               edgecolors="black", linewidths=0.5, zorder=4)
        for key, color, ls in [("s", C_SUCCESS, "-"), ("f", C_FAILURE, "-"),
                               ("ss", C_STEERED, "--"), ("sf", C_STEERED_FAIL, "--")]:
            if key in proj_bot and proj_bot[key].shape[0] >= 2:
                draw_ellipse(ax_bot, proj_bot[key], color, n_std=1.5, alpha_fill=0.08, ls=ls)
        for key, color, marker in [("s", C_SUCCESS, "o"), ("f", C_FAILURE, "o"),
                                    ("ss", C_STEERED, "D"), ("sf", C_STEERED_FAIL, "^")]:
            if key in proj_bot:
                c = proj_bot[key].mean(0)
                ax_bot.plot(*c, marker, color=color, markersize=14,
                            markeredgecolor="white", markeredgewidth=2, zorder=11)

        # Episode counts
        counts = f"s={E_s.shape[0]}, f={E_f.shape[0]}, ss={E_ss.shape[0]}, sf={E_sf.shape[0]}"
        ax_bot.set_title(counts, fontsize=8)
        ax_bot.set_xlabel("Traj. PC 1", fontsize=8)
        if idx == 0:
            ax_bot.set_ylabel("Per-episode trajectories\nTraj. PC 2", fontsize=9)
        ax_bot.tick_params(labelsize=7)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_SUCCESS, alpha=0.5, label="Baseline success"),
        mpatches.Patch(color=C_FAILURE, alpha=0.5, label="Baseline failure"),
        mpatches.Patch(color=C_STEERED, alpha=0.5, label="Steered success"),
        mpatches.Patch(color=C_STEERED_FAIL, alpha=0.5, label="Steered failure"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    mode_desc = {
        "mean": "mean-pooled",
        "phases": "3-phase (early/mid/late)",
        "mean_std": "[mean, std]",
    }.get(embedding_mode, embedding_mode)
    fig.suptitle(f"Top: per-step activations   |   Bottom: per-episode trajectories ({mode_desc} embedding)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, out_name)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 15: Cosine similarity to success centroid (violin/box plots)
# ──────────────────────────────────────────────────────────────────────────────

def fig15_cosine_similarity(conceptors_npz, ds=0):
    """Violin plots: cos(h, mu_success) distribution for each of the 4 groups."""
    if not _check_steered_data():
        return

    results = load_steering_results()
    available = [t for t in TASKS
                 if (ACTIVATIONS_DIR / t).is_dir() and (STEERED_ACTIVATIONS_DIR / t).is_dir()]

    # Aggregate cosine similarities across all tasks
    cos_sims = {"Baseline\nsuccess": [], "Baseline\nfailure": [],
                "Steered\nsuccess": [], "Steered\nfailure": []}
    # Also per-task for subplot version
    per_task_cos = {}

    for task in available:
        r = results.get(task, {})
        L = r.get("best_global_L", 11)
        layer_idx = LAYER_MAP[L]

        data = _load_all_four_groups(task, layer_idx, ds, conceptors_npz, results)
        if data is None:
            continue

        # Success centroid from baseline
        mu_s = data["X_s"].mean(0)
        mu_s_norm = mu_s / (np.linalg.norm(mu_s) + 1e-12)

        task_cos = {}
        for label, X in [("Baseline\nsuccess", data["X_s"]),
                         ("Baseline\nfailure", data["X_f"]),
                         ("Steered\nsuccess", data["X_ss"]),
                         ("Steered\nfailure", data["X_sf"])]:
            if X.shape[0] > 0:
                norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                cosines = (X / norms) @ mu_s_norm
                cos_sims[label].extend(cosines.tolist())
                task_cos[label] = cosines
        per_task_cos[TASK_SHORT.get(task, task[:20])] = task_cos

    # ---- Aggregate violin plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    group_labels = ["Baseline\nsuccess", "Baseline\nfailure",
                    "Steered\nsuccess", "Steered\nfailure"]
    colors = [C_SUCCESS, C_FAILURE, C_STEERED, C_STEERED_FAIL]
    plot_data = [np.array(cos_sims[g]) for g in group_labels]

    # Filter out empty groups
    valid = [(g, d, c) for g, d, c in zip(group_labels, plot_data, colors) if len(d) > 0]
    if len(valid) < 2:
        print("[fig15] Insufficient data. Skipping.")
        return

    positions = list(range(len(valid)))
    parts = ax.violinplot([d for _, d, _ in valid], positions=positions,
                          showmeans=True, showmedians=True, showextrema=False)
    for i, (body_key, pc) in enumerate(zip(parts["bodies"], [c for _, _, c in valid])):
        body_key.set_facecolor(pc)
        body_key.set_alpha(0.4)
        body_key.set_edgecolor(pc)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")

    ax.set_xticks(positions)
    ax.set_xticklabels([g for g, _, _ in valid], fontsize=10)
    ax.set_ylabel(r"$\cos(\mathbf{h}, \boldsymbol{\mu}_{\mathrm{success}})$", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("Cosine Similarity to Baseline Success Centroid\n"
                 "Steering shifts failure activations toward the success direction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "cosine_similarity_violin")

    # ---- Per-task box plot version ----
    if len(per_task_cos) >= 2:
        n_tasks = len(per_task_cos)
        fig2, axes2 = plt.subplots(1, n_tasks, figsize=(2.2 * n_tasks, 4.5),
                                    sharey=True, squeeze=False)
        for i, (tname, task_cos) in enumerate(per_task_cos.items()):
            ax2 = axes2[0, i]
            box_data, box_colors, box_labels = [], [], []
            for label, color in zip(group_labels, colors):
                if label in task_cos and len(task_cos[label]) > 0:
                    box_data.append(task_cos[label])
                    box_colors.append(color)
                    box_labels.append(label.replace("\n", " "))

            if box_data:
                bp = ax2.boxplot(box_data, patch_artist=True, widths=0.6,
                                 showfliers=False, medianprops=dict(color="black"))
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
            ax2.set_title(tname, fontsize=8, fontweight="bold")
            ax2.set_xticks([])
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            if i == 0:
                ax2.set_ylabel(r"$\cos(\mathbf{h}, \boldsymbol{\mu}_{\mathrm{success}})$",
                               fontsize=9)

        handles = [mpatches.Patch(color=c, alpha=0.5, label=l.replace("\n", " "))
                   for l, c in zip(group_labels, colors)]
        fig2.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                    bbox_to_anchor=(0.5, -0.06))
        fig2.suptitle("Per-Task Cosine Similarity to Success Centroid",
                      fontsize=12, fontweight="bold", y=1.02)
        fig2.tight_layout()
        _save(fig2, "cosine_similarity_per_task")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300, pad_inches=0.1)
        print(f"  Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Conceptor Ellipsoid Visualization")
    print("=" * 70)

    LAYER, ALPHA, BETA, DS = 11, 1.0, 0.3, 0

    print(f"\nDefault params: layer={LAYER}, alpha={ALPHA}, beta={BETA}, ds={DS}")
    print(f"Activations: {ACTIVATIONS_DIR}")
    print(f"Conceptors:  {CONCEPTORS_PATH}")
    print(f"Output:      {OUTPUT_DIR}")

    avail = [t for t in TASKS if (ACTIVATIONS_DIR / t).is_dir()]
    print(f"\nAvailable tasks: {len(avail)}/{len(TASKS)}")
    for t in avail:
        print(f"  + {TASK_SHORT.get(t, t[:40])}")

    if not avail:
        print("\nERROR: No activation directories found!")
        return

    print(f"\nLoading conceptors...")
    cnpz = np.load(CONCEPTORS_PATH, allow_pickle=True)
    print(f"  {len(cnpz.files)} arrays")

    for name, fn, args in [
        ("Fig 1: PCA ellipsoids", fig1_pca_ellipsoids, (cnpz, LAYER, ALPHA, BETA, DS)),
        ("Fig 2: Subspace energy", fig2_subspace_energy, (cnpz, LAYER, ALPHA, BETA, DS)),
        ("Fig 3: Eigenspectrum", fig3_eigenspectrum, (cnpz, LAYER, ALPHA)),
        ("Fig 4: Paper panel", fig4_paper_panel, (cnpz, DS)),
        ("Fig 5: Shift scatter", fig5_shift_scatter, (cnpz, LAYER, ALPHA, BETA, DS)),
        ("Fig 6: Conceptor geometry", fig6_conceptor_geometry, (cnpz, LAYER, ALPHA, BETA, DS)),
        ("Fig 7: Steered success", fig7_steered_success, (cnpz, DS)),
        ("Fig 8+9: Joint t-SNE (steered success & failure)", fig8_9_joint_tsne, (cnpz, DS)),
        # ---- Real steered activations (skip gracefully if not collected yet) ----
        ("Fig 10: 4-group t-SNE (real steered)", fig10_four_group_tsne, (cnpz, DS)),
        ("Fig 11: 4-group subspace energy bars", fig11_four_group_energy, (cnpz, DS)),
        ("Fig 12: Shift vs. performance gain", fig12_shift_vs_gain, (cnpz, DS)),
        ("Fig 13: Centroid trajectories", fig13_centroid_trajectories, (cnpz, DS)),
        ("Fig 14: Population-level view", fig14_population_view, (cnpz, DS)),
        ("Fig 15: Cosine similarity", fig15_cosine_similarity, (cnpz, DS)),
        ("Fig 16: Per-task population view", fig16_per_task_population_view, (cnpz, DS)),
    ]:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        fn(*args)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
