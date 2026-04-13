"""Overlap-driven diagnostic charts for picking conceptor (alpha, beta) cheaply.

Story: conceptor overlap sim(C_s, C_f) is a one-shot, GPU-free statistic that
parameterises the aperture alpha. If overlap predicts steering success, and if
there is a consistent best beta region, one can pick (alpha, beta) without a
full GPU sweep.

Panels:
  (a) Overlap vs alpha (per layer, mean over tasks +/- std).
  (b) Overlap vs mean success rate (scatter, one point per (task, layer, alpha),
      with Pearson r).
  (c) Success rate vs beta, stratified by overlap tertile (low/mid/high).
  (d) Heatmap of mean success rate in (overlap_bin x beta) space.
  (e) Full-sweep vs overlap-guided pick: success with cheap rule vs best
      success from the full (alpha, beta) grid, per (task, layer).
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CONCEPTORS_PATH = "/vast/projects/ungar/stellar/miaom/.cache/openpi/libero_conceptors.npz"
RESULTS_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results"
OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYERS_CONCEPTOR = [0, 5, 11, 17]
LAYERS_STEER = [5, 11, 17]
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
BETAS = [0.1, 0.3, 0.5]

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

COLORS = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))
LAYER_COLORS = {L: c for L, c in zip(LAYERS_CONCEPTOR, plt.cm.viridis([0.15, 0.4, 0.65, 0.9]))}

_all_result_dirs = [d for d in os.listdir(RESULTS_DIR)
                    if os.path.isdir(os.path.join(RESULTS_DIR, d))
                    and d not in ("scripts", "logs")]


def result_dir_for(task: str) -> str:
    if task in _all_result_dirs:
        return os.path.join(RESULTS_DIR, task)
    for d in _all_result_dirs:
        if task.startswith(d) or d.startswith(task[:40]):
            return os.path.join(RESULTS_DIR, d)
    raise FileNotFoundError(f"No result dir for {task}")


# ── Load conceptors and compute overlaps ────────────────────────
print("Loading conceptors and computing overlaps ...")
conceptors = np.load(CONCEPTORS_PATH, allow_pickle=True)


def overlap(Cs: np.ndarray, Cf: np.ndarray) -> float:
    # Normalised Frobenius inner product: tr(Cs Cf) / sqrt(tr(Cs^2) tr(Cf^2))
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    return num / np.sqrt(ns * nf)


ovl = {}  # ovl[(task, layer, alpha)] = overlap
for t in TASKS:
    for L in LAYERS_CONCEPTOR:
        for a in ALPHAS:
            Cs = conceptors[f"{t}__L{L}__{a}__C_success"]
            Cf = conceptors[f"{t}__L{L}__{a}__C_failure"]
            ovl[(t, L, a)] = overlap(Cs, Cf)

# ── Load steering success rates (global strategy only) ──────────
print("Loading steering results ...")
cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
success = {t: {} for t in TASKS}
for t in TASKS:
    with open(os.path.join(result_dir_for(t), "summary.json")) as fh:
        data = json.load(fh)
    for entry in data["conditions"]:
        m = cond_re.match(entry["condition"])
        if not m:
            continue
        L, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3))
        success[t][(L, a, b)] = float(entry["success_rate"])

# ── Derived tables ──────────────────────────────────────────────
# Per (task, layer, alpha): overlap and mean-over-beta success.
rows = []  # list of (task, layer, alpha, beta, success, overlap)
for t in TASKS:
    for L in LAYERS_STEER:
        for a in ALPHAS:
            o = ovl[(t, L, a)]
            for b in BETAS:
                rows.append((t, L, a, b, success[t][(L, a, b)], o))
# columns: task idx, layer, alpha, beta, success, overlap
r_task = np.array([TASKS.index(r[0]) for r in rows])
r_layer = np.array([r[1] for r in rows])
r_alpha = np.array([r[2] for r in rows])
r_beta = np.array([r[3] for r in rows])
r_succ = np.array([r[4] for r in rows])
r_ovl = np.array([r[5] for r in rows])

# ── Figure: 2 rows x 3 cols (last cell blank, 5 panels shown) ───
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, wspace=0.40, hspace=0.45)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, direction="out")


def panel_label(ax, txt):
    ax.text(-0.18, 1.07, txt, transform=ax.transAxes,
            fontsize=12, fontweight="bold")


# ── Panel (a): Overlap vs alpha, mean +/- std over tasks, per layer.
for L in LAYERS_CONCEPTOR:
    y = np.array([[ovl[(t, L, a)] for a in ALPHAS] for t in TASKS])
    mu = y.mean(0)
    sd = y.std(0)
    c = LAYER_COLORS[L]
    ax1.plot(ALPHAS, mu, color=c, lw=1.5, marker="o", markersize=4,
             label=f"L{L}")
    ax1.fill_between(ALPHAS, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
ax1.set_xscale("log")
ax1.set_xticks(ALPHAS)
ax1.set_xticklabels([f"{a:g}" for a in ALPHAS])
ax1.set_xlabel(r"Aperture $\alpha$", fontsize=11)
ax1.set_ylabel(r"Overlap $\mathrm{sim}(C_s, C_f)$", fontsize=11)
ax1.set_title("Overlap vs. Aperture", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, frameon=False, loc="best", title="Layer",
           title_fontsize=8)
panel_label(ax1, "(a)")

# ── Panel (b): Overlap vs success (mean over beta).
# One point per (task, layer, alpha); y = mean over beta.
b_ovl, b_succ, b_layer, b_task = [], [], [], []
for t in TASKS:
    for L in LAYERS_STEER:
        for a in ALPHAS:
            b_ovl.append(ovl[(t, L, a)])
            b_succ.append(np.mean([success[t][(L, a, b)] for b in BETAS]))
            b_layer.append(L)
            b_task.append(TASKS.index(t))
b_ovl = np.array(b_ovl); b_succ = np.array(b_succ)
b_layer = np.array(b_layer); b_task = np.array(b_task)

for L in LAYERS_STEER:
    m = b_layer == L
    ax2.scatter(b_ovl[m], b_succ[m], s=30, color=LAYER_COLORS[L],
                alpha=0.75, edgecolors="black", linewidths=0.3,
                label=f"L{L}")
r, p = stats.pearsonr(b_ovl, b_succ)
slope, intercept = np.polyfit(b_ovl, b_succ, 1)
xf = np.linspace(b_ovl.min(), b_ovl.max(), 100)
ax2.plot(xf, slope * xf + intercept, "k--", lw=1.0)
ann_x, ha = (0.95, "right") if r > 0 else (0.05, "left")
ax2.text(ann_x, 0.06, rf"$r = {r:.2f}$,  $p = {p:.1e}$",
         transform=ax2.transAxes, fontsize=9, ha=ha)
ax2.set_xlabel(r"Overlap $\mathrm{sim}(C_s, C_f)$", fontsize=11)
ax2.set_ylabel(r"Mean Success Rate (over $\beta$)", fontsize=11)
ax2.set_title("Overlap vs. Steering Success", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, frameon=False, title="Layer", title_fontsize=8)
panel_label(ax2, "(b)")

# ── Panel (c): Success vs beta, stratified by overlap tertile.
# For each (task, layer, alpha), we have one overlap and 3 beta points.
# Assign tertile by overlap across all (task, layer, alpha).
unique_tla = [(t, L, a) for t in TASKS for L in LAYERS_STEER for a in ALPHAS]
ov_vals = np.array([ovl[k] for k in unique_tla])
q1, q2 = np.quantile(ov_vals, [1/3, 2/3])

tertile_data = {"low": defaultdict(list), "mid": defaultdict(list),
                "high": defaultdict(list)}
for (t, L, a) in unique_tla:
    o = ovl[(t, L, a)]
    bucket = "low" if o <= q1 else ("mid" if o <= q2 else "high")
    for b in BETAS:
        tertile_data[bucket][b].append(success[t][(L, a, b)])

tert_colors = {"low": "#3b7dd8", "mid": "#f2a541", "high": "#c73e3a"}
tert_labels = {
    "low":  rf"low (ovl $\leq$ {q1:.2f})",
    "mid":  rf"mid ({q1:.2f}–{q2:.2f})",
    "high": rf"high (ovl $>$ {q2:.2f})",
}
for bucket in ["low", "mid", "high"]:
    mus = np.array([np.mean(tertile_data[bucket][b]) for b in BETAS])
    ses = np.array([np.std(tertile_data[bucket][b]) /
                    np.sqrt(len(tertile_data[bucket][b])) for b in BETAS])
    ax3.plot(BETAS, mus, color=tert_colors[bucket], lw=1.5,
             marker="o", markersize=5, label=tert_labels[bucket])
    ax3.fill_between(BETAS, mus - ses, mus + ses,
                     color=tert_colors[bucket], alpha=0.18, linewidth=0)
ax3.set_xticks(BETAS)
ax3.set_xlabel(r"Steering Strength $\beta$", fontsize=11)
ax3.set_ylabel("Mean Success Rate", fontsize=11)
ax3.set_title(r"Success vs. $\beta$ by Overlap Tertile",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, frameon=False, title="Overlap regime",
           title_fontsize=8, loc="best")
panel_label(ax3, "(c)")

# ── Panel (d): Heatmap (overlap_bin x beta) → mean success.
n_ovl_bins = 4
ovl_edges = np.quantile(r_ovl, np.linspace(0, 1, n_ovl_bins + 1))
ovl_edges[0] -= 1e-9; ovl_edges[-1] += 1e-9
heatmap = np.full((n_ovl_bins, len(BETAS)), np.nan)
for i in range(n_ovl_bins):
    lo, hi = ovl_edges[i], ovl_edges[i + 1]
    for j, b in enumerate(BETAS):
        m = (r_ovl > lo) & (r_ovl <= hi) & (r_beta == b)
        if m.sum():
            heatmap[i, j] = r_succ[m].mean()

im = ax4.imshow(heatmap, cmap="YlOrRd", aspect="auto",
                origin="lower", vmin=0, vmax=1)
for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        v = heatmap[i, j]
        color = "white" if v > 0.6 else "black"
        ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=8, color=color)
ax4.set_xticks(range(len(BETAS)))
ax4.set_xticklabels([f"{b:g}" for b in BETAS])
ylabels = [f"[{ovl_edges[i]:.2f}, {ovl_edges[i+1]:.2f}]"
           for i in range(n_ovl_bins)]
ax4.set_yticks(range(n_ovl_bins))
ax4.set_yticklabels(ylabels)
ax4.set_xlabel(r"Steering Strength $\beta$", fontsize=11)
ax4.set_ylabel("Overlap Quartile", fontsize=11)
ax4.set_title("Sweet-Spot Map", fontsize=11, fontweight="bold")
cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label("Success Rate", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.6)
panel_label(ax4, "(d)")

# Identify sweet-spot cell.
best_cell = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
best_ovl_lo, best_ovl_hi = ovl_edges[best_cell[0]], ovl_edges[best_cell[0] + 1]
best_beta = BETAS[best_cell[1]]
print(f"Sweet-spot: overlap in [{best_ovl_lo:.2f}, {best_ovl_hi:.2f}], "
      f"beta={best_beta} → mean success {heatmap[best_cell]:.2f}")

# ── Panel (e): Cheap pick vs full sweep.
# Rule: for each (task, layer), pick α whose overlap is closest to the midpoint
# of the sweet-spot overlap bin; fix β = best_beta. Compare that success rate
# to the best success rate over the full (α, β) grid.
target_ovl = 0.5 * (best_ovl_lo + best_ovl_hi)

cheap, best, label_tl = [], [], []
for t in TASKS:
    for L in LAYERS_STEER:
        ovs = np.array([ovl[(t, L, a)] for a in ALPHAS])
        a_pick = ALPHAS[int(np.argmin(np.abs(ovs - target_ovl)))]
        cheap.append(success[t][(L, a_pick, best_beta)])
        best.append(max(success[t][(L, a, b)] for a in ALPHAS for b in BETAS))
        label_tl.append((t, L))
cheap = np.array(cheap); best = np.array(best)
regret = best - cheap

# Colour dots by layer for readability.
for L in LAYERS_STEER:
    m = np.array([tl[1] == L for tl in label_tl])
    ax5.scatter(best[m], cheap[m], s=40, color=LAYER_COLORS[L],
                alpha=0.8, edgecolors="black", linewidths=0.3,
                label=f"L{L}")
lims = [0, 1]
ax5.plot(lims, lims, "k--", lw=0.8, alpha=0.7, label="y = x")
ax5.set_xlim(lims); ax5.set_ylim(lims)
ax5.set_xlabel("Best Success (full 5×3 sweep)", fontsize=11)
ax5.set_ylabel("Cheap-Pick Success", fontsize=11)
ax5.set_title("Cheap Rule vs. Full Sweep", fontsize=11, fontweight="bold")

# Stats annotation for panel e.
mean_regret = regret.mean()
median_regret = np.median(regret)
# GPU cost ratio: cheap rule evaluates 1 cell instead of 15 per (task, layer).
cost_ratio = 1.0 / (len(ALPHAS) * len(BETAS))
ann = (
    rf"mean regret $=$ {mean_regret:.2f}"
    "\n"
    rf"median regret $=$ {median_regret:.2f}"
    "\n"
    rf"GPU cost $=$ {cost_ratio:.0%} of full sweep"
)
ax5.text(0.05, 0.75, ann, transform=ax5.transAxes, fontsize=9,
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="lightgray", linewidth=0.6))
ax5.legend(fontsize=8, frameon=False, loc="lower right", title="Layer",
           title_fontsize=8)
panel_label(ax5, "(e)")

# Hide the unused 6th cell.
fig.add_subplot(gs[1, 2]).axis("off")

# ── Save ────────────────────────────────────────────────────────
pdf = os.path.join(OUTPUT_DIR, "libero_overlap_diagnostics.pdf")
png = os.path.join(OUTPUT_DIR, "libero_overlap_diagnostics.png")
fig.savefig(pdf, bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig(png, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"Saved: {pdf}")
print(f"Saved: {png}")

# Dump underlying table for further analysis.
import csv
csv_path = os.path.join(OUTPUT_DIR, "libero_overlap_table.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "layer", "alpha", "beta", "overlap", "success"])
    for (t, L, a, b, s, o) in rows:
        w.writerow([t, L, a, b, o, s])
print(f"Saved: {csv_path}")
