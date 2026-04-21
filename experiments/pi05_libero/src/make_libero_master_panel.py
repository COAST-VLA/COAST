"""Master 1x4 diagnostic panel for the paper.

Story: three cheap, GPU-free diagnostics jointly pick (layer, alpha, beta)
for conceptor steering without a full sweep.

  (a) Quota per layer + per-layer mean success  → pick the layer
  (b) Overlap vs success (bar chart at L*)       → overlap as guard-rail
  (c) Aperture alpha vs overlap (per layer)      → alpha controls overlap
  (d) Alpha x beta success heatmap at L*         → joint validation
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
QUOTA_ALPHA = 10.0
CONCEPTOR_TYPE_FOR_QUOTA = "contrastive"

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

LAYER_COLORS = {L: c for L, c in
                zip(LAYERS_CONCEPTOR, plt.cm.viridis([0.15, 0.4, 0.65, 0.9]))}

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


# ── Load conceptors; compute quota + overlap ────────────────────
print("Loading conceptors ...")
conceptors = np.load(CONCEPTORS_PATH, allow_pickle=True)


def quota(task: str, layer: int, alpha: float) -> float:
    C = conceptors[f"{task}__L{layer}__{alpha}__C_{CONCEPTOR_TYPE_FOR_QUOTA}"]
    return float(np.trace(C)) / C.shape[0]


def overlap(task: str, layer: int, alpha: float) -> float:
    Cs = conceptors[f"{task}__L{layer}__{alpha}__C_success"]
    Cf = conceptors[f"{task}__L{layer}__{alpha}__C_failure"]
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    return num / np.sqrt(ns * nf)


print("Computing quotas and overlaps ...")
Q = {(t, L): quota(t, L, QUOTA_ALPHA) for t in TASKS for L in LAYERS_CONCEPTOR}
O = {(t, L, a): overlap(t, L, a)
     for t in TASKS for L in LAYERS_CONCEPTOR for a in ALPHAS}

# ── Load steering success rates (global strategy) ───────────────
print("Loading steering results ...")
cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")
S = {t: {} for t in TASKS}
for t in TASKS:
    with open(os.path.join(result_dir_for(t), "summary.json")) as fh:
        data = json.load(fh)
    for e in data["conditions"]:
        m = cond_re.match(e["condition"])
        if m:
            S[t][(int(m.group(1)), float(m.group(2)), float(m.group(3)))] = \
                float(e["success_rate"])

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 3.5),
                         gridspec_kw={"wspace": 0.45})
ax_a, ax_b, ax_c, ax_d = axes

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, direction="out")


def panel_label(ax, txt, dx=-0.22):
    ax.text(dx, 1.08, txt, transform=ax.transAxes,
            fontsize=12, fontweight="bold")


# ── (a) Quota per layer (box) + mean success per layer (line) ───
# Only layers with steering success (L5, L11, L17) so both axes share support.
quota_by_layer = [[Q[(t, L)] for t in TASKS] for L in LAYERS_STEER]
bp = ax_a.boxplot(
    quota_by_layer, positions=range(len(LAYERS_STEER)),
    widths=0.55, patch_artist=True, showfliers=False,
    medianprops=dict(color="black", lw=1.2),
    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    boxprops=dict(lw=0.6),
)
for patch, L in zip(bp["boxes"], LAYERS_STEER):
    patch.set_facecolor(LAYER_COLORS[L]); patch.set_alpha(0.55)

ax_a.set_xticks(range(len(LAYERS_STEER)))
ax_a.set_xticklabels([f"L{L}" for L in LAYERS_STEER])
ax_a.set_xlabel("Layer", fontsize=11)
ax_a.set_ylabel(r"Conceptor Quota $q(C)$", fontsize=11, color="#2b4a78")
ax_a.tick_params(axis="y", labelcolor="#2b4a78")
ax_a.set_title("Quota and Success per Layer", fontsize=11, fontweight="bold")

# Right axis: per-layer mean success rate across tasks and (alpha, beta).
ax_ar = ax_a.twinx()
ax_ar.spines["top"].set_visible(False)
ax_ar.spines["right"].set_visible(True)
ax_ar.spines["right"].set_color("#b0352e")
ax_ar.spines["left"].set_visible(False)
succ_by_layer = {L: [S[t][(L, a, b)] for t in TASKS
                     for a in ALPHAS for b in BETAS]
                 for L in LAYERS_STEER}
mu = [np.mean(succ_by_layer[L]) for L in LAYERS_STEER]
sd = [np.std(succ_by_layer[L]) / np.sqrt(len(succ_by_layer[L]))
      for L in LAYERS_STEER]
ax_ar.errorbar(range(len(LAYERS_STEER)), mu, yerr=sd,
               color="#b0352e", lw=1.8, marker="s", markersize=6,
               capsize=3, label="Mean success")
ax_ar.set_ylabel("Mean Success Rate", fontsize=11, color="#b0352e")
ax_ar.tick_params(axis="y", labelsize=9, labelcolor="#b0352e",
                  direction="out")
ax_ar.set_ylim(0, max(mu) * 1.25)
panel_label(ax_a, "(a)")

best_L = LAYERS_STEER[int(np.argmax(mu))]

# ── (b) Success per overlap at L=best_L (bar chart) ─────────────
alpha_ovl_b = [float(np.mean([O[(t, best_L, a)] for t in TASKS])) for a in ALPHAS]
task_means_b = []
for a in ALPHAS:
    per_task = [float(np.mean([S[t][(best_L, a, b)] for b in BETAS]))
                for t in TASKS]
    task_means_b.append(per_task)
bar_mu = np.array([np.mean(tm) for tm in task_means_b])
bar_se = np.array([np.std(tm) / np.sqrt(len(tm)) for tm in task_means_b])

# Sort by overlap ascending
order = np.argsort(alpha_ovl_b)
ovs_b = [alpha_ovl_b[i] for i in order]
mus_b = bar_mu[order]
ses_b = bar_se[order]
als_b = [ALPHAS[i] for i in order]

# Color: gold for sweet-spot overlap, light blue otherwise
colors_b = ["#f6ad55" if 0.85 <= ov <= 0.95 else "#90cdf4" for ov in ovs_b]

bars_b = ax_b.bar(range(len(ovs_b)), mus_b, yerr=ses_b, capsize=3,
                  color=colors_b, edgecolor="black", linewidth=0.6)
ax_b.set_xticks(range(len(ovs_b)))
ax_b.set_xticklabels([f"{ov:.2f}" for ov in ovs_b], fontsize=9)

# Annotate alpha values above each bar
for i, (a, m_val, s_val) in enumerate(zip(als_b, mus_b, ses_b)):
    ax_b.text(i, m_val + s_val + 0.015, rf"$\alpha\!=$\!{a:g}",
              ha="center", va="bottom", fontsize=7.5, color="#555")

ax_b.set_xlabel(r"Overlap $\mathrm{sim}(C_s, C_f)$ at $L{=}%d$" % best_L,
                fontsize=11)
ax_b.set_ylabel("Mean Success Rate", fontsize=11)
ax_b.set_title("Overlap vs. Success", fontsize=11, fontweight="bold")
panel_label(ax_b, "(b)")

# ── (c) Alpha vs overlap (per layer, mean ± std over tasks) ─────
for L in LAYERS_CONCEPTOR:
    Y = np.array([[O[(t, L, a)] for a in ALPHAS] for t in TASKS])
    m, s = Y.mean(0), Y.std(0)
    c = LAYER_COLORS[L]
    ax_c.plot(ALPHAS, m, color=c, lw=1.6, marker="o", markersize=4,
              label=f"L{L}")
    ax_c.fill_between(ALPHAS, m - s, m + s, color=c, alpha=0.15, linewidth=0)
ax_c.set_xscale("log")
ax_c.set_xticks(ALPHAS)
ax_c.set_xticklabels([f"{a:g}" for a in ALPHAS])
ax_c.set_xlabel(r"Aperture $\alpha$", fontsize=11)
ax_c.set_ylabel(r"Overlap $\mathrm{sim}(C_s, C_f)$", fontsize=11)
ax_c.set_title("Aperture Controls Overlap", fontsize=11, fontweight="bold")
ax_c.legend(fontsize=7.5, frameon=False, loc="lower left",
            title="Layer", title_fontsize=8)
panel_label(ax_c, "(c)")

# ── (d) Alpha x beta success heatmap at the best layer, with each
#        alpha column annotated by its mean overlap across tasks. ──
ab_H = np.zeros((len(BETAS), len(ALPHAS)))
for j, a in enumerate(ALPHAS):
    for i, b in enumerate(BETAS):
        ab_H[i, j] = np.mean([S[t][(best_L, a, b)] for t in TASKS])

# Mean overlap per alpha across the 10 tasks at best_L.
alpha_ovl = [float(np.mean([O[(t, best_L, a)] for t in TASKS])) for a in ALPHAS]

im = ax_d.imshow(ab_H, cmap="YlOrRd", aspect="auto", origin="lower",
                 vmin=0, vmax=1)
for i in range(ab_H.shape[0]):
    for j in range(ab_H.shape[1]):
        v = ab_H[i, j]
        col = "white" if v > 0.6 else "black"
        ax_d.text(j, i, f"{v:.2f}", ha="center", va="center",
                  fontsize=8, color=col)
# Primary x-axis: alpha values.
ax_d.set_xticks(range(len(ALPHAS)))
ax_d.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=9)
ax_d.set_yticks(range(len(BETAS)))
ax_d.set_yticklabels([f"{b:g}" for b in BETAS])
ax_d.set_xlabel(r"Aperture $\alpha$", fontsize=11)
ax_d.set_ylabel(r"Steering Strength $\beta$", fontsize=11)
ax_d.set_title(rf"Success at $L{{=}}{best_L}$",
               fontsize=11, fontweight="bold")
# Secondary x-axis below showing mean overlap per alpha column.
sec = ax_d.secondary_xaxis(-0.22)
sec.set_xticks(range(len(ALPHAS)))
sec.set_xticklabels([f"{o:.2f}" for o in alpha_ovl], fontsize=8,
                    color="#555")
sec.tick_params(length=0, pad=2, colors="#555")
sec.set_xlabel(r"mean overlap at $L{=}%d$" % best_L,
               fontsize=9, color="#555", labelpad=2)
sec.spines["bottom"].set_visible(False)
cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cbar.set_label("Success Rate", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.6)
panel_label(ax_d, "(d)")

# ── Save ────────────────────────────────────────────────────────
pdf = os.path.join(OUTPUT_DIR, "libero_master_panel.pdf")
png = os.path.join(OUTPUT_DIR, "libero_master_panel.png")
fig.savefig(pdf, bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig(png, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"Saved: {pdf}")
print(f"Saved: {png}")
