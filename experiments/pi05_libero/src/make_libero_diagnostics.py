"""Generate LIBERO diagnostic figure: quota, correlation, and sensitivity.

Panel A: conceptor quota across Gemma action-expert layers (one line per task).
Panel B: quota vs. steering success rate (one point per task-layer).
Panel C: heatmap of success rate across (aperture alpha, steering strength beta)
         at the best layer, averaged over the 10 LIBERO tasks.
"""

import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── RC params ───────────────────────────────────────────────────
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

# ── Paths ───────────────────────────────────────────────────────
CONCEPTORS_PATH = "/vast/projects/ungar/stellar/miaom/.cache/openpi/libero_conceptors.npz"
RESULTS_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results"
OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/diagnostic_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants discovered from data exploration ──────────────────
LAYERS_CONCEPTOR = [0, 5, 11, 17]        # layers with saved conceptors
LAYERS_STEER = [5, 11, 17]                # layers evaluated in steering sweep
ALPHAS = [0.1, 0.5, 1.0, 2.0, 10.0]
BETAS = [0.1, 0.3, 0.5]
QUOTA_ALPHA = 10.0                        # default aperture for quota computation
CONCEPTOR_TYPE = "contrastive"            # type used for steering

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

# Short aliases for the legend.
TASK_ALIAS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": "K3: stove+moka",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": "K4: bowl-drawer",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": "K6: mug-microwave",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "K8: 2 moka",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": "L1: soup+cheese",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": "L2: soup+tomato",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": "L2: cheese+butter",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "L5: 2 mugs",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": "L6: mug+pudding",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": "S1: book-caddy",
}

COLORS = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))

# ── Task directory resolution ───────────────────────────────────
# Steering result dirs truncate long task names. Map full → dir.
_all_result_dirs = [d for d in os.listdir(RESULTS_DIR)
                    if os.path.isdir(os.path.join(RESULTS_DIR, d))
                    and d not in ("scripts", "logs")]

def result_dir_for(task: str) -> str:
    # exact match first
    if task in _all_result_dirs:
        return os.path.join(RESULTS_DIR, task)
    # prefix match (some dirs are truncated)
    for d in _all_result_dirs:
        if task.startswith(d) or d.startswith(task[:40]):
            return os.path.join(RESULTS_DIR, d)
    raise FileNotFoundError(f"No result dir for {task}")


# ── Load conceptors and compute quotas ──────────────────────────
print("Loading conceptors ...")
conceptors = np.load(CONCEPTORS_PATH, allow_pickle=True)

def quota_for(task: str, layer: int, alpha: float) -> float:
    key = f"{task}__L{layer}__{alpha}__C_{CONCEPTOR_TYPE}"
    C = conceptors[key]
    return float(np.trace(C)) / C.shape[0]

# quotas[task] = array of quota per layer in LAYERS_CONCEPTOR (at QUOTA_ALPHA)
quotas = {t: np.array([quota_for(t, L, QUOTA_ALPHA) for L in LAYERS_CONCEPTOR])
          for t in TASKS}

# ── Load steering success rates ─────────────────────────────────
print("Loading steering results ...")
cond_re = re.compile(r"^global_L(\d+)_a([\d.]+)_b([\d.]+)$")

# success[task][(layer, alpha, beta)] = success_rate
success = {t: {} for t in TASKS}
for t in TASKS:
    summary_path = os.path.join(result_dir_for(t), "summary.json")
    with open(summary_path) as fh:
        data = json.load(fh)
    for entry in data["conditions"]:
        m = cond_re.match(entry["condition"])
        if not m:
            continue
        L, a, b = int(m.group(1)), float(m.group(2)), float(m.group(3))
        success[t][(L, a, b)] = float(entry["success_rate"])

# Sanity: ensure each task has all 45 global entries.
for t in TASKS:
    assert len(success[t]) == len(LAYERS_STEER) * len(ALPHAS) * len(BETAS), \
        f"{t}: got {len(success[t])} entries"

# Per-(task, layer) mean success rate over (alpha, beta).
mean_success_tl = {
    (t, L): float(np.mean([success[t][(L, a, b)] for a in ALPHAS for b in BETAS]))
    for t in TASKS for L in LAYERS_STEER
}

# Per-layer mean success rate across all tasks and (alpha, beta).
layer_mean_success = {
    L: float(np.mean([mean_success_tl[(t, L)] for t in TASKS]))
    for L in LAYERS_STEER
}
best_layer = max(layer_mean_success, key=layer_mean_success.get)
print(f"Per-layer mean success: {layer_mean_success}  → best layer: L{best_layer}")

# ── Create figure ───────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.35}
)

# ── Panel A: Quota across layers ────────────────────────────────
for i, t in enumerate(TASKS):
    ax1.plot(
        LAYERS_CONCEPTOR, quotas[t],
        color=COLORS[i], lw=1.5, marker="o", markersize=3.5,
        alpha=0.85, label=TASK_ALIAS[t],
    )
ax1.set_xticks(LAYERS_CONCEPTOR)
ax1.set_xlabel("Layer", fontsize=11)
ax1.set_ylabel(r"Conceptor Quota $q(C)$", fontsize=11)
ax1.set_title("Quota Across Layers", fontsize=11, fontweight="bold")
ax1.tick_params(labelsize=9)
ax1.text(-0.18, 1.06, "(a)", transform=ax1.transAxes,
         fontsize=12, fontweight="bold")

# ── Panel B: Quota vs. Success Rate ─────────────────────────────
all_q, all_s = [], []
for i, t in enumerate(TASKS):
    # align by layer: only steer layers have success rate
    qs = [float(np.trace(conceptors[f"{t}__L{L}__{QUOTA_ALPHA}__C_{CONCEPTOR_TYPE}"]))
          / conceptors[f"{t}__L{L}__{QUOTA_ALPHA}__C_{CONCEPTOR_TYPE}"].shape[0]
          for L in LAYERS_STEER]
    ss = [mean_success_tl[(t, L)] for L in LAYERS_STEER]
    face = COLORS[i]
    edge = tuple(np.clip(np.asarray(face[:3]) * 0.6, 0, 1)) + (1.0,)
    ax2.scatter(qs, ss, s=32, color=face, alpha=0.7,
                edgecolors=edge, linewidths=0.6, label=TASK_ALIAS[t])
    all_q.extend(qs); all_s.extend(ss)

all_q, all_s = np.asarray(all_q), np.asarray(all_s)
r, p = stats.pearsonr(all_q, all_s)
slope, intercept = np.polyfit(all_q, all_s, 1)
x_fit = np.linspace(all_q.min(), all_q.max(), 100)
ax2.plot(x_fit, slope * x_fit + intercept, "k--", lw=1.0)

# Place annotation where it doesn't overlap the data.
# If correlation is negative, data trends down-right, so top-right is free.
ann_x, ann_ha = (0.95, "right") if r < 0 else (0.05, "left")
ax2.text(ann_x, 0.92, rf"$r = {r:.2f}$,  $p = {p:.1e}$",
         transform=ax2.transAxes, fontsize=9, ha=ann_ha)
ax2.set_xlabel(r"Conceptor Quota $q(C)$", fontsize=11)
ax2.set_ylabel("Mean Success Rate", fontsize=11)
ax2.set_title("Quota vs. Steering Success", fontsize=11, fontweight="bold")
ax2.tick_params(labelsize=9)
ax2.text(-0.18, 1.06, "(b)", transform=ax2.transAxes,
         fontsize=12, fontweight="bold")

# ── Panel C: Parameter Sensitivity Heatmap ──────────────────────
# Rows = beta (ascending, origin='lower'), cols = alpha (ascending).
heatmap = np.zeros((len(BETAS), len(ALPHAS)))
for i, b in enumerate(BETAS):
    for j, a in enumerate(ALPHAS):
        heatmap[i, j] = np.mean(
            [success[t][(best_layer, a, b)] for t in TASKS]
        )

im = ax3.imshow(heatmap, cmap="YlOrRd", aspect="auto",
                origin="lower", vmin=0, vmax=1)
for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        val = heatmap[i, j]
        color = "white" if val > 0.6 else "black"
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7, color=color)
ax3.set_xticks(range(len(ALPHAS)))
ax3.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=9)
ax3.set_yticks(range(len(BETAS)))
ax3.set_yticklabels([f"{b:g}" for b in BETAS], fontsize=9)
ax3.set_xlabel(r"Aperture $\alpha$", fontsize=11)
ax3.set_ylabel(r"Steering Strength $\beta$", fontsize=11)
ax3.set_title(f"Parameter Sensitivity (L{best_layer})",
              fontsize=11, fontweight="bold")
ax3.text(-0.18, 1.06, "(c)", transform=ax3.transAxes,
         fontsize=12, fontweight="bold")
cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Success Rate", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.6)

# ── Shared legend below figure ──────────────────────────────────
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5,
           fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.10))

# ── Save ────────────────────────────────────────────────────────
pdf_path = os.path.join(OUTPUT_DIR, "libero_diagnostics.pdf")
png_path = os.path.join(OUTPUT_DIR, "libero_diagnostics.png")
fig.savefig(pdf_path, bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig(png_path, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
