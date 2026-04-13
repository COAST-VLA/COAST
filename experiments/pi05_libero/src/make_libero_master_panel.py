"""Master 1x5 diagnostic panel for the paper.

Story: three cheap, GPU-free diagnostics jointly pick (layer, alpha, beta)
for conceptor steering without a full sweep.

  (a) Quota per layer + per-layer mean success  → pick the layer
  (b) Overlap vs success rate                    → overlap predicts success
  (c) Aperture alpha vs overlap (per layer)      → alpha controls overlap
  (d) Success vs beta, stratified by overlap     → beta behaviour given regime
  (e) Sweet-spot heatmap (overlap bin x beta)    → joint pick
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
fig, axes = plt.subplots(1, 5, figsize=(20, 3.7),
                         gridspec_kw={"wspace": 0.55})
ax_a, ax_b, ax_c, ax_d, ax_e = axes

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, direction="out")


def panel_label(ax, txt, dx=-0.22):
    ax.text(dx, 1.08, txt, transform=ax.transAxes,
            fontsize=12, fontweight="bold")


# ── (a) Quota per layer (box) + mean success per layer (line) ───
# Box shows distribution of quota across 10 tasks per layer.
quota_by_layer = [[Q[(t, L)] for t in TASKS] for L in LAYERS_CONCEPTOR]
bp = ax_a.boxplot(
    quota_by_layer, positions=range(len(LAYERS_CONCEPTOR)),
    widths=0.55, patch_artist=True, showfliers=False,
    medianprops=dict(color="black", lw=1.2),
    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
    boxprops=dict(lw=0.6),
)
for patch, L in zip(bp["boxes"], LAYERS_CONCEPTOR):
    patch.set_facecolor(LAYER_COLORS[L]); patch.set_alpha(0.55)

ax_a.set_xticks(range(len(LAYERS_CONCEPTOR)))
ax_a.set_xticklabels([f"L{L}" for L in LAYERS_CONCEPTOR])
ax_a.set_xlabel("Layer", fontsize=11)
ax_a.set_ylabel(r"Conceptor Quota $q(C)$", fontsize=11, color="#2b4a78")
ax_a.tick_params(axis="y", labelcolor="#2b4a78")
ax_a.set_title("Quota and Success per Layer", fontsize=11, fontweight="bold")

# Right axis: per-layer mean success rate across tasks and (alpha, beta).
ax_ar = ax_a.twinx()
ax_ar.spines["top"].set_visible(False)
# twin axis needs its own right spine visible; colour it.
ax_ar.spines["right"].set_visible(True)
ax_ar.spines["right"].set_color("#b0352e")
ax_ar.spines["left"].set_visible(False)
succ_by_layer = {L: [S[t][(L, a, b)] for t in TASKS
                     for a in ALPHAS for b in BETAS]
                 for L in LAYERS_STEER}
mu = [np.mean(succ_by_layer[L]) for L in LAYERS_STEER]
sd = [np.std(succ_by_layer[L]) / np.sqrt(len(succ_by_layer[L]))
      for L in LAYERS_STEER]
# Map LAYERS_STEER back onto x-positions of LAYERS_CONCEPTOR.
xpos = [LAYERS_CONCEPTOR.index(L) for L in LAYERS_STEER]
ax_ar.errorbar(xpos, mu, yerr=sd, color="#b0352e", lw=1.8,
               marker="s", markersize=6, capsize=3, label="Mean success")
ax_ar.set_ylabel("Mean Success Rate", fontsize=11, color="#b0352e")
ax_ar.tick_params(axis="y", labelsize=9, labelcolor="#b0352e",
                  direction="out")
ax_ar.set_ylim(0, max(mu) * 1.25)
panel_label(ax_a, "(a)")

# ── (b) Overlap vs success rate scatter, r annotation ───────────
xs, ys, cs = [], [], []
for t in TASKS:
    for L in LAYERS_STEER:
        for a in ALPHAS:
            o = O[(t, L, a)]
            s = float(np.mean([S[t][(L, a, b)] for b in BETAS]))
            xs.append(o); ys.append(s); cs.append(LAYER_COLORS[L])
xs = np.array(xs); ys = np.array(ys)
for L in LAYERS_STEER:
    m = np.array([cs[i] is LAYER_COLORS[L] for i in range(len(cs))])
    ax_b.scatter(xs[m], ys[m], s=28, color=LAYER_COLORS[L],
                 alpha=0.75, edgecolors="black", linewidths=0.3,
                 label=f"L{L}")
r, p = stats.pearsonr(xs, ys)
slope, intercept = np.polyfit(xs, ys, 1)
xf = np.linspace(xs.min(), xs.max(), 100)
ax_b.plot(xf, slope * xf + intercept, "k--", lw=1.0)
ax_b.text(0.04, 0.94, rf"$r = {r:.2f}$,  $p = {p:.1e}$",
          transform=ax_b.transAxes, fontsize=9, ha="left", va="top")
ax_b.set_xlabel(r"Overlap $\mathrm{sim}(C_s, C_f)$", fontsize=11)
ax_b.set_ylabel(r"Mean Success Rate", fontsize=11)
ax_b.set_title("Overlap vs. Success", fontsize=11, fontweight="bold")
ax_b.legend(fontsize=7.5, frameon=False, loc="lower right",
            title="Layer", title_fontsize=8)
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

# ── (d) Success vs beta, stratified by overlap tertile ──────────
tla_keys = [(t, L, a) for t in TASKS for L in LAYERS_STEER for a in ALPHAS]
ovs = np.array([O[k] for k in tla_keys])
q1, q2 = np.quantile(ovs, [1/3, 2/3])
tertile = {"low": defaultdict(list), "mid": defaultdict(list),
           "high": defaultdict(list)}
for (t, L, a) in tla_keys:
    o = O[(t, L, a)]
    bucket = "low" if o <= q1 else ("mid" if o <= q2 else "high")
    for b in BETAS:
        tertile[bucket][b].append(S[t][(L, a, b)])
tert_c = {"low": "#3b7dd8", "mid": "#f2a541", "high": "#c73e3a"}
tert_l = {
    "low":  rf"low ($\leq {q1:.2f}$)",
    "mid":  rf"mid ({q1:.2f}–{q2:.2f})",
    "high": rf"high ($> {q2:.2f}$)",
}
for bucket in ["low", "mid", "high"]:
    mus = np.array([np.mean(tertile[bucket][b]) for b in BETAS])
    ses = np.array([np.std(tertile[bucket][b]) /
                    np.sqrt(len(tertile[bucket][b])) for b in BETAS])
    ax_d.plot(BETAS, mus, color=tert_c[bucket], lw=1.6,
              marker="o", markersize=5, label=tert_l[bucket])
    ax_d.fill_between(BETAS, mus - ses, mus + ses,
                      color=tert_c[bucket], alpha=0.18, linewidth=0)
ax_d.set_xticks(BETAS)
ax_d.set_xlabel(r"Steering Strength $\beta$", fontsize=11)
ax_d.set_ylabel("Mean Success Rate", fontsize=11)
ax_d.set_title(r"$\beta$ Given Overlap Regime", fontsize=11, fontweight="bold")
ax_d.legend(fontsize=7.5, frameon=False, loc="lower left",
            title="Overlap", title_fontsize=8)
panel_label(ax_d, "(d)")

# ── (e) Sweet-spot heatmap ──────────────────────────────────────
r_ovl, r_beta, r_succ = [], [], []
for t in TASKS:
    for L in LAYERS_STEER:
        for a in ALPHAS:
            for b in BETAS:
                r_ovl.append(O[(t, L, a)])
                r_beta.append(b)
                r_succ.append(S[t][(L, a, b)])
r_ovl = np.array(r_ovl); r_beta = np.array(r_beta); r_succ = np.array(r_succ)
n_ovl_bins = 4
edges = np.quantile(r_ovl, np.linspace(0, 1, n_ovl_bins + 1))
edges[0] -= 1e-9; edges[-1] += 1e-9
H = np.full((n_ovl_bins, len(BETAS)), np.nan)
for i in range(n_ovl_bins):
    lo, hi = edges[i], edges[i + 1]
    for j, b in enumerate(BETAS):
        m = (r_ovl > lo) & (r_ovl <= hi) & (r_beta == b)
        if m.sum():
            H[i, j] = r_succ[m].mean()

im = ax_e.imshow(H, cmap="YlOrRd", aspect="auto", origin="lower",
                 vmin=0, vmax=1)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H[i, j]
        col = "white" if v > 0.6 else "black"
        ax_e.text(j, i, f"{v:.2f}", ha="center", va="center",
                  fontsize=8, color=col)
ax_e.set_xticks(range(len(BETAS)))
ax_e.set_xticklabels([f"{b:g}" for b in BETAS])
ax_e.set_yticks(range(n_ovl_bins))
ax_e.set_yticklabels([f"[{edges[i]:.2f}, {edges[i+1]:.2f}]"
                      for i in range(n_ovl_bins)])
ax_e.set_xlabel(r"Steering Strength $\beta$", fontsize=11)
ax_e.set_ylabel("Overlap Quartile", fontsize=11)
ax_e.set_title("Sweet-Spot Map", fontsize=11, fontweight="bold")
cbar = fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
cbar.set_label("Success Rate", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_linewidth(0.6)
panel_label(ax_e, "(e)")

# ── Save ────────────────────────────────────────────────────────
pdf = os.path.join(OUTPUT_DIR, "libero_master_panel.pdf")
png = os.path.join(OUTPUT_DIR, "libero_master_panel.png")
fig.savefig(pdf, bbox_inches="tight", dpi=300, pad_inches=0.05)
fig.savefig(png, bbox_inches="tight", dpi=300, pad_inches=0.05)
print(f"Saved: {pdf}")
print(f"Saved: {png}")
