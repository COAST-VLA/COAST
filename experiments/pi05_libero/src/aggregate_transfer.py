#!/usr/bin/env python3
"""
Aggregate Design-B transfer results into NxN matrices + heatmaps.

For each strategy {global, per_step_0, per_step_9}, builds a 10x10 matrix where
row=source task, column=target task. Off-diagonal cells come from
experiments/pi05_libero/transfer_results/target_*/summary.json. Diagonal comes
from each task's own best under that strategy in experiments/pi05_libero/steering_results/.

Writes:
  experiments/pi05_libero/transfer_results/analysis/
    matrix_{strategy}.csv
    heatmap_{strategy}.png
    delta_vs_diag_{strategy}.png  (transfer SR - target's self-best)
    stats.json
"""

import csv
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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

ALIAS = {
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

STRATEGIES = ["global", "per_step_0", "per_step_9"]


def short(t: str) -> str:
    return t[:60]


def lbl(t: str) -> str:
    return ALIAS.get(t, t[:15])


def load_transfer_cells(transfer_root: pathlib.Path):
    """Returns dict strategy -> {(source, target): sr}."""
    data = {s: {} for s in STRATEGIES}
    for target in TASKS:
        summary = transfer_root / f"target_{short(target)}" / "summary.json"
        if not summary.exists():
            print(f"[warn] missing {summary}")
            continue
        with open(summary) as f:
            d = json.load(f)
        for cell in d.get("cells", []):
            strat = cell["strategy"]
            if strat not in data:
                continue
            sr = cell.get("success_rate")
            if sr is None or (isinstance(sr, float) and np.isnan(sr)):
                sr = float("nan")
            data[strat][(cell["source"], cell["target"])] = float(sr)
    return data


def load_diagonal(steering_root: pathlib.Path):
    """Per-task self-best under each strategy."""
    diag = {s: {} for s in STRATEGIES}
    for task in TASKS:
        summary = steering_root / short(task) / "summary.json"
        if not summary.exists():
            print(f"[warn] missing diag source {summary}")
            continue
        with open(summary) as f:
            d = json.load(f)
        bests = {s: 0.0 for s in STRATEGIES}
        for c in d["conditions"]:
            name = c["condition"]
            sr = c["success_rate"]
            if name.startswith("global_"):
                bests["global"] = max(bests["global"], sr)
            elif name.startswith("per_step_0_"):
                bests["per_step_0"] = max(bests["per_step_0"], sr)
            elif name.startswith("per_step_9_"):
                bests["per_step_9"] = max(bests["per_step_9"], sr)
        for s in STRATEGIES:
            diag[s][task] = bests[s]
    return diag


def build_matrix(strat_cells, diag_for_strat):
    N = len(TASKS)
    M = np.full((N, N), np.nan)
    for i, src in enumerate(TASKS):
        for j, tgt in enumerate(TASKS):
            if src == tgt:
                if tgt in diag_for_strat:
                    M[i, j] = diag_for_strat[tgt]
            elif (src, tgt) in strat_cells:
                M[i, j] = strat_cells[(src, tgt)]
    return M


def save_csv(M, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = [""] + [f"tgt:{lbl(t)}" for t in TASKS]
        w.writerow(hdr)
        for i, src in enumerate(TASKS):
            row = [f"src:{lbl(src)}"]
            for j in range(len(TASKS)):
                row.append(f"{M[i, j]:.3f}" if not np.isnan(M[i, j]) else "")
            w.writerow(row)


def save_heatmap(M, path, title, vmin=0.0, vmax=1.0, cmap="viridis", center=None):
    N = len(TASKS)
    fig, ax = plt.subplots(figsize=(12, 10))
    if center is not None:
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    labels = [lbl(t) for t in TASKS]
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(title)
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            if not np.isnan(v):
                if center is not None:
                    txt_color = "black" if abs(v) < 0.3 else "white"
                else:
                    txt_color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v:+.2f}" if center is not None else f"{v:.2f}",
                        ha="center", va="center", color=txt_color, fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def summarize(M):
    diag = np.diag(M)
    off = M.copy()
    np.fill_diagonal(off, np.nan)
    row_mean = np.nanmean(off, axis=1)  # source's mean transfer-out
    col_mean = np.nanmean(off, axis=0)  # target's mean recipient SR
    return {
        "diag_mean": float(np.nanmean(diag)),
        "off_diag_mean": float(np.nanmean(off)),
        "off_diag_median": float(np.nanmedian(off)),
        "off_diag_min": float(np.nanmin(off)),
        "off_diag_max": float(np.nanmax(off)),
        "n_filled": int(np.sum(~np.isnan(off))),
        "n_expected": int(off.size - len(TASKS)),
        "source_givers": {
            lbl(TASKS[i]): float(row_mean[i]) for i in range(len(TASKS))
        },
        "target_receivers": {
            lbl(TASKS[j]): float(col_mean[j]) for j in range(len(TASKS))
        },
    }


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    transfer_root = repo_root / "experiments" / "pi05_libero" / "transfer_results"
    steering_root = repo_root / "experiments" / "pi05_libero" / "steering_results"
    out_root = transfer_root / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    cells = load_transfer_cells(transfer_root)
    diag = load_diagonal(steering_root)

    stats = {}
    for strat in STRATEGIES:
        M = build_matrix(cells[strat], diag[strat])
        save_csv(M, out_root / f"matrix_{strat}.csv")
        save_heatmap(M, out_root / f"heatmap_{strat}.png",
                     f"Transfer SR ({strat}) — source → target  [design B]")
        # delta vs diagonal (per target)
        D = M - np.diag(M)[np.newaxis, :]
        save_heatmap(D, out_root / f"delta_vs_diag_{strat}.png",
                     f"Transfer SR − target's self-best ({strat})",
                     vmin=-1.0, vmax=1.0, cmap="RdBu_r", center=0.0)
        stats[strat] = summarize(M)

    with open(out_root / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nAnalysis written to {out_root}\n")
    print(f"{'Strategy':<14s} {'diag':>8s} {'off-diag':>10s} {'range':>20s} {'filled':>10s}")
    for strat, s in stats.items():
        rng = f"[{s['off_diag_min']:.2f},{s['off_diag_max']:.2f}]"
        print(f"{strat:<14s} {s['diag_mean']:>8.3f} {s['off_diag_mean']:>10.3f} {rng:>20s} "
              f"{s['n_filled']}/{s['n_expected']}")


if __name__ == "__main__":
    main()
