#!/usr/bin/env python3
"""Aggregate Design-B transfer results for pi0.5 RoboCasa into NxN matrices + heatmaps."""

import csv
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASKS = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]

ALIAS = {
    "CloseFridge": "CloseFridge",
    "CoffeeSetupMug": "CoffeeSetup",
    "OpenDrawer": "OpenDrawer",
    "OpenStandMixerHead": "StandMixer",
    "PickPlaceCounterToCabinet": "PP_Cabinet",
    "PickPlaceCounterToStove": "PP_Stove",
    "TurnOnElectricKettle": "Kettle",
}

STRATEGIES = ["global", "per_step_0", "per_step_9"]


def lbl(t: str) -> str:
    return ALIAS.get(t, t[:15])


def load_transfer_cells(transfer_root: pathlib.Path):
    data = {s: {} for s in STRATEGIES}
    for target in TASKS:
        summary = transfer_root / f"target_{target}" / "summary.json"
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
    diag = {s: {} for s in STRATEGIES}
    for task in TASKS:
        summary = steering_root / task / "summary.json"
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
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    labels = [lbl(t) for t in TASKS]
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(labels)
    ax.set_xlabel("target"); ax.set_ylabel("source"); ax.set_title(title)
    for i in range(N):
        for j in range(N):
            v = M[i, j]
            if not np.isnan(v):
                if center is not None:
                    txt_color = "black" if abs(v) < 0.3 else "white"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", color=txt_color, fontsize=9)
                else:
                    txt_color = "white" if v < 0.5 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def summarize(M):
    diag = np.diag(M)
    off = M.copy()
    np.fill_diagonal(off, np.nan)
    return {
        "diag_mean": float(np.nanmean(diag)),
        "off_diag_mean": float(np.nanmean(off)),
        "off_diag_median": float(np.nanmedian(off)),
        "off_diag_min": float(np.nanmin(off)),
        "off_diag_max": float(np.nanmax(off)),
        "n_filled": int(np.sum(~np.isnan(off))),
        "n_expected": int(off.size - len(TASKS)),
        "source_givers": {
            lbl(TASKS[i]): float(np.nanmean(off[i])) for i in range(len(TASKS))
        },
        "target_receivers": {
            lbl(TASKS[j]): float(np.nanmean(off[:, j])) for j in range(len(TASKS))
        },
    }


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    transfer_root = repo_root / "experiments" / "pi05_robocasa" / "transfer_results"
    steering_root = repo_root / "experiments" / "pi05_robocasa" / "steering_results"
    out_root = transfer_root / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    cells = load_transfer_cells(transfer_root)
    diag = load_diagonal(steering_root)

    stats = {}
    for strat in STRATEGIES:
        M = build_matrix(cells[strat], diag[strat])
        save_csv(M, out_root / f"matrix_{strat}.csv")
        save_heatmap(M, out_root / f"heatmap_{strat}.png",
                     f"Transfer SR ({strat}) — source → target  [pi0.5 RoboCasa, design B]")
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
