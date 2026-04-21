#!/usr/bin/env python3
"""
Generate LaTeX tables for Design-B transfer experiment results.

For each benchmark (pi0.5 LIBERO, pi0.5 RoboCasa), writes a table with:
  Target | Baseline | Self-best | Top-1..Top-4 transfer cells

Transfer cell format: "source_alias (strategy) SR"
  strategy ∈ {G, PS0, PS9}  (global / per_step_0 / per_step_9)

Output:
  experiments/pi05_libero/transfer_results/analysis/transfer_table.tex
  experiments/pi05_robocasa/transfer_results/analysis/transfer_table.tex
"""

import argparse
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # openpi-new/

STRAT_SHORT = {"global": "G", "per_step_0": "PS0", "per_step_9": "PS9"}


LIBERO = {
    "model_name": r"$\pi_{0.5}$ LIBERO",
    "tasks": [
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
    ],
    "alias": {
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": "K3\\_stove\\_moka",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": "K4\\_bowl\\_drawer",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": "K6\\_mug\\_micro",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": "K8\\_two\\_moka",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": "LR1\\_soup\\_cheese",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": "LR2\\_soup\\_tomato",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": "LR2\\_cheese\\_butter",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "LR5\\_mugs\\_plates",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": "LR6\\_mug\\_choc",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": "S1\\_book\\_caddy",
    },
    "steering_root": REPO_ROOT / "experiments" / "pi05_libero" / "steering_results",
    "transfer_root": REPO_ROOT / "experiments" / "pi05_libero" / "transfer_results",
    "target_dir_fn": lambda t: f"target_{t[:60]}",
    "steering_dir_fn": lambda t: t[:60],
    "label": "tab:transfer_pi05_libero",
}

ROBOCASA = {
    "model_name": r"$\pi_{0.5}$ RoboCasa",
    "tasks": [
        "CloseFridge",
        "CoffeeSetupMug",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "TurnOnElectricKettle",
    ],
    "alias": {
        "CloseFridge": "CloseFridge",
        "CoffeeSetupMug": "CoffeeSetup",
        "OpenDrawer": "OpenDrawer",
        "OpenStandMixerHead": "StandMixer",
        "PickPlaceCounterToCabinet": "PP\\_Cabinet",
        "PickPlaceCounterToStove": "PP\\_Stove",
        "TurnOnElectricKettle": "Kettle",
    },
    "steering_root": REPO_ROOT / "experiments" / "pi05_robocasa" / "steering_results",
    "transfer_root": REPO_ROOT / "experiments" / "pi05_robocasa" / "transfer_results",
    "target_dir_fn": lambda t: f"target_{t}",
    "steering_dir_fn": lambda t: t,
    "label": "tab:transfer_pi05_robocasa",
}


def load_baseline_and_self_best(steering_root, steering_dir_fn, task):
    summary = steering_root / steering_dir_fn(task) / "summary.json"
    if not summary.exists():
        return None, None
    d = json.load(open(summary))
    baseline = None
    self_best = 0.0
    for c in d["conditions"]:
        n, sr = c["condition"], c["success_rate"]
        if n == "baseline":
            baseline = sr
        elif n.startswith(("global_", "per_step_")):
            if sr > self_best:
                self_best = sr
    return baseline, self_best


def load_transfer_cells(transfer_root, target_dir_fn, target):
    summary = transfer_root / target_dir_fn(target) / "summary.json"
    if not summary.exists():
        return []
    d = json.load(open(summary))
    out = []
    for c in d.get("cells", []):
        sr = c.get("success_rate")
        if sr is None:
            continue
        out.append({
            "source": c["source"],
            "strategy": c["strategy"],
            "sr": float(sr),
        })
    return out


def fmt_cell(cell, alias):
    src_alias = alias.get(cell["source"], cell["source"][:15])
    strat = STRAT_SHORT.get(cell["strategy"], cell["strategy"])
    return f"{src_alias} ({strat}) {cell['sr']:.2f}"


def row_highlights(row):
    """Return (best_val, second_val) among baseline/self/top4 cell SRs, or (None, None).

    Ties are resolved by distinct value: best = max unique, second = 2nd max unique.
    """
    vals = []
    if row["baseline"] is not None:
        vals.append(round(row["baseline"], 4))
    if row["self_best"] is not None:
        vals.append(round(row["self_best"], 4))
    for c in row["top4"]:
        if c is not None:
            vals.append(round(c["sr"], 4))
    uniq = sorted(set(vals), reverse=True)
    best = uniq[0] if len(uniq) >= 1 else None
    second = uniq[1] if len(uniq) >= 2 else None
    return best, second


def wrap(text, val, best, second):
    if val is None:
        return text
    v = round(val, 4)
    if best is not None and v == best:
        return r"\textbf{" + text + "}"
    if second is not None and v == second:
        return r"\underline{" + text + "}"
    return text


def build_table(cfg):
    rows = []
    for task in cfg["tasks"]:
        baseline, self_best = load_baseline_and_self_best(
            cfg["steering_root"], cfg["steering_dir_fn"], task
        )
        cells = load_transfer_cells(
            cfg["transfer_root"], cfg["target_dir_fn"], task
        )
        cells.sort(key=lambda c: -c["sr"])
        top4 = cells[:4]
        while len(top4) < 4:
            top4.append(None)
        rows.append({
            "target": task,
            "target_alias": cfg["alias"].get(task, task[:15]),
            "baseline": baseline,
            "self_best": self_best,
            "top4": top4,
        })
    return rows


def emit_subtable(cfg, rows, width=None):
    """Emit one subtable (no \\caption — caller puts it inside subtable env).

    Per row, bold the best SR and underline the second-best SR, comparing across
    baseline, self-best, and the four transfer cells. If ``width`` is provided
    (fraction of \\textwidth), the tabular is wrapped in \\resizebox.
    """
    lines = []
    if width is not None:
        lines.append(rf"\resizebox{{{width}\textwidth}}{{!}}{{%")
    lines.append(r"\begin{tabular}{lcc|cccc}")
    lines.append(r"\toprule")
    lines.append(r"Target & Base. & Self & Top-1 & Top-2 & Top-3 & Top-4 \\")
    lines.append(r"\midrule")
    for r in rows:
        best, second = row_highlights(r)
        b_txt = f"{r['baseline']:.2f}" if r["baseline"] is not None else "--"
        sb_txt = f"{r['self_best']:.2f}" if r["self_best"] is not None else "--"
        b = wrap(b_txt, r["baseline"], best, second)
        sb = wrap(sb_txt, r["self_best"], best, second)
        t_cells = []
        for c in r["top4"]:
            if c is None:
                t_cells.append("--")
            else:
                t_cells.append(wrap(fmt_cell(c, cfg["alias"]), c["sr"], best, second))
        lines.append(f"{r['target_alias']} & {b} & {sb} & " + " & ".join(t_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if width is not None:
        lines.append(r"}")
    return "\n".join(lines)


def emit_master(libero_rows, robocasa_rows, width=None):
    lines = []
    lines.append(r"% Requires: \usepackage{booktabs,subcaption,graphicx}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Transfer experiment (Design B) across benchmarks. For each target "
                 r"task, we report the no-steering baseline (Base.), the self-best self-steered "
                 r"result (Self), and the top-4 transfer sources. Each transfer cell is "
                 r"\emph{source\_alias (strategy) SR}, with strategy "
                 r"$\in \{\mathrm{G}=\text{global},\ \mathrm{PS0}=\text{per-step }t{=}0,\ "
                 r"\mathrm{PS9}=\text{per-step }t{=}9\}$. All values are success rates over "
                 r"15 episodes. Per row, the highest SR across \{Base., Self, Top-1..Top-4\} "
                 r"is shown in \textbf{bold} and the second-highest is \underline{underlined}.}")
    lines.append(r"\label{tab:transfer_master}")
    # Libero subtable (stacked top)
    lines.append(r"\begin{subtable}[t]{\linewidth}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{LIBERO['model_name']}}}")
    lines.append(rf"\label{{{LIBERO['label']}}}")
    lines.append(emit_subtable(LIBERO, libero_rows, width=width))
    lines.append(r"\end{subtable}")
    lines.append(r"\vspace{1em}")
    # Robocasa subtable (stacked bottom)
    lines.append(r"\begin{subtable}[t]{\linewidth}")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{ROBOCASA['model_name']}}}")
    lines.append(rf"\label{{{ROBOCASA['label']}}}")
    lines.append(emit_subtable(ROBOCASA, robocasa_rows, width=width))
    lines.append(r"\end{subtable}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Fraction of \\textwidth to shrink each tabular to (e.g. 0.8). "
             "If omitted, tabulars use their natural width.",
    )
    args = parser.parse_args()

    libero_rows = build_table(LIBERO)
    robocasa_rows = build_table(ROBOCASA)

    tex = emit_master(libero_rows, robocasa_rows, width=args.width) + "\n"

    # Canonical output: pi05_libero analysis dir
    canonical_path = LIBERO["transfer_root"] / "analysis" / "transfer_master_table.tex"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(tex)

    # Copies: shared master dir + pi05_robocasa analysis dir
    master_dir = REPO_ROOT / "experiments" / "shared" / "analysis_output"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_path = master_dir / "transfer_master_table.tex"
    master_path.write_text(tex)

    robocasa_copy = ROBOCASA["transfer_root"] / "analysis" / "transfer_master_table.tex"
    robocasa_copy.parent.mkdir(parents=True, exist_ok=True)
    robocasa_copy.write_text(tex)

    # Console preview
    print(f"\nCanonical table written to {canonical_path}")
    print(f"Copies: {master_path}, {robocasa_copy}")
    for cfg, rows in [(LIBERO, libero_rows), (ROBOCASA, robocasa_rows)]:
        print(f"\n{'='*80}\n{cfg['model_name']}\n{'='*80}")
        print(f"{'Target':<22s} {'Base':>5s} {'Self':>5s}  Top-4 transfer")
        for r in rows:
            b = f"{r['baseline']:.2f}" if r["baseline"] is not None else "  -- "
            sb = f"{r['self_best']:.2f}" if r["self_best"] is not None else "  -- "
            top4 = [fmt_cell(c, cfg["alias"]).replace("\\_", "_") if c else "--" for c in r["top4"]]
            print(f"{r['target_alias'].replace(chr(92)+'_','_'):<22s} {b:>5s} {sb:>5s}  | " + "  |  ".join(top4))


if __name__ == "__main__":
    main()
