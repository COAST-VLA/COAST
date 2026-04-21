#!/usr/bin/env python3
"""
Oracle Gap Table
================

For each benchmark, compare the success rate achieved by the geometric
selection rule against the exhaustive sweep's best (oracle).

Report: "Geometric selection achieves X% of oracle performance while
evaluating Y% of configurations."

Geometric selection rule (from select_parameters.py):
  1. Layer:  highest mean quota across tasks
  2. Alpha:  overlap in sweet-spot band [0.85, 0.95]
  3. Beta:   {0.1, 0.3}  (drop 0.5)

The selected subset is all conditions matching the selected (layer, alphas, betas)
across all steering strategies (global, per_step, pos_only, etc.).

Benchmarks:
  - groot_robocasa  (GR-1 + RoboCasa)
  - pi05_robocasa   (pi0.5 + RoboCasa)
  - pi05_libero     (pi0.5 + LIBERO)
"""

import json
import os
import re
import sys

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "groot_robocasa": {
        "results_dir": "/vast/projects/ungar/stellar/miaom/openpi-groot/experiments/groot_robocasa/steering_results",
        "selected_params": "/vast/projects/ungar/stellar/miaom/openpi-groot/experiments/groot_robocasa/selected_params/selected_params.json",
    },
    "pi05_robocasa": {
        "results_dir": "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_robocasa/steering_results",
        "selected_params": "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_robocasa/selected_params.json",
    },
    "pi05_libero": {
        "results_dir": "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results",
        "selected_params": "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/selected_params.json",
    },
}

OUTPUT_DIR = "/vast/projects/ungar/stellar/miaom/openpi-new/experiments/shared/analysis_output"


# ──────────────────────────────────────────────────────────────────────────────
# Condition matching
# ──────────────────────────────────────────────────────────────────────────────

def parse_condition(cond_str):
    """Parse a condition string into (strategy, layer, alpha, beta) or None."""
    # global_L11_a0.5_b0.3
    m = re.match(r"^(global|pos_only)_L(\d+)_a([\d.]+)_b([\d.]+)$", cond_str)
    if m:
        return {
            "strategy": m.group(1),
            "layer": int(m.group(2)),
            "alpha": float(m.group(3)),
            "beta": float(m.group(4)),
        }
    # per_step_0_L11_a0.5_b0.3  or  per_step_L10_b0.3
    m = re.match(r"^(per_step)(?:_(\d+))?_L(\d+)(?:_a([\d.]+))?_b([\d.]+)$", cond_str)
    if m:
        return {
            "strategy": "per_step",
            "layer": int(m.group(3)),
            "alpha": float(m.group(4)) if m.group(4) else None,
            "beta": float(m.group(5)),
        }
    # linear_L10_la1.0
    m = re.match(r"^(linear)_L(\d+)_la([\d.]+)$", cond_str)
    if m:
        return {
            "strategy": "linear",
            "layer": int(m.group(2)),
            "alpha": float(m.group(3)),
            "beta": None,
        }
    # random_L10_b0.1
    m = re.match(r"^(random)_L(\d+)_b([\d.]+)$", cond_str)
    if m:
        return {
            "strategy": "random",
            "layer": int(m.group(2)),
            "alpha": None,
            "beta": float(m.group(3)),
        }
    # baseline
    if cond_str == "baseline":
        return {"strategy": "baseline", "layer": None, "alpha": None, "beta": None}
    return None


def condition_in_selected(parsed, sel_layer, sel_alphas, sel_betas):
    """Check if a parsed condition is within the geometric-selected subset."""
    if parsed is None:
        return False
    strategy = parsed["strategy"]

    # Baseline is always included
    if strategy == "baseline":
        return True

    # Layer must match
    if parsed["layer"] != sel_layer:
        return False

    # For strategies with alpha: alpha must be in selected set
    if parsed["alpha"] is not None:
        if parsed["alpha"] not in sel_alphas:
            return False

    # For strategies with beta: beta must be in selected set
    if parsed["beta"] is not None:
        if parsed["beta"] not in sel_betas:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_benchmark(name, results_dir, selected_params_path):
    """Compute oracle gap for one benchmark."""
    with open(selected_params_path) as f:
        sel = json.load(f)

    sel_layer = sel["best_layer"]
    sel_alphas = [float(a) for a in sel["selected_alphas"]]
    sel_betas = [float(b) for b in sel["selected_betas"]]

    # Load all task results
    task_data = {}
    for rd in sorted(os.listdir(results_dir)):
        sp = os.path.join(results_dir, rd, "summary.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            data = json.load(f)
        task_data[rd] = data["conditions"]

    # Per-task analysis
    per_task = []
    all_full_conds = set()
    all_selected_conds = set()

    for task, conditions in task_data.items():
        # Build lookup: condition_str → SR
        cond_sr = {}
        for entry in conditions:
            cond_sr[entry["condition"]] = entry["success_rate"]

        # Split into oracle (all) vs selected (geometric subset)
        full_conditions = set()
        selected_conditions = set()
        baseline_sr = cond_sr.get("baseline", 0.0)

        for cond_str, sr in cond_sr.items():
            parsed = parse_condition(cond_str)
            full_conditions.add(cond_str)
            all_full_conds.add(cond_str)

            if condition_in_selected(parsed, sel_layer, sel_alphas, sel_betas):
                selected_conditions.add(cond_str)
                all_selected_conds.add(cond_str)

        # Oracle: best SR across all conditions (excluding baseline)
        steering_conds = {c: s for c, s in cond_sr.items() if c != "baseline"}
        if not steering_conds:
            continue
        oracle_sr = max(steering_conds.values())
        oracle_cond = max(steering_conds, key=steering_conds.get)

        # Geometric: best SR across selected conditions (excluding baseline)
        selected_steering = {c: s for c, s in cond_sr.items()
                           if c in selected_conditions and c != "baseline"}
        if selected_steering:
            geo_sr = max(selected_steering.values())
            geo_cond = max(selected_steering, key=selected_steering.get)
        else:
            geo_sr = baseline_sr
            geo_cond = "baseline (no selected conditions)"

        per_task.append({
            "task": task,
            "baseline_sr": baseline_sr,
            "oracle_sr": oracle_sr,
            "oracle_cond": oracle_cond,
            "geo_sr": geo_sr,
            "geo_cond": geo_cond,
            "n_full": len(full_conditions) - 1,  # exclude baseline
            "n_selected": len(selected_conditions) - 1,  # exclude baseline
        })

    return {
        "name": name,
        "sel_layer": sel_layer,
        "sel_alphas": sel_alphas,
        "sel_betas": sel_betas,
        "per_task": per_task,
        "n_full_unique": len(all_full_conds) - 1,
        "n_selected_unique": len(all_selected_conds) - 1,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for name, cfg in BENCHMARKS.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")
        r = analyze_benchmark(name, cfg["results_dir"], cfg["selected_params"])
        results.append(r)

        # Print per-task detail
        print(f"\n  Geometric selection: L={r['sel_layer']}, "
              f"α={r['sel_alphas']}, β={r['sel_betas']}")
        print(f"  Grid: {r['n_selected_unique']} / {r['n_full_unique']} conditions "
              f"({100*r['n_selected_unique']/r['n_full_unique']:.1f}%)\n")

        print(f"  {'Task':<40s} {'Baseline':>8s} {'Oracle':>8s} {'Geo':>8s} {'%Oracle':>8s}  Oracle Cond")
        print(f"  {'-'*110}")
        for t in r["per_task"]:
            pct = 100 * t["geo_sr"] / t["oracle_sr"] if t["oracle_sr"] > 0 else 0
            print(f"  {t['task']:<40s} {t['baseline_sr']:>8.3f} {t['oracle_sr']:>8.3f} "
                  f"{t['geo_sr']:>8.3f} {pct:>7.1f}%  {t['oracle_cond']}")

    # ── Aggregate table ──────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  ORACLE GAP TABLE — SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Benchmark':<20s} {'Tasks':>5s} {'Configs':>12s} {'Selected':>10s} "
          f"{'%Grid':>7s} {'Oracle SR':>10s} {'Geo SR':>8s} {'%Oracle':>8s} {'Baseline':>9s}")
    print(f"  {'-'*90}")

    table_rows = []
    for r in results:
        pts = r["per_task"]
        n_tasks = len(pts)
        mean_baseline = np.mean([t["baseline_sr"] for t in pts])
        mean_oracle = np.mean([t["oracle_sr"] for t in pts])
        mean_geo = np.mean([t["geo_sr"] for t in pts])
        pct_oracle = 100 * mean_geo / mean_oracle if mean_oracle > 0 else 0
        pct_grid = 100 * r["n_selected_unique"] / r["n_full_unique"]

        print(f"  {r['name']:<20s} {n_tasks:>5d} {r['n_full_unique']:>12d} "
              f"{r['n_selected_unique']:>10d} {pct_grid:>6.1f}% "
              f"{mean_oracle:>10.3f} {mean_geo:>8.3f} {pct_oracle:>7.1f}% "
              f"{mean_baseline:>9.3f}")

        table_rows.append({
            "benchmark": r["name"],
            "n_tasks": n_tasks,
            "n_configs_full": r["n_full_unique"],
            "n_configs_selected": r["n_selected_unique"],
            "pct_grid": round(pct_grid, 1),
            "mean_baseline_sr": round(mean_baseline, 3),
            "mean_oracle_sr": round(mean_oracle, 3),
            "mean_geo_sr": round(mean_geo, 3),
            "pct_oracle": round(pct_oracle, 1),
            "sel_layer": r["sel_layer"],
            "sel_alphas": r["sel_alphas"],
            "sel_betas": r["sel_betas"],
        })

    print(f"\n  Interpretation: geometric selection achieves X% of oracle")
    print(f"  performance while evaluating Y% of the full grid.\n")

    # ── Per-task detail for appendix ─────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  PER-TASK ORACLE vs GEOMETRIC — Full Detail")
    print(f"{'='*90}")
    for r in results:
        print(f"\n  --- {r['name']} (L={r['sel_layer']}, α={r['sel_alphas']}, β={r['sel_betas']}) ---")
        pts = r["per_task"]
        for t in pts:
            gap = t["oracle_sr"] - t["geo_sr"]
            pct = 100 * t["geo_sr"] / t["oracle_sr"] if t["oracle_sr"] > 0 else 0
            geo_vs_base = t["geo_sr"] - t["baseline_sr"]
            oracle_vs_base = t["oracle_sr"] - t["baseline_sr"]
            print(f"    {t['task']:<40s}  base={t['baseline_sr']:.2f}  oracle={t['oracle_sr']:.2f}  "
                  f"geo={t['geo_sr']:.2f}  gap={gap:+.2f}  "
                  f"({pct:.0f}% oracle)")
            if gap > 0.05:
                print(f"      ↳ oracle used: {t['oracle_cond']}")
                print(f"      ↳ geo used:    {t['geo_cond']}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    # Nice display names for benchmarks
    DISPLAY_NAMES = {
        "groot_robocasa": "GR-1 RoboCasa",
        "pi05_robocasa": "$\\pi_{0.5}$ RoboCasa",
        "pi05_libero": "$\\pi_{0.5}$ LIBERO",
    }

    # Build caption with selected params
    param_parts = []
    for tr in table_rows:
        dname = DISPLAY_NAMES.get(tr['benchmark'], tr['benchmark'])
        alphas = tr['sel_alphas']
        if len(alphas) == 1:
            a_str = f"$\\alpha{{=}}{alphas[0]}$"
        else:
            a_str = "$\\alpha \\in \\{" + ", ".join(str(a) for a in alphas) + "\\}$"
        param_parts.append(f"{dname} uses $\\ell{{=}}{tr['sel_layer']}$, {a_str}")
    betas_str = ", ".join(str(b) for b in table_rows[0]['sel_betas'])

    tex_path = os.path.join(OUTPUT_DIR, "oracle_gap_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        pct_lo = min(tr['pct_oracle'] for tr in table_rows)
        pct_hi = max(tr['pct_oracle'] for tr in table_rows)
        grid_lo = min(tr['pct_grid'] for tr in table_rows)
        grid_hi = max(tr['pct_grid'] for tr in table_rows)
        caption = (f"\\caption{{Oracle gap analysis. Geometric selection achieves "
                   f"{pct_lo:.0f}--{pct_hi:.0f}\\% of oracle performance while evaluating "
                   f"only {grid_lo:.0f}--{grid_hi:.0f}\\% of the full configuration grid. "
                   f"Selected parameters: {'; '.join(param_parts)}. "
                   f"All benchmarks use $\\beta \\in \\{{{betas_str}\\}}$.}}\n")
        f.write(caption)
        f.write("\\label{tab:oracle-gap}\n")
        f.write("\\begin{tabular}{lrrrrc}\n")
        f.write("\\toprule\n")
        f.write("Benchmark & Tasks & \\shortstack{Full\\\\Grid} & \\shortstack{Selected\\\\Grid} & \\shortstack{\\% Grid\\\\Evaluated} & \\shortstack{\\% of Oracle\\\\SR} \\\\\n")
        f.write("\\midrule\n")
        for tr in table_rows:
            dname = DISPLAY_NAMES.get(tr['benchmark'], tr['benchmark'])
            f.write(f"{dname} & {tr['n_tasks']} & "
                    f"{tr['n_configs_full']} & {tr['n_configs_selected']} & "
                    f"{tr['pct_grid']:.1f}\\% & {tr['pct_oracle']:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"\nSaved LaTeX: {tex_path}")

    # ── JSON output ──────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "oracle_gap_table.json")
    full_output = {
        "summary": table_rows,
        "per_benchmark": [],
    }
    for r in results:
        full_output["per_benchmark"].append({
            "name": r["name"],
            "sel_layer": r["sel_layer"],
            "sel_alphas": r["sel_alphas"],
            "sel_betas": r["sel_betas"],
            "per_task": [{
                "task": t["task"],
                "baseline_sr": t["baseline_sr"],
                "oracle_sr": t["oracle_sr"],
                "oracle_cond": t["oracle_cond"],
                "geo_sr": t["geo_sr"],
                "geo_cond": t["geo_cond"],
                "n_full": t["n_full"],
                "n_selected": t["n_selected"],
            } for t in r["per_task"]],
        })
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
