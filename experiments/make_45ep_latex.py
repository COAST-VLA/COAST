#!/usr/bin/env python3
"""
Generate LaTeX tables with statistical significance from 45-episode steering results.

Reads summary.json from each task directory, computes per-task success rates,
then runs paired t-tests (baseline vs each steering mode) and bootstrap CIs.

Usage:
    python experiments/make_45ep_latex.py
"""

import json
import os
import pathlib
import sys

import numpy as np
from scipy import stats

BASE = "/vast/projects/ungar/stellar/miaom"

EXPERIMENT_SETS = {
    "pi05_libero": {
        "label": r"$\pi$0.5 LIBERO",
        "results_dir": f"{BASE}/openpi-new/experiments/pi05_libero/steering_results_45ep",
        "mode": "oracle",
    },
    "pi05_robocasa": {
        "label": r"$\pi$0.5 RoboCasa",
        "results_dir": f"{BASE}/openpi-new/experiments/pi05_robocasa/steering_results_45ep",
        "mode": "oracle",
    },
    "groot_robocasa": {
        "label": "GR00T RoboCasa",
        "results_dir": f"{BASE}/openpi-groot/experiments/groot_robocasa/steering_results_45ep",
        "mode": "oracle",
    },
}

CONDITION_PREFIXES = {
    "Baseline": "baseline",
    "Global": "global_",
    "Per-Step": "per_step",
    "Pos.-Only": "pos_only_",
}


def load_results(results_dir: str) -> dict[str, dict[str, float]]:
    """Load summary.json from each task dir.

    Returns {task: {label: sr}} where label is one of Baseline/Global/Per-Step/Pos.-Only,
    resolved by prefix matching against CONDITION_PREFIXES.
    """
    results = {}
    rdir = pathlib.Path(results_dir)
    for task_dir in sorted(rdir.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("scripts", "logs"):
            continue
        summary = task_dir / "summary.json"
        if not summary.exists():
            print(f"  WARNING: no summary.json in {task_dir.name}", file=sys.stderr)
            continue
        with open(summary) as f:
            data = json.load(f)
        raw_map = {}
        for c in data.get("conditions", []):
            if c is not None and "condition" in c:
                raw_map[c["condition"]] = c["success_rate"]

        label_map = {}
        for label, prefix in CONDITION_PREFIXES.items():
            if label == "Baseline":
                if "baseline" in raw_map:
                    label_map[label] = raw_map["baseline"]
            else:
                matches = [(k, v) for k, v in raw_map.items() if k.startswith(prefix) and k != "baseline"]
                if len(matches) == 1:
                    label_map[label] = matches[0][1]
                elif len(matches) > 1:
                    print(f"  WARNING: multiple {label} matches in {task_dir.name}: {[m[0] for m in matches]}", file=sys.stderr)
                    label_map[label] = matches[0][1]

        results[task_dir.name] = label_map
    return results


def bootstrap_ci(values, n_boot=10000, ci=0.95, rng=None):
    """Bootstrap confidence interval for the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])


def significance_marker(p: float) -> str:
    if p < 0.001:
        return r"$^{***}$"
    elif p < 0.01:
        return r"$^{**}$"
    elif p < 0.05:
        return r"$^{*}$"
    return ""


def make_table_one_experiment(exp_key: str, exp_cfg: dict) -> str:
    """Generate a LaTeX table for one experiment set."""
    results = load_results(exp_cfg["results_dir"])
    if not results:
        return f"% No results found for {exp_key}\n"

    tasks = sorted(results.keys())
    cond_labels = list(CONDITION_PREFIXES.keys())

    n_tasks = len(tasks)
    print(f"\n{exp_key}: {n_tasks} tasks loaded", file=sys.stderr)

    # Build per-task arrays
    per_task = {cl: [] for cl in cond_labels}
    for task in tasks:
        for cl in cond_labels:
            sr = results[task].get(cl, float("nan"))
            per_task[cl].append(sr)

    baseline_arr = np.array(per_task["Baseline"])

    # Header
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + exp_cfg["label"] + r" --- 45-episode evaluation with statistical significance.}")
    lines.append(r"\label{tab:" + exp_key + r"_45ep}")
    col_spec = "l" + "c" * len(cond_labels)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append("Task & " + " & ".join(cond_labels) + r" \\")
    lines.append(r"\midrule")

    # Per-task rows
    for i, task in enumerate(tasks):
        short = task[:40].replace("_", r"\_")
        cells = [short]
        for cl in cond_labels:
            sr = results[task].get(cl, float("nan"))
            if np.isnan(sr):
                cells.append("---")
            else:
                cells.append(f"{sr:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # Mean row with significance
    mean_cells = [r"\textbf{Mean}"]
    for cl in cond_labels:
        arr = np.array(per_task[cl])
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            mean_cells.append("---")
            continue
        m = valid.mean()
        se = valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0

        if cl == "Baseline":
            mean_cells.append(f"{m:.3f} $\\pm$ {se:.3f}")
        else:
            # Paired t-test vs baseline (only on tasks where both are valid)
            mask = ~np.isnan(arr) & ~np.isnan(baseline_arr)
            if mask.sum() >= 2:
                t_stat, p_val = stats.ttest_rel(arr[mask], baseline_arr[mask])
                sig = significance_marker(p_val)
                ci_lo, ci_hi = bootstrap_ci(arr[mask] - baseline_arr[mask])
                delta = arr[mask].mean() - baseline_arr[mask].mean()
                sign = "+" if delta >= 0 else ""
                mean_cells.append(
                    f"{m:.3f} $\\pm$ {se:.3f}{sig}"
                )
            else:
                mean_cells.append(f"{m:.3f} $\\pm$ {se:.3f}")

    lines.append(" & ".join(mean_cells) + r" \\")

    # Delta row
    delta_cells = [r"$\Delta$ vs Baseline"]
    for cl in cond_labels:
        if cl == "Baseline":
            delta_cells.append("---")
            continue
        arr = np.array(per_task[cl])
        mask = ~np.isnan(arr) & ~np.isnan(baseline_arr)
        if mask.sum() >= 2:
            delta = arr[mask].mean() - baseline_arr[mask].mean()
            t_stat, p_val = stats.ttest_rel(arr[mask], baseline_arr[mask])
            sig = significance_marker(p_val)
            ci_lo, ci_hi = bootstrap_ci(arr[mask] - baseline_arr[mask])
            sign = "+" if delta >= 0 else ""
            delta_cells.append(
                f"{sign}{delta:.3f}{sig} [{ci_lo:+.3f}, {ci_hi:+.3f}]"
            )
        else:
            delta_cells.append("---")
    lines.append(" & ".join(delta_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\raggedright\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (paired $t$-test vs.\ baseline). 95\% bootstrap CI on $\Delta$.}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_combined_table() -> str:
    """Generate one combined table across all 3 experiment sets."""
    all_data = {}
    cond_labels = list(CONDITION_PREFIXES.keys())
    for exp_key, exp_cfg in EXPERIMENT_SETS.items():
        results = load_results(exp_cfg["results_dir"])
        if not results:
            print(f"  WARNING: no results for {exp_key}", file=sys.stderr)
            continue
        tasks = sorted(results.keys())
        per_task = {cl: [] for cl in cond_labels}
        for task in tasks:
            for cl in cond_labels:
                per_task[cl].append(results[task].get(cl, float("nan")))
        all_data[exp_key] = {
            "label": exp_cfg["label"],
            "per_task": per_task,
            "cond_labels": cond_labels,
            "n_tasks": len(tasks),
        }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Conceptor steering results (45 episodes). Mean success rate $\pm$ SE across tasks.}")
    lines.append(r"\label{tab:steering_45ep_combined}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & Baseline & Global & Per-Step & Pos.-Only \\")
    lines.append(r"\midrule")

    for exp_key, d in all_data.items():
        pt = d["per_task"]
        baseline_arr = np.array(pt["Baseline"])
        cells = [d["label"]]
        for cl in ["Baseline", "Global", "Per-Step", "Pos.-Only"]:
            arr = np.array(pt[cl])
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                cells.append("---")
                continue
            m = valid.mean()
            se = valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0
            if cl == "Baseline":
                cells.append(f"{m:.3f} $\\pm$ {se:.3f}")
            else:
                mask = ~np.isnan(arr) & ~np.isnan(baseline_arr)
                if mask.sum() >= 2:
                    t_stat, p_val = stats.ttest_rel(arr[mask], baseline_arr[mask])
                    sig = significance_marker(p_val)
                    cells.append(f"{m:.3f} $\\pm$ {se:.3f}{sig}")
                else:
                    cells.append(f"{m:.3f} $\\pm$ {se:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\raggedright\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (paired $t$-test vs.\ baseline, 45 episodes per condition per task).}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("45-Episode Steering Results — LaTeX Tables")
    print("=" * 70)

    # Individual tables
    for exp_key, exp_cfg in EXPERIMENT_SETS.items():
        table = make_table_one_experiment(exp_key, exp_cfg)
        print(f"\n{'='*70}")
        print(f"  {exp_key}")
        print(f"{'='*70}")
        print(table)

        out_path = pathlib.Path(exp_cfg["results_dir"]) / "table_45ep.tex"
        with open(out_path, "w") as f:
            f.write(table + "\n")
        print(f"\n  → Saved to {out_path}", file=sys.stderr)

    # Combined table
    combined = make_combined_table()
    print(f"\n{'='*70}")
    print("  COMBINED")
    print(f"{'='*70}")
    print(combined)

    combined_path = f"{BASE}/openpi-new/experiments/table_steering_45ep_combined.tex"
    with open(combined_path, "w") as f:
        f.write(combined + "\n")
    print(f"\n  → Saved to {combined_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
