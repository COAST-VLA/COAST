#!/usr/bin/env python3
"""Success-rate aggregator for the per-checkpoint × 18-task eval sweep.

Two modes:

  per-checkpoint   Walk ONE checkpoint's activation tree and write
                   success_rates.json (used at the end of each sbatch job).

  all              Walk ALL per-checkpoint success_rates.json files under an
                   activations root and write a combined results.json (used
                   by the top-level aggregator job).

The per-task success rate is computed from the per-episode metadata files
written by collect_activations_robocasa.py:
    <ckpt_dir>/<task>/episode_NNN_env_NNN/metadata.json
        -> {"episode_success": bool, ...}

Only episodes under indices [0, num_rollouts) are counted — envs that ran
only because a chunk fell short of num_envs but weren't part of the requested
rollout quota don't get included. (collect_activations_robocasa.py preserves
them but sets their episode_id accordingly; the filter is tolerant of that.)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def _task_success_rate(task_dir: pathlib.Path) -> dict[str, Any]:
    """Scan one task directory and return {n_episodes, n_success, success_rate}."""
    eps = sorted(p for p in task_dir.iterdir()
                 if p.is_dir() and p.name.startswith("episode_"))
    successes = []
    for ep in eps:
        meta_path = ep / "metadata.json"
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "episode_success" in meta:
            successes.append(bool(meta["episode_success"]))
    n = len(successes)
    ns = sum(successes)
    sr = (ns / n) if n else float("nan")
    return {"n_episodes": n, "n_success": ns, "success_rate": sr}


def per_checkpoint(checkpoint_dir: pathlib.Path) -> None:
    """Walk <checkpoint_dir>/<task>/episode_*/metadata.json and write
    <checkpoint_dir>/success_rates.json."""
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint dir not found: {checkpoint_dir}")

    tasks = sorted(p for p in checkpoint_dir.iterdir()
                   if p.is_dir() and not p.name.startswith("_")
                   and p.name != "success_rates.json")
    out = {"checkpoint_stem": checkpoint_dir.name, "tasks": {}}
    all_rates = []
    for task_dir in tasks:
        info = _task_success_rate(task_dir)
        out["tasks"][task_dir.name] = info
        if info["n_episodes"] > 0:
            all_rates.append(info["success_rate"])
    out["mean_success_rate"] = (sum(all_rates) / len(all_rates)) if all_rates else float("nan")
    out["num_tasks"] = len(out["tasks"])

    out_path = checkpoint_dir / "success_rates.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable stdout summary.
    print(f"\n[success_rates] {checkpoint_dir.name}  (mean={out['mean_success_rate']:.3f})")
    print(f"  {'task':<35s} {'n_succ':>7s} {'n_eps':>6s} {'SR':>7s}")
    for t, info in sorted(out["tasks"].items()):
        sr = info["success_rate"]
        sr_str = f"{sr:.3f}" if sr == sr else "  nan"
        print(f"  {t:<35s} {info['n_success']:>7d} {info['n_episodes']:>6d} {sr_str:>7s}")
    print(f"  {'MEAN':<35s} {'':>7s} {'':>6s} {out['mean_success_rate']:>7.3f}")
    print(f"\nwritten: {out_path}")


def aggregate_all(activations_root: pathlib.Path, output_json: pathlib.Path) -> None:
    """Collect every <activations_root>/<ckpt>/success_rates.json into one file.

    If a per-checkpoint file is missing (e.g. that job hasn't finished or
    failed), fall back to computing it on the fly from metadata.
    """
    if not activations_root.is_dir():
        raise SystemExit(f"activations root not found: {activations_root}")

    ckpt_dirs = sorted(p for p in activations_root.iterdir()
                       if p.is_dir() and not p.name.startswith("_"))
    combined = {"activations_root": str(activations_root), "checkpoints": {}}

    for ck in ckpt_dirs:
        sr_file = ck / "success_rates.json"
        if sr_file.is_file():
            with open(sr_file) as f:
                data = json.load(f)
        else:
            # Fallback: materialize on demand so partial sweeps still show
            # something useful in results.json.
            print(f"  note: {sr_file} missing, computing from metadata ...")
            per_checkpoint(ck)
            with open(sr_file) as f:
                data = json.load(f)
        combined["checkpoints"][ck.name] = data

    # Cross-checkpoint summary table: {task -> {ckpt_stem -> success_rate}}
    all_tasks: set[str] = set()
    for data in combined["checkpoints"].values():
        all_tasks.update(data.get("tasks", {}).keys())
    per_task_table = {}
    for t in sorted(all_tasks):
        per_task_table[t] = {
            ck: combined["checkpoints"][ck]["tasks"].get(t, {}).get("success_rate", None)
            for ck in combined["checkpoints"]
        }
    combined["per_task_table"] = per_task_table
    combined["mean_per_checkpoint"] = {
        ck: combined["checkpoints"][ck].get("mean_success_rate")
        for ck in combined["checkpoints"]
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(combined, f, indent=2)

    # Human-readable stdout: task × checkpoint table.
    stems = list(combined["checkpoints"].keys())
    print(f"\n[aggregate] {len(stems)} checkpoint(s), {len(all_tasks)} task(s)")
    print(f"\n{'task':<35s}  " + "  ".join(f"{s[:16]:>16s}" for s in stems))
    for t in sorted(all_tasks):
        row = per_task_table[t]
        cells = "  ".join(
            f"{(v*100):>15.1f}%" if isinstance(v, (int, float)) and v == v else f"{'nan':>16s}"
            for v in (row[s] for s in stems)
        )
        print(f"{t:<35s}  {cells}")
    mean_row = "  ".join(
        f"{(v*100):>15.1f}%" if isinstance(v, (int, float)) and v == v else f"{'nan':>16s}"
        for v in (combined["mean_per_checkpoint"][s] for s in stems)
    )
    print(f"{'MEAN':<35s}  {mean_row}")
    print(f"\nwritten: {output_json}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("per-checkpoint", help="compute SR for one checkpoint")
    pc.add_argument("--checkpoint-dir", type=pathlib.Path, required=True,
                    help="<activations_root>/<ckpt_stem>")

    ag = sub.add_parser("all", help="combine per-checkpoint SRs into results.json")
    ag.add_argument("--activations-root", type=pathlib.Path, required=True,
                    help="dir whose children are <ckpt_stem>/ subdirs")
    ag.add_argument("--output-json", type=pathlib.Path, default=None,
                    help="path to write results.json (default: <root>/results.json)")

    args = p.parse_args()
    if args.cmd == "per-checkpoint":
        per_checkpoint(args.checkpoint_dir)
    else:
        out = args.output_json or (args.activations_root / "results.json")
        aggregate_all(args.activations_root, out)


if __name__ == "__main__":
    main()
