#!/usr/bin/env python3
"""Per-task variant of `select_parameters.py`.

Runs the same overlap-and-quota selection logic, but separately for each task
present in the conceptor `.npz`, instead of averaging metrics across tasks.

Use when you want a per-(checkpoint, task) recipe rather than one recipe
that's averaged across all tasks in a single conceptor file.

Usage
-----
    python experiments/robocasa_steering/select_parameters_per_task.py \\
        --conceptor-npz path/to/conceptors.npz \\
        --output-json   path/to/selected_params__per_task.json

Output JSON
-----------
{
  "<task1>": {best_layer, selected_alphas, selected_betas, overlap_band, diagnostics},
  "<task2>": {...},
  ...
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow importing the sibling select_parameters module regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_parameters import select_parameters  # noqa: E402


class _NpzSubset:
    """`.files` / `__getitem__` view of an NpzFile filtered by key prefix."""

    def __init__(self, npz, prefix: str):
        self._npz = npz
        self._files = [k for k in npz.files if k.startswith(prefix)]

    @property
    def files(self):
        return list(self._files)

    def __getitem__(self, k):
        return self._npz[k]


def discover_tasks(npz) -> list[str]:
    """Tasks present in the conceptor file (matches build_conceptors key format)."""
    seen: set[str] = set()
    for k in npz.files:
        if k.startswith("_"):
            continue
        # keys look like {task}__L{layer}__{alpha}__C_{kind}
        head = k.split("__L", 1)[0]
        if head:
            seen.add(head)
    return sorted(seen)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conceptor-npz", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--overlap-low", type=float, default=0.85)
    p.add_argument("--overlap-high", type=float, default=0.95)
    p.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--quota-alpha", type=float, default=10.0)
    args = p.parse_args()

    npz = np.load(args.conceptor_npz)
    tasks = discover_tasks(npz)
    print(f"[per-task selector] {args.conceptor_npz}")
    print(f"  tasks ({len(tasks)}): {tasks}")

    per_task: dict[str, dict] = {}
    for t in tasks:
        print("\n" + "=" * 60)
        print(f"TASK: {t}")
        print("=" * 60)
        sub = _NpzSubset(npz, f"{t}__")
        try:
            per_task[t] = select_parameters(
                sub,
                overlap_low=args.overlap_low,
                overlap_high=args.overlap_high,
                candidate_betas=args.betas,
                quota_alpha=args.quota_alpha,
            )
        except SystemExit as e:
            # select_parameters sys.exits on missing quota data — record + continue.
            print(f"  [skipped] selection failed: {e}", file=sys.stderr)
            per_task[t] = {"error": f"select_parameters failed: {e}"}

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(per_task, f, indent=2)
    print(f"\nWritten: {args.output_json}")


if __name__ == "__main__":
    main()
