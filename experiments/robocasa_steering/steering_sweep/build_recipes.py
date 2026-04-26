#!/usr/bin/env python3
"""Generate per-(ckpt, task) steering recipe JSON from the conceptor manifest.

Rules (encoded from the user's spec):
  * beta = 0.1 for all.
  * For CoffeeSetupMug: use the FIRST alpha from each ckpt's
    selected_params__per_task.json (i.e. per-(task, epoch) selection).
  * For all other tasks: use the FIRST alpha from
    common_params__per_task.json (the alpha that overlapped across epochs).
  * Layer: take whatever common_params__per_task.json says (mode across ckpts).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONCEPTORS = HERE.parent / "conceptors"
MANIFEST = CONCEPTORS / "manifest.json"
COMMON = CONCEPTORS / "common_params__per_task.json"
OUT = HERE / "recipes.json"

PER_TASK_FIRST_ALPHA_RULE = "first_per_ckpt"  # CoffeeSetupMug
COMMON_FIRST_ALPHA_RULE = "first_common"     # all other tasks
TASKS_USING_PER_CKPT_ALPHA = {"CoffeeSetupMug"}
BETA = 0.1


def main():
    if not MANIFEST.is_file() or not COMMON.is_file():
        sys.exit(f"missing manifest or common_params at {CONCEPTORS}")
    manifest = json.loads(MANIFEST.read_text())
    common = json.loads(COMMON.read_text())

    recipes: dict[str, dict[str, dict]] = {}
    for ck, info in manifest["checkpoints"].items():
        per_ck: dict[str, dict] = {}
        for task, sel in info["tasks"].items():
            if "error" in sel:
                continue
            layer = sel["best_layer"]  # may differ from common; we'll override below
            if task in TASKS_USING_PER_CKPT_ALPHA:
                # First alpha from this ckpt's per-task selection.
                alphas = sel["selected_alphas"]
                if not alphas:
                    continue
                alpha = float(alphas[0])
                # Use this ckpt's chosen layer (per-(task,epoch) selection).
                layer_use = layer
            else:
                if task not in common:
                    continue  # task not present in common (shouldn't happen)
                comm = common[task]
                alphas = comm["selected_alphas"]
                if not alphas:
                    continue
                alpha = float(alphas[0])
                layer_use = comm["best_layer"]
            per_ck[task] = {
                "layer": int(layer_use),
                "alpha": float(alpha),
                "beta": float(BETA),
            }
        recipes[ck] = per_ck

    OUT.write_text(json.dumps(recipes, indent=2))
    # Pretty-print summary.
    all_tasks = sorted({t for r in recipes.values() for t in r})
    ckpts = list(recipes.keys())
    print(f"recipes written to {OUT}")
    print()
    w = 16
    print(f'{"task":<28s} ' + ' '.join(f'{c[:14]:>{w}s}' for c in ckpts))
    print('-' * (28 + (w+1) * len(ckpts)))
    for t in all_tasks:
        cells = []
        for c in ckpts:
            r = recipes[c].get(t)
            cells.append(f'L{r["layer"]} a={r["alpha"]} b={r["beta"]}'.center(w) if r else '—'.center(w))
        print(f'{t:<28s} ' + ' '.join(cells))


if __name__ == "__main__":
    main()
