#!/usr/bin/env python3
"""Summarize success/failure episode counts per task from activation datasets."""
import json
import sys
from pathlib import Path


def summarize(root: Path):
    if not root.exists():
        print(f"  Not found: {root}")
        return
    tasks = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"  {len(tasks)} tasks")
    total_s, total_f = 0, 0
    for task in tasks:
        task_dir = root / task
        eps = sorted(d for d in task_dir.iterdir() if d.is_dir())
        n_success = 0
        n_failure = 0
        n_unknown = 0
        for ep in eps:
            meta_path = ep / "metadata.json"
            if not meta_path.exists():
                n_unknown += 1
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("episode_success", False):
                n_success += 1
            else:
                n_failure += 1
        total = n_success + n_failure + n_unknown
        total_s += n_success
        total_f += n_failure
        print(f"  {task:70s}  {total:3d} eps  {n_success:3d} succ  {n_failure:3d} fail  {n_unknown:3d} unk")
    print(f"  {'TOTAL':70s}  {total_s + total_f:3d} eps  {total_s:3d} succ  {total_f:3d} fail")


if __name__ == "__main__":
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/openpi"

    print("\n=== LIBERO (ckpt 2000) ===")
    summarize(cache / "pi0fast-libero-activations-v1-2000-15env" / "2000")

    print("\n=== MetaWorld ML45 (ckpt 2500) ===")
    summarize(cache / "pi0fast-metaworld-activations-v1-ml45train-16env" / "2500")

    print("\n=== MetaWorld ML45 task list (for run_steering.sh) ===")
    mw_root = cache / "pi0fast-metaworld-activations-v1-ml45train-16env" / "2500"
    if mw_root.exists():
        tasks = sorted(d.name for d in mw_root.iterdir() if d.is_dir())
        for i, t in enumerate(tasks):
            print(f"    {t}")
