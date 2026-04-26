#!/usr/bin/env python3
"""Build conceptors + per-task parameter selections for every checkpoint
under an activations root, and write a single manifest.json the steering
runner can dispatch from.

Usage
-----
    python experiments/robocasa_steering/build_and_select_all.py \\
        --activations-root /mnt/bird_home/kim34/eval_sweep_results \\
        --output-dir       experiments/robocasa_steering/conceptors

For each checkpoint subdir <activations_root>/<ckpt_stem>/ this script:
  1. shells out to build_conceptors.py        --activations-dir=<ckpt_dir>
                                              --output-npz=<output_dir>/<ckpt_stem>/conceptors.npz
  2. shells out to select_parameters_per_task.py
                                              --conceptor-npz=<above>
                                              --output-json=<output_dir>/<ckpt_stem>/selected_params__per_task.json
  3. records the (ckpt_stem, task) -> (conceptor_npz, selected_params) mapping
     into <output_dir>/manifest.json.

The manifest schema (canonical lookup key: manifest["checkpoints"][ckpt_stem]["tasks"][task]):
  {
    "activations_root": "...",
    "checkpoints": {
        "<ckpt_stem>": {
          "conceptor_npz": "...",                     # absolute path
          "selected_params_json": "...",              # absolute path
          "tasks": {
            "<task>": {
              "best_layer": int,
              "selected_alphas": [...],
              "selected_betas": [...],
              "overlap_band": [low, high],
            },
            ...
          },
          "tasks_skipped_no_success": [...]           # tasks with 0 success (build_conceptors drops these)
        },
        ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def discover_tasks_under_ckpt(ckpt_dir: Path) -> list[str]:
    out = []
    for p in sorted(ckpt_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("_"):
            out.append(p.name)
    return out


def task_success_count(ckpt_dir: Path, task: str) -> tuple[int, int]:
    metas = list((ckpt_dir / task).glob("episode_*_env_*/metadata.json"))
    successes = 0
    for m in metas:
        try:
            d = json.loads(m.read_text())
            if d.get("episode_success"):
                successes += 1
        except Exception:
            pass
    return successes, len(metas)


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations-root", required=True, type=Path,
                    help="Root containing one subdir per checkpoint stem.")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write per-ckpt conceptors + manifest.json.")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Pass-through to build_conceptors.py.")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Pass-through to build_conceptors.py.")
    ap.add_argument("--overlap-low", type=float, default=0.85)
    ap.add_argument("--overlap-high", type=float, default=0.95)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip the build step (assume conceptors.npz already on disk).")
    ap.add_argument("--skip-select", action="store_true",
                    help="Skip the selection step.")
    args = ap.parse_args()

    if not args.activations_root.is_dir():
        sys.exit(f"activations_root not a dir: {args.activations_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_stems = sorted(p.name for p in args.activations_root.iterdir()
                        if p.is_dir() and not p.name.startswith("_"))
    print(f"[build_and_select_all] checkpoints found: {ckpt_stems}")

    manifest = {
        "activations_root": str(args.activations_root.resolve()),
        "checkpoints": {},
    }

    build_script = HERE / "build_conceptors.py"
    select_script = HERE / "select_parameters_per_task.py"

    for stem in ckpt_stems:
        ckpt_dir = args.activations_root / stem
        out_subdir = args.output_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)
        conceptor_npz = out_subdir / "conceptors.npz"
        selected_json = out_subdir / "selected_params__per_task.json"

        # Pre-scan the on-disk activation tree to find tasks with 0 successes.
        # build_conceptors.py auto-drops these; we record them in the manifest.
        all_tasks_on_disk = discover_tasks_under_ckpt(ckpt_dir)
        skipped_no_success = []
        for t in all_tasks_on_disk:
            ns, ne = task_success_count(ckpt_dir, t)
            if ne >= 1 and ns == 0:
                skipped_no_success.append(t)

        # 1. Build conceptors.
        if not args.skip_build:
            cmd = [str(PYTHON), str(build_script),
                   "--activations-dir", str(ckpt_dir),
                   "--output-npz", str(conceptor_npz)]
            if args.alphas: cmd += ["--alphas", *map(str, args.alphas)]
            if args.layers: cmd += ["--layers", *map(str, args.layers)]
            run(cmd)

        if not conceptor_npz.is_file():
            print(f"[skip] {stem}: no conceptor npz at {conceptor_npz}", file=sys.stderr)
            continue

        # 2. Run per-task selector.
        if not args.skip_select:
            cmd = [str(PYTHON), str(select_script),
                   "--conceptor-npz", str(conceptor_npz),
                   "--output-json", str(selected_json),
                   "--overlap-low", str(args.overlap_low),
                   "--overlap-high", str(args.overlap_high),
                   "--betas", *map(str, args.betas)]
            run(cmd)

        # 3. Slot into manifest.
        per_task_entries: dict = {}
        if selected_json.is_file():
            data = json.loads(selected_json.read_text())
            for task, sel in data.items():
                if "error" in sel:
                    per_task_entries[task] = sel
                    continue
                per_task_entries[task] = {
                    "best_layer": sel["best_layer"],
                    "selected_alphas": sel["selected_alphas"],
                    "selected_betas": sel["selected_betas"],
                    "overlap_band": sel["overlap_band"],
                }

        manifest["checkpoints"][stem] = {
            "conceptor_npz": str(conceptor_npz.resolve()),
            "selected_params_json": str(selected_json.resolve()),
            "tasks": per_task_entries,
            "tasks_skipped_no_success": skipped_no_success,
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[build_and_select_all] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
