#!/usr/bin/env python3
"""
Pick the best-performing steering configuration per task from the sweep
``summary.json`` files and write a consolidated ``selected_params.json`` ready
to feed into downstream experiments (e.g. steered activation collection).

Input layout (written by conceptor_steering.py):
  experiments/pi0_fast_libero/steering_results/{task_short}/summary.json
      {"task": "...", "conditions": [{"condition": str, "success_rate": float}, ...]}

Output:
  experiments/pi0_fast_libero/selected_params.json
      {"<task>": {"strategy": str, "alpha": float, "beta": float,
                  "kind": "C_contrastive", "success_rate": float,
                  "beat_baseline_by": float}, ...}

Usage (from repo root):
    uv run experiments/pi0_fast_libero/src/select_best_configs.py
"""
import dataclasses
import json
import logging
import pathlib
import re
from typing import Optional

import tyro

logger = logging.getLogger(__name__)

# Parses condition strings written by conceptor_steering.py
RE_CONST = re.compile(
    r"^(?P<strategy>global|per_token_first|per_token_mid|per_token_last)"
    r"_a(?P<alpha>[0-9.]+)_b(?P<beta>[0-9.]+)$"
)
RE_LINEAR = re.compile(r"^linear_b(?P<beta>[0-9.]+)_be(?P<beta_end>[0-9.]+)$")
RE_RANDOM = re.compile(r"^random_b(?P<beta>[0-9.]+)$")


def parse_condition(name: str) -> Optional[dict]:
    """Parse a condition string into its hyperparameters. Returns None if it
    isn't a recognized steering condition (e.g. ``baseline`` or ``random_*``)."""
    m = RE_CONST.match(name)
    if m:
        return {
            "kind": "constant",
            "strategy": m["strategy"],
            "alpha": float(m["alpha"]),
            "beta": float(m["beta"]),
        }
    m = RE_LINEAR.match(name)
    if m:
        return {
            "kind": "linear",
            "strategy": "global",
            "alpha": 1.0,
            "beta": float(m["beta"]),
            "beta_end": float(m["beta_end"]),
        }
    return None


@dataclasses.dataclass
class Args:
    results_dir: str = "experiments/pi0_fast_libero/steering_results"
    output_path: str = "experiments/pi0_fast_libero/selected_params.json"
    # Only select configs that beat baseline by at least this margin.
    min_baseline_gain: float = 0.0


def main(args: Args) -> None:
    results_root = pathlib.Path(args.results_dir)
    if not results_root.exists():
        raise FileNotFoundError(f"No steering results at {results_root}")

    selected: dict = {}
    for task_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            logger.warning("No summary.json in %s", task_dir.name)
            continue
        with open(summary_path) as f:
            data = json.load(f)
        task = data.get("task") or task_dir.name
        conds = data.get("conditions", [])
        if not conds:
            logger.warning("Empty conditions for %s", task)
            continue

        baseline_sr = next(
            (c["success_rate"] for c in conds if c["condition"] == "baseline"), None
        )
        # Rank steered (non-random) conditions by success_rate.
        ranked = [
            c for c in conds
            if parse_condition(c["condition"]) is not None
               and not c["condition"].startswith("random")
               and c["success_rate"] == c["success_rate"]  # filter NaN
        ]
        ranked.sort(key=lambda c: c["success_rate"], reverse=True)
        best = ranked[0] if ranked else None
        if best is None:
            logger.warning("No valid steered condition for %s", task)
            continue

        gain = (
            best["success_rate"] - baseline_sr
            if baseline_sr is not None
            else float("nan")
        )
        if baseline_sr is not None and gain < args.min_baseline_gain:
            logger.info(
                "%s: best %s (SR=%.2f) does NOT beat baseline (%.2f) by >=%.2f — skipping",
                task[:40], best["condition"], best["success_rate"], baseline_sr,
                args.min_baseline_gain,
            )
            continue

        parsed = parse_condition(best["condition"])
        entry = {
            "condition": best["condition"],
            "success_rate": best["success_rate"],
            "baseline_sr": baseline_sr,
            "beat_baseline_by": gain,
            **parsed,
            "kind_conceptor": "C_contrastive",
        }
        selected[task] = entry
        logger.info(
            "%s: best %s (SR=%.2f, baseline=%.2f, +%.2f)",
            task[:40], best["condition"], best["success_rate"],
            baseline_sr if baseline_sr is not None else float("nan"), gain,
        )

    out_path = pathlib.Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info("\nWrote %d selections → %s", len(selected), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
