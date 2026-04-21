#!/usr/bin/env python3
"""
Transfer Experiment (Design B) — pi0.5 RoboCasa
=================================================

Mirrors experiments/pi05_libero/src/transfer_steering.py. For each target task
j, sweeps all source tasks i and applies source i's best conceptor to j.

For each (i, j):
  * Global:     C = {i}__L{L_j^global}__{alpha_i^global}__C_contrastive,
                beta = beta_i^global
  * Per-step:   for step in [0, 9]:
                C = {i}__L{L_j^per_step}__per_step_{step}__C_contrastive,
                beta = beta_i^per_step
"""

import dataclasses
import json
import logging
import pathlib
import re
import sys
from typing import List

import tyro

_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from conceptor_steering import (  # noqa: E402
    ConceptorSteeringHook,
    ROBOCASA_TASKS,
    SteeredPolicyWrapper,
    get_global_contrastive,
    get_per_step_contrastive,
    load_npz,
    run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

CONDITION_RE = re.compile(
    r"^(?P<strategy>global|per_step_\d+)_L(?P<layer>\d+)_a(?P<alpha>[\d.]+)_b(?P<beta>[\d.]+)$"
)


def parse_condition(name: str):
    m = CONDITION_RE.match(name)
    if not m:
        return None
    return {
        "strategy": m.group("strategy"),
        "layer": int(m.group("layer")),
        "alpha": float(m.group("alpha")),
        "beta": float(m.group("beta")),
    }


def extract_best_configs(summary_path: pathlib.Path):
    with open(summary_path) as f:
        data = json.load(f)
    best = {"global": None, "per_step": None}
    for entry in data["conditions"]:
        parsed = parse_condition(entry["condition"])
        if parsed is None:
            continue
        key = "global" if parsed["strategy"] == "global" else "per_step"
        sr = entry["success_rate"]
        if best[key] is None or sr > best[key]["sr"]:
            best[key] = {**parsed, "sr": sr}
    if best["global"] is None or best["per_step"] is None:
        raise ValueError(f"Missing best global/per-step for {summary_path}")
    return best


@dataclasses.dataclass
class Args:
    target_task: str = ROBOCASA_TASKS[0]
    config: str = "pi05_robocasa"
    checkpoint_dir: str = (
        "/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
    )
    num_episodes: int = 15
    port: int = 8300

    results_root: str = "experiments/pi05_robocasa/steering_results"
    output_dir: str = "experiments/pi05_robocasa/transfer_results"

    per_step_values: List[int] = dataclasses.field(default_factory=lambda: [0, 9])


def main(args: Args):
    target = args.target_task
    target_out = pathlib.Path(args.output_dir) / f"target_{target}"
    target_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target: {target}")
    logger.info(f"Output: {target_out}")

    results_root = pathlib.Path(args.results_root)
    best_configs = {}
    for t in ROBOCASA_TASKS:
        summary_p = results_root / t / "summary.json"
        if not summary_p.exists():
            raise FileNotFoundError(f"Summary not found for task {t}: {summary_p}")
        best_configs[t] = extract_best_configs(summary_p)

    target_cfg = best_configs[target]
    target_L_global = target_cfg["global"]["layer"]
    target_L_per_step = target_cfg["per_step"]["layer"]
    logger.info(f"Target best-L (global={target_L_global}, per_step={target_L_per_step})")

    with open(target_out / "transfer_args.json", "w") as f:
        json.dump(
            {
                "args": dataclasses.asdict(args),
                "target": target,
                "target_best_configs": target_cfg,
                "all_best_configs": best_configs,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info("Loading policy (one-time cost)...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    npz = load_npz()
    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    all_cells = []

    def save_progress():
        with open(target_out / "summary.json", "w") as f:
            json.dump(
                {"target": target, "target_best": target_cfg, "cells": all_cells},
                f,
                indent=2,
            )

    sources = [s for s in ROBOCASA_TASKS if s != target]
    total = len(sources) * (1 + len(args.per_step_values))
    idx = 0

    for source in sources:
        src_cfg = best_configs[source]

        # ── Global transfer ──
        idx += 1
        alpha_g = src_cfg["global"]["alpha"]
        beta_g = src_cfg["global"]["beta"]
        cond_name = f"source_{source}__global"
        logger.info(f"[{idx}/{total}] {cond_name} (L={target_L_global}, a={alpha_g}, b={beta_g})")

        try:
            C = get_global_contrastive(npz, source, target_L_global, alpha_g)
        except KeyError as e:
            logger.warning(f"  Skipping (missing conceptor): {e}")
            all_cells.append(
                {
                    "source": source, "target": target, "strategy": "global",
                    "layer": target_L_global, "alpha": alpha_g, "beta": beta_g,
                    "success_rate": None, "error": str(e),
                }
            )
            save_progress()
        else:
            hook = ConceptorSteeringHook(C, beta=beta_g, device=device)
            wrapper.update_hooks([(target_L_global, hook)])
            r = run_condition(
                target, args.port, args.num_episodes, cond_name, target_out,
            )
            all_cells.append(
                {
                    "source": source, "target": target, "strategy": "global",
                    "layer": target_L_global, "alpha": alpha_g, "beta": beta_g,
                    "success_rate": r["success_rate"],
                }
            )
            save_progress()

        # ── Per-step transfer (full time-indexed stack) ──
        beta_p = src_cfg["per_step"]["beta"]
        for step in args.per_step_values:
            idx += 1
            cond_name = f"source_{source}__per_step_{step}"
            logger.info(f"[{idx}/{total}] {cond_name} (L={target_L_per_step}, b={beta_p})")
            try:
                C = get_per_step_contrastive(npz, source, target_L_per_step, step)
            except KeyError as e:
                logger.warning(f"  Skipping (missing conceptor): {e}")
                all_cells.append(
                    {
                        "source": source, "target": target,
                        "strategy": f"per_step_{step}",
                        "layer": target_L_per_step, "beta": beta_p,
                        "success_rate": None, "error": str(e),
                    }
                )
                save_progress()
                continue
            hook = ConceptorSteeringHook(C, beta=beta_p, device=device)
            wrapper.update_hooks([(target_L_per_step, hook)])
            r = run_condition(
                target, args.port, args.num_episodes, cond_name, target_out,
            )
            all_cells.append(
                {
                    "source": source, "target": target,
                    "strategy": f"per_step_{step}",
                    "layer": target_L_per_step, "beta": beta_p,
                    "success_rate": r["success_rate"],
                }
            )
            save_progress()

    save_progress()
    logger.info(f"\n{'='*70}\nDone target={target}: {len(all_cells)} cells\n{'='*70}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
