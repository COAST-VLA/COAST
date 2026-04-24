"""Compute conceptor NPZ from a RoboCasa activation tree.

Parallel to experiments/libero/compute_conceptors.py. See that script's docstring
for the end-to-end pipeline. Canonical NPZs live at
``brandonyang/robocasa-conceptors`` on HuggingFace; this script is for
reproducing or rebuilding them.

Usage (from repo root)::

    uv run python experiments/robocasa/compute_conceptors.py \\
        --activation_root activations/pi05_pretrain_human300_75000 \\
        --output_path conceptors/robocasa_conceptors_fresh.npz
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import tyro

from openpi.serving.conceptors import DEFAULT_ALPHAS
from openpi.serving.conceptors import DEFAULT_COLLECT_LAYERS
from openpi.serving.conceptors import DEFAULT_PER_STEP_INDICES
from openpi.serving.conceptors import compute_all_conceptors

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    activation_root: pathlib.Path = pathlib.Path("activations")
    output_path: pathlib.Path = pathlib.Path("conceptors/robocasa_conceptors_fresh.npz")

    layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    per_step_indices: tuple[int, ...] = DEFAULT_PER_STEP_INDICES
    collect_layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS
    min_episodes_per_class: int = 2

    task_filter: tuple[str, ...] = ()


def main(args: Args) -> None:
    summary = compute_all_conceptors(
        activation_root=args.activation_root,
        output_path=args.output_path,
        layers=args.layers,
        alphas=args.alphas,
        per_step_indices=args.per_step_indices,
        collect_layers=args.collect_layers,
        min_episodes_per_class=args.min_episodes_per_class,
        task_filter=tuple(args.task_filter) if args.task_filter else None,
    )
    logger.info("=" * 60)
    logger.info("Included tasks: %d", summary["num_tasks"])
    logger.info("Total keys written: %d", summary["num_keys"])
    if summary["skipped_tasks"]:
        logger.warning("Skipped tasks (insufficient episodes): %s", summary["skipped_tasks"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
