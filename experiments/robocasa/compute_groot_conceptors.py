"""Compute a GR00T N1.5 conceptor NPZ from a RoboCasa ``groot_v1`` activation tree.

Usage (from repo root)::

    uv run python experiments/robocasa/compute_groot_conceptors.py \\
        --activation_root activations/groot_n15-robocasa-activations-v1-15env \\
        --output_path conceptors/groot_robocasa_conceptors_fresh.npz
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import tyro

from openpi.serving.conceptors import DEFAULT_ALPHAS
from openpi.serving.conceptors import GROOT_COLLECT_LAYERS
from openpi.serving.conceptors import GROOT_PER_STEP_INDICES
from openpi.serving.conceptors import compute_all_conceptors

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    activation_root: pathlib.Path = pathlib.Path("activations")
    output_path: pathlib.Path = pathlib.Path("conceptors/groot_robocasa_conceptors_fresh.npz")

    # Layer 11 matches the RoboCasa client/server default steering layer while
    # keeping the output NPZ compact. Pass more layers for sweeps.
    layers: tuple[int, ...] = (11,)
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    per_step_indices: tuple[int, ...] = GROOT_PER_STEP_INDICES
    collect_layers: tuple[int, ...] = GROOT_COLLECT_LAYERS
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
