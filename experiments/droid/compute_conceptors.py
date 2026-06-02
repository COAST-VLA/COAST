"""Compute conceptor NPZ from a DROID activation tree.

This script PRODUCES ``droid_conceptors.npz`` from activations collected
during real-robot rollouts. DROID evaluation is real-robot only, so the
collection step requires an operator running `examples/droid/main.py --collect`
against a `--collect_activations --pytorch` server. Once activations are on
disk this script is identical in shape to the LIBERO / MetaWorld equivalents.

Note: DROID has ``seq_len=15`` action tokens per inference (vs 10 elsewhere).
``compute_all_conceptors`` flattens over all tokens, so no parameterization
is needed — the pipeline is transparent to the difference.

Pipeline:
  1. Collect activations: server + `examples/droid/main.py --collect` over
     many rollouts per instruction. Task names are slugs of the instruction.
  2. Run this script pointing at the resulting activation root.
  3. The output NPZ is drop-in compatible with ``src/openpi/serving/steering.py``.

Usage (from repo root)::

    uv run python experiments/droid/compute_conceptors.py \\
        --activation_root activations/pi05_droid \\
        --output_path conceptors/droid_conceptors_fresh.npz

All math lives in ``src/openpi/serving/conceptors.py``.
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
    # Root directory written by the collection server during real-robot rollouts.
    # Layout: <activation_root>/<checkpoint_step>/<slug>/episode_NNN_env_NNN/...
    activation_root: pathlib.Path = pathlib.Path("activations/pi05_droid")

    # Output NPZ (parent dir will be created).
    output_path: pathlib.Path = pathlib.Path("conceptors/droid_conceptors_fresh.npz")

    layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    per_step_indices: tuple[int, ...] = DEFAULT_PER_STEP_INDICES

    collect_layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS

    # DROID rollouts are expensive. Loosen the per-class floor to 2 so a
    # small collection still produces conceptors. Raise if you have data.
    min_episodes_per_class: int = 2

    # Optional subset of task slugs (default: all found).
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
