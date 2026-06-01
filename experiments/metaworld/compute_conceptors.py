"""Compute conceptor NPZ from a MetaWorld activation tree.

This is the script that PRODUCES ``metaworld_conceptors.npz``. The canonical
NPZ lives at ``brandonyang/metaworld-conceptors`` on HuggingFace (download via
``hf download brandonyang/metaworld-conceptors ...``). Run this script only
when rebuilding from fresh activations (e.g., a new checkpoint).

Pipeline:
  1. Collect activations *in-process* with:
     ``CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
         --collect --split train --num_envs 16 \
         --policy.config=pi05_metaworld \
         --policy.dir=checkpoints/openpi-metaworld-5000``
     (MetaWorld collection is in-process — no separate server — but the on-disk
     tree layout matches LIBERO/RoboCasa's server-side collection.)
  2. Run this script pointing at the resulting activation root.
  3. The output NPZ is drop-in compatible with ``src/openpi/serving/steering.py``.

Usage (from repo root)::

    uv run python experiments/metaworld/compute_conceptors.py \\
        --activation_root activations/openpi-metaworld-5000 \\
        --output_path conceptors/metaworld_conceptors_fresh.npz

All math lives in ``src/openpi/serving/conceptors.py``; this is just a CLI
wrapper with MetaWorld-appropriate defaults.
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
    # Root directory written by `examples/metaworld/eval_all.py --collect`.
    # Layout: <activation_root>/<checkpoint_step>/<env_name>/episode_NNN_env_NNN/...
    activation_root: pathlib.Path = pathlib.Path("activations/openpi-metaworld-5000")

    # Output NPZ (parent dir will be created).
    output_path: pathlib.Path = pathlib.Path("conceptors/metaworld_conceptors_fresh.npz")

    # Real transformer layer indices to compute conceptors for. Each must be
    # in collect_layers below (activations only exist for those).
    layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    per_step_indices: tuple[int, ...] = DEFAULT_PER_STEP_INDICES

    # Axis-1 → real-layer mapping baked into the activation tensor.
    collect_layers: tuple[int, ...] = DEFAULT_COLLECT_LAYERS

    # Tasks with fewer than this many successes OR failures are skipped.
    min_episodes_per_class: int = 2

    # Optional subset of task (env_name) values to include (default: all found).
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
