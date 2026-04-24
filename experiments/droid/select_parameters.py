# ruff: noqa: RUF002
"""Narrow DROID steering hyperparameters via diagnostic metrics.

Thin entrypoint over ``experiments/shared/select_parameters.py`` with
DROID-appropriate defaults. The shared utility computes per-layer quota and
per-α success/failure overlap, then returns a shortlist of promising
``(layer, alphas, betas)`` values. This is the tuning tool for DROID because
real-robot evaluation is manual — there is no automated sweep driver.

Usage (from repo root)::

    uv run python experiments/droid/select_parameters.py \\
        --conceptor_npz conceptors/droid_conceptors.npz \\
        --output_json experiments/droid/selected_params.json

See ``experiments/droid/README.md`` for the full manual-eval workflow.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import tyro

# Ensure the sibling `experiments/shared/` dir is importable regardless of cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from shared.select_parameters import run_selection  # noqa: E402

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    conceptor_npz: pathlib.Path = pathlib.Path("conceptors/droid_conceptors.npz")
    output_json: pathlib.Path = pathlib.Path("experiments/droid/selected_params.json")
    overlap_low: float = 0.85
    overlap_high: float = 0.95
    betas: tuple[float, ...] = (0.1, 0.3)
    quota_alpha: float = 10.0


def main(args: Args) -> None:
    run_selection(
        conceptor_npz=args.conceptor_npz,
        output_json=args.output_json,
        overlap_low=args.overlap_low,
        overlap_high=args.overlap_high,
        candidate_betas=args.betas,
        quota_alpha=args.quota_alpha,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
