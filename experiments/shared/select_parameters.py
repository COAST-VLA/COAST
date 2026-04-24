"""Narrow a (layer, alpha, beta) steering grid via diagnostic metrics.

Reads a pre-computed conceptor NPZ (the output of any ``compute_conceptors.py``)
and uses two diagnostics to pick a promising slice of the hyperparameter grid:

  1. **Layer** — pick the single layer with the highest mean *quota*
     ``q(C) = tr(C)/d`` at ``α = quota_alpha`` (the largest α, where conceptors
     are sharpest). Higher quota ⇒ the conceptor covers more of the state space.
  2. **Alphas** — at the chosen layer, keep α values whose mean success/failure
     *overlap* ``sim(C_s, C_f) = tr(C_s C_f) / sqrt(tr(C_s²) tr(C_f²))`` falls
     in a sweet-spot band (default ``[0.85, 0.95]``). Too-low overlap means the
     two conceptors cover disjoint regions (no useful contrast); too-high means
     the success/failure subspaces collapse (no information).
  3. **Betas** — fixed shortlist (default ``[0.1, 0.3]``); ``β = 0.5`` was
     empirically harmful in the LIBERO diagnostic runs that motivated this tool.

This is the tuning entrypoint for environments where an automated sweep driver
is impractical (real-robot eval, e.g. DROID). Sim envs should use the
``find_best_configs.py`` sweep driver instead — it produces ``best_configs.json``
with per-task winners, which is strictly more informative than a shortlist.

Usage (from repo root)::

    uv run python experiments/shared/select_parameters.py \\
        --conceptor_npz conceptors/droid_conceptors.npz \\
        --output_json experiments/droid/selected_params.json

Callable from other scripts via ``run_selection(...)``.
"""

# ruff: noqa: N803, N806, RUF001, RUF002, RUF003
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import re
from typing import Any

import numpy as np
import tyro

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^(.+?)__L(\d+)__(.+?)__(C_.+)$")


def _quota(C: np.ndarray) -> float:
    """q(C) = tr(C) / d."""
    return float(np.trace(C)) / C.shape[0]


def _overlap(Cs: np.ndarray, Cf: np.ndarray) -> float:
    """sim(C_s, C_f) = tr(C_s C_f) / sqrt(tr(C_s²) tr(C_f²))."""
    num = float(np.einsum("ij,ji->", Cs, Cf))
    ns = float(np.einsum("ij,ji->", Cs, Cs))
    nf = float(np.einsum("ij,ji->", Cf, Cf))
    if ns * nf == 0:
        return 0.0
    return num / float(np.sqrt(ns * nf))


def _parse_npz_structure(npz: np.lib.npyio.NpzFile) -> tuple[list[str], list[int], list[float]]:
    """Discover ``(tasks, layers, numeric_alphas)`` from the NPZ key format."""
    tasks: set[str] = set()
    layers: set[int] = set()
    alphas: set[float] = set()
    for key in npz.files:
        m = _KEY_RE.match(key)
        if not m:
            continue
        tasks.add(m.group(1))
        layers.add(int(m.group(2)))
        alpha_str = m.group(3)
        try:
            alphas.add(float(alpha_str))
        except ValueError:
            # Non-numeric α slot (e.g. "per_step_0"); skip.
            continue
    return sorted(tasks), sorted(layers), sorted(alphas)


def run_selection(
    conceptor_npz: pathlib.Path,
    output_json: pathlib.Path,
    *,
    overlap_low: float = 0.85,
    overlap_high: float = 0.95,
    candidate_betas: tuple[float, ...] = (0.1, 0.3),
    quota_alpha: float = 10.0,
    conceptor_type_for_quota: str = "contrastive",
) -> dict[str, Any]:
    """Run the three-step selection rule. Writes JSON, returns the result dict."""
    npz = np.load(conceptor_npz, allow_pickle=True)
    tasks, layers, alphas = _parse_npz_structure(npz)
    logger.info("Found %d tasks, layers=%s, alphas=%s", len(tasks), layers, alphas)

    layer_quotas: dict[int, float] = {}
    for L in layers:
        quotas = [
            _quota(npz[key])
            for t in tasks
            if (key := f"{t}__L{L}__{quota_alpha}__C_{conceptor_type_for_quota}") in npz.files
        ]
        if quotas:
            layer_quotas[L] = float(np.mean(quotas))

    if not layer_quotas:
        raise ValueError(
            f"No quota data found in NPZ — expected keys matching "
            f"'<task>__L<layer>__{quota_alpha}__C_{conceptor_type_for_quota}'."
        )

    best_layer = max(layer_quotas, key=layer_quotas.__getitem__)
    logger.info("Step 1 — best layer by mean quota: L=%d (quota=%.4f)", best_layer, layer_quotas[best_layer])

    alpha_overlaps: dict[float, float] = {}
    for a in alphas:
        per_task = []
        for t in tasks:
            cs_key = f"{t}__L{best_layer}__{a}__C_success"
            cf_key = f"{t}__L{best_layer}__{a}__C_failure"
            if cs_key in npz.files and cf_key in npz.files:
                per_task.append(_overlap(npz[cs_key], npz[cf_key]))
        if per_task:
            alpha_overlaps[a] = float(np.mean(per_task))

    selected_alphas = [a for a, ov in alpha_overlaps.items() if overlap_low <= ov <= overlap_high]
    if not selected_alphas and alpha_overlaps:
        band_center = 0.5 * (overlap_low + overlap_high)
        closest = min(alpha_overlaps, key=lambda a: abs(alpha_overlaps[a] - band_center))
        selected_alphas = [closest]
        logger.warning("No α in overlap band [%g, %g]; fell back to α=%g", overlap_low, overlap_high, closest)

    selected_betas = list(candidate_betas)
    logger.info(
        "Selected: layer=%d, alphas=%s, betas=%s",
        best_layer,
        selected_alphas,
        selected_betas,
    )

    result: dict[str, Any] = {
        "best_layer": best_layer,
        "selected_alphas": selected_alphas,
        "selected_betas": selected_betas,
        "overlap_band": [overlap_low, overlap_high],
        "diagnostics": {
            "layer_quotas": {str(L): v for L, v in layer_quotas.items()},
            "alpha_overlaps_at_best_layer": {str(a): v for a, v in alpha_overlaps.items()},
            "tasks": tasks,
            "all_layers": layers,
            "all_alphas": alphas,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", output_json)
    return result


@dataclasses.dataclass
class Args:
    # Pre-computed conceptors NPZ (from any compute_conceptors.py).
    conceptor_npz: pathlib.Path
    # Where to write the selected-parameters JSON.
    output_json: pathlib.Path
    # Overlap sweet-spot band.
    overlap_low: float = 0.85
    overlap_high: float = 0.95
    # Candidate betas (kept as a shortlist).
    betas: tuple[float, ...] = (0.1, 0.3)
    # Alpha used for the quota computation — largest α = sharpest conceptor.
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
