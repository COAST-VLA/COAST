"""Client-side steering protocol and schema validators.

This module is the single source of truth for the on-wire steering protocol:
the set of allowed strategy names, the magic-key constant, and the
``best_configs.json`` schema. It has zero openpi or torch dependencies —
pure stdlib — so it imports cleanly from:

- The main ``openpi`` package (root venv, Python 3.11) where the server lives.
- ``examples/libero_env`` (Python 3.8 sub-venv).
- ``examples/robocasa_env`` (Python 3.11 sub-venv, isolated deps).
- Any future env client.

The server-side wrapper (``openpi.serving.steering.SteeredPolicyWrapper``)
imports ``ALLOWED_STRATEGIES`` and ``STEERING_KEY`` from here so the two
sides of the wire can't drift out of sync.

Example client use:

    from openpi_client.steering import STEERING_KEY, build_steering_payload

    element = {"observation/state": ..., "prompt": ...}
    if args.steer:
        element[STEERING_KEY] = build_steering_payload(
            task=task_name,
            layer=args.steering_layer,
            alpha=args.steering_alpha,
            beta=args.steering_beta,
            strategy=args.steering_strategy,
        )
    result = policy.infer(element)
"""

# ruff: noqa: RUF001, RUF002, RUF003
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Protocol constants
# ──────────────────────────────────────────────────────────────────────────────

# On-wire magic key — matches the ``__collect__`` / ``__finalize_episode__``
# pattern in openpi.serving.activation_collector. The server's
# SteeredPolicyWrapper pops this from obs before dispatching.
STEERING_KEY = "__steering__"

# Strategies (see openpi.serving.steering for the math):
#   global          — h' = (1-β)h + β(h @ C_contrastive.T), α selects aperture
#   per_step_{0,9}  — same but C is per-denoise-step (α baked into NPZ)
#   positive_only   — h' = (1-β)h + β(h @ C_success.T), α selects aperture
#   random_matched  — h' = (1-β)h + β(h @ C_rand.T), C_rand has same spectrum
#                     as C_contrastive at α but random eigenvectors
#   linear          — h' = h + α · v, v = unit(mean_success - mean_failure). β ignored.
ALLOWED_STRATEGIES = (
    "global",
    "per_step_0",
    "per_step_9",
    "positive_only",
    "random_matched",
    "linear",
)


def build_steering_payload(
    task: str,
    layer: int,
    alpha: float,
    beta: float,
    strategy: str,
) -> Dict[str, Any]:
    """Construct the obs[STEERING_KEY] dict with the correct schema.

    Small helper so callers don't have to remember the field names.
    """
    return {
        "task": task,
        "layer": int(layer),
        "alpha": float(alpha),
        "beta": float(beta),
        "strategy": strategy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# best_configs.json — load + schema-check
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_CONFIG_FIELDS = {
    "layer": int,
    "alpha": (int, float),
    "beta": (int, float),
    "strategy": str,
}


def load_and_validate_steering_config(path: str) -> Dict[str, Any]:
    """Parse and schema-check a ``best_configs.json`` file.

    Fail-fast: callers (e.g. eval_all.py) should invoke this before spawning
    any subprocesses so a malformed config surfaces immediately.

    Expected schema::

        {
          "task_suite": "libero_10",                           # informational
          "defaults": {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
          "tasks": {
            "<task_name>": {"layer": ..., "alpha": ..., "beta": ..., "strategy": ...},
            ...
          }
        }

    Returns the parsed dict. Raises ``FileNotFoundError`` or ``ValueError``
    with a specific message on any schema violation.
    """
    cfg_path = pathlib.Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError("steering_config not found: {}".format(cfg_path))
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("tasks"), dict):
        raise ValueError("{}: root must be a dict with a 'tasks' dict".format(cfg_path))
    for name, entry in cfg["tasks"].items():
        if not isinstance(entry, dict):
            raise ValueError("{}: tasks[{!r}] must be a dict".format(cfg_path, name))
        for key, expected_type in _REQUIRED_CONFIG_FIELDS.items():
            if key not in entry:
                raise ValueError("{}: tasks[{!r}] missing {!r}".format(cfg_path, name, key))
            if not isinstance(entry[key], expected_type):
                raise ValueError("{}: tasks[{!r}].{} wrong type".format(cfg_path, name, key))
        if entry["strategy"] not in ALLOWED_STRATEGIES:
            raise ValueError("{}: tasks[{!r}].strategy not in {}".format(cfg_path, name, ALLOWED_STRATEGIES))
    if "defaults" in cfg:
        for key, expected_type in _REQUIRED_CONFIG_FIELDS.items():
            if key not in cfg["defaults"] or not isinstance(cfg["defaults"][key], expected_type):
                raise ValueError("{}: defaults.{} missing or wrong type".format(cfg_path, key))
        if cfg["defaults"]["strategy"] not in ALLOWED_STRATEGIES:
            raise ValueError("{}: defaults.strategy not in {}".format(cfg_path, ALLOWED_STRATEGIES))
    return cfg


def resolve_steering_for_task(
    fallback: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    task_name: str,
) -> Dict[str, Any]:
    """Pick the steering params for a given task from a (possibly None) config.

    Resolution order:
      1. ``config["tasks"][task_name]`` if present
      2. ``config["defaults"]`` if present
      3. ``fallback`` (e.g. the CLI scalar flags)

    Args:
        fallback: dict with keys ``layer``, ``alpha``, ``beta``, ``strategy``.
                  Typically built from the caller's CLI args.
        config: parsed ``best_configs.json`` (from ``load_and_validate_steering_config``)
                or ``None`` if no config file was provided.
        task_name: the task to look up.
    Returns:
        A dict with ``layer``, ``alpha``, ``beta``, ``strategy`` — ready to
        pass to ``build_steering_payload``.
    """
    if config is None:
        return fallback
    if task_name in config["tasks"]:
        return config["tasks"][task_name]
    if "defaults" in config:
        return config["defaults"]
    return fallback
