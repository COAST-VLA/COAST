"""Tests for packages/openpi-client/src/openpi_client/steering.py.

Pure-stdlib tests — must pass in both the root (Python 3.11) venv and the
libero (Python 3.8) sub-venv where openpi_client lives. No torch, no JAX.
"""

# ruff: noqa: N802, N806, PT018, RUF001, RUF002, RUF003
from __future__ import annotations

import json
import pathlib

from openpi_client import steering as client_steering
from openpi_client.steering import ALLOWED_STRATEGIES
from openpi_client.steering import DEFAULT_STEERING_ALPHA
from openpi_client.steering import DEFAULT_STEERING_BETA
from openpi_client.steering import DEFAULT_STEERING_LAYER
from openpi_client.steering import DEFAULT_STEERING_STRATEGY
from openpi_client.steering import STEERING_KEY
from openpi_client.steering import build_steering_payload
from openpi_client.steering import load_and_validate_steering_config
from openpi_client.steering import resolve_steering_for_task
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol constants
# ═══════════════════════════════════════════════════════════════════════════════


def test_steering_key_matches_wire_string():
    """Regression: the on-wire key is an implementation detail shared with the server.
    If anyone changes it on one side, this test fires."""
    assert STEERING_KEY == "__steering__"


def test_allowed_strategies_expected_set():
    assert set(ALLOWED_STRATEGIES) == {
        "global",
        "per_step_0",
        "per_step_9",
        "positive_only",
        "random_matched",
        "linear",
    }


def test_defaults_are_plain_literals():
    """Sub-venv main.py scripts use these as dataclass field defaults, so they
    must stay simple JSON-serializable literals."""
    assert isinstance(DEFAULT_STEERING_LAYER, int)
    assert isinstance(DEFAULT_STEERING_ALPHA, float)
    assert isinstance(DEFAULT_STEERING_BETA, float)
    assert isinstance(DEFAULT_STEERING_STRATEGY, str)
    assert DEFAULT_STEERING_STRATEGY in ALLOWED_STRATEGIES


def test_server_and_client_strategies_agree():
    """The server-side ALLOWED_STRATEGIES must be the exact same tuple as the
    client's. This is the "drift guard" that the decl-with-docs pattern
    was aspirational before the refactor."""
    from openpi.serving.steering import ALLOWED_STRATEGIES as SERVER_ALLOWED

    assert SERVER_ALLOWED == ALLOWED_STRATEGIES


# ═══════════════════════════════════════════════════════════════════════════════
# build_steering_payload
# ═══════════════════════════════════════════════════════════════════════════════


def test_build_steering_payload_shape():
    p = build_steering_payload(task="t", layer=11, alpha=0.1, beta=0.3, strategy="global")
    assert set(p.keys()) == {"task", "layer", "alpha", "beta", "strategy"}


def test_build_steering_payload_coerces_types():
    """ints/floats that come in as other numeric types should be normalized."""
    p = build_steering_payload(task="t", layer=11.0, alpha=1, beta=1, strategy="linear")
    assert isinstance(p["layer"], int) and p["layer"] == 11
    assert isinstance(p["alpha"], float) and p["alpha"] == 1.0
    assert isinstance(p["beta"], float) and p["beta"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# load_and_validate_steering_config
# ═══════════════════════════════════════════════════════════════════════════════


def _valid_config_dict():
    return {
        "task_suite": "libero_10",
        "defaults": {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
        "tasks": {
            "taskA": {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
            "taskB": {"layer": 17, "alpha": 0.5, "beta": 0.1, "strategy": "per_step_0"},
        },
    }


def test_valid_config_parses(tmp_path: pathlib.Path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps(_valid_config_dict()))
    cfg = load_and_validate_steering_config(str(path))
    assert "taskA" in cfg["tasks"]


def test_missing_file(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        load_and_validate_steering_config(str(tmp_path / "nope.json"))


def test_missing_tasks_field(tmp_path: pathlib.Path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"defaults": {}}))
    with pytest.raises(ValueError, match="tasks"):
        load_and_validate_steering_config(str(path))


def test_task_missing_field(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    del cfg["tasks"]["taskA"]["layer"]
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="layer"):
        load_and_validate_steering_config(str(path))


def test_task_bad_strategy(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    cfg["tasks"]["taskA"]["strategy"] = "nope"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="strategy"):
        load_and_validate_steering_config(str(path))


def test_task_wrong_type(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    cfg["tasks"]["taskA"]["layer"] = "eleven"  # str instead of int
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="layer"):
        load_and_validate_steering_config(str(path))


def test_defaults_wrong_type(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    cfg["defaults"]["layer"] = "not_an_int"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="defaults"):
        load_and_validate_steering_config(str(path))


def test_config_without_defaults_is_valid(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    del cfg["defaults"]
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    result = load_and_validate_steering_config(str(path))
    assert "defaults" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# resolve_steering_for_task
# ═══════════════════════════════════════════════════════════════════════════════


_FALLBACK = {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"}
_CUSTOM = {"layer": 17, "alpha": 0.5, "beta": 0.1, "strategy": "per_step_0"}


def test_resolve_no_config_returns_fallback():
    assert resolve_steering_for_task(_FALLBACK, None, "anytask") == _FALLBACK


def test_resolve_task_present_returns_task_entry():
    cfg = {"tasks": {"taskA": _CUSTOM}, "defaults": _FALLBACK}
    assert resolve_steering_for_task(_FALLBACK, cfg, "taskA") == _CUSTOM


def test_resolve_task_missing_falls_back_to_defaults():
    cfg = {"tasks": {"taskA": _CUSTOM}, "defaults": _FALLBACK}
    assert resolve_steering_for_task({"different": "fallback"}, cfg, "taskB") == _FALLBACK


def test_resolve_task_missing_no_defaults_returns_fallback():
    cfg = {"tasks": {"taskA": _CUSTOM}}
    assert resolve_steering_for_task(_FALLBACK, cfg, "taskB") == _FALLBACK


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level coverage sanity
# ═══════════════════════════════════════════════════════════════════════════════


def test_module_exports():
    """Guardrail: the documented public API must be importable."""
    expected_attrs = [
        "STEERING_KEY",
        "ALLOWED_STRATEGIES",
        "DEFAULT_STEERING_LAYER",
        "DEFAULT_STEERING_ALPHA",
        "DEFAULT_STEERING_BETA",
        "DEFAULT_STEERING_STRATEGY",
        "build_steering_payload",
        "load_and_validate_steering_config",
        "resolve_steering_for_task",
    ]
    for attr in expected_attrs:
        assert hasattr(client_steering, attr), f"missing {attr}"
