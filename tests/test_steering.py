"""Unit tests for src/openpi/serving/steering.py.

Pure-Python / torch-on-CPU — no GPU or checkpoint required. Covers:

- ConceptorSteeringHook math (identity, β=0, β=1, tuple outputs, log reset)
- Payload / config schema validation
- SteeredPolicyWrapper:
    - passthrough when no __steering__ key is present
    - routes through infer_with_steering when __steering__ is present
    - strips __steering__ before the underlying policy sees it
    - caches hooks on (task, layer, alpha, beta, strategy)
    - validates payloads and rejects malformed ones
"""

# ruff: noqa: N802, N806, PT018, RUF001, RUF002, RUF003
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from openpi.serving import steering
from openpi.serving.steering import ALLOWED_STRATEGIES
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.serving.steering import available_tasks
from openpi.serving.steering import compute_random_conceptor
from openpi.serving.steering import get_conceptor_matrix
from openpi.serving.steering import validate_best_configs_json
from openpi.serving.steering import validate_steering_payload

# ═══════════════════════════════════════════════════════════════════════════════
# ConceptorSteeringHook
# ═══════════════════════════════════════════════════════════════════════════════


def test_identity_conceptor_is_no_op():
    """C = I → h' = (1-β)h + β·h = h for any β."""
    d = 16
    hook = ConceptorSteeringHook(np.eye(d, dtype=np.float32), beta=0.5, device="cpu")
    h = torch.randn(2, 4, d)
    out = hook(None, None, h)
    torch.testing.assert_close(out, h, rtol=1e-5, atol=1e-6)


def test_zero_beta_is_no_op_even_with_random_C():
    d = 16
    C = compute_random_conceptor(d=d, alpha=1.0, seed=1)
    hook = ConceptorSteeringHook(C, beta=0.0, device="cpu")
    h = torch.randn(2, 4, d)
    torch.testing.assert_close(hook(None, None, h), h, rtol=1e-5, atol=1e-6)


def test_full_beta_projects_through_C():
    """β=1 → h' = h @ C^T (C is symmetric so equivalently h @ C)."""
    d = 16
    C_np = compute_random_conceptor(d=d, alpha=1.0, seed=2)
    hook = ConceptorSteeringHook(C_np, beta=1.0, device="cpu")
    h = torch.randn(2, 4, d)
    out = hook(None, None, h)
    expected = h @ torch.from_numpy(C_np).to(h.dtype).T
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_records_intervention_norms():
    d = 16
    C = compute_random_conceptor(d=d, alpha=1.0, seed=3)
    hook = ConceptorSteeringHook(C, beta=0.5, device="cpu")
    h = torch.randn(2, 4, d)
    hook(None, None, h)
    hook(None, None, h)
    assert len(hook.intervention_norms) == 2
    assert all(n > 0 for n in hook.intervention_norms)


def test_reset_logs_clears():
    d = 16
    hook = ConceptorSteeringHook(np.eye(d, dtype=np.float32), beta=0.5, device="cpu")
    hook(None, None, torch.randn(2, 4, d))
    assert len(hook.intervention_norms) == 1
    hook.reset_logs()
    assert hook.intervention_norms == []


def test_preserves_tuple_output_extras():
    """When layer output is (hidden, cache, ...), only hidden is transformed."""
    d = 16
    hook = ConceptorSteeringHook(np.eye(d, dtype=np.float32), beta=0.0, device="cpu")
    h = torch.randn(2, 4, d)
    extras = ("cache", 42)
    out = hook(None, None, (h, *extras))
    assert isinstance(out, tuple)
    assert out[1:] == extras
    torch.testing.assert_close(out[0], h)


def test_set_denoise_step():
    hook = ConceptorSteeringHook(np.eye(8, dtype=np.float32), beta=0.3, device="cpu")
    hook.set_denoise_step(5)
    assert hook.current_denoise_step == 5


def test_repr_mentions_beta_and_dim():
    hook = ConceptorSteeringHook(np.eye(32, dtype=np.float32), beta=0.42, device="cpu")
    s = repr(hook)
    assert "0.42" in s
    assert "32" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Steering payload validation
# ═══════════════════════════════════════════════════════════════════════════════


def _valid_payload():
    return {"task": "taskA", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"}


def test_valid_payload_passes():
    validate_steering_payload(_valid_payload(), {"taskA", "taskB"})


def test_missing_field_raises():
    p = _valid_payload()
    del p["layer"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_steering_payload(p, {"taskA"})


def test_wrong_type_raises():
    p = _valid_payload()
    p["layer"] = "eleven"  # str instead of int
    with pytest.raises(ValueError, match="layer"):
        validate_steering_payload(p, {"taskA"})


def test_unknown_strategy_raises():
    p = _valid_payload()
    p["strategy"] = "not_a_strategy"
    with pytest.raises(ValueError, match="strategy"):
        validate_steering_payload(p, {"taskA"})


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="not found in conceptor"):
        validate_steering_payload(_valid_payload(), {"someOtherTask"})


def test_non_dict_payload_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_steering_payload(["not", "a", "dict"], {"taskA"})


# ═══════════════════════════════════════════════════════════════════════════════
# best_configs.json validation
# ═══════════════════════════════════════════════════════════════════════════════


def _valid_config_dict():
    return {
        "task_suite": "libero_10",
        "defaults": {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
        "tasks": {
            "taskA": {
                "layer": 11,
                "alpha": 0.1,
                "beta": 0.3,
                "strategy": "global",
                "baseline_sr": 0.5,
                "steered_sr": 0.9,
            },
            "taskB": {"layer": 17, "alpha": 0.5, "beta": 0.1, "strategy": "per_step_0"},
        },
    }


def test_valid_config_parses(tmp_path: pathlib.Path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps(_valid_config_dict()))
    cfg = validate_best_configs_json(path)
    assert "taskA" in cfg["tasks"]
    assert cfg["defaults"]["layer"] == 11


def test_missing_tasks_field_raises(tmp_path: pathlib.Path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"defaults": {}}))
    with pytest.raises(ValueError, match="tasks"):
        validate_best_configs_json(path)


def test_task_missing_field_raises(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    del cfg["tasks"]["taskA"]["layer"]
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="layer"):
        validate_best_configs_json(path)


def test_task_bad_strategy_raises(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    cfg["tasks"]["taskA"]["strategy"] = "nope"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="strategy"):
        validate_best_configs_json(path)


def test_defaults_bad_field_raises(tmp_path: pathlib.Path):
    cfg = _valid_config_dict()
    cfg["defaults"]["layer"] = "not_an_int"
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="defaults.layer"):
        validate_best_configs_json(path)


def test_missing_file_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        validate_best_configs_json(tmp_path / "nope.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Conceptor NPZ helpers (synthesized mini-NPZ, no download required)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mini_npz(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a tiny conceptor NPZ with two tasks, one layer, matching miranda-v2 key format."""
    d = 8
    arrays = {}
    for task in ("taskA", "taskB"):
        for alpha in ("0.1", "0.5", "1.0"):
            arrays[f"{task}__L11__{alpha}__C_contrastive"] = np.eye(d, dtype=np.float32) * 0.5
        for step in (0, 9):
            arrays[f"{task}__L11__per_step_{step}__C_contrastive"] = np.eye(d, dtype=np.float32) * 0.3
    path = tmp_path / "mini.npz"
    np.savez(path, **arrays)
    return path


def test_load_conceptor_npz_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        steering.load_conceptor_npz(tmp_path / "nope.npz")


def test_available_tasks_extracts_keys(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    assert available_tasks(npz) == {"taskA", "taskB"}


def test_get_conceptor_matrix_global(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    C = get_conceptor_matrix(npz, "taskA", 11, 0.1, "global")
    assert C.shape == (8, 8)
    np.testing.assert_allclose(C, np.eye(8) * 0.5)


def test_get_conceptor_matrix_per_step(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    C = get_conceptor_matrix(npz, "taskB", 11, 0.1, "per_step_9")  # alpha ignored for per_step
    np.testing.assert_allclose(C, np.eye(8) * 0.3)


def test_get_conceptor_matrix_unknown_strategy_raises(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(ValueError, match="Unknown steering strategy"):
        get_conceptor_matrix(npz, "taskA", 11, 0.1, "bogus")


def test_get_conceptor_matrix_missing_key_raises(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(KeyError, match="not in NPZ"):
        get_conceptor_matrix(npz, "taskA", 99, 0.1, "global")


# ═══════════════════════════════════════════════════════════════════════════════
# SteeredPolicyWrapper
# ═══════════════════════════════════════════════════════════════════════════════


class _StubPolicy:
    def __init__(self):
        self.infer_calls = 0
        self.steering_calls = 0
        self.last_steering_hooks = None
        self.last_infer_obs = None
        self.last_steering_obs = None
        self._metadata = {"stub": True}

    def infer(self, obs):
        self.infer_calls += 1
        self.last_infer_obs = dict(obs)
        return {"actions": np.zeros((1, 4))}

    def infer_with_steering(self, obs, *, steering_hooks):
        self.steering_calls += 1
        self.last_steering_hooks = steering_hooks
        self.last_steering_obs = dict(obs)
        return {"actions": np.ones((1, 4))}, {}

    @property
    def metadata(self):
        return self._metadata


def test_wrapper_passthrough_when_no_steering_key(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    result = w.infer({"obs": "data"})
    assert p.infer_calls == 1 and p.steering_calls == 0
    assert np.all(result["actions"] == 0)


def test_wrapper_routes_through_steering(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    obs = {
        "obs": "data",
        "__steering__": {"task": "taskA", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
    }
    result = w.infer(obs)
    assert p.infer_calls == 0 and p.steering_calls == 1
    assert p.last_steering_hooks is not None
    layer_idx, hook = p.last_steering_hooks[0]
    assert layer_idx == 11
    assert isinstance(hook, ConceptorSteeringHook)
    assert np.all(result["actions"] == 1)


def test_wrapper_strips_magic_key_before_underlying_policy(mini_npz: pathlib.Path):
    """The underlying policy must never see __steering__ — transforms would choke on it."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    obs = {
        "obs": "data",
        "__steering__": {"task": "taskA", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
    }
    w.infer(obs)
    assert "__steering__" not in p.last_steering_obs


def test_wrapper_caches_hooks(mini_npz: pathlib.Path):
    """Repeat calls with the same (task, layer, alpha, beta, strategy) reuse one hook."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    payload = {"task": "taskA", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"}

    w.infer({"__steering__": dict(payload)})
    first_hook = p.last_steering_hooks[0][1]
    w.infer({"__steering__": dict(payload)})
    second_hook = p.last_steering_hooks[0][1]
    assert first_hook is second_hook
    assert len(w._hook_cache) == 1  # noqa: SLF001

    # Different payload → new cache entry
    payload2 = {**payload, "beta": 0.1}
    w.infer({"__steering__": payload2})
    assert len(w._hook_cache) == 2  # noqa: SLF001


def test_wrapper_validates_payload(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    with pytest.raises(ValueError, match="strategy"):
        w.infer({"__steering__": {"task": "taskA", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "bad"}})
    with pytest.raises(ValueError, match="not found in conceptor"):
        w.infer({"__steering__": {"task": "unknown", "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"}})


def test_wrapper_metadata_extends_underlying(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    meta = w.metadata
    assert meta["stub"] is True
    assert meta["steering_enabled"] is True
    assert meta["num_conceptor_tasks"] == 2


def test_defaults_are_module_constants():
    """Regression: sub-venv scripts hard-code these defaults, so they must stay literal ints/floats."""
    assert isinstance(steering.DEFAULT_STEERING_LAYER, int)
    assert isinstance(steering.DEFAULT_STEERING_ALPHA, float)
    assert isinstance(steering.DEFAULT_STEERING_BETA, float)
    assert steering.DEFAULT_STEERING_STRATEGY in ALLOWED_STRATEGIES
