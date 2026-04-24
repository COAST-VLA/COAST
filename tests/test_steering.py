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

import pathlib

import numpy as np
import pytest
import torch

from openpi.serving import steering
from openpi.serving.steering import ConceptorSteeringHook
from openpi.serving.steering import SteeredPolicyWrapper
from openpi.serving.steering import available_tasks
from openpi.serving.steering import compute_random_conceptor
from openpi.serving.steering import get_conceptor_matrix
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


def test_hook_per_step_mode_selects_correct_matrix():
    """Build a hook with 3 distinct matrices; verify each one is applied at its step."""
    d = 4
    # Three identity-matrices scaled so their outputs are distinguishable:
    # M_t = (1-β)I + β(scale_t · I) = (1 - β + β·scale_t) · I
    # With β=1, M_t = scale_t · I → output = h * scale_t
    matrices = [np.eye(d, dtype=np.float32) * s for s in (0.1, 0.5, 0.9)]
    hook = ConceptorSteeringHook(beta=1.0, device="cpu", matrices_per_step=matrices)
    h = torch.ones(1, 1, d)
    for t, expected_scale in enumerate((0.1, 0.5, 0.9)):
        hook.set_denoise_step(t)
        out = hook(None, None, h)
        torch.testing.assert_close(out, h * expected_scale)


def test_hook_per_step_mode_out_of_bounds_raises():
    hook = ConceptorSteeringHook(
        beta=0.3,
        device="cpu",
        matrices_per_step=[np.eye(4, dtype=np.float32)] * 3,
    )
    hook.set_denoise_step(7)  # list length = 3
    with pytest.raises(IndexError):
        hook(None, None, torch.ones(1, 1, 4))


def test_hook_rejects_both_args():
    with pytest.raises(ValueError, match="exactly one"):
        ConceptorSteeringHook(
            np.eye(4, dtype=np.float32),
            beta=0.3,
            device="cpu",
            matrices_per_step=[np.eye(4, dtype=np.float32)],
        )


def test_hook_rejects_neither_arg():
    with pytest.raises(ValueError, match="exactly one"):
        ConceptorSteeringHook(beta=0.3, device="cpu")


def test_hook_repr_per_step_mentions_count():
    hook = ConceptorSteeringHook(
        beta=0.3,
        device="cpu",
        matrices_per_step=[np.eye(4, dtype=np.float32)] * 5,
    )
    s = repr(hook)
    assert "per_step=5" in s


def test_repr_mentions_beta_and_dim():
    hook = ConceptorSteeringHook(np.eye(32, dtype=np.float32), beta=0.42, device="cpu")
    s = repr(hook)
    assert "0.42" in s
    assert "32" in s


def test_hook_handles_h_on_different_device():
    """Hook built with device="cpu" must still work when h is explicitly moved
    to "cpu" after the fact (the `.to(device=h.device, ...)` call should cover
    any mismatch without raising). Multi-GPU tests require a GPU and live in
    tests/models/; this one exercises the CPU-CPU path and confirms the
    device= kwarg is threaded through."""
    hook = ConceptorSteeringHook(np.eye(4, dtype=np.float32), beta=0.5, device="cpu")
    h = torch.randn(1, 2, 4).to(device="cpu")
    out = hook(None, None, h)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, h, atol=1e-6, rtol=1e-5)  # C=I, so h' == h


def test_linear_hook_handles_h_on_different_device():
    direction = np.zeros(4, dtype=np.float32)
    direction[0] = 1.0
    from openpi.serving.steering import LinearSteeringHook

    hook = LinearSteeringHook(direction, alpha=0.0, device="cpu")  # alpha=0 → no-op
    h = torch.randn(1, 2, 4).to(device="cpu")
    out = hook(None, None, h)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, h, atol=1e-6, rtol=1e-5)


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


def test_nan_alpha_rejected():
    """NaN alpha would poison the M matrix. The wire validator must reject it."""
    p = _valid_payload()
    p["alpha"] = float("nan")
    with pytest.raises(ValueError, match="alpha.*finite"):
        validate_steering_payload(p, {"taskA"})


def test_inf_alpha_rejected():
    p = _valid_payload()
    p["alpha"] = float("inf")
    with pytest.raises(ValueError, match="alpha.*finite"):
        validate_steering_payload(p, {"taskA"})


def test_nan_beta_rejected():
    p = _valid_payload()
    p["beta"] = float("nan")
    with pytest.raises(ValueError, match="beta.*finite"):
        validate_steering_payload(p, {"taskA"})


def test_neg_inf_beta_rejected():
    p = _valid_payload()
    p["beta"] = float("-inf")
    with pytest.raises(ValueError, match="beta.*finite"):
        validate_steering_payload(p, {"taskA"})


def test_bool_alpha_rejected():
    """Python bool is a subclass of int — isinstance(True, (int, float)) is True.
    Without an explicit bool guard, `{"alpha": True}` would silently be treated
    as α=1.0 downstream. The validator must reject it explicitly."""
    p = _valid_payload()
    p["alpha"] = True
    with pytest.raises(ValueError, match="alpha"):
        validate_steering_payload(p, {"taskA"})


def test_bool_layer_rejected():
    p = _valid_payload()
    p["layer"] = True
    with pytest.raises(ValueError, match="layer"):
        validate_steering_payload(p, {"taskA"})


# ═══════════════════════════════════════════════════════════════════════════════
# Conceptor NPZ helpers (synthesized mini-NPZ, no download required)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mini_npz(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a tiny conceptor NPZ with two tasks, one layer, matching miranda-v2 key format.

    Contains all per-strategy keys: C_contrastive + C_success (+ C_failure) per
    alpha, per_step_{0,9} variants, and a linear_direction per layer.
    """
    d = 8
    arrays = {}
    for task in ("taskA", "taskB"):
        for alpha in ("0.1", "0.5", "1.0"):
            arrays[f"{task}__L11__{alpha}__C_contrastive"] = np.eye(d, dtype=np.float32) * 0.5
            arrays[f"{task}__L11__{alpha}__C_success"] = np.eye(d, dtype=np.float32) * 0.4
            arrays[f"{task}__L11__{alpha}__C_failure"] = np.eye(d, dtype=np.float32) * 0.35
        # All 10 per-step conceptors. Distinct scaling per step so
        # the per_step strategy test can confirm the correct matrix is selected.
        for step in range(10):
            arrays[f"{task}__L11__per_step_{step}__C_contrastive"] = np.eye(d, dtype=np.float32) * (0.1 + 0.05 * step)
        # Linear direction: unit vector, distinct per task
        v = np.zeros(d, dtype=np.float32)
        v[0 if task == "taskA" else 1] = 1.0
        arrays[f"{task}__L11__linear_direction"] = v
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


def test_get_conceptor_matrix_unknown_strategy_raises(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(ValueError, match="Unknown steering strategy"):
        get_conceptor_matrix(npz, "taskA", 11, 0.1, "bogus")


def test_get_conceptor_matrix_missing_key_raises(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(KeyError, match="not in NPZ"):
        get_conceptor_matrix(npz, "taskA", 99, 0.1, "global")


def test_get_conceptor_matrix_per_step_rejects_direct_lookup(mini_npz: pathlib.Path):
    """per_step returns a LIST of matrices via get_per_step_conceptor_matrices;
    routing it through get_conceptor_matrix is a caller bug."""
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(ValueError, match="per_step"):
        get_conceptor_matrix(npz, "taskA", 11, 0.1, "per_step")


def test_get_per_step_conceptor_matrices_returns_all_10(mini_npz: pathlib.Path):
    """Returns the 10 per-step conceptors in denoising order (t=0 first)."""
    npz = steering.load_conceptor_npz(mini_npz)
    mats = steering.get_per_step_conceptor_matrices(npz, "taskA", 11)
    assert len(mats) == 10
    # Fixture sets scale = 0.1 + 0.05 * step, so mats[t] = I * (0.1 + 0.05 * t).
    for t in range(10):
        np.testing.assert_allclose(mats[t], np.eye(8) * (0.1 + 0.05 * t), atol=1e-7)


def test_get_per_step_conceptor_matrices_missing_key_raises(tmp_path: pathlib.Path):
    """Hit an NPZ that's missing some of the per_step_0..per_step_9 keys."""
    d = 4
    arrays = {
        "taskA__L11__per_step_0__C_contrastive": np.eye(d, dtype=np.float32) * 0.1,
        "taskA__L11__per_step_9__C_contrastive": np.eye(d, dtype=np.float32) * 0.9,
    }
    path = tmp_path / "legacy.npz"
    np.savez(path, **arrays)
    npz = steering.load_conceptor_npz(path)
    with pytest.raises(KeyError, match="missing NPZ key"):
        steering.get_per_step_conceptor_matrices(npz, "taskA", 11)


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


# ═══════════════════════════════════════════════════════════════════════════════
# LinearSteeringHook
# ═══════════════════════════════════════════════════════════════════════════════


def test_linear_hook_alpha_zero_is_no_op():
    d = 8
    v = np.zeros(d, dtype=np.float32)
    v[0] = 1.0
    hook = steering.LinearSteeringHook(v, alpha=0.0, device="cpu")
    h = torch.randn(2, 4, d)
    out = hook(None, None, h)
    torch.testing.assert_close(out, h, rtol=1e-6, atol=1e-6)


def test_linear_hook_additive_math():
    """h' = h + alpha * v elementwise (v broadcasts across batch and sequence)."""
    d = 8
    v = np.zeros(d, dtype=np.float32)
    v[0] = 1.0
    hook = steering.LinearSteeringHook(v, alpha=2.5, device="cpu")
    h = torch.zeros(1, 1, d)
    out = hook(None, None, h)
    expected = torch.zeros(1, 1, d)
    expected[..., 0] = 2.5
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_linear_hook_preserves_tuple_output():
    d = 4
    v = np.zeros(d, dtype=np.float32)
    hook = steering.LinearSteeringHook(v, alpha=1.0, device="cpu")
    h = torch.randn(1, 2, d)
    extras = ("kv", 99)
    out = hook(None, None, (h, *extras))
    assert isinstance(out, tuple)
    assert out[1:] == extras
    torch.testing.assert_close(out[0], h, rtol=1e-6, atol=1e-6)


def test_linear_hook_records_intervention_norms():
    d = 8
    v = np.zeros(d, dtype=np.float32)
    v[0] = 1.0
    hook = steering.LinearSteeringHook(v, alpha=1.0, device="cpu")
    h = torch.randn(2, 4, d)
    hook(None, None, h)
    hook(None, None, h)
    assert len(hook.intervention_norms) == 2
    assert all(n > 0 for n in hook.intervention_norms)


def test_linear_hook_rejects_non_1d_direction():
    with pytest.raises(ValueError, match="1-D"):
        steering.LinearSteeringHook(np.zeros((3, 3), dtype=np.float32), alpha=1.0, device="cpu")


def test_linear_hook_repr():
    hook = steering.LinearSteeringHook(np.zeros(32, dtype=np.float32), alpha=0.5, device="cpu")
    s = repr(hook)
    assert "0.5" in s
    assert "32" in s


# ═══════════════════════════════════════════════════════════════════════════════
# get_conceptor_matrix — new strategies
# ═══════════════════════════════════════════════════════════════════════════════


def test_positive_only_looks_up_C_success(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    C = steering.get_conceptor_matrix(npz, "taskA", 11, 0.1, "positive_only")
    # Fixture sets C_success = 0.4 * I; C_contrastive = 0.5 * I. Must pick C_success.
    np.testing.assert_allclose(C, 0.4 * np.eye(8), atol=1e-6)


def test_random_matched_same_spectrum_as_contrastive(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    # Fixture's C_contrastive = 0.5 * I → eigenvalues all 0.5.
    C = steering.get_conceptor_matrix(npz, "taskA", 11, 0.1, "random_matched", random_seed=7)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (C + C.T)))
    np.testing.assert_allclose(eig, np.full(8, 0.5), atol=1e-5)


def test_random_matched_requires_seed(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(ValueError, match="random_seed"):
        steering.get_conceptor_matrix(npz, "taskA", 11, 0.1, "random_matched")


def test_linear_strategy_routed_to_helper(mini_npz: pathlib.Path):
    """get_conceptor_matrix should refuse 'linear'; caller must use get_linear_direction."""
    npz = steering.load_conceptor_npz(mini_npz)
    with pytest.raises(ValueError, match="linear"):
        steering.get_conceptor_matrix(npz, "taskA", 11, 0.1, "linear")


def test_get_linear_direction(mini_npz: pathlib.Path):
    npz = steering.load_conceptor_npz(mini_npz)
    v_a = steering.get_linear_direction(npz, "taskA", 11)
    v_b = steering.get_linear_direction(npz, "taskB", 11)
    # Fixture: taskA → e0, taskB → e1
    np.testing.assert_allclose(v_a, np.eye(8)[0])
    np.testing.assert_allclose(v_b, np.eye(8)[1])


def test_get_linear_direction_missing_raises(tmp_path: pathlib.Path):
    """NPZ without linear_direction keys must raise a helpful KeyError."""
    path = tmp_path / "legacy.npz"
    np.savez(path, **{"taskA__L11__0.1__C_contrastive": np.eye(8, dtype=np.float32)})
    npz = steering.load_conceptor_npz(path)
    with pytest.raises(KeyError, match="predate"):
        steering.get_linear_direction(npz, "taskA", 11)


# ═══════════════════════════════════════════════════════════════════════════════
# SteeredPolicyWrapper — end-to-end dispatch across all 5 strategies
# ═══════════════════════════════════════════════════════════════════════════════


def _payload(task: str, strategy: str, alpha: float = 0.1, beta: float = 0.3) -> dict:
    return {"task": task, "layer": 11, "alpha": alpha, "beta": beta, "strategy": strategy}


def test_wrapper_dispatches_positive_only(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "positive_only")})
    assert p.steering_calls == 1
    layer_idx, hook = p.last_steering_hooks[0]
    assert layer_idx == 11
    assert isinstance(hook, ConceptorSteeringHook)


def test_wrapper_dispatches_random_matched(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "random_matched")})
    assert p.steering_calls == 1
    _, hook = p.last_steering_hooks[0]
    assert isinstance(hook, ConceptorSteeringHook)


def test_wrapper_random_matched_deterministic_across_calls(mini_npz: pathlib.Path):
    """Same (task, layer, α, β, strategy) → same random matrix on cache hit."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "random_matched")})
    first = p.last_steering_hooks[0][1]
    w.infer({"__steering__": _payload("taskA", "random_matched")})
    second = p.last_steering_hooks[0][1]
    assert first is second  # cache hit


def test_wrapper_does_not_mutate_caller_obs(mini_npz: pathlib.Path):
    """The `__steering__` payload is part of the inbound obs dict. The wrapper
    must strip it before passing to the underlying policy BUT must not mutate
    the caller's dict (matches CollectingPolicy's .get() contract). Verify the
    key is still present in the caller's dict after infer()."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    payload = _payload("taskA", "global")
    obs = {"some_obs": "data", "__steering__": payload}
    w.infer(obs)
    # Caller's dict still has the steering key:
    assert "__steering__" in obs
    # But the downstream policy did not see it:
    assert "__steering__" not in p.last_steering_obs


def test_wrapper_cache_key_per_step_ignores_alpha(mini_npz: pathlib.Path):
    """per_step NPZ keys bake α=1.0; runtime α is irrelevant. Two per_step
    requests differing only in α must share one cache entry."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "per_step", alpha=0.1)})
    w.infer({"__steering__": _payload("taskA", "per_step", alpha=0.5)})
    w.infer({"__steering__": _payload("taskA", "per_step", alpha=1.0)})
    assert len(w._hook_cache) == 1  # noqa: SLF001


def test_wrapper_cache_key_linear_ignores_beta(mini_npz: pathlib.Path):
    """LinearSteeringHook has no β term; runtime β is irrelevant. Three linear
    requests differing only in β must share one cache entry."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "linear", alpha=0.5, beta=0.1)})
    w.infer({"__steering__": _payload("taskA", "linear", alpha=0.5, beta=0.3)})
    w.infer({"__steering__": _payload("taskA", "linear", alpha=0.5, beta=0.5)})
    assert len(w._hook_cache) == 1  # noqa: SLF001


def test_wrapper_cache_key_global_keys_on_both_alpha_beta(mini_npz: pathlib.Path):
    """Non-per_step/non-linear strategies still differentiate on both α and β —
    a change in either must produce a distinct hook."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "global", alpha=0.1, beta=0.1)})
    w.infer({"__steering__": _payload("taskA", "global", alpha=0.1, beta=0.3)})  # different β
    w.infer({"__steering__": _payload("taskA", "global", alpha=0.5, beta=0.1)})  # different α
    assert len(w._hook_cache) == 3  # noqa: SLF001


def test_wrapper_random_matched_seed_stable_across_processes(mini_npz: pathlib.Path):
    """The ``random_matched`` seed must be a function of (task, layer, α, β, strategy)
    that is stable across processes — Python's built-in hash() salts strings per-process
    under the default PYTHONHASHSEED=random, which would break reproducibility of the
    control-baseline experiment across server restarts.

    Rather than spawn a subprocess, verify the seed derivation directly: it should
    use a stable hash (blake2b/hashlib), not Python's salted hash()."""
    import hashlib

    w1 = SteeredPolicyWrapper(_StubPolicy(), conceptor_npz_path=mini_npz, device="cpu")
    w2 = SteeredPolicyWrapper(_StubPolicy(), conceptor_npz_path=mini_npz, device="cpu")
    payload = _payload("taskA", "random_matched")
    w1.infer({"__steering__": payload})
    w2.infer({"__steering__": dict(payload)})  # fresh dict to avoid aliasing
    # Within a single process two wrappers always agree; this is a baseline.
    M1 = w1._hook_cache[("taskA", 11, 0.1, 0.3, "random_matched")].M  # noqa: SLF001
    M2 = w2._hook_cache[("taskA", 11, 0.1, 0.3, "random_matched")].M  # noqa: SLF001
    torch.testing.assert_close(M1, M2, atol=0.0, rtol=0.0)

    # Stronger check: confirm the seed is derived from a stable hash, not builtin
    # hash(). The seed key deliberately excludes β (see
    # _get_or_build_hook.random_matched branch) so that β sweeps don't confound
    # the interpolation effect with the random-basis identity.
    seed_key = ("taskA", 11, 0.1, "random_matched")  # (task, layer, α, strategy) — β intentionally dropped
    expected_seed = int(hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=4).hexdigest(), 16)
    # Regenerate the random_matched matrix with that seed and confirm it matches.
    from openpi.serving.conceptors import random_matched_conceptor

    npz = steering.load_conceptor_npz(mini_npz)
    C_ref = np.asarray(npz["taskA__L11__0.1__C_contrastive"])
    C_expected = random_matched_conceptor(C_ref, seed=expected_seed)
    # The hook's M = (1-β)I + β·C, so invert that: C_recovered = (M - (1-β)I) / β
    d = C_expected.shape[0]
    M_cpu = M1.cpu().numpy()
    beta = 0.3
    C_recovered = (M_cpu - (1 - beta) * np.eye(d, dtype=np.float32)) / beta
    np.testing.assert_allclose(C_recovered, C_expected.astype(np.float32), atol=1e-5)


def test_random_matched_seed_is_beta_independent(mini_npz: pathlib.Path):
    """β sweeps for random_matched must hold the random eigenbasis fixed.

    Regression for a confound in which the seed derivation included β, so
    every β value got a different random control matrix — conflating the
    interpolation-strength effect with "we happened to sample a different
    random basis at this β."

    Given fixed (task, layer, α), the recovered conceptor from the hook's
    M matrix must be identical across β values (up to the β scaling of M).
    """
    w = SteeredPolicyWrapper(_StubPolicy(), conceptor_npz_path=mini_npz, device="cpu")
    payload_b01 = _payload("taskA", "random_matched", beta=0.1)
    payload_b03 = _payload("taskA", "random_matched", beta=0.3)
    w.infer({"__steering__": payload_b01})
    w.infer({"__steering__": payload_b03})

    M_b01 = w._hook_cache[("taskA", 11, 0.1, 0.1, "random_matched")].M.cpu().numpy()  # noqa: SLF001
    M_b03 = w._hook_cache[("taskA", 11, 0.1, 0.3, "random_matched")].M.cpu().numpy()  # noqa: SLF001

    # Invert M_β = (1-β)I + β·C to recover C at each β; the two C matrices
    # must match if and only if the random basis is β-independent.
    d = M_b01.shape[0]
    I = np.eye(d, dtype=np.float32)  # noqa: E741
    C_recovered_b01 = (M_b01 - 0.9 * I) / 0.1
    C_recovered_b03 = (M_b03 - 0.7 * I) / 0.3
    np.testing.assert_allclose(C_recovered_b01, C_recovered_b03, atol=1e-5)


def test_wrapper_dispatches_linear(mini_npz: pathlib.Path):
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "linear", alpha=0.5)})
    assert p.steering_calls == 1
    layer_idx, hook = p.last_steering_hooks[0]
    assert layer_idx == 11
    assert isinstance(hook, steering.LinearSteeringHook)
    # Fixture's taskA direction is e0; alpha=0.5 → hook.alpha == 0.5
    assert hook.alpha == 0.5


def test_wrapper_all_five_strategies_cache_separately(mini_npz: pathlib.Path):
    """Different strategies at same (task, layer, α, β) yield 5 distinct hooks."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    for strat in ("global", "per_step", "positive_only", "random_matched", "linear"):
        w.infer({"__steering__": _payload("taskA", strat)})
    assert len(w._hook_cache) == 5  # noqa: SLF001


def test_wrapper_dispatches_per_step(mini_npz: pathlib.Path):
    """per_step builds a hook holding all 10 per-step matrices."""
    p = _StubPolicy()
    w = SteeredPolicyWrapper(p, conceptor_npz_path=mini_npz, device="cpu")
    w.infer({"__steering__": _payload("taskA", "per_step")})
    assert p.steering_calls == 1
    layer_idx, hook = p.last_steering_hooks[0]
    assert layer_idx == 11
    assert isinstance(hook, ConceptorSteeringHook)
    assert hook._Ms is not None and len(hook._Ms) == 10  # noqa: SLF001
