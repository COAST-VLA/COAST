"""Unit tests for GR00T N1.5 steering support."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from openpi_client.steering import STEERING_KEY, build_steering_payload

import groot_steering
import serve


@pytest.fixture
def conceptor_npz(tmp_path: pathlib.Path) -> pathlib.Path:
    task = "OpenDrawer"
    layer = 3
    I = np.eye(4, dtype=np.float32)
    data = {
        f"{task}__L{layer}__0.1__C_contrastive": I,
        f"{task}__L{layer}__0.1__C_success": 0.5 * I,
        f"{task}__L{layer}__linear_direction": np.ones(4, dtype=np.float32) / 2.0,
    }
    for t in range(4):
        data[f"{task}__L{layer}__per_step_{t}__C_contrastive"] = I
    path = tmp_path / "groot_conceptors.npz"
    np.savez(path, **data)
    return path


class _StubPolicy:
    def __init__(self):
        self.metadata = {"underlying": "stub"}
        self.infer_calls = 0
        self.steering_calls = 0
        self.last_obs = None
        self.last_steering_hooks = None

    def infer(self, obs):
        self.infer_calls += 1
        self.last_obs = dict(obs)
        return {"actions": np.zeros((16, 12), dtype=np.float32)}

    def infer_with_steering(self, obs, *, steering_hooks):
        self.steering_calls += 1
        self.last_obs = dict(obs)
        self.last_steering_hooks = steering_hooks
        return {"actions": np.ones((16, 12), dtype=np.float32)}, {}


def _payload(strategy: str = "global") -> dict:
    return build_steering_payload(
        task="OpenDrawer",
        layer=3,
        alpha=0.1,
        beta=0.3,
        strategy=strategy,
    )


def test_available_tasks_reads_layer_keys(conceptor_npz: pathlib.Path):
    npz = groot_steering.load_conceptor_npz(conceptor_npz)
    assert groot_steering.available_tasks(npz) == {"OpenDrawer"}


def test_identity_conceptor_hook_is_no_op():
    hook = groot_steering.GrootConceptorSteeringHook(
        np.eye(4, dtype=np.float32), beta=0.7, device="cpu"
    )
    x = torch.randn(2, 5, 4)
    y = hook(None, (), x)
    torch.testing.assert_close(y, x)
    assert hook.intervention_norms[-1] == pytest.approx(0.0, abs=1e-6)


def test_linear_hook_adds_direction():
    hook = groot_steering.GrootLinearSteeringHook(
        np.ones(4, dtype=np.float32), alpha=0.25, device="cpu"
    )
    x = torch.zeros(1, 2, 4)
    y = hook(None, (), x)
    torch.testing.assert_close(y, torch.ones_like(x) * 0.25)


def test_wrapper_passthrough_without_steering_key(conceptor_npz: pathlib.Path):
    stub = _StubPolicy()
    wrapper = groot_steering.SteeredGrootPolicyWrapper(
        stub, conceptor_npz, device="cpu", num_denoising_steps=4
    )
    result = wrapper.infer({"observation/state": np.zeros(16, dtype=np.float32)})
    assert result["actions"].shape == (16, 12)
    assert stub.infer_calls == 1
    assert stub.steering_calls == 0


def test_wrapper_routes_steering_and_strips_magic_key(conceptor_npz: pathlib.Path):
    stub = _StubPolicy()
    wrapper = groot_steering.SteeredGrootPolicyWrapper(
        stub, conceptor_npz, device="cpu", num_denoising_steps=4
    )
    obs = {"foo": 1, STEERING_KEY: _payload()}
    result = wrapper.infer(obs)

    assert result["actions"].shape == (16, 12)
    assert stub.infer_calls == 0
    assert stub.steering_calls == 1
    assert STEERING_KEY not in stub.last_obs
    layer, hook = stub.last_steering_hooks[0]
    assert layer == 3
    assert isinstance(hook, groot_steering.GrootConceptorSteeringHook)


def test_wrapper_builds_per_step_hook_for_groot_denoising_steps(
    conceptor_npz: pathlib.Path,
):
    stub = _StubPolicy()
    wrapper = groot_steering.SteeredGrootPolicyWrapper(
        stub, conceptor_npz, device="cpu", num_denoising_steps=4
    )
    wrapper.infer({"foo": 1, STEERING_KEY: _payload("per_step")})
    _, hook = stub.last_steering_hooks[0]
    assert isinstance(hook, groot_steering.GrootConceptorSteeringHook)
    assert len(hook._Ms) == 4


def test_wrapper_rejects_unknown_task(conceptor_npz: pathlib.Path):
    stub = _StubPolicy()
    wrapper = groot_steering.SteeredGrootPolicyWrapper(
        stub, conceptor_npz, device="cpu", num_denoising_steps=4
    )
    bad = _payload()
    bad["task"] = "MissingTask"
    with pytest.raises(ValueError, match="not found in conceptor NPZ"):
        wrapper.infer({STEERING_KEY: bad})


def test_wrapper_metadata_reports_groot_backend(conceptor_npz: pathlib.Path):
    wrapper = groot_steering.SteeredGrootPolicyWrapper(
        _StubPolicy(), conceptor_npz, device="cpu", num_denoising_steps=4
    )
    meta = wrapper.metadata
    assert meta["underlying"] == "stub"
    assert meta["steering_enabled"] is True
    assert meta["steering_model_type"] == "groot_n15"
    assert meta["steering_backend"] == "groot_dit_hooks"
    assert meta["num_conceptor_tasks"] == 1


def test_serve_rejects_steer_and_collect_together():
    args = serve.Args(
        steer=True,
        collect_activations=True,
        conceptor_npz="/tmp/fake.npz",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        serve._validate_args(args)


def test_serve_rejects_steer_without_conceptor():
    args = serve.Args(steer=True, conceptor_npz=None)
    with pytest.raises(ValueError, match="requires --conceptor_npz"):
        serve._validate_args(args)
