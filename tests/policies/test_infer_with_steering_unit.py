"""Pure-Python unit tests for Policy.infer_with_steering argument forwarding.

These tests bypass model construction and the Observation/jax.tree data pipeline
by heavy-mocking, to isolate the one thing we care about: does the `**self._sample_kwargs`
expansion actually thread custom sampler arguments through to
`model.sample_actions_with_steering`. CPU-only; no checkpoint or GPU required.
Complements the manual GPU end-to-end tests in tests/policies/test_infer_with_steering.py.
"""

# Test must poke at Policy's private attributes to construct a minimal stub
# without running the full __init__ (which requires a real model + checkpoint).
# ruff: noqa: SLF001

from __future__ import annotations

from unittest import mock

import numpy as np
import torch

from openpi.policies import policy as _policy_mod
from openpi.policies.policy import Policy


def _make_bare_policy(sample_kwargs: dict | None):
    """Build a Policy without running __init__ (no real model / transforms).

    Returns (policy, captured_kwargs_holder). The holder is a mutable dict
    populated by the stub sampler when ``infer_with_steering`` calls it.
    """
    p = Policy.__new__(Policy)
    p._is_pytorch_model = True
    p._pytorch_device = "cpu"
    p._sample_kwargs = dict(sample_kwargs or {})
    p._input_transform = lambda x: x
    p._output_transform = lambda x: x

    captured: dict = {}

    def _sample_actions_with_steering(device, observation, *, steering_hooks=None, **kwargs):
        captured["kwargs"] = kwargs
        captured["hooks"] = steering_hooks
        # Return a (actions, diagnostics) tuple compatible with infer_with_steering's
        # downstream .detach() / .cpu() pipeline.
        return torch.zeros((1, 10, 7), dtype=torch.float32), {}

    stub_model = mock.MagicMock()
    stub_model.sample_actions_with_steering = _sample_actions_with_steering
    p._model = stub_model
    return p, captured


def _run_infer(policy, steering_hooks):
    """Call policy.infer_with_steering while patching out the data pipeline.

    The real method runs jax.tree.map, input transforms, tensor conversion,
    Observation.from_dict (jaxtyping-validated), and then the sample call
    we actually care about. Everything except the sample call is orthogonal
    to the sample_kwargs-forwarding contract being tested, so we patch them
    to identity / trivial stubs.
    """

    # Observation.from_dict normally jaxtypes the tensor shapes; in this unit
    # test we just pass whatever dict it receives through.
    class _StubObs:
        pass

    # 'observation/state' gates the single-vs-batched branch; 'state' is read
    # by the output-dict construction after the sampler call.
    obs = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "state": np.zeros(8, dtype=np.float32),
    }

    with (
        mock.patch.object(_policy_mod._model.Observation, "from_dict", return_value=_StubObs()),
        mock.patch.object(_policy_mod.jax, "tree", new=mock.MagicMock(map=lambda f, x: x)),
        # Second jax.tree.map branch (output conversion) — same stub.
        mock.patch("openpi.policies.policy.torch.from_numpy", side_effect=lambda arr: torch.zeros(1)),
    ):
        # Provide an `inputs["state"]` dict entry the method reads for the output
        # dict. The post-sample output-processing walks ``outputs`` via
        # jax.tree.map (stubbed to identity) and does .detach()/.cpu() only on
        # PyTorch tensors, which our stub returns.
        policy.infer_with_steering(obs, steering_hooks=steering_hooks)


class _Hook:
    """Minimal hook stand-in — SteeredPolicyWrapper / sample_actions_with_steering
    would normally expect a (layer_idx, hook_callable) pair; this test only cares
    that the list survives unchanged."""


def test_infer_with_steering_forwards_sample_kwargs():
    """Regression for a silent bug where `Policy.infer_with_steering` dropped
    the policy's `sample_kwargs` on the floor. If a user passed
    ``create_trained_policy(sample_kwargs={"num_steps": 20})`` the steered
    path would run at the sampler default (10) and baseline vs steered SR
    would differ for sampler reasons unrelated to the steering hook itself.
    """
    policy, captured = _make_bare_policy(sample_kwargs={"num_steps": 20})
    _run_infer(policy, steering_hooks=[(5, _Hook())])
    assert captured.get("kwargs") == {"num_steps": 20}, (
        f"Expected num_steps=20 to reach the sampler, got {captured.get('kwargs')!r}"
    )


def test_infer_with_steering_default_sample_kwargs_is_empty():
    """With no sample_kwargs set, no extra kwargs are forwarded; the sampler
    falls back to its own defaults (e.g., num_steps=10)."""
    policy, captured = _make_bare_policy(sample_kwargs=None)
    _run_infer(policy, steering_hooks=[(5, _Hook())])
    assert captured.get("kwargs") == {}, (
        f"Expected empty kwargs when sample_kwargs is None, got {captured.get('kwargs')!r}"
    )


def test_infer_with_steering_forwards_arbitrary_extra_kwargs():
    """sample_kwargs is a forward-compatible escape hatch; anything set at
    policy construction should reach the model sampler, not just num_steps."""
    policy, captured = _make_bare_policy(sample_kwargs={"num_steps": 15, "custom_flag": True})
    _run_infer(policy, steering_hooks=[(5, _Hook())])
    assert captured.get("kwargs") == {"num_steps": 15, "custom_flag": True}, (
        f"Expected all sample_kwargs forwarded intact, got {captured.get('kwargs')!r}"
    )


def test_infer_with_steering_passes_steering_hooks_separately():
    """``steering_hooks`` and ``sample_kwargs`` must not collide — hooks flow
    through their own explicit parameter; sample_kwargs becomes **kwargs."""
    policy, captured = _make_bare_policy(sample_kwargs={"num_steps": 20})
    hooks = [(5, _Hook()), (11, _Hook())]
    _run_infer(policy, steering_hooks=hooks)
    assert captured.get("hooks") is hooks
    assert "steering_hooks" not in (captured.get("kwargs") or {}), "steering_hooks must not leak into **sample_kwargs"
