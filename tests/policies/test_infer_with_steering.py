"""Manual GPU tests for Policy.infer_with_steering.

Requires a free GPU and a local pi05_libero checkpoint. Skipped by default
via the ``manual`` marker; run explicitly with::

    uv run pytest tests/policies/test_infer_with_steering.py -m manual -v
"""

# ruff: noqa: RUF001, RUF002, RUF003

import jax
import numpy as np
import pytest
import torch

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.serving.steering import ConceptorSteeringHook
from openpi.training import config as _config

_CHECKPOINT_DIR = "checkpoints/openpi-libero-2000"


def _prepare_observation(policy):
    """Prepare a LIBERO example as a batched Observation on the policy's device."""
    example = libero_policy.make_libero_example()
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...],  # noqa: SLF001
        inputs,
    )
    return _model.Observation.from_dict(inputs)


@pytest.mark.manual
def test_identity_hook_with_shared_noise_matches_plain_sample_actions():
    """With an identity conceptor and β=0, sample_actions_with_steering and sample_actions
    must produce identical outputs *when given the same noise*.

    This is the core correctness guarantee for the hook plumbing: a no-op steering hook
    should not change actions. Pinning the noise is essential — without it, both methods
    sample fresh gaussian noise and produce different (but individually correct) outputs.
    """
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, _CHECKPOINT_DIR)
    observation = _prepare_observation(policy)
    device = policy._pytorch_device  # noqa: SLF001

    # Fixed noise, shared between both calls.
    bsize = observation.state.shape[0]
    actions_shape = (bsize, policy._model.config.action_horizon, policy._model.config.action_dim)  # noqa: SLF001
    torch.manual_seed(42)
    noise = torch.randn(actions_shape, dtype=torch.float32, device=device)

    # Plain sample_actions with the pinned noise.
    with torch.no_grad():
        plain_actions = policy._model.sample_actions(device, observation, noise=noise)  # noqa: SLF001

    # Identity hook (C=I, β=0 → h' = h exactly) with the same noise.
    hook = ConceptorSteeringHook(np.eye(1024, dtype=np.float32), beta=0.0, device=str(device))
    steered_actions, _ = policy._model.sample_actions_with_steering(  # noqa: SLF001
        device, observation, noise=noise, steering_hooks=[(5, hook)]
    )

    np.testing.assert_allclose(
        steered_actions.cpu().numpy(),
        plain_actions.cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
        err_msg="β=0 identity hook changed the actions — hook plumbing is wrong",
    )


@pytest.mark.manual
def test_identity_matrix_beta_nonzero_still_identity():
    """With C=I, any β produces M = (1-β)I + βI = I, so h @ M.T = h.
    Output must match plain sample_actions when noise is shared.
    """
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, _CHECKPOINT_DIR)
    observation = _prepare_observation(policy)
    device = policy._pytorch_device  # noqa: SLF001

    bsize = observation.state.shape[0]
    actions_shape = (bsize, policy._model.config.action_horizon, policy._model.config.action_dim)  # noqa: SLF001
    torch.manual_seed(42)
    noise = torch.randn(actions_shape, dtype=torch.float32, device=device)

    with torch.no_grad():
        plain_actions = policy._model.sample_actions(device, observation, noise=noise)  # noqa: SLF001

    hook = ConceptorSteeringHook(np.eye(1024, dtype=np.float32), beta=0.7, device=str(device))
    steered_actions, _ = policy._model.sample_actions_with_steering(  # noqa: SLF001
        device, observation, noise=noise, steering_hooks=[(5, hook)]
    )

    np.testing.assert_allclose(
        steered_actions.cpu().numpy(),
        plain_actions.cpu().numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    # Hook fires once per denoise step (default 10).
    assert len(hook.intervention_norms) == 10
    # With C=I and β=0.7, M=I so h @ M.T - h = 0 → intervention norms ≈ 0.
    assert all(n < 1e-3 for n in hook.intervention_norms), (
        f"identity steering produced nonzero intervention norms: {hook.intervention_norms}"
    )


@pytest.mark.manual
def test_infer_with_steering_rejects_jax_policy():
    """JAX policies must raise NotImplementedError — the hook path is PyTorch-only."""

    cfg = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    from openpi.policies import aloha_policy

    example = aloha_policy.make_aloha_example()
    hook = ConceptorSteeringHook(np.eye(1024, dtype=np.float32), beta=0.0, device="cpu")

    with pytest.raises(NotImplementedError, match="PyTorch"):
        policy.infer_with_steering(example, steering_hooks=[(5, hook)])
