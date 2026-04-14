"""Manual GPU tests for Policy.infer_with_steering.

Requires a free GPU and a local pi05_libero checkpoint. Skipped by default
via the ``manual`` marker; run explicitly with::

    uv run pytest tests/policies/test_infer_with_steering.py -m manual -v
"""

# ruff: noqa: RUF001, RUF002, RUF003

import pathlib
import sys

import numpy as np
import pytest

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# Let us import ConceptorSteeringHook from the experiment dir without packaging it.
_EXP_DIR = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "pi05_libero"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

_CHECKPOINT_DIR = "checkpoints/pi05_libero/libero_b200_bs512/2000"


@pytest.mark.manual
def test_infer_with_steering_identity_matches_plain_infer():
    """With C=I and β=0, steered output must be numerically equal to plain output.

    This is the core correctness guarantee for the hook plumbing: a no-op
    steering hook should not change actions. If it does, something in the
    hook registration, tensor conversion, or output merging is wrong.
    """
    from conceptor_steering import ConceptorSteeringHook

    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, _CHECKPOINT_DIR)

    example = libero_policy.make_libero_example()

    # Plain inference first (no hooks installed).
    baseline = policy.infer(example)

    # Same example, same checkpoint, β=0 steering hook → must match.
    hook = ConceptorSteeringHook(
        np.eye(1024, dtype=np.float32),
        beta=0.0,
        device=str(policy._pytorch_device),  # noqa: SLF001
    )
    steered, diagnostics = policy.infer_with_steering(example, steering_hooks=[(5, hook)])

    assert isinstance(diagnostics, dict)
    np.testing.assert_allclose(
        steered["actions"],
        baseline["actions"],
        rtol=1e-4,
        atol=1e-5,
        err_msg="β=0 steering hook changed the actions — hook plumbing is wrong",
    )


@pytest.mark.manual
def test_infer_with_steering_identity_matrix_beta_nonzero_is_no_op():
    """With C=I, any β produces h' = (1-β)h + β·h = h. Output should match plain infer."""
    from conceptor_steering import ConceptorSteeringHook

    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, _CHECKPOINT_DIR)

    example = libero_policy.make_libero_example()
    baseline = policy.infer(example)

    hook = ConceptorSteeringHook(
        np.eye(1024, dtype=np.float32),
        beta=0.7,  # non-zero β; with C=I the math still collapses to identity
        device=str(policy._pytorch_device),  # noqa: SLF001
    )
    steered, _ = policy.infer_with_steering(example, steering_hooks=[(5, hook)])

    np.testing.assert_allclose(
        steered["actions"],
        baseline["actions"],
        rtol=1e-4,
        atol=1e-5,
    )
    # The hook should have fired once per denoise step (10 by default).
    assert len(hook.intervention_norms) == 10
    # Intervention norms should be ~0 since C=I makes the steering a no-op.
    assert all(n < 1e-3 for n in hook.intervention_norms), (
        f"identity steering produced nonzero intervention norms: {hook.intervention_norms}"
    )


@pytest.mark.manual
def test_infer_with_steering_rejects_jax_policy():
    """JAX policies must raise NotImplementedError — the hook path is PyTorch-only."""
    from conceptor_steering import ConceptorSteeringHook

    cfg = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(cfg, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    from openpi.policies import aloha_policy

    example = aloha_policy.make_aloha_example()
    hook = ConceptorSteeringHook(np.eye(1024, dtype=np.float32), beta=0.0, device="cpu")

    with pytest.raises(NotImplementedError, match="PyTorch"):
        policy.infer_with_steering(example, steering_hooks=[(5, hook)])
