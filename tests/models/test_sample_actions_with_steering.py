"""Manual GPU tests for PI0Pytorch.sample_actions_with_steering.

These verify low-level hook lifecycle guarantees: hooks must be removed
from the action-expert layers whether the call completes successfully or
raises. Skipped by default; run explicitly with::

    uv run pytest tests/models/test_sample_actions_with_steering.py -m manual -v
"""

import numpy as np
import pytest
import torch

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.serving.steering import ConceptorSteeringHook
from openpi.training import config as _config

_CHECKPOINT_DIR = "checkpoints/coast-libero-2000"


@pytest.fixture(scope="module")
def loaded_policy_and_obs():
    """Load the pi05_libero policy + a prepared Observation once per module."""
    import jax

    from openpi.models import model as _model

    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, _CHECKPOINT_DIR)

    example = libero_policy.make_libero_example()
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)  # noqa: SLF001
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...],  # noqa: SLF001
        inputs,
    )
    observation = _model.Observation.from_dict(inputs)
    return policy, observation


def _expert_layers(policy):
    model = policy._model  # noqa: SLF001
    return model.paligemma_with_expert.gemma_expert.model.layers


def _hook_count(layers) -> list[int]:
    return [len(layer._forward_hooks) for layer in layers]  # noqa: SLF001


@pytest.mark.manual
def test_hooks_removed_after_successful_call(loaded_policy_and_obs):
    """After sample_actions_with_steering returns, zero user hooks remain."""
    policy, observation = loaded_policy_and_obs
    layers = _expert_layers(policy)
    before = _hook_count(layers)

    hook = ConceptorSteeringHook(
        np.eye(1024, dtype=np.float32),
        beta=0.0,
        device=str(policy._pytorch_device),  # noqa: SLF001
    )
    policy._model.sample_actions_with_steering(  # noqa: SLF001
        policy._pytorch_device,  # noqa: SLF001
        observation,
        steering_hooks=[(5, hook)],
    )

    after = _hook_count(layers)
    assert before == after, f"sample_actions_with_steering leaked forward hooks: before={before}, after={after}"


@pytest.mark.manual
def test_hooks_removed_after_failing_call(loaded_policy_and_obs):
    """If the hook raises, the try/finally block must still remove it."""
    policy, observation = loaded_policy_and_obs
    layers = _expert_layers(policy)
    before = _hook_count(layers)

    class _BoomHook:
        def __init__(self):
            self.current_denoise_step = -1

        def __call__(self, module, input, output):
            raise RuntimeError("boom")

        def set_denoise_step(self, t):
            self.current_denoise_step = t

    boom = _BoomHook()
    with pytest.raises(RuntimeError, match="boom"):
        policy._model.sample_actions_with_steering(  # noqa: SLF001
            policy._pytorch_device,  # noqa: SLF001
            observation,
            steering_hooks=[(5, boom)],
        )

    after = _hook_count(layers)
    assert before == after, f"hook leaked after exception: before={before}, after={after}"
