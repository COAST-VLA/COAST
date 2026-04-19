from openpi_client import action_chunk_broker
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _BaselineModel(torch.nn.Module):
    """Stand-in for a baseline (e.g., DP) that does not expose sample_actions_with_intermediates."""

    def sample_actions(self, device, observation):
        raise RuntimeError("unused in this test")


def test_infer_with_intermediates_raises_for_baseline_without_method():
    """Models without sample_actions_with_intermediates get a clear NotImplementedError (not AttributeError)."""
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._is_pytorch_model = True  # noqa: SLF001
    policy._model = _BaselineModel()  # noqa: SLF001
    with pytest.raises(NotImplementedError, match="sample_actions_with_intermediates"):
        policy.infer_with_intermediates({"observation/state": None})


def test_infer_with_intermediates_v2_raises_for_baseline_without_method():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._is_pytorch_model = True  # noqa: SLF001
    policy._model = _BaselineModel()  # noqa: SLF001
    with pytest.raises(NotImplementedError, match="sample_actions_with_intermediates_v2"):
        policy.infer_with_intermediates_v2({"observation/state": None})


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
