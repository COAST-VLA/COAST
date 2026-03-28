"""
Test that sample_actions_with_intermediates() produces the same actions as sample_actions().

This test loads the real model and runs both methods with the same noise to verify
the intermediates path doesn't change model behavior.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    MUJOCO_GL=egl uv run pytest tests/test_action_equivalence.py -v -s
"""

import logging
import pathlib

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import pytest
import torch

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"
CONFIG_NAME = "pi05_metaworld"


def _prepare_observation(policy, obs_dict):
    """Prepare observation tensors from raw obs dict (same as infer_with_intermediates)."""
    import jax

    from openpi.models import model as _model
    from openpi.policies.policy import collate_transformed_singles

    inputs = jax.tree.map(lambda x: x, obs_dict)
    ex = {k: v[0] for k, v in inputs.items()}
    singles = [policy._input_transform(ex)]  # noqa: SLF001
    inputs = collate_transformed_singles(singles)
    device = policy._pytorch_device  # noqa: SLF001
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device), inputs)
    return _model.Observation.from_dict(inputs), device


@pytest.fixture(scope="module")
def policy():
    if not pathlib.Path(CHECKPOINT_DIR).exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_DIR}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    train_config = _config.get_config(CONFIG_NAME)
    return _policy_config.create_trained_policy(train_config, CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def env_and_obs():
    """Create a MetaWorld env and get one observation."""
    from examples.metaworld.main import MultiCameraWrapper

    env = MultiCameraWrapper(
        gym.make("Meta-World/MT1", env_name="reach-v3", seed=42, width=224, height=224),
        ["corner", "corner4", "gripperPOV"],
    )
    obs, info = env.reset(seed=42)
    camera_views = info["cameras"]
    obs_dict = {
        "observation/image": camera_views["corner4"][np.newaxis],
        "observation/wrist_image": camera_views["gripperPOV"][np.newaxis],
        "observation/state": obs.astype(np.float32)[np.newaxis, :4],
        "prompt": ["reach the goal position"],
    }
    yield obs_dict
    env.close()


@pytest.mark.manual
class TestActionEquivalence:
    def test_same_actions_with_same_noise(self, policy, env_and_obs):
        """Both methods should produce identical (or near-identical) actions given the same noise."""
        observation, device = _prepare_observation(policy, env_and_obs)
        model = policy._model  # noqa: SLF001

        # Generate fixed noise
        noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), device)

        # Run intermediates method (eager mode)
        actions_intermed, intermediates = model.sample_actions_with_intermediates(
            device, observation, noise=noise.clone()
        )
        actions_intermed = actions_intermed.detach().cpu().float().numpy()

        # Run normal sample_actions (torch.compiled — first call triggers warmup)
        actions_normal = model.sample_actions(device, observation, noise=noise.clone())
        actions_normal = actions_normal.detach().cpu().float().numpy()

        # They should be very close (small numerical differences from torch.compile)
        max_diff = np.max(np.abs(actions_intermed - actions_normal))
        mean_diff = np.mean(np.abs(actions_intermed - actions_normal))
        logger.info(f"Max action diff: {max_diff:.6e}, Mean action diff: {mean_diff:.6e}")

        # torch.compile(mode="max-autotune") uses different triton matmul kernels than
        # eager mode, causing small numerical differences (~1e-3). This is expected.
        assert max_diff < 0.01, f"Actions differ too much! max_diff={max_diff:.6e}"
        assert mean_diff < 0.005, f"Actions differ too much! mean_diff={mean_diff:.6e}"
        logger.info("Action equivalence verified (within torch.compile tolerance)!")

    def test_intermediates_have_expected_keys(self, policy, env_and_obs):
        """Check that intermediates dict has all expected keys."""
        observation, device = _prepare_observation(policy, env_and_obs)
        model = policy._model  # noqa: SLF001

        _, intermediates = model.sample_actions_with_intermediates(device, observation)

        expected_keys = {"all_x_t", "all_v_t", "all_adarms_cond", "all_suffix_residual", "all_suffix_mlp_hidden"}
        assert set(intermediates.keys()) == expected_keys
