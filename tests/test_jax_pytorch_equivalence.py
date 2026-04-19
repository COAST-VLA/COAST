"""
Test that PyTorch inference produces the same actions as JAX inference.

Loads both JAX and PyTorch models from the same checkpoint, runs inference
with identical noise, and compares the output actions.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    MUJOCO_GL=egl uv run pytest tests/test_jax_pytorch_equivalence.py -v -s -m manual
"""

import logging
import os
import pathlib

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import pytest

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

# Override with PI05_TEST_CKPT when the checkpoint lives elsewhere.
CHECKPOINT_DIR = pathlib.Path(os.environ.get("PI05_TEST_CKPT", "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"))
CONFIG_NAME = "pi05_metaworld"


@pytest.fixture(scope="module")
def policies():
    """Load both JAX and PyTorch policies from the same checkpoint."""
    import torch

    if not CHECKPOINT_DIR.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_DIR}")
    safetensors_path = CHECKPOINT_DIR / "model.safetensors"
    if not safetensors_path.exists():
        pytest.skip("model.safetensors not found — need both JAX and PyTorch weights")
    if not (CHECKPOINT_DIR / "params").exists():
        pytest.skip("JAX params not found — need both JAX and PyTorch weights")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    train_config = _config.get_config(CONFIG_NAME)

    # Load JAX policy: temporarily hide safetensors so auto-detection picks JAX
    hidden_path = CHECKPOINT_DIR / "model.safetensors.hidden"
    safetensors_path.rename(hidden_path)
    try:
        jax_policy = _policy_config.create_trained_policy(train_config, CHECKPOINT_DIR)
        assert not jax_policy._is_pytorch_model  # noqa: SLF001
    finally:
        hidden_path.rename(safetensors_path)

    # Load PyTorch policy (safetensors is back)
    pytorch_policy = _policy_config.create_trained_policy(train_config, CHECKPOINT_DIR)
    assert pytorch_policy._is_pytorch_model  # noqa: SLF001

    return jax_policy, pytorch_policy


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
class TestJaxPytorchEquivalence:
    def test_same_actions_with_same_noise(self, policies, env_and_obs):
        """JAX and PyTorch should produce similar actions given the same noise."""
        jax_policy, pytorch_policy = policies

        # Generate fixed noise as numpy (both policies accept numpy noise)
        rng = np.random.default_rng(seed=12345)
        noise = rng.standard_normal((1, 32, 32)).astype(np.float32)

        # Run JAX inference
        jax_result = jax_policy.infer(env_and_obs, noise=noise.copy())
        jax_actions = jax_result["actions"]

        # Run PyTorch inference
        pytorch_result = pytorch_policy.infer(env_and_obs, noise=noise.copy())
        pytorch_actions = pytorch_result["actions"]

        logger.info(f"JAX actions shape: {jax_actions.shape}, dtype: {jax_actions.dtype}")
        logger.info(f"PyTorch actions shape: {pytorch_actions.shape}, dtype: {pytorch_actions.dtype}")

        max_diff = np.max(np.abs(jax_actions - pytorch_actions))
        mean_diff = np.mean(np.abs(jax_actions - pytorch_actions))
        logger.info(f"Max action diff (JAX vs PyTorch): {max_diff:.6e}")
        logger.info(f"Mean action diff (JAX vs PyTorch): {mean_diff:.6e}")

        # Allow tolerance for JAX vs PyTorch numerical differences
        # (different matmul implementations, bfloat16 rounding, etc.)
        assert max_diff < 0.05, f"Actions differ too much! max_diff={max_diff:.6e}"
        assert mean_diff < 0.01, f"Actions differ too much! mean_diff={mean_diff:.6e}"
        logger.info("JAX vs PyTorch equivalence verified!")
