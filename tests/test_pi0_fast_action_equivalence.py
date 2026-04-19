"""Test that pi0-fast ``sample_actions_with_intermediates`` produces the same
decoded actions as ``sample_actions``.

Commit 24d8668 claims bit-exact decoded actions between the two paths (while
noting that raw tokens can drift after ~90 tokens due to XLA fp-reorder). This
test locks in that guarantee at the Policy level so the intermediate-collection
path can be trusted to produce the same behavior as the eval path.

GPU-required (model is Pi0-FAST on Gemma-2B), so marked ``manual``. To run locally::

    export CUDA_VISIBLE_DEVICES=0
    hf download brandonyang/pi0fast-metaworld-checkpoints \\
        --include "pi0_fast_metaworld_b200_bs512/2500/*" \\
        --local-dir checkpoints/pi0_fast_metaworld
    MUJOCO_GL=egl uv run pytest tests/test_pi0_fast_action_equivalence.py -v -s -m manual

Override the checkpoint path with ``PI0_FAST_TEST_CKPT`` if needed.
"""

from __future__ import annotations

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

CHECKPOINT_DIR = pathlib.Path(
    os.environ.get(
        "PI0_FAST_TEST_CKPT",
        "checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500/",
    )
)
CONFIG_NAME = "pi0_fast_metaworld"


@pytest.fixture(scope="module")
def policy():
    """Load the pi0-fast policy via the JAX path (no safetensors needed).

    pi0-fast has no PyTorch port (see load_policy in examples/metaworld/main.py),
    so the policy is always loaded with use_pytorch=False and the JAX
    while_loop decode handles both the eval and collection paths.
    """
    import jax

    if not CHECKPOINT_DIR.exists():
        pytest.skip(
            f"Checkpoint not found: {CHECKPOINT_DIR} "
            f"(download with: hf download brandonyang/pi0fast-metaworld-checkpoints "
            f'--include "pi0_fast_metaworld_b200_bs512/2500/*" '
            f"--local-dir checkpoints/pi0_fast_metaworld)"
        )
    if not (CHECKPOINT_DIR / "params").exists():
        pytest.skip(f"JAX params dir not found at {CHECKPOINT_DIR / 'params'}")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("No GPU visible to JAX")

    train_config = _config.get_config(CONFIG_NAME)
    pol = _policy_config.create_trained_policy(train_config, str(CHECKPOINT_DIR), use_pytorch=False)
    assert not pol._is_pytorch_model, "pi0-fast must load as a JAX policy"  # noqa: SLF001
    assert hasattr(pol, "_sample_actions_with_intermediates"), (
        "Policy must have _sample_actions_with_intermediates JIT'd"
    )
    return pol


@pytest.fixture(scope="module")
def obs_dict():
    """Grab one MetaWorld observation (batch-of-1) for ``reach-v3``."""
    from examples.metaworld.main import MultiCameraWrapper

    env = MultiCameraWrapper(
        gym.make("Meta-World/MT1", env_name="reach-v3", seed=42, width=224, height=224),
        ["corner", "corner4", "gripperPOV"],
    )
    try:
        obs, info = env.reset(seed=42)
        camera_views = info["cameras"]
        yield {
            "observation/image": camera_views["corner4"][np.newaxis],
            "observation/wrist_image": camera_views["gripperPOV"][np.newaxis],
            "observation/state": obs.astype(np.float32)[np.newaxis, :4],
            "prompt": ["reach the goal position"],
        }
    finally:
        env.close()


@pytest.mark.manual
class TestPi0FastActionEquivalence:
    def test_decoded_actions_bit_exact(self, policy, obs_dict):
        """Policy.infer() vs Policy.infer_with_intermediates() decoded actions
        must match bit-exactly at temperature=0.

        Commit 24d8668 verified this manually (max_abs_diff=0.0 on 3 test
        observations). Raw tokens may differ slightly between the two while_loops
        due to XLA fp-reorder, but the FAST tokenizer maps those near-miss token
        sequences back to identical (32, 4) action chunks. This test pins that
        invariant — any future change to sample_actions_with_intermediates that
        breaks action equivalence should fail here rather than silently drift
        the activation dataset away from the eval dataset.
        """
        # Reset RNG between calls so both paths consume the same key. The Policy
        # advances self._rng on each call, so we snapshot and restore.
        rng_before = policy._rng  # noqa: SLF001

        plain = policy.infer(obs_dict)
        actions_plain = np.asarray(plain["actions"])

        policy._rng = rng_before  # noqa: SLF001
        with_intermediates, intermediates = policy.infer_with_intermediates(obs_dict)
        actions_interm = np.asarray(with_intermediates["actions"])

        assert actions_plain.shape == actions_interm.shape, (
            f"shape mismatch: plain={actions_plain.shape}, interm={actions_interm.shape}"
        )
        max_diff = float(np.max(np.abs(actions_plain - actions_interm)))
        mean_diff = float(np.mean(np.abs(actions_plain - actions_interm)))
        logger.info("pi0-fast action diff: max=%.3e mean=%.3e", max_diff, mean_diff)
        # The two paths decode through the same FAST tokenizer, which is a
        # discrete lookup — identical token sequences ⇒ identical actions. Even
        # when raw tokens drift by one or two IDs due to fp-reorder, the
        # tokenizer decoder tends to still map back to the same (action_horizon,
        # action_dim) chunk. Assert bit-exactness; fall back to near-zero if
        # a future XLA change breaks it.
        assert max_diff == 0.0, (
            f"Decoded actions diverged between sample_actions and "
            f"sample_actions_with_intermediates (max_diff={max_diff:.3e}). "
            f"If this is an XLA change and not a real regression, loosen to "
            f"a small tolerance and update the commit-message claim in 24d8668."
        )

        # Intermediates sanity: must contain the fast_v1 schema keys.
        for key in ("generated_tokens", "token_pre_logits", "token_logprobs", "num_tokens"):
            assert key in intermediates, f"missing intermediate {key!r}"
        num_tokens = int(intermediates["num_tokens"])
        assert num_tokens >= 1
        # After the Python-side slicing, shapes should be aligned:
        #   tokens/logprobs: (num_tokens, batch=1)
        #   pre_logits:      (num_tokens - 1, batch=1, width)
        assert intermediates["generated_tokens"].shape[0] == num_tokens
        assert intermediates["token_logprobs"].shape[0] == num_tokens
        assert intermediates["token_pre_logits"].shape[0] == max(num_tokens - 1, 0)

    def test_three_observations_deterministic(self, policy, obs_dict):
        """Repeat the comparison on a few perturbed observations. Catches
        regressions where equivalence holds for one sample but not others
        (e.g. a bug that only manifests when num_tokens varies)."""
        rng_snapshots = []
        for i in range(3):
            # Slightly perturb the state so each call exercises a different
            # decode trajectory / num_tokens.
            perturbed = dict(obs_dict)
            perturbed["observation/state"] = obs_dict["observation/state"] + np.float32(0.01 * i)

            rng_before = policy._rng  # noqa: SLF001
            rng_snapshots.append(rng_before)

            plain = policy.infer(perturbed)
            actions_plain = np.asarray(plain["actions"])

            policy._rng = rng_before  # noqa: SLF001
            with_intermediates, _ = policy.infer_with_intermediates(perturbed)
            actions_interm = np.asarray(with_intermediates["actions"])

            max_diff = float(np.max(np.abs(actions_plain - actions_interm)))
            logger.info("pi0-fast perturb %d: max_diff=%.3e", i, max_diff)
            assert max_diff == 0.0, f"Sample {i}: decoded actions diverged (max_diff={max_diff:.3e})"
