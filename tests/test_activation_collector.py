"""Unit tests for src/openpi/serving/activation_collector.py.

These tests do not need a GPU or a real checkpoint. They cover:
- save_step_activations / save_episode_files write the expected on-disk schema
- env_id slicing on the batch dim is correct
- CollectingPolicy.infer dispatching:
    - rejects requests without magic keys
    - rejects requests with both magic keys
    - __collect__: calls infer_with_intermediates, saves activations, returns actions
    - __finalize_episode__: writes episode files, returns ack
- _batch_single_example handles 1-D state and adds the leading batch dim
- End-to-end CollectionSession <-> CollectingPolicy round-trip with a stub
  underlying policy

The dedicated CollectionSession state-tracking unit tests live in
tests/client/test_collection_session.py (next to the other openpi-client
unit tests). The end-to-end pipeline tests stay here because they exercise
both the client helper and the server wrapper together.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time

import numpy as np
from openpi_client.collection_session import BatchCollectionSession
from openpi_client.collection_session import CollectionSession
import pytest

from openpi.models import model as _model
from openpi.serving.activation_collector import CollectingPolicy
from openpi.serving.activation_collector import save_episode_files
from openpi.serving.activation_collector import save_step_activations
from openpi.serving.activation_collector import save_step_activations_fast

# ----------------------------------------------------------------------- helpers


def _fake_intermediates(num_steps: int = 10, batch: int = 2) -> dict:
    """Mimic the shapes returned by Pi0Pytorch.sample_actions_with_intermediates (v1)."""
    return {
        "all_x_t": np.random.randn(num_steps, batch, 10, 32).astype(np.float32),
        "all_v_t": np.random.randn(num_steps, batch, 10, 32).astype(np.float32),
        "all_adarms_cond": np.random.randn(num_steps, batch, 1024).astype(np.float32),
        "all_suffix_residual": np.random.randn(num_steps, 4, batch, 10, 1024).astype(np.float32),
        "all_suffix_mlp_hidden": np.random.randn(num_steps, 4, batch, 10, 4096).astype(np.float32),
    }


class _StubPolicy:
    """Records calls to infer_with_intermediates and returns canned outputs."""

    def __init__(self, action_horizon: int = 10, action_dim: int = 7) -> None:
        self.calls: list[dict] = []
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self.metadata = {"underlying": "stub"}

    def infer_with_intermediates(self, obs: dict) -> tuple[dict, dict]:
        self.calls.append(dict(obs))
        batch_size = int(np.asarray(obs["observation/state"]).shape[0])
        actions = np.zeros((batch_size, self._action_horizon, self._action_dim), dtype=np.float32)
        return {"actions": actions, "policy_timing": {"infer_ms": 1.23}}, _fake_intermediates(batch=batch_size)


class _StubFastPolicy:
    """Like _StubPolicy but returns pi0-fast shaped intermediates.

    Mimics Policy.infer_with_intermediates for a JAX pi0-fast model after
    the Python-side slicing in policy.py: generated_tokens / token_logprobs
    are (num_tokens, batch), token_pre_logits is (num_tokens-1, batch, width),
    num_tokens is an int.
    """

    def __init__(self, action_horizon: int = 10, action_dim: int = 7, num_tokens: int = 4, width: int = 64) -> None:
        self.calls: list[dict] = []
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._num_tokens = num_tokens
        self._width = width
        self.metadata = {"underlying": "stub-fast"}

    def infer_with_intermediates(self, obs: dict) -> tuple[dict, dict]:
        self.calls.append(dict(obs))
        batch_size = int(np.asarray(obs["observation/state"]).shape[0])
        actions = np.zeros((batch_size, self._action_horizon, self._action_dim), dtype=np.float32)
        intermediates = {
            "generated_tokens": np.arange(self._num_tokens * batch_size, dtype=np.int32).reshape(
                self._num_tokens, batch_size
            ),
            "token_logprobs": np.linspace(-3.0, -0.1, self._num_tokens * batch_size, dtype=np.float32).reshape(
                self._num_tokens, batch_size
            ),
            "token_pre_logits": np.random.randn(max(self._num_tokens - 1, 0), batch_size, self._width).astype(
                np.float32
            ),
            "num_tokens": self._num_tokens,
        }
        return {"actions": actions, "policy_timing": {"infer_ms": 2.34}}, intermediates


# --------------------------------------------------------------- writer tests


class TestSaveStepActivations:
    def test_writes_all_files(self, tmp_path: pathlib.Path) -> None:
        intermediates = _fake_intermediates(num_steps=10, batch=2)
        step_dir = tmp_path / "ep" / "step_0000"
        save_step_activations(
            step_dir,
            intermediates,
            env_id=0,
            step_metadata={"task_name": "t", "step": 0},
        )
        for fname in [
            "denoising.npz",
            "adarms_cond.npz",
            "suffix_residual.npz",
            "suffix_mlp_hidden.npz",
            "metadata.json",
        ]:
            assert (step_dir / fname).exists(), f"missing {fname}"

    def test_env_id_slicing_takes_correct_batch_index(self, tmp_path: pathlib.Path) -> None:
        intermediates = _fake_intermediates(num_steps=10, batch=3)
        step_dir = tmp_path / "step"
        save_step_activations(step_dir, intermediates, env_id=2, step_metadata={"step": 0})

        # all_x_t shape (num_steps, batch, ah, dim) -> sliced to (num_steps, ah, dim)
        saved = np.load(step_dir / "denoising.npz")
        np.testing.assert_array_equal(saved["all_x_t"], intermediates["all_x_t"][:, 2])
        np.testing.assert_array_equal(saved["all_v_t"], intermediates["all_v_t"][:, 2])

        # all_adarms_cond shape (num_steps, batch, hidden) -> (num_steps, hidden)
        cond = np.load(step_dir / "adarms_cond.npz")
        np.testing.assert_array_equal(cond["all_adarms_cond"], intermediates["all_adarms_cond"][:, 2])

        # all_suffix_residual shape (num_steps, num_layers, batch, ah, hidden) -> (num_steps, num_layers, ah, hidden)
        res = np.load(step_dir / "suffix_residual.npz")
        np.testing.assert_array_equal(res["all_suffix_residual"], intermediates["all_suffix_residual"][:, :, 2])

        mlp = np.load(step_dir / "suffix_mlp_hidden.npz")
        np.testing.assert_array_equal(mlp["all_suffix_mlp_hidden"], intermediates["all_suffix_mlp_hidden"][:, :, 2])

    def test_metadata_json_round_trips(self, tmp_path: pathlib.Path) -> None:
        meta = {"task_name": "foo", "step": 42, "cumulative_reward": 0.5, "success_so_far": False}
        save_step_activations(
            tmp_path / "step",
            _fake_intermediates(),
            env_id=0,
            step_metadata=meta,
        )
        with open(tmp_path / "step" / "metadata.json") as f:
            loaded = json.load(f)
        assert loaded == meta


def _fake_fast_intermediates(num_tokens: int = 6, batch: int = 2, width: int = 2048) -> dict:
    """Mimic the shapes returned by Policy.infer_with_intermediates for pi0-fast
    after the Python-side slicing (generated_tokens/logprobs are (num_tokens, batch),
    token_pre_logits is (num_tokens-1, batch, width), num_tokens is an int).
    """
    return {
        "generated_tokens": np.arange(num_tokens * batch, dtype=np.int32).reshape(num_tokens, batch),
        "token_logprobs": np.linspace(-5.0, -0.1, num_tokens * batch, dtype=np.float32).reshape(num_tokens, batch),
        "token_pre_logits": np.random.randn(max(num_tokens - 1, 0), batch, width).astype(np.float32),
        "num_tokens": num_tokens,
    }


class TestSaveStepActivationsFast:
    """Locks in the ``fast_v1`` on-disk schema for pi0-fast activations."""

    def test_writes_all_files(self, tmp_path: pathlib.Path) -> None:
        intermediates = _fake_fast_intermediates(num_tokens=6, batch=2)
        step_dir = tmp_path / "ep" / "step_0000"
        save_step_activations_fast(
            step_dir,
            intermediates,
            env_id=0,
            step_metadata={"task_name": "t", "step": 0},
        )
        for fname in ["tokens.npz", "hidden_states.npz", "token_logprobs.npz", "metadata.json"]:
            assert (step_dir / fname).exists(), f"missing {fname}"

    def test_env_id_slicing_picks_batch_column(self, tmp_path: pathlib.Path) -> None:
        intermediates = _fake_fast_intermediates(num_tokens=5, batch=3)
        step_dir = tmp_path / "step"
        save_step_activations_fast(step_dir, intermediates, env_id=2, step_metadata={})

        tokens = np.load(step_dir / "tokens.npz")["generated_tokens"]
        np.testing.assert_array_equal(tokens, intermediates["generated_tokens"][:, 2])

        logprobs = np.load(step_dir / "token_logprobs.npz")["token_logprobs"]
        np.testing.assert_allclose(logprobs, intermediates["token_logprobs"][:, 2])

        pre_logits = np.load(step_dir / "hidden_states.npz")["token_pre_logits"]
        np.testing.assert_allclose(
            pre_logits,
            intermediates["token_pre_logits"][:, 2].astype(np.float16),
        )

    def test_dtypes_match_schema(self, tmp_path: pathlib.Path) -> None:
        # Source arrays deliberately use wider dtypes; the writer should downcast
        # (tokens→int32, logprobs→float32, pre_logits→float16) so the on-disk
        # schema is stable regardless of what the JAX side produces.
        intermediates = {
            "generated_tokens": np.ones((4, 1), dtype=np.int64),
            "token_logprobs": np.ones((4, 1), dtype=np.float64),
            "token_pre_logits": np.ones((3, 1, 16), dtype=np.float32),
            "num_tokens": 4,
        }
        step_dir = tmp_path / "step"
        save_step_activations_fast(step_dir, intermediates, env_id=0, step_metadata={})
        assert np.load(step_dir / "tokens.npz")["generated_tokens"].dtype == np.int32
        assert np.load(step_dir / "token_logprobs.npz")["token_logprobs"].dtype == np.float32
        assert np.load(step_dir / "hidden_states.npz")["token_pre_logits"].dtype == np.float16

    def test_metadata_has_num_tokens_and_collection_version(self, tmp_path: pathlib.Path) -> None:
        intermediates = _fake_fast_intermediates(num_tokens=7, batch=1)
        step_dir = tmp_path / "step"
        save_step_activations_fast(
            step_dir,
            intermediates,
            env_id=0,
            step_metadata={"task_name": "t", "step": 3, "inference_step": 1},
        )
        with open(step_dir / "metadata.json") as f:
            loaded = json.load(f)
        # Caller-supplied fields pass through unchanged...
        assert loaded["task_name"] == "t"
        assert loaded["step"] == 3
        assert loaded["inference_step"] == 1
        # ...and the writer adds the two schema-identifying fields.
        assert loaded["num_tokens"] == 7
        assert loaded["collection_version"] == "fast_v1"

    def test_does_not_mutate_caller_metadata(self, tmp_path: pathlib.Path) -> None:
        """The writer adds num_tokens/collection_version to metadata. It must
        not mutate the caller's dict in place — otherwise repeated calls in a
        rollout would accumulate fields or trigger surprising ordering bugs.
        """
        intermediates = _fake_fast_intermediates(num_tokens=3, batch=1)
        caller_meta = {"task_name": "t", "step": 0}
        save_step_activations_fast(tmp_path / "step", intermediates, env_id=0, step_metadata=caller_meta)
        assert caller_meta == {"task_name": "t", "step": 0}

    def test_num_tokens_one_skips_hidden_states_file(self, tmp_path: pathlib.Path) -> None:
        """Edge case: if only one token was generated (EOS immediately), pre_logits
        has leading shape 0 and hidden_states.npz must not be written — otherwise
        np.load later chokes on a zero-length array with no recorded shape."""
        intermediates = {
            "generated_tokens": np.array([[5]], dtype=np.int32),
            "token_logprobs": np.array([[-0.1]], dtype=np.float32),
            "token_pre_logits": np.zeros((0, 1, 16), dtype=np.float32),
            "num_tokens": 1,
        }
        step_dir = tmp_path / "step"
        save_step_activations_fast(step_dir, intermediates, env_id=0, step_metadata={})
        assert (step_dir / "tokens.npz").exists()
        assert (step_dir / "token_logprobs.npz").exists()
        assert (step_dir / "metadata.json").exists()
        assert not (step_dir / "hidden_states.npz").exists(), (
            "hidden_states.npz should be omitted when token_pre_logits has length 0"
        )


class TestSaveEpisodeFiles:
    def test_writes_metadata_and_rewards(self, tmp_path: pathlib.Path) -> None:
        episode_dir = tmp_path / "ep"
        episode_metadata = {"task_name": "foo", "episode_id": 1, "total_reward": 1.0}
        per_step_reward = [0.0, 0.0, 0.0, 0.5, 0.5]
        per_step_success = [False, False, False, False, True]

        save_episode_files(episode_dir, episode_metadata, per_step_reward, per_step_success)

        with open(episode_dir / "metadata.json") as f:
            assert json.load(f) == episode_metadata

        rew = np.load(episode_dir / "rewards.npz")
        assert rew["per_step_reward"].dtype == np.float32
        assert rew["cumulative_reward"].dtype == np.float32
        assert rew["success_at_step"].dtype == bool

        np.testing.assert_array_equal(rew["per_step_reward"], np.array(per_step_reward, dtype=np.float32))
        np.testing.assert_array_equal(rew["cumulative_reward"], np.cumsum(per_step_reward).astype(np.float32))
        np.testing.assert_array_equal(rew["success_at_step"], np.array(per_step_success, dtype=bool))

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        save_episode_files(deep, {}, [], [])
        assert (deep / "metadata.json").exists()
        assert (deep / "rewards.npz").exists()


# ---------------------------------------------------- batch helper tests


class TestBatchSingleExample:
    def test_1d_state_gets_batch_dim(self) -> None:
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "do the thing",
        }
        batched = CollectingPolicy._batch_single_example(obs)  # noqa: SLF001
        assert batched["observation/state"].shape == (1, 8)
        assert batched["observation/image"].shape == (1, 224, 224, 3)
        assert batched["prompt"] == ["do the thing"]

    def test_2d_state_passes_through(self) -> None:
        obs = {
            "observation/state": np.zeros((4, 8), dtype=np.float32),
            "observation/image": np.zeros((4, 224, 224, 3), dtype=np.uint8),
            "prompt": ["a", "b", "c", "d"],
        }
        batched = CollectingPolicy._batch_single_example(obs)  # noqa: SLF001
        assert batched["observation/state"].shape == (4, 8)
        assert batched["observation/image"].shape == (4, 224, 224, 3)
        assert batched["prompt"] == ["a", "b", "c", "d"]

    def test_missing_state_returns_obs_unchanged(self) -> None:
        obs = {"foo": "bar"}
        assert CollectingPolicy._batch_single_example(obs) is obs  # noqa: SLF001

    def test_batch_single_example_droid_keys(self) -> None:
        """Droid sends observation/joint_position + observation/gripper_position
        instead of observation/state. Verify all arrays gain a batch dim."""
        obs = {
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "pick up cup",
        }
        batched = CollectingPolicy._batch_single_example(obs)  # noqa: SLF001
        assert batched["observation/joint_position"].shape == (1, 7)
        assert batched["observation/gripper_position"].shape == (1, 1)
        assert batched["observation/exterior_image_1_left"].shape == (1, 224, 224, 3)
        assert batched["prompt"] == ["pick up cup"]

    def test_batch_single_example_already_batched_droid_keys(self) -> None:
        """If droid obs already has a batch dim, pass through unchanged."""
        obs = {
            "observation/joint_position": np.zeros((1, 7), dtype=np.float32),
            "observation/gripper_position": np.zeros((1, 1), dtype=np.float32),
            "observation/exterior_image_1_left": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "prompt": ["pick up cup"],
        }
        batched = CollectingPolicy._batch_single_example(obs)  # noqa: SLF001
        assert batched["observation/joint_position"].shape == (1, 7)
        assert batched["observation/gripper_position"].shape == (1, 1)
        assert batched["observation/exterior_image_1_left"].shape == (1, 224, 224, 3)
        assert batched["prompt"] == ["pick up cup"]

    def test_batch_single_example_no_observation_keys(self) -> None:
        """If no observation arrays exist, return unchanged."""
        obs = {"prompt": "hello"}
        result = CollectingPolicy._batch_single_example(obs)  # noqa: SLF001
        assert result is obs


# ------------------------------------------- CollectingPolicy dispatch tests


@pytest.fixture
def policy_setup(tmp_path: pathlib.Path):
    stub = _StubPolicy()
    wrapper = CollectingPolicy(
        policy=stub,
        output_root=tmp_path,
        checkpoint_step="ckpt-step",
        policy_dir="/fake/policy/dir",
        config_name="fake_config",
        model_type=_model.ModelType.PI05,
    )
    return stub, wrapper


class TestCollectingPolicyMetadata:
    def test_metadata_merges_underlying_and_adds_collection_fields(self, policy_setup, tmp_path) -> None:
        _, wrapper = policy_setup
        meta = wrapper.metadata
        assert meta["underlying"] == "stub"
        assert meta["policy_dir"] == "/fake/policy/dir"
        assert meta["config_name"] == "fake_config"
        # PI05 fixture → diffusion schema identifier.
        assert meta["collection_mode"] == "v1"
        assert meta["model_type"] == "pi05"
        assert meta["checkpoint_step"] == "ckpt-step"
        assert meta["output_root"] == str(tmp_path)


class TestCollectingPolicyDispatch:
    def test_rejects_request_without_magic_keys(self, policy_setup) -> None:
        _, wrapper = policy_setup
        obs = {"observation/state": np.zeros(8, dtype=np.float32), "prompt": "x"}
        with pytest.raises(ValueError, match=r"(?i)collection-only"):
            wrapper.infer(obs)

    def test_rejects_request_with_both_magic_keys(self, policy_setup) -> None:
        _, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "x",
            "__collect__": {"task_name": "t", "episode_id": 0, "env_id": 0, "step": 0},
            "__finalize_episode__": {"task_name": "t", "episode_id": 0, "env_id": 0},
        }
        with pytest.raises(ValueError, match="both"):
            wrapper.infer(obs)

    def test_rejects_batched_obs_to_avoid_silent_corruption(self, policy_setup) -> None:
        """The __collect__ payload carries one env_id, so we cannot label
        per-element activations from a multi-env batch. The wrapper must reject
        batched obs (state.shape[0] > 1) loudly rather than silently slicing
        batch index 0 and writing it under metadata's env_id, which would
        silently corrupt the on-disk dataset.
        """
        _, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros((4, 8), dtype=np.float32),
            "observation/image": np.zeros((4, 224, 224, 3), dtype=np.uint8),
            "prompt": ["a", "b", "c", "d"],
            "__collect__": {
                "task_name": "t",
                "episode_id": 0,
                "env_id": 0,
                "step": 0,
                "inference_step": 0,
                "prompt": "x",
                "cumulative_reward": 0.0,
                "success_so_far": False,
                "reward_since_last_inference": 0.0,
            },
        }
        with pytest.raises(ValueError, match="single-example"):
            wrapper.infer(obs)


class TestCollectingPolicyInferCollect:
    def test_calls_underlying_with_clean_obs_and_saves_activations(self, policy_setup, tmp_path: pathlib.Path) -> None:
        stub, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "do thing",
            "__collect__": {
                "task_name": "task_a",
                "episode_id": 0,
                "env_id": 0,
                "step": 5,
                "inference_step": 1,
                "prompt": "do thing",
                "cumulative_reward": 0.0,
                "success_so_far": False,
                "reward_since_last_inference": 0.0,
            },
        }
        result = wrapper.infer(obs)

        # Underlying policy should have been called once with magic keys stripped and a leading batch dim added.
        assert len(stub.calls) == 1
        sent = stub.calls[0]
        assert "__collect__" not in sent
        assert "__finalize_episode__" not in sent
        assert sent["observation/state"].shape == (1, 8)
        assert sent["observation/image"].shape == (1, 224, 224, 3)
        assert sent["prompt"] == ["do thing"]

        # Returned actions should have the batch dim stripped.
        assert result["actions"].shape == (10, 7)

        # On-disk activations should be at <output_root>/ckpt-step/task_a/episode_000_env_000/step_0005/
        step_dir = tmp_path / "ckpt-step" / "task_a" / "episode_000_env_000" / "step_0005"
        for fname in [
            "denoising.npz",
            "adarms_cond.npz",
            "suffix_residual.npz",
            "suffix_mlp_hidden.npz",
            "metadata.json",
        ]:
            assert (step_dir / fname).exists(), f"missing {fname}"

        with open(step_dir / "metadata.json") as f:
            saved_meta = json.load(f)
        assert saved_meta["task_name"] == "task_a"
        assert saved_meta["step"] == 5
        assert saved_meta["inference_step"] == 1


class TestCollectingPolicyInferCollectBatch:
    """Batched __collect__ (list) path used by the metaworld vectorized client.

    When __collect__ is a list, the obs is pre-batched (B, ...) and the server
    writes one step_dir per entry indexed into the batch dim. Backwards-
    compatible with the dict-shaped path tested above.
    """

    def test_writes_one_step_dir_per_env_in_batch(self, policy_setup, tmp_path: pathlib.Path) -> None:
        stub, wrapper = policy_setup
        num_envs = 3
        obs = {
            "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
            "observation/image": np.zeros((num_envs, 224, 224, 3), dtype=np.uint8),
            "prompt": ["do thing"] * num_envs,
            "__collect__": [
                {
                    "task_name": "task_b",
                    "episode_id": 0,
                    "env_id": env_id,
                    "step": 7,
                    "inference_step": 1,
                    "prompt": "do thing",
                    "cumulative_reward": float(env_id),
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for env_id in range(num_envs)
            ],
        }
        result = wrapper.infer(obs)

        # Underlying policy called exactly once with magic keys stripped.
        assert len(stub.calls) == 1
        assert "__collect__" not in stub.calls[0]
        # Pre-batched obs is forwarded as-is — no extra batch dim added.
        assert stub.calls[0]["observation/state"].shape == (num_envs, 8)

        # Returned actions retain the batch dim so the client can dispatch per-env.
        assert result["actions"].shape == (num_envs, 10, 7)

        # Each env_id should land in its own step_dir with its own metadata.
        for env_id in range(num_envs):
            step_dir = tmp_path / "ckpt-step" / "task_b" / f"episode_000_env_{env_id:03d}" / "step_0007"
            for fname in [
                "denoising.npz",
                "adarms_cond.npz",
                "suffix_residual.npz",
                "suffix_mlp_hidden.npz",
                "metadata.json",
            ]:
                assert (step_dir / fname).exists(), f"missing {step_dir / fname}"
            with open(step_dir / "metadata.json") as f:
                step_meta = json.load(f)
            assert step_meta["env_id"] == env_id
            assert step_meta["cumulative_reward"] == float(env_id)

    def test_env_id_slicing_picks_correct_batch_index(self, policy_setup, tmp_path: pathlib.Path) -> None:
        """Each env_id's saved activations must come from its own batch index,
        not an off-by-one slice. Probes the env_id == 2 case in a 3-env batch.
        """
        stub, wrapper = policy_setup
        num_envs = 3

        # Replace the stub's intermediates with a deterministic tensor whose
        # per-batch-index slice is identifiable.
        canned_intermediates = {
            "all_x_t": np.arange(10 * num_envs * 32 * 32, dtype=np.float32).reshape(10, num_envs, 32, 32),
            "all_v_t": np.zeros((10, num_envs, 32, 32), dtype=np.float32),
            "all_adarms_cond": np.zeros((10, num_envs, 1024), dtype=np.float32),
            "all_suffix_residual": np.zeros((10, 4, num_envs, 32, 1024), dtype=np.float32),
            "all_suffix_mlp_hidden": np.zeros((10, 4, num_envs, 32, 4096), dtype=np.float32),
        }

        def _stub_infer(obs: dict):
            return (
                {"actions": np.zeros((num_envs, 10, 7), dtype=np.float32)},
                canned_intermediates,
            )

        stub.infer_with_intermediates = _stub_infer  # type: ignore[assignment]

        obs = {
            "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
            "prompt": ["p"] * num_envs,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": env_id,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for env_id in range(num_envs)
            ],
        }
        wrapper.infer(obs)

        for env_id in range(num_envs):
            step_dir = tmp_path / "ckpt-step" / "t" / f"episode_000_env_{env_id:03d}" / "step_0000"
            saved = np.load(step_dir / "denoising.npz")
            np.testing.assert_array_equal(saved["all_x_t"], canned_intermediates["all_x_t"][:, env_id])

    def test_batch_size_mismatch_raises(self, policy_setup) -> None:
        _, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros((4, 8), dtype=np.float32),
            "prompt": ["p"] * 4,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": i,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for i in range(3)  # 3 entries vs batch of 4
            ],
        }
        with pytest.raises(ValueError, match="Batch size mismatch"):
            wrapper.infer(obs)

    def test_unbatched_obs_with_list_collect_raises(self, policy_setup) -> None:
        """List __collect__ requires pre-batched obs with shape (B, ...)."""
        _, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),  # 1-D
            "prompt": "p",
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 0,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
            ],
        }
        with pytest.raises(ValueError, match="pre-batched"):
            wrapper.infer(obs)

    def test_empty_list_collect_raises(self, policy_setup) -> None:
        _, wrapper = policy_setup
        obs = {
            "observation/state": np.zeros((0, 8), dtype=np.float32),
            "prompt": [],
            "__collect__": [],
        }
        with pytest.raises(ValueError, match="must not be empty"):
            wrapper.infer(obs)


class TestCollectingPolicyFinalizeBatch:
    def test_finalizes_one_episode_per_entry(self, policy_setup, tmp_path: pathlib.Path) -> None:
        _, wrapper = policy_setup
        num_envs = 3
        finalize_payload = {
            "__finalize_episode__": [
                {
                    "task_name": "task_b",
                    "episode_id": 5,
                    "env_id": env_id,
                    "prompt": "p",
                    "episode_success": env_id == 0,
                    "total_reward": float(env_id),
                    "steps_to_success": 4 if env_id == 0 else -1,
                    "total_env_steps": 6,
                    "total_inference_steps": 2,
                    "per_step_reward": [0.0, 0.0, 0.0, 0.0, 1.0 if env_id == 0 else 0.0, 0.0],
                    "per_step_success": [False] * 4 + [env_id == 0, env_id == 0],
                }
                for env_id in range(num_envs)
            ],
        }
        result = wrapper.infer(finalize_payload)
        assert result["ack"] is True
        assert len(result["episode_dirs"]) == num_envs

        for env_id in range(num_envs):
            episode_dir = tmp_path / "ckpt-step" / "task_b" / f"episode_005_env_{env_id:03d}"
            assert (episode_dir / "metadata.json").exists()
            assert (episode_dir / "rewards.npz").exists()
            with open(episode_dir / "metadata.json") as f:
                ep_meta = json.load(f)
            assert ep_meta["env_id"] == env_id
            assert ep_meta["episode_success"] is (env_id == 0)

    def test_empty_list_finalize_raises(self, policy_setup) -> None:
        _, wrapper = policy_setup
        with pytest.raises(ValueError, match="must not be empty"):
            wrapper.infer({"__finalize_episode__": []})


class TestCollectingPolicyFinalize:
    def test_writes_episode_files_and_returns_ack(self, policy_setup, tmp_path: pathlib.Path) -> None:
        stub, wrapper = policy_setup
        finalize_payload = {
            "__finalize_episode__": {
                "task_name": "task_b",
                "episode_id": 3,
                "env_id": 0,
                "prompt": "do other thing",
                "episode_success": True,
                "total_reward": 1.0,
                "steps_to_success": 7,
                "total_env_steps": 10,
                "total_inference_steps": 2,
                "per_step_reward": [0.0] * 7 + [1.0, 0.0, 0.0],
                "per_step_success": [False] * 7 + [True, True, True],
            },
        }
        result = wrapper.infer(finalize_payload)

        # No inference call should have happened.
        assert stub.calls == []
        assert result["ack"] is True
        assert "episode_dir" in result

        episode_dir = tmp_path / "ckpt-step" / "task_b" / "episode_003_env_000"
        assert (episode_dir / "metadata.json").exists()
        assert (episode_dir / "rewards.npz").exists()

        with open(episode_dir / "metadata.json") as f:
            ep_meta = json.load(f)
        # Server adds checkpoint_dir + config_name from its own startup config.
        assert ep_meta["checkpoint_dir"] == "/fake/policy/dir"
        assert ep_meta["config_name"] == "fake_config"
        assert ep_meta["episode_success"] is True
        assert ep_meta["steps_to_success"] == 7
        assert ep_meta["total_env_steps"] == 10

        rewards = np.load(episode_dir / "rewards.npz")
        assert rewards["per_step_reward"].shape == (10,)
        assert float(rewards["cumulative_reward"][-1]) == pytest.approx(1.0)
        assert bool(rewards["success_at_step"][7]) is True


class TestCollectingPolicyPi0Fast:
    """Verifies CollectingPolicy routes to the fast_v1 writer when
    constructed with model_type=PI0_FAST.

    The dispatch decision is made once at construction time (self._save_step_fn),
    not by probing the intermediates dict shape — these tests pin that contract
    so a future change can't silently fall back to the diffusion writer and
    corrupt the on-disk dataset.
    """

    @pytest.fixture
    def fast_policy_setup(self, tmp_path: pathlib.Path):
        stub = _StubFastPolicy()
        wrapper = CollectingPolicy(
            policy=stub,
            output_root=tmp_path,
            checkpoint_step="fast-ckpt",
            policy_dir="/fake/policy/dir",
            config_name="pi0_fast_libero",
            model_type=_model.ModelType.PI0_FAST,
        )
        return stub, wrapper

    def test_metadata_reports_fast_v1_collection_mode(self, fast_policy_setup) -> None:
        _, wrapper = fast_policy_setup
        meta = wrapper.metadata
        assert meta["collection_mode"] == "fast_v1"
        assert meta["model_type"] == "pi0_fast"

    def test_infer_writes_fast_v1_schema(self, fast_policy_setup, tmp_path: pathlib.Path) -> None:
        stub, wrapper = fast_policy_setup
        obs = {
            "observation/state": np.zeros(8, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "do the thing",
            "__collect__": {
                "task_name": "t",
                "episode_id": 0,
                "env_id": 0,
                "step": 3,
                "inference_step": 1,
                "prompt": "do the thing",
                "cumulative_reward": 0.0,
                "success_so_far": False,
                "reward_since_last_inference": 0.0,
            },
        }
        result = wrapper.infer(obs)

        assert len(stub.calls) == 1
        assert result["actions"].shape == (10, 7)  # 3D stripped to (ah, ad)

        step_dir = tmp_path / "fast-ckpt" / "t" / "episode_000_env_000" / "step_0003"
        # fast_v1 files present
        assert (step_dir / "tokens.npz").exists()
        assert (step_dir / "token_logprobs.npz").exists()
        assert (step_dir / "hidden_states.npz").exists()
        assert (step_dir / "metadata.json").exists()
        # diffusion files NOT present — locks in that we didn't also write the v1 layout
        assert not (step_dir / "denoising.npz").exists()
        assert not (step_dir / "adarms_cond.npz").exists()
        assert not (step_dir / "suffix_residual.npz").exists()
        assert not (step_dir / "suffix_mlp_hidden.npz").exists()

        with open(step_dir / "metadata.json") as f:
            step_meta = json.load(f)
        assert step_meta["collection_version"] == "fast_v1"
        assert step_meta["num_tokens"] == 4

    def test_fast_writer_handles_single_token_no_hidden_states(self, tmp_path: pathlib.Path) -> None:
        """If the model emits a single EOS token, token_pre_logits has leading
        shape 0 and hidden_states.npz must be omitted. Regression for the
        edge-case branch inside save_step_activations_fast."""
        stub = _StubFastPolicy(num_tokens=1)
        wrapper = CollectingPolicy(
            policy=stub,
            output_root=tmp_path,
            checkpoint_step="ckpt",
            policy_dir="/d",
            config_name="pi0_fast_libero",
            model_type=_model.ModelType.PI0_FAST,
        )
        wrapper.infer(
            {
                "observation/state": np.zeros(8, dtype=np.float32),
                "prompt": "x",
                "__collect__": {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 0,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "x",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                },
            }
        )
        step_dir = tmp_path / "ckpt" / "t" / "episode_000_env_000" / "step_0000"
        assert (step_dir / "tokens.npz").exists()
        assert (step_dir / "token_logprobs.npz").exists()
        assert not (step_dir / "hidden_states.npz").exists()


class TestPathConstruction:
    def test_step_dir_format(self, policy_setup, tmp_path: pathlib.Path) -> None:
        _, wrapper = policy_setup
        meta = {"task_name": "my_task", "episode_id": 12, "env_id": 3, "step": 142}
        expected = tmp_path / "ckpt-step" / "my_task" / "episode_012_env_003" / "step_0142"
        assert wrapper._step_dir(meta) == expected  # noqa: SLF001

    def test_episode_dir_format(self, policy_setup, tmp_path: pathlib.Path) -> None:
        _, wrapper = policy_setup
        meta = {"task_name": "my_task", "episode_id": 12, "env_id": 3}
        expected = tmp_path / "ckpt-step" / "my_task" / "episode_012_env_003"
        assert wrapper._episode_dir(meta) == expected  # noqa: SLF001

    @pytest.mark.parametrize("task_name", ["/tmp/evil", "../evil", "nested/task", r"..\\evil"])
    def test_episode_dir_rejects_unsafe_task_name(self, policy_setup, task_name: str) -> None:
        _, wrapper = policy_setup
        meta = {"task_name": task_name, "episode_id": 12, "env_id": 3}
        with pytest.raises(ValueError, match="Invalid task_name"):
            wrapper._episode_dir(meta)  # noqa: SLF001


# ----------------------- end-to-end client+server pipeline tests


class TestEndToEndPipeline:
    """Drives a full rollout through CollectionSession (client) and CollectingPolicy
    (server), with the underlying policy stubbed out. Verifies that the on-disk
    schema produced by this client-server pair matches what tests/test_activations.py
    asserts -- so any drift in the protocol or the writers is caught in CI without
    needing a GPU, a real checkpoint, or a running WebSocket server.
    """

    def _drive_rollout(
        self,
        wrapper: CollectingPolicy,
        task_name: str,
        episode_id: int,
        prompt: str,
        steps_per_inference: int,
        num_inferences: int,
        success_at_step: int | None,
    ) -> None:
        """Mimic eval_task: alternate make_collect_metadata + record_step + finalize_episode."""
        session = CollectionSession(wrapper)
        session.start_episode(task_name=task_name, task_id=0, episode_id=episode_id, prompt=prompt)

        env_step = 0
        for inf_idx in range(num_inferences):
            obs = {
                "observation/state": np.zeros(8, dtype=np.float32),
                "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "prompt": prompt,
                "__collect__": session.make_collect_metadata(env_step),
            }
            wrapper.infer(obs)
            for _ in range(steps_per_inference):
                done = success_at_step is not None and env_step == success_at_step
                session.record_step(env_step, 1.0 if done else 0.0, done=done)
                env_step += 1
                if done:
                    break
            if success_at_step is not None and env_step > success_at_step:
                break
            del inf_idx
        session.finalize_episode()

    def test_round_trip_writes_schema_compliant_files(self, policy_setup, tmp_path: pathlib.Path) -> None:
        stub, wrapper = policy_setup
        self._drive_rollout(
            wrapper=wrapper,
            task_name="task_int",
            episode_id=0,
            prompt="do the integration thing",
            steps_per_inference=5,
            num_inferences=3,
            success_at_step=12,
        )

        # Underlying policy should have been called once per inference call.
        assert len(stub.calls) == 3

        episode_dir = tmp_path / "ckpt-step" / "task_int" / "episode_000_env_000"
        # Step dirs at 0, 5, 10 (every replan_steps starting from 0).
        step_dirs = sorted(episode_dir.glob("step_*"))
        assert [p.name for p in step_dirs] == ["step_0000", "step_0005", "step_0010"]

        # Each step dir has all 5 expected files.
        for step_dir in step_dirs:
            for fname in [
                "denoising.npz",
                "adarms_cond.npz",
                "suffix_residual.npz",
                "suffix_mlp_hidden.npz",
                "metadata.json",
            ]:
                assert (step_dir / fname).exists(), f"missing {fname} in {step_dir}"

        # Episode-level files exist with the schema test_activations.py expects.
        assert (episode_dir / "metadata.json").exists()
        assert (episode_dir / "rewards.npz").exists()

        with open(episode_dir / "metadata.json") as f:
            ep_meta = json.load(f)
        # Required fields per tests/test_activations.py::TestEpisodeMetadata.test_required_fields
        for field in [
            "task_name",
            "episode_id",
            "env_id",
            "episode_success",
            "total_reward",
            "steps_to_success",
            "total_env_steps",
            "total_inference_steps",
            "prompt",
            "checkpoint_dir",
            "config_name",
        ]:
            assert field in ep_meta, f"missing {field}"
        assert ep_meta["episode_success"] is True
        assert ep_meta["steps_to_success"] == 12  # first index where done=True
        assert ep_meta["total_reward"] == pytest.approx(1.0)
        # The last record_step before finalize was at step 12 (the done step), so the array length is 13.
        assert ep_meta["total_env_steps"] == 13

        rewards = np.load(episode_dir / "rewards.npz")
        # rewards.npz alignment: per_step_reward[steps_to_success] should be the success reward
        # (matches the metaworld convention test_activations.py validates).
        assert rewards["per_step_reward"].shape == (ep_meta["total_env_steps"],)
        assert float(rewards["per_step_reward"][ep_meta["steps_to_success"]]) == pytest.approx(1.0)
        assert bool(rewards["success_at_step"][ep_meta["steps_to_success"]]) is True
        # Cumulative final matches total_reward (catches arithmetic drift in save_episode_files).
        assert float(rewards["cumulative_reward"][-1]) == pytest.approx(ep_meta["total_reward"])

        # Step metadata required fields per test_activations.py::TestStepMetadata.test_required_fields
        with open(step_dirs[0] / "metadata.json") as f:
            step_meta = json.load(f)
        for field in [
            "task_name",
            "episode_id",
            "env_id",
            "step",
            "inference_step",
            "prompt",
            "cumulative_reward",
            "success_so_far",
            "reward_since_last_inference",
        ]:
            assert field in step_meta, f"missing {field}"
        assert step_meta["step"] == 0
        assert step_meta["inference_step"] == 0

    def test_round_trip_batch_session_writes_per_env_dirs(self, policy_setup, tmp_path: pathlib.Path) -> None:
        """End-to-end: BatchCollectionSession (client) + CollectingPolicy (server)
        drive a vectorized rollout with N envs in one inference call. Verifies
        the on-disk schema is identical to the single-env path, just with N
        episode dirs per task.
        """
        stub, wrapper = policy_setup
        num_envs = 3
        session = BatchCollectionSession(wrapper, num_envs=num_envs)
        session.start_episode(task_name="task_batch", episode_id=0, prompt="batch prompt")

        env_step = 0
        steps_per_inference = 5
        num_inferences = 3
        # env 0 succeeds at step 12; envs 1,2 never succeed.
        success_at = {0: 12}

        for _ in range(num_inferences):
            obs = {
                "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
                "observation/image": np.zeros((num_envs, 224, 224, 3), dtype=np.uint8),
                "prompt": ["batch prompt"] * num_envs,
                "__collect__": session.make_collect_metadata(env_step),
            }
            wrapper.infer(obs)
            for _ in range(steps_per_inference):
                rewards = np.zeros(num_envs, dtype=np.float32)
                dones = np.zeros(num_envs, dtype=bool)
                for env_id, target in success_at.items():
                    if env_step == target:
                        rewards[env_id] = 1.0
                        dones[env_id] = True
                session.record_step(env_step, rewards, dones)
                env_step += 1
        session.finalize_episode()

        # Underlying policy should have been called once per inference call.
        assert len(stub.calls) == num_inferences

        for env_id in range(num_envs):
            episode_dir = tmp_path / "ckpt-step" / "task_batch" / f"episode_000_env_{env_id:03d}"
            assert episode_dir.is_dir(), f"missing {episode_dir}"
            step_dirs = sorted(episode_dir.glob("step_*"))
            assert [p.name for p in step_dirs] == ["step_0000", "step_0005", "step_0010"]

            with open(episode_dir / "metadata.json") as f:
                ep_meta = json.load(f)
            assert ep_meta["task_name"] == "task_batch"
            assert ep_meta["env_id"] == env_id
            if env_id == 0:
                assert ep_meta["episode_success"] is True
                assert ep_meta["steps_to_success"] == 12
                assert ep_meta["total_reward"] == pytest.approx(1.0)
            else:
                assert ep_meta["episode_success"] is False
                assert ep_meta["steps_to_success"] == -1
                assert ep_meta["total_reward"] == pytest.approx(0.0)

    def test_failure_episode_writes_steps_to_success_minus_one(self, policy_setup, tmp_path: pathlib.Path) -> None:
        _, wrapper = policy_setup
        self._drive_rollout(
            wrapper=wrapper,
            task_name="task_fail",
            episode_id=2,
            prompt="fail",
            steps_per_inference=5,
            num_inferences=2,
            success_at_step=None,
        )

        episode_dir = tmp_path / "ckpt-step" / "task_fail" / "episode_002_env_000"
        with open(episode_dir / "metadata.json") as f:
            ep_meta = json.load(f)
        assert ep_meta["episode_success"] is False
        assert ep_meta["steps_to_success"] == -1
        assert ep_meta["total_reward"] == pytest.approx(0.0)
        rewards = np.load(episode_dir / "rewards.npz")
        assert not bool(np.any(rewards["success_at_step"]))


# --------------------- parallel-clients (libero_env eval_all) regression tests


class _ConcurrencyProbingStub:
    """Underlying-policy stub that fails the test if two calls overlap.

    The real PyTorch sample_actions_with_intermediates registers forward hooks
    on shared module instances and writes into a local dict via closure. Two
    concurrent calls would alias the hook target and cross-contaminate their
    captures. CollectingPolicy._intermediates_lock must serialize calls to
    protect this, even though the production asyncio server happens to also
    serialize them implicitly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0
        self.total_calls = 0
        self.metadata = {"underlying": "probe"}

    def infer_with_intermediates(self, obs: dict) -> tuple[dict, dict]:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        # Sleep long enough that unsynchronized callers would reliably interleave.
        time.sleep(0.01)
        with self._lock:
            self._in_flight -= 1
            self.total_calls += 1
        batch_size = int(np.asarray(obs["observation/state"]).shape[0])
        actions = np.zeros((batch_size, 10, 7), dtype=np.float32)
        return (
            {"actions": actions, "policy_timing": {"infer_ms": 10.0}},
            _fake_intermediates(batch=batch_size),
        )


class TestParallelCollectionSessions:
    """Regression tests for the libero_env/eval_all.py parallel-subprocess setup.

    Production layout: one --collect_activations policy server, N libero
    subprocesses each running a distinct task_id, each with its own
    CollectionSession and its own WebSocket connection. The server must handle
    interleaved __collect__ / __finalize_episode__ payloads from disjoint
    task_names without corruption. We emulate the interleaving in-process with
    N real threads hitting one shared CollectingPolicy — this covers everything
    except the WebSocket transport, which is orthogonal to the collection
    invariants.
    """

    def test_parallel_sessions_disjoint_tasks_no_contamination(self, tmp_path: pathlib.Path) -> None:
        stub = _ConcurrencyProbingStub()
        wrapper = CollectingPolicy(
            policy=stub,
            output_root=tmp_path,
            checkpoint_step="ckpt",
            policy_dir="/fake/policy/dir",
            config_name="fake",
            model_type=_model.ModelType.PI05,
        )

        num_clients = 8
        inferences_per_client = 3
        steps_per_inference = 5

        errors: list[BaseException] = []

        def drive_client(task_idx: int) -> None:
            try:
                session = CollectionSession(wrapper)
                session.start_episode(
                    task_name=f"task_{task_idx:02d}",
                    task_id=task_idx,
                    episode_id=0,
                    prompt=f"prompt for task {task_idx}",
                )
                env_step = 0
                for _ in range(inferences_per_client):
                    obs = {
                        "observation/state": np.zeros(8, dtype=np.float32),
                        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                        "prompt": f"prompt for task {task_idx}",
                        "__collect__": session.make_collect_metadata(env_step),
                    }
                    wrapper.infer(obs)
                    for _ in range(steps_per_inference):
                        session.record_step(env_step, 0.0, done=False)
                        env_step += 1
                session.finalize_episode()
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=drive_client, args=(i,)) for i in range(num_clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"worker threads raised: {errors}"

        # 1. _intermediates_lock must serialize infer_with_intermediates. If this
        #    fails, hook-based activation capture can cross-contaminate between
        #    concurrent clients.
        assert stub.max_in_flight == 1, (
            f"CollectingPolicy._intermediates_lock did not serialize calls "
            f"(max_in_flight={stub.max_in_flight}). Concurrent clients would "
            f"corrupt each other's hook-captured activations."
        )
        assert stub.total_calls == num_clients * inferences_per_client

        # 2. Every client's activations must land at its own disjoint path, and
        #    every step's metadata must carry its own client's task_name/prompt.
        #    This rules out path collisions and metadata cross-contamination.
        for task_idx in range(num_clients):
            episode_dir = tmp_path / "ckpt" / f"task_{task_idx:02d}" / "episode_000_env_000"
            assert episode_dir.is_dir(), f"missing episode dir for task {task_idx}"

            step_dirs = sorted(episode_dir.glob("step_*"))
            assert [p.name for p in step_dirs] == [
                "step_0000",
                "step_0005",
                "step_0010",
            ], f"task {task_idx}: unexpected step dirs {[p.name for p in step_dirs]}"

            for step_dir in step_dirs:
                with open(step_dir / "metadata.json") as f:
                    step_meta = json.load(f)
                assert step_meta["task_name"] == f"task_{task_idx:02d}", (
                    f"cross-contamination at {step_dir}: task_name="
                    f"{step_meta['task_name']!r}, expected task_{task_idx:02d}"
                )
                assert step_meta["prompt"] == f"prompt for task {task_idx}"
                assert step_meta["episode_id"] == 0
                assert step_meta["env_id"] == 0

            with open(episode_dir / "metadata.json") as f:
                ep_meta = json.load(f)
            assert ep_meta["task_name"] == f"task_{task_idx:02d}"
            assert ep_meta["total_env_steps"] == inferences_per_client * steps_per_inference
            assert ep_meta["total_inference_steps"] == inferences_per_client
            assert ep_meta["prompt"] == f"prompt for task {task_idx}"

    def test_parallel_inference_step_counters_are_per_session(self, tmp_path: pathlib.Path) -> None:
        """Verifies CollectionSession's inference_step counter is per-instance.

        Each libero subprocess has its own CollectionSession, so even when N
        sessions run concurrently the inference_step counters must not share
        state. This catches accidental class-level state in CollectionSession.
        """
        stub = _ConcurrencyProbingStub()
        wrapper = CollectingPolicy(
            policy=stub,
            output_root=tmp_path,
            checkpoint_step="ckpt",
            policy_dir="/fake/policy/dir",
            config_name="fake",
            model_type=_model.ModelType.PI05,
        )

        num_clients = 6
        inferences_per_client = 4

        def drive_client(task_idx: int) -> None:
            session = CollectionSession(wrapper)
            session.start_episode(
                task_name=f"task_{task_idx:02d}",
                task_id=task_idx,
                episode_id=0,
                prompt="p",
            )
            for inf_idx in range(inferences_per_client):
                obs = {
                    "observation/state": np.zeros(8, dtype=np.float32),
                    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "prompt": "p",
                    "__collect__": session.make_collect_metadata(inf_idx * 5),
                }
                wrapper.infer(obs)
                session.record_step(inf_idx * 5, 0.0, done=False)
            session.finalize_episode()

        threads = [threading.Thread(target=drive_client, args=(i,)) for i in range(num_clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every client must have recorded inference_step=0,1,2,3 in its own
        # step metadata. If CollectionSession leaked state between threads,
        # the counters would skip or repeat values.
        for task_idx in range(num_clients):
            episode_dir = tmp_path / "ckpt" / f"task_{task_idx:02d}" / "episode_000_env_000"
            step_dirs = sorted(episode_dir.glob("step_*"))
            observed = []
            for step_dir in step_dirs:
                with open(step_dir / "metadata.json") as f:
                    observed.append(json.load(f)["inference_step"])
            assert observed == list(range(inferences_per_client)), (
                f"task {task_idx}: expected inference_step counter {list(range(inferences_per_client))}, got {observed}"
            )


# ----------------------------------------- Protocol edge-case regression tests
#
# These tests cover gaps surfaced during a code-review pass on the unification
# refactor: input validation that the server should reject loudly instead of
# letting it propagate as a confusing KeyError or silently corrupting outputs.


class TestBatchProtocolEdgeCases:
    """Regression tests for `_handle_collect_infer_batch` validation. The
    server is exposed over WebSocket — a buggy client must get a clear
    ValueError, not an obscure KeyError or a half-written step_dir."""

    def test_duplicate_env_ids_overwrite_same_step_dir(self, policy_setup, tmp_path: pathlib.Path) -> None:
        """Documents current behavior: if a client sends duplicate env_ids in
        __collect__, the server happily writes to the same step_dir multiple
        times (last entry wins). This is a silent corruption mode worth
        knowing about — ideally the server would reject duplicates.
        """
        stub, wrapper = policy_setup
        num_envs = 3
        # All entries claim env_id=1 — same step_dir three times.
        obs = {
            "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
            "prompt": ["p"] * num_envs,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 1,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": float(i),
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for i in range(num_envs)
            ],
        }
        wrapper.infer(obs)
        # Only one step_dir exists — env_001/step_0000 — and its metadata is
        # whichever entry was written last.
        episode_dir = tmp_path / "ckpt-step" / "t" / "episode_000_env_001"
        step_dirs = list(episode_dir.glob("step_*"))
        assert len(step_dirs) == 1
        with open(step_dirs[0] / "metadata.json") as f:
            saved = json.load(f)
        # Last entry's cumulative_reward wins (== num_envs - 1).
        assert saved["cumulative_reward"] == float(num_envs - 1)
        # No env_000 or env_002 dirs should exist — confirms data loss.
        assert not (tmp_path / "ckpt-step" / "t" / "episode_000_env_000").exists()
        assert not (tmp_path / "ckpt-step" / "t" / "episode_000_env_002").exists()

    def test_out_of_range_env_id_raises(self, policy_setup) -> None:
        """env_id beyond batch_size-1 should fail; currently it triggers
        IndexError deep in save_step_activations rather than a clear
        ValueError at the protocol boundary."""
        _, wrapper = policy_setup
        num_envs = 3
        obs = {
            "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
            "prompt": ["p"] * num_envs,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 0,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                },
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 1,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                },
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": 99,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                },
            ],
        }
        # IndexError today; ideally a ValueError would catch this earlier.
        with pytest.raises((IndexError, ValueError)):
            wrapper.infer(obs)

    def test_missing_observation_state_with_only_images_kerrors(self, policy_setup) -> None:
        """The probe-key fallback in _handle_collect_infer_batch only checks
        for ``observation/state`` then iterates non-image observation/* keys.
        If the obs has only image keys, the probe falls through and the
        following indexing raises KeyError (not the user-friendly ValueError
        the rest of the protocol returns).
        """
        _, wrapper = policy_setup
        num_envs = 2
        obs = {
            # only images — no proprio key the probe can use
            "observation/image": np.zeros((num_envs, 32, 32, 3), dtype=np.uint8),
            "prompt": ["p"] * num_envs,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": i,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for i in range(num_envs)
            ],
        }
        # KeyError today: lookup of clean_obs["observation/state"] after the
        # probe loop fails to reassign probe_key.
        with pytest.raises((KeyError, ValueError)):
            wrapper.infer(obs)


class TestBatchSessionCrossTaskReuse:
    """Eval_all.py instantiates ONE BatchCollectionSession at process start and
    reuses it across all 45 ML45 tasks. Correctness depends on
    ``start_episode`` resetting every per-env state field. A future contributor
    that adds a new field to ``__init__`` but forgets to mirror it in
    ``start_episode`` would silently leak state across tasks. These tests
    guard the contract."""

    def test_inference_step_resets_between_tasks(self, policy_setup) -> None:
        """First task ends with inference_step >= 1; second task's first
        ``__collect__`` payload must report inference_step=0 again."""
        stub, wrapper = policy_setup
        session = BatchCollectionSession(wrapper, num_envs=2)

        # Task A: drive a few infer calls so inference_step advances.
        session.start_episode("task_a", episode_id=0, prompt="prompt-a")
        for step in (0, 5):
            wrapper.infer(
                {
                    "observation/state": np.zeros((2, 8), dtype=np.float32),
                    "prompt": ["prompt-a"] * 2,
                    "__collect__": session.make_collect_metadata(step),
                }
            )
        # Task B: starts fresh.
        session.start_episode("task_b", episode_id=0, prompt="prompt-b")
        meta = session.make_collect_metadata(step=0)
        assert all(entry["inference_step"] == 0 for entry in meta), (
            f"inference_step did not reset between tasks: {[e['inference_step'] for e in meta]}"
        )

    def test_cumulative_reward_resets_between_tasks(self, policy_setup) -> None:
        _, wrapper = policy_setup
        session = BatchCollectionSession(wrapper, num_envs=2)
        session.start_episode("task_a", 0, "p-a")
        session.record_step(0, [1.0, 2.0], [False, False])
        session.record_step(1, [0.5, 0.5], [True, False])
        # Task A leaves cumulative_reward = [1.5, 2.5], success = [True, False].
        session.start_episode("task_b", 0, "p-b")
        meta = session.make_collect_metadata(step=0)
        for entry in meta:
            assert entry["cumulative_reward"] == 0.0, (
                f"cumulative_reward leaked from task_a: {entry['cumulative_reward']}"
            )
            assert entry["success_so_far"] is False, "success_so_far leaked from task_a"
            assert entry["reward_since_last_inference"] == 0.0

    def test_full_round_trip_writes_separate_task_dirs(self, policy_setup, tmp_path: pathlib.Path) -> None:
        """End-to-end across two tasks: BatchCollectionSession + CollectingPolicy
        should produce two distinct task subtrees with no cross-contamination
        in the per-step / per-episode metadata."""
        stub, wrapper = policy_setup
        num_envs = 2
        session = BatchCollectionSession(wrapper, num_envs=num_envs)

        for task_name in ("task_a", "task_b"):
            session.start_episode(task_name, episode_id=0, prompt=f"prompt-{task_name}")
            for step in (0, 5):
                wrapper.infer(
                    {
                        "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
                        "prompt": [f"prompt-{task_name}"] * num_envs,
                        "__collect__": session.make_collect_metadata(step),
                    }
                )
                for s in range(5):
                    session.record_step(step + s, np.array([0.1, 0.2]), np.array([False, False]))
            session.finalize_episode()

        # Verify each task's metadata is correct and isolated.
        for task_name in ("task_a", "task_b"):
            for env_id in range(num_envs):
                ep_dir = tmp_path / "ckpt-step" / task_name / f"episode_000_env_{env_id:03d}"
                assert ep_dir.is_dir()
                with open(ep_dir / "metadata.json") as f:
                    ep_meta = json.load(f)
                assert ep_meta["task_name"] == task_name
                assert ep_meta["prompt"] == f"prompt-{task_name}"
                # Per-step metadata.json should also carry the right task_name.
                step_dirs = sorted(ep_dir.glob("step_*"))
                assert [p.name for p in step_dirs] == ["step_0000", "step_0005"]
                with open(step_dirs[0] / "metadata.json") as f:
                    step_meta = json.load(f)
                assert step_meta["task_name"] == task_name


class TestBatchSessionDoneSemantics:
    """The ``done`` parameter on BatchCollectionSession.record_step is named
    after gym's done flag but is interpreted as task-success (sticky
    ``_success[env_id]=True``, locks ``steps_to_success``). Tests document
    this so a future caller that confuses done-vs-success doesn't silently
    corrupt the sticky-success bookkeeping."""

    def test_done_true_sets_sticky_success(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=2)
        session.start_episode("t", 0, "p")
        session.record_step(5, [0.0, 0.0], [True, False])
        # After done=True at step 5 for env 0, success_so_far should remain True
        # even when subsequent record_step calls report done=False (post-success
        # steps in a metaworld rollout).
        session.record_step(6, [0.0, 0.0], [False, False])
        meta = session.make_collect_metadata(step=7)
        assert meta[0]["success_so_far"] is True
        assert meta[1]["success_so_far"] is False

    def test_steps_to_success_locks_first_success_only(self) -> None:
        client = _RecordingClient()
        session = BatchCollectionSession(client, num_envs=1)
        session.start_episode("t", 0, "p")
        session.record_step(0, [0.0], [False])
        session.record_step(3, [1.0], [True])  # first success at step 3
        session.record_step(4, [0.0], [False])
        session.record_step(7, [0.0], [True])  # later "success" must not overwrite
        session.finalize_episode()
        finalize_entry = client.calls[-1]["__finalize_episode__"][0]
        assert finalize_entry["steps_to_success"] == 3, (
            f"expected first success step (3) to be locked, got {finalize_entry['steps_to_success']}"
        )

    def test_record_step_without_any_success_emits_minus_one(self) -> None:
        client = _RecordingClient()
        session = BatchCollectionSession(client, num_envs=2)
        session.start_episode("t", 0, "p")
        for step in range(5):
            session.record_step(step, [0.1, 0.1], [False, False])
        session.finalize_episode()
        for entry in client.calls[-1]["__finalize_episode__"]:
            assert entry["steps_to_success"] == -1
            assert entry["episode_success"] is False


class TestBatchProbeKey:
    """Even if the probe-key error message is suboptimal (KeyError vs ValueError),
    the SUCCESS path with various proprio key names should still work.
    Regression test for a maintenance change to the fallback heuristic."""

    def test_probe_finds_alternate_proprio_key(self, policy_setup) -> None:
        """The fallback iterates non-image observation/* keys. A client that
        uses ``observation/proprio`` instead of ``observation/state`` should
        still pass the batch-size check."""
        stub, wrapper = policy_setup
        num_envs = 2
        # NB: stub policy keys on observation/state, so we keep that for the
        # forward pass but ALSO add observation/proprio so the probe loop has
        # an alternate to find. Real clients send only one proprio key — this
        # just exercises the fallback heuristic in isolation.
        obs = {
            "observation/state": np.zeros((num_envs, 8), dtype=np.float32),
            "observation/proprio": np.zeros((num_envs, 4), dtype=np.float32),
            "prompt": ["p"] * num_envs,
            "__collect__": [
                {
                    "task_name": "t",
                    "episode_id": 0,
                    "env_id": i,
                    "step": 0,
                    "inference_step": 0,
                    "prompt": "p",
                    "cumulative_reward": 0.0,
                    "success_so_far": False,
                    "reward_since_last_inference": 0.0,
                }
                for i in range(num_envs)
            ],
        }
        result = wrapper.infer(obs)
        assert result["actions"].shape == (num_envs, 10, 7)


# `_RecordingClient` is defined in tests/client/test_collection_session.py.
# Re-define a minimal one here so this file does not import across tests/.
class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, payload: dict) -> dict:
        self.calls.append(dict(payload))
        return {"ack": True, "episode_dirs": ["dummy"] * len(payload.get("__finalize_episode__", []) or [None])}
