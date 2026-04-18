"""Unit tests for groot_env/groot_activation_collector.py.

Groot-side analog of `tests/test_activation_collector.py`. Stub-based: no GPU,
no model, no checkpoint. Covers:

- `save_step_activations` writes the GR00T schema (denoising.npz,
  backbone_cond.npz, dit_hidden_states.npz, dit_mlp_hidden.npz,
  metadata.json) with correct env-id slicing
- `save_episode_files` writes rewards.npz + metadata.json
- `CollectingPolicy.infer` dispatch:
    - rejects plain requests (no magic keys)
    - rejects requests with both magic keys
    - routes `__collect__` to `infer_with_intermediates` and saves activations
    - routes `__finalize_episode__` to the episode writer
- `_batch_single_example` adds a batch dim to 1-D state
- End-to-end round-trip between `CollectionSession` (client) and
  `CollectingPolicy` (server) with a stub underlying policy
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from groot_activation_collector import (
    CollectingPolicy,
    save_episode_files,
    save_step_activations,
)

# --- helpers


def _fake_groot_intermediates(num_denoising: int = 4, batch: int = 1) -> dict:
    """Mimic the shapes returned by GR00TAdapterPolicy.infer_with_intermediates.

    See `groot_adapter._get_action_with_intermediates` for the real producer.
    Shapes below are the robocasa N1.5 defaults:
      - action_horizon=16, action_dim=32 (padded)
      - vl_seq_len=8 (stub), vl_hidden=16 (stub)
      - num_dit_layers=16, sa_seq_len=10 (stub), dit_hidden=32 (stub), ff_inner=128 (stub)
    `all_x_t` and `all_v_t` both have num_denoising entries (NOT num_denoising+1)
    to match pi0's `sample_actions_with_intermediates` convention. The DiT layer
    axis is exactly num_layers (no leading-input entry), also matching pi0.
    """
    return {
        "all_x_t": np.random.randn(num_denoising, batch, 16, 32).astype(np.float32),
        "all_v_t": np.random.randn(num_denoising, batch, 16, 32).astype(np.float32),
        "backbone_features": np.random.randn(batch, 8, 16).astype(np.float16),
        "all_dit_hidden_states": np.random.randn(
            num_denoising, 16, batch, 10, 32
        ).astype(np.float16),
        "all_dit_mlp_hidden": np.random.randn(num_denoising, 16, batch, 10, 128).astype(
            np.float16
        ),
    }


class _StubPolicy:
    """Stub GR00TAdapterPolicy. Records infer_with_intermediates calls and
    returns canned actions + intermediates."""

    def __init__(self):
        self.calls: list[dict] = []
        self.metadata = {"underlying": "stub"}

    def infer_with_intermediates(self, obs: dict) -> tuple[dict, dict]:
        self.calls.append(dict(obs))
        batch_size = int(np.asarray(obs["observation/state"]).shape[0])
        actions = np.zeros((batch_size, 16, 12), dtype=np.float32)
        return {
            "actions": actions,
            "policy_timing": {"infer_ms": 1.23},
        }, _fake_groot_intermediates(batch=batch_size)


# --- writer tests


class TestSaveStepActivations:
    def test_writes_all_files(self, tmp_path: pathlib.Path):
        interm = _fake_groot_intermediates()
        step_dir = tmp_path / "ep" / "step_0000"
        save_step_activations(
            step_dir, interm, env_id=0, step_metadata={"task_name": "t", "step": 0}
        )
        for fname in (
            "denoising.npz",
            "backbone_cond.npz",
            "dit_hidden_states.npz",
            "dit_mlp_hidden.npz",
            "metadata.json",
        ):
            assert (step_dir / fname).exists(), f"missing {fname}"

    def test_slices_env_id_on_batch_dim(self, tmp_path: pathlib.Path):
        """With batch=2, env_id=1 should slice the second example everywhere."""
        interm = _fake_groot_intermediates(batch=2)
        step_dir = tmp_path / "step"
        save_step_activations(step_dir, interm, env_id=1, step_metadata={})
        d = np.load(step_dir / "denoising.npz")
        np.testing.assert_array_equal(d["all_x_t"], interm["all_x_t"][:, 1])
        np.testing.assert_array_equal(d["all_v_t"], interm["all_v_t"][:, 1])
        bc = np.load(step_dir / "backbone_cond.npz")
        np.testing.assert_array_equal(
            bc["backbone_features"], interm["backbone_features"][1]
        )
        dh = np.load(step_dir / "dit_hidden_states.npz")
        np.testing.assert_array_equal(
            dh["all_dit_hidden_states"], interm["all_dit_hidden_states"][:, :, 1]
        )
        dm = np.load(step_dir / "dit_mlp_hidden.npz")
        np.testing.assert_array_equal(
            dm["all_dit_mlp_hidden"], interm["all_dit_mlp_hidden"][:, :, 1]
        )

    def test_metadata_json_content(self, tmp_path: pathlib.Path):
        step_dir = tmp_path / "step"
        save_step_activations(
            step_dir,
            _fake_groot_intermediates(),
            env_id=0,
            step_metadata={"task_name": "open_drawer", "step": 42, "inference_step": 8},
        )
        meta = json.load(open(step_dir / "metadata.json"))
        assert meta["task_name"] == "open_drawer"
        assert meta["step"] == 42
        assert meta["inference_step"] == 8


class TestSaveEpisodeFiles:
    def test_writes_metadata_and_rewards(self, tmp_path: pathlib.Path):
        ep_dir = tmp_path / "ep"
        save_episode_files(
            ep_dir,
            episode_metadata={
                "task_name": "t",
                "episode_id": 0,
                "episode_success": True,
            },
            per_step_reward=[0.0, 0.1, 0.3],
            per_step_success=[False, False, True],
        )
        assert (ep_dir / "metadata.json").exists()
        assert (ep_dir / "rewards.npz").exists()
        r = np.load(ep_dir / "rewards.npz")
        np.testing.assert_allclose(r["per_step_reward"], [0.0, 0.1, 0.3])
        np.testing.assert_allclose(r["cumulative_reward"], [0.0, 0.1, 0.4])
        np.testing.assert_array_equal(r["success_at_step"], [False, False, True])


# --- dispatcher tests


def _make_policy(tmp_path: pathlib.Path) -> tuple[CollectingPolicy, _StubPolicy]:
    stub = _StubPolicy()
    policy = CollectingPolicy(
        policy=stub,
        output_root=tmp_path,
        checkpoint_step="checkpoint-120000",
        policy_dir="/fake/ckpt",
        config_name="groot_n15_robocasa",
    )
    return policy, stub


class TestCollectingPolicyDispatch:
    def test_rejects_plain_request(self, tmp_path: pathlib.Path):
        policy, _ = _make_policy(tmp_path)
        with pytest.raises(ValueError, match="requires either __collect__"):
            policy.infer({"observation/state": np.zeros(16, dtype=np.float32)})

    def test_rejects_both_magic_keys(self, tmp_path: pathlib.Path):
        policy, _ = _make_policy(tmp_path)
        with pytest.raises(ValueError, match="both"):
            policy.infer(
                {
                    "observation/state": np.zeros(16, dtype=np.float32),
                    "__collect__": {},
                    "__finalize_episode__": {},
                }
            )

    def test_collect_writes_activations_and_returns_actions(
        self, tmp_path: pathlib.Path
    ):
        policy, stub = _make_policy(tmp_path)
        collect_meta = {
            "task_name": "open_drawer",
            "episode_id": 0,
            "env_id": 0,
            "step": 5,
            "inference_step": 1,
        }
        obs = {
            "observation/state": np.zeros(16, dtype=np.float32),
            "__collect__": collect_meta,
        }
        result = policy.infer(obs)

        # Actions forwarded from the stub (batch dim stripped since B=1).
        assert result["actions"].shape == (16, 12)
        assert "policy_timing" in result
        # Magic keys not leaked into the stub.
        assert "__collect__" not in stub.calls[0]
        assert "__finalize_episode__" not in stub.calls[0]
        # Files present on disk.
        step_dir = (
            tmp_path
            / "checkpoint-120000"
            / "open_drawer"
            / "episode_000_env_000"
            / "step_0005"
        )
        assert (step_dir / "denoising.npz").exists()
        assert (step_dir / "backbone_cond.npz").exists()
        assert (step_dir / "dit_hidden_states.npz").exists()
        assert (step_dir / "dit_mlp_hidden.npz").exists()

    def test_finalize_writes_episode_files(self, tmp_path: pathlib.Path):
        policy, stub = _make_policy(tmp_path)
        final_meta = {
            "task_name": "open_drawer",
            "episode_id": 0,
            "env_id": 0,
            "episode_success": True,
            "total_reward": 1.0,
            "steps_to_success": 37,
            "total_env_steps": 50,
            "total_inference_steps": 10,
            "prompt": "open the drawer",
            "per_step_reward": [0.0] * 49 + [1.0],
            "per_step_success": [False] * 49 + [True],
        }
        result = policy.infer({"__finalize_episode__": final_meta})
        assert result["ack"] is True
        ep_dir = tmp_path / "checkpoint-120000" / "open_drawer" / "episode_000_env_000"
        assert (ep_dir / "metadata.json").exists()
        assert (ep_dir / "rewards.npz").exists()
        # Finalize does NOT touch the underlying policy.
        assert stub.calls == []

    def test_rejects_multi_env_batch(self, tmp_path: pathlib.Path):
        """Collection mode enforces B=1 because the __collect__ payload carries
        one env_id; a multi-env batch can't be disambiguated per-element."""
        policy, _ = _make_policy(tmp_path)
        obs = {
            # 2-D state with batch>1 means pre-batched multi-env input.
            "observation/state": np.zeros((3, 16), dtype=np.float32),
            "__collect__": {"task_name": "t", "episode_id": 0, "env_id": 0, "step": 0},
        }
        with pytest.raises(ValueError, match="single-example"):
            policy.infer(obs)


class TestCollectingPolicyMetadata:
    def test_metadata_merges_underlying_with_collection_info(
        self, tmp_path: pathlib.Path
    ):
        policy, _ = _make_policy(tmp_path)
        meta = policy.metadata
        assert meta["underlying"] == "stub"  # from stub
        assert meta["collection_mode"] == "v1"
        assert meta["checkpoint_step"] == "checkpoint-120000"
        assert meta["config_name"] == "groot_n15_robocasa"


class TestBatchSingleExample:
    def test_adds_batch_dim_to_1d_state(self, tmp_path: pathlib.Path):
        policy, _ = _make_policy(tmp_path)
        obs = {
            "observation/state": np.zeros(16, dtype=np.float32),
            "prompt": "open",
        }
        batched = policy._batch_single_example(obs)
        assert batched["observation/state"].shape == (1, 16)
        assert batched["prompt"] == ["open"]

    def test_passthrough_when_already_batched(self, tmp_path: pathlib.Path):
        policy, _ = _make_policy(tmp_path)
        obs = {"observation/state": np.zeros((2, 16), dtype=np.float32)}
        batched = policy._batch_single_example(obs)
        assert batched["observation/state"].shape == (2, 16)


# --- end-to-end with CollectionSession


class TestCollectionSessionRoundtrip:
    def test_session_talks_to_collecting_policy(self, tmp_path: pathlib.Path):
        """CollectionSession (the client helper) should produce magic-key
        payloads that CollectingPolicy accepts and routes correctly."""
        from openpi_client.collection_session import CollectionSession

        policy, _ = _make_policy(tmp_path)
        # CollectionSession takes the server-facing policy as its constructor
        # arg (finalize_episode() calls policy.infer() directly). Wire it to
        # the CollectingPolicy we're testing so the round-trip stays in-process.
        session = CollectionSession(policy)
        session.start_episode(
            task_name="open_drawer",
            task_id=0,
            episode_id=0,
            prompt="open the drawer",
        )

        # Three inference steps.
        for step in range(3):
            collect_meta = session.make_collect_metadata(step)
            obs = {
                "observation/state": np.zeros(16, dtype=np.float32),
                "__collect__": collect_meta,
            }
            result = policy.infer(obs)
            assert result["actions"].shape == (16, 12)
            session.record_step(step, reward=0.0 if step < 2 else 1.0, done=step == 2)

        # Finalize -- the session sends the __finalize_episode__ payload via
        # the same policy.infer(), so no second call is needed.
        ack = session.finalize_episode()
        assert ack["ack"] is True

        # Disk layout reflects the three steps + episode file.
        ep_dir = tmp_path / "checkpoint-120000" / "open_drawer" / "episode_000_env_000"
        for step in range(3):
            step_dir = ep_dir / f"step_{step:04d}"
            assert (step_dir / "denoising.npz").exists()
            assert (step_dir / "backbone_cond.npz").exists()
            assert (step_dir / "dit_hidden_states.npz").exists()
            assert (step_dir / "dit_mlp_hidden.npz").exists()
        assert (ep_dir / "metadata.json").exists()
        assert (ep_dir / "rewards.npz").exists()
