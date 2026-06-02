"""Unit tests for openpi_client.collection_session.CollectionSession.

These tests run in the root COAST venv against the editable-installed
openpi-client package — no libero/robocasa/droid venv needed. They cover the
state-tracking and protocol-payload-shaping behavior of the helper, with a
stub policy that just records every infer() call so we can inspect the
payloads it would have sent over the wire.
"""

from __future__ import annotations

import json

import numpy as np
from openpi_client.collection_session import BatchCollectionSession
from openpi_client.collection_session import CollectionSession
import pytest


class _RecordingClient:
    """Records every payload that .infer is called with. Returns a canned
    action chunk so callers can iterate further if they want."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, payload: dict) -> dict:
        # Deep-ish copy so subsequent mutations by the caller don't change history.
        self.calls.append(json.loads(json.dumps(payload, default=str)))
        return {"actions": np.zeros((10, 7), dtype=np.float32), "ack": True}


# --------------------------------------------------- start_episode tests


class TestStartEpisode:
    def test_resets_state_for_new_episode(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)

        # First episode: do some work, then reset.
        session.start_episode("task_a", task_id=1, episode_id=0, prompt="prompt-a")
        session.make_collect_metadata(step=0)  # bumps inference_step
        session.record_step(0, 0.5, done=False)  # accumulates reward

        session.start_episode("task_b", task_id=2, episode_id=5, prompt="prompt-b")
        # All counters should be reset.
        meta = session.make_collect_metadata(step=0)
        assert meta["task_name"] == "task_b"
        assert meta["episode_id"] == 5
        assert meta["env_id"] == 0
        assert meta["inference_step"] == 0
        assert meta["cumulative_reward"] == 0.0
        assert meta["success_so_far"] is False
        assert meta["reward_since_last_inference"] == 0.0
        assert meta["prompt"] == "prompt-b"

    def test_env_id_can_be_overridden(self) -> None:
        session = CollectionSession(_RecordingClient())
        session.start_episode("t", task_id=0, episode_id=0, prompt="p", env_id=7)
        meta = session.make_collect_metadata(step=0)
        assert meta["env_id"] == 7


# --------------------------------------------------- make_collect_metadata tests


class TestMakeCollectMetadata:
    def test_inference_step_increments_per_call(self) -> None:
        session = CollectionSession(_RecordingClient())
        session.start_episode("t", 0, 0, "p")
        m0 = session.make_collect_metadata(step=0)
        m1 = session.make_collect_metadata(step=5)
        m2 = session.make_collect_metadata(step=10)
        assert [m0["inference_step"], m1["inference_step"], m2["inference_step"]] == [0, 1, 2]
        assert [m0["step"], m1["step"], m2["step"]] == [0, 5, 10]

    def test_reward_since_last_inference_resets_after_each_call(self) -> None:
        session = CollectionSession(_RecordingClient())
        session.start_episode("t", 0, 0, "p")
        # First inference: nothing recorded yet, delta is 0.
        m0 = session.make_collect_metadata(step=0)
        assert m0["cumulative_reward"] == 0.0
        assert m0["reward_since_last_inference"] == 0.0

        # Record some env steps then inference again.
        session.record_step(0, 0.3, done=False)
        session.record_step(1, 0.2, done=False)
        m1 = session.make_collect_metadata(step=2)
        assert m1["cumulative_reward"] == pytest.approx(0.5)
        assert m1["reward_since_last_inference"] == pytest.approx(0.5)

        # No new env steps -> next call's delta should be 0 again.
        m2 = session.make_collect_metadata(step=3)
        assert m2["cumulative_reward"] == pytest.approx(0.5)
        assert m2["reward_since_last_inference"] == pytest.approx(0.0)


# --------------------------------------------------- record_step tests


class TestRecordStep:
    def test_accumulates_rewards_and_tracks_first_success(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        session.record_step(0, 0.0, done=False)
        session.record_step(1, 0.0, done=False)
        session.record_step(2, 1.0, done=True)  # success
        session.record_step(3, 0.0, done=True)  # still done

        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["per_step_reward"] == [0.0, 0.0, 1.0, 0.0]
        assert finalize["per_step_success"] == [False, False, True, True]
        assert finalize["episode_success"] is True
        assert finalize["steps_to_success"] == 2  # first index where done=True
        assert finalize["total_reward"] == pytest.approx(1.0)
        assert finalize["total_env_steps"] == 4

    def test_no_success_marks_steps_to_success_as_minus_one(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        session.record_step(0, 0.1, done=False)
        session.record_step(1, 0.2, done=False)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is False
        assert finalize["steps_to_success"] == -1
        assert finalize["total_reward"] == pytest.approx(0.3)


# --------------------------------------------------- finalize_episode tests


class TestFinalizeEpisode:
    def test_payload_shape_matches_server_expectations(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task_x", 0, 4, "prompt-x", env_id=0)

        # Two inferences and three env steps.
        session.make_collect_metadata(step=0)
        session.record_step(0, 0.0, done=False)
        session.record_step(1, 0.0, done=False)
        session.make_collect_metadata(step=2)
        session.record_step(2, 1.0, done=True)

        session.finalize_episode()

        assert len(client.calls) == 1
        finalize = client.calls[-1]["__finalize_episode__"]
        # Required fields the server's CollectingPolicy reads.
        for field in [
            "task_name",
            "episode_id",
            "env_id",
            "prompt",
            "episode_success",
            "total_reward",
            "steps_to_success",
            "total_env_steps",
            "total_inference_steps",
            "per_step_reward",
            "per_step_success",
        ]:
            assert field in finalize, f"missing {field}"

        assert finalize["task_name"] == "task_x"
        assert finalize["episode_id"] == 4
        assert finalize["env_id"] == 0
        assert finalize["prompt"] == "prompt-x"
        assert finalize["total_inference_steps"] == 2  # two make_collect_metadata calls
        assert finalize["total_env_steps"] == 3
        assert finalize["per_step_success"] == [False, False, True]
        assert finalize["per_step_reward"] == [0.0, 0.0, 1.0]


# ----------------------------------------------- BatchCollectionSession tests


class TestBatchCollectionSession:
    """Vectorized rollouts (one inference call per N envs) use list-shaped
    __collect__ / __finalize_episode__ payloads. The bookkeeping mirrors the
    single-env CollectionSession but tracks per-env state."""

    def test_make_collect_metadata_returns_list_per_env(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=3)
        session.start_episode("t", episode_id=0, prompt="p")
        meta = session.make_collect_metadata(step=5)
        assert isinstance(meta, list)
        assert len(meta) == 3
        for env_id, entry in enumerate(meta):
            assert entry["task_name"] == "t"
            assert entry["episode_id"] == 0
            assert entry["env_id"] == env_id
            assert entry["step"] == 5
            assert entry["inference_step"] == 0
            assert entry["prompt"] == "p"
            assert entry["cumulative_reward"] == 0.0
            assert entry["success_so_far"] is False
            assert entry["reward_since_last_inference"] == 0.0

    def test_inference_step_advances_per_call_not_per_env(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=2)
        session.start_episode("t", 0, "p")
        m0 = session.make_collect_metadata(step=0)
        m1 = session.make_collect_metadata(step=5)
        assert {entry["inference_step"] for entry in m0} == {0}
        assert {entry["inference_step"] for entry in m1} == {1}

    def test_record_step_tracks_per_env_reward_and_success(self) -> None:
        client = _RecordingClient()
        session = BatchCollectionSession(client, num_envs=3)
        session.start_episode("t", 0, "p")
        session.record_step(0, [1.0, 0.0, 0.5], [False, False, False])
        session.record_step(1, [0.0, 2.0, 0.5], [True, False, False])
        session.record_step(2, [0.0, 0.0, 0.0], [True, True, False])

        # cumulative_reward should be visible in the next make_collect_metadata.
        meta = session.make_collect_metadata(step=3)
        assert meta[0]["cumulative_reward"] == pytest.approx(1.0)
        assert meta[1]["cumulative_reward"] == pytest.approx(2.0)
        assert meta[2]["cumulative_reward"] == pytest.approx(1.0)
        # env 0 saw success at step 1, env 1 at step 2, env 2 never.
        assert meta[0]["success_so_far"] is True
        assert meta[1]["success_so_far"] is True
        assert meta[2]["success_so_far"] is False

    def test_finalize_episode_emits_list_payload_per_env(self) -> None:
        client = _RecordingClient()
        session = BatchCollectionSession(client, num_envs=3)
        session.start_episode("task_x", episode_id=4, prompt="prompt-x")

        session.make_collect_metadata(step=0)
        session.record_step(0, [0.0, 0.0, 0.0], [False, False, False])
        session.record_step(1, [1.0, 0.0, 0.0], [True, False, False])
        session.make_collect_metadata(step=2)
        session.record_step(2, [0.0, 0.5, 0.5], [True, False, False])

        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert isinstance(finalize, list)
        assert len(finalize) == 3

        # env 0: succeeded at step 1, total reward 1.0
        assert finalize[0]["env_id"] == 0
        assert finalize[0]["episode_success"] is True
        assert finalize[0]["steps_to_success"] == 1
        assert finalize[0]["total_reward"] == pytest.approx(1.0)
        assert finalize[0]["per_step_reward"] == [0.0, 1.0, 0.0]
        assert finalize[0]["per_step_success"] == [False, True, True]

        # env 1: never succeeded, total reward 0.5
        assert finalize[1]["env_id"] == 1
        assert finalize[1]["episode_success"] is False
        assert finalize[1]["steps_to_success"] == -1
        assert finalize[1]["total_reward"] == pytest.approx(0.5)

        # env 2: never succeeded, total reward 0.5
        assert finalize[2]["episode_success"] is False
        assert finalize[2]["total_reward"] == pytest.approx(0.5)

        # All three envs share the same task_name / episode_id / prompt.
        for entry in finalize:
            assert entry["task_name"] == "task_x"
            assert entry["episode_id"] == 4
            assert entry["prompt"] == "prompt-x"
            assert entry["total_inference_steps"] == 2
            assert entry["total_env_steps"] == 3

    def test_zero_num_envs_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_envs"):
            BatchCollectionSession(_RecordingClient(), num_envs=0)

    def test_record_step_wrong_length_rejected(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=3)
        session.start_episode("t", 0, "p")
        with pytest.raises(ValueError, match="reward must have shape"):
            session.record_step(0, [0.0, 0.0], [False, False, False])
        with pytest.raises(ValueError, match="done must have shape"):
            session.record_step(0, [0.0, 0.0, 0.0], [False, False])

    def test_start_episode_resets_state(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=2)
        session.start_episode("a", 0, "p-a")
        session.make_collect_metadata(step=0)
        session.record_step(0, [1.0, 1.0], [True, False])

        session.start_episode("b", 5, "p-b")
        meta = session.make_collect_metadata(step=0)
        for entry in meta:
            assert entry["task_name"] == "b"
            assert entry["episode_id"] == 5
            assert entry["prompt"] == "p-b"
            assert entry["inference_step"] == 0
            assert entry["cumulative_reward"] == 0.0
            assert entry["success_so_far"] is False

    def test_reward_since_last_inference_resets_per_env(self) -> None:
        session = BatchCollectionSession(_RecordingClient(), num_envs=2)
        session.start_episode("t", 0, "p")
        session.make_collect_metadata(step=0)  # snapshot 0
        session.record_step(0, [0.3, 0.1], [False, False])
        session.record_step(1, [0.2, 0.4], [False, False])
        m1 = session.make_collect_metadata(step=2)
        assert m1[0]["reward_since_last_inference"] == pytest.approx(0.5)
        assert m1[1]["reward_since_last_inference"] == pytest.approx(0.5)
        # No new env steps -> next snapshot's delta is 0.
        m2 = session.make_collect_metadata(step=3)
        assert m2[0]["reward_since_last_inference"] == pytest.approx(0.0)
        assert m2[1]["reward_since_last_inference"] == pytest.approx(0.0)

    def test_record_step_accepts_numpy_arrays(self) -> None:
        """Vectorized envs return reward / done as numpy arrays. The session
        must handle them without manual list conversion."""
        session = BatchCollectionSession(_RecordingClient(), num_envs=3)
        session.start_episode("t", 0, "p")
        session.record_step(
            0,
            np.array([0.5, 1.0, 0.0], dtype=np.float32),
            np.array([False, True, False]),
        )
        meta = session.make_collect_metadata(step=1)
        assert meta[0]["cumulative_reward"] == pytest.approx(0.5)
        assert meta[1]["cumulative_reward"] == pytest.approx(1.0)
        assert meta[1]["success_so_far"] is True
