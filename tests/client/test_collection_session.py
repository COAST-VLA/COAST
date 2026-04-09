"""Unit tests for openpi_client.collection_session.CollectionSession.

These tests run in the main openpi venv against the editable-installed
openpi-client package — no libero/robocasa/droid venv needed. They cover the
state-tracking and protocol-payload-shaping behavior of the helper, with a
stub policy that just records every infer() call so we can inspect the
payloads it would have sent over the wire.
"""

from __future__ import annotations

import json

import numpy as np
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


# --------------------------------------------------- set_episode_outcome tests


class TestSetEpisodeOutcome:
    """Out-of-band terminal outcome injection. Used by clients like droid where
    `env.step()` returns no per-step reward/done flag and success is determined
    by a human grade after the rollout loop ends."""

    def test_droid_pattern_zero_step_rewards_then_success(self) -> None:
        """Mimic droid's flow: many record_step calls with reward=0, done=False,
        then a single set_episode_outcome(success=True, total_reward=1.0) before
        finalize. The final payload should reflect the user's grade, not the
        all-zeros per-step recording."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("pick-up-the-block", 0, 0, "pick up the block")
        for t in range(5):
            session.record_step(t, 0.0, done=False)
        session.set_episode_outcome(success=True, total_reward=1.0)
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is True
        assert finalize["total_reward"] == pytest.approx(1.0)
        assert finalize["steps_to_success"] == 4  # last step
        assert finalize["total_env_steps"] == 5
        # per_step_reward sum must equal total_reward (test_activations.py
        # asserts this invariant via test_rewards_cumulative_matches_total).
        assert sum(finalize["per_step_reward"]) == pytest.approx(1.0)
        assert finalize["per_step_reward"] == [0.0, 0.0, 0.0, 0.0, 1.0]
        # Last step's success flag flipped to True.
        assert finalize["per_step_success"] == [False, False, False, False, True]

    def test_droid_pattern_zero_step_rewards_then_failure(self) -> None:
        """Same pattern but the user grades the rollout as a failure (success=False).
        steps_to_success stays -1, no per_step_success flag is set, total_reward is 0."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        for t in range(3):
            session.record_step(t, 0.0, done=False)
        session.set_episode_outcome(success=False, total_reward=0.0)
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is False
        assert finalize["total_reward"] == pytest.approx(0.0)
        assert finalize["steps_to_success"] == -1
        assert finalize["per_step_reward"] == [0.0, 0.0, 0.0]
        assert finalize["per_step_success"] == [False, False, False]

    def test_partial_success_score(self) -> None:
        """Droid's prompt accepts a numeric grade in [0, 1] (not just binary).
        The session should record the float total_reward and flag success as
        whatever the caller passes for the bool argument."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        for t in range(4):
            session.record_step(t, 0.0, done=False)
        # User entered 75% — caller decides whether to consider that "success".
        session.set_episode_outcome(success=True, total_reward=0.75)
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is True
        assert finalize["total_reward"] == pytest.approx(0.75)
        assert finalize["per_step_reward"][-1] == pytest.approx(0.75)
        assert sum(finalize["per_step_reward"]) == pytest.approx(0.75)

    def test_overrides_existing_per_step_rewards_to_match_total(self) -> None:
        """If record_step recorded non-zero rewards AND set_episode_outcome is
        also called with a different total, the last entry of per_step_reward
        is adjusted (delta added) so the sum matches the new total. This is
        the documented contract — preserves test_activations.py's
        cumulative_reward[-1] == total_reward invariant."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        session.record_step(0, 0.1, done=False)
        session.record_step(1, 0.2, done=False)
        session.record_step(2, 0.3, done=False)  # sum so far: 0.6
        session.set_episode_outcome(success=True, total_reward=1.0)  # delta = +0.4
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["total_reward"] == pytest.approx(1.0)
        # Last reward bumped from 0.3 to 0.7 to make the sum match.
        assert finalize["per_step_reward"][-1] == pytest.approx(0.7)
        assert sum(finalize["per_step_reward"]) == pytest.approx(1.0)

    def test_idempotent_when_called_twice_with_different_values(self) -> None:
        """Calling set_episode_outcome twice should leave the session in the
        state of the second call, not double-apply the delta."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        for t in range(3):
            session.record_step(t, 0.0, done=False)
        session.set_episode_outcome(success=True, total_reward=1.0)
        session.set_episode_outcome(success=False, total_reward=0.5)  # downgrade
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is False
        assert finalize["total_reward"] == pytest.approx(0.5)
        assert finalize["steps_to_success"] == -1
        assert sum(finalize["per_step_reward"]) == pytest.approx(0.5)

    def test_empty_per_step_arrays_only_sets_scalars(self) -> None:
        """Edge case: set_episode_outcome called before any record_step.
        per_step_reward is empty, so we can only set the episode-level scalars."""
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        session.set_episode_outcome(success=True, total_reward=1.0)
        session.finalize_episode()

        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is True
        assert finalize["total_reward"] == pytest.approx(1.0)
        assert finalize["steps_to_success"] == -1
        assert finalize["per_step_reward"] == []
        assert finalize["per_step_success"] == []
        assert finalize["total_env_steps"] == 0


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
