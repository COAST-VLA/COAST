"""Unit tests for CollectionSession.set_episode_result.

These tests verify the new set_episode_result method that allows real-robot
environments (droid, future hardware) to set episode success from a post-hoc
human label without faking env steps.
"""

from __future__ import annotations

from openpi_client.collection_session import CollectionSession
import pytest


class _RecordingClient:
    def __init__(self):
        self.calls = []

    def infer(self, obs):
        self.calls.append(obs)
        return {"ack": True}


class TestSetEpisodeResult:
    def test_set_episode_result_true(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task", task_id=0, episode_id=0, prompt="p")
        session.set_episode_result(True)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is True

    def test_set_episode_result_false(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task", task_id=0, episode_id=0, prompt="p")
        session.set_episode_result(False)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is False

    def test_set_episode_result_does_not_add_fake_steps(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task", task_id=0, episode_id=0, prompt="p")
        for i in range(5):
            session.record_step(i, reward=0.0, done=False)
        session.set_episode_result(True)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert len(finalize["per_step_reward"]) == 5
        assert len(finalize["per_step_success"]) == 5

    def test_set_episode_result_with_custom_reward(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task", task_id=0, episode_id=0, prompt="p")
        session.set_episode_result(True, total_reward=42.0)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["total_reward"] == pytest.approx(42.0)

    def test_set_episode_result_sets_steps_to_success(self) -> None:
        client = _RecordingClient()
        session = CollectionSession(client)
        session.start_episode("task", task_id=0, episode_id=0, prompt="p")
        for i in range(3):
            session.record_step(i, reward=0.0, done=False)
        session.set_episode_result(True)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        # steps_to_success is the last valid 0-based index into per_step_reward (len - 1 = 2)
        assert finalize["steps_to_success"] == 2
