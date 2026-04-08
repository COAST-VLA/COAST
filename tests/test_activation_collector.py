"""Unit tests for src/openpi/serving/activation_collector.py and the matching
client-side helper at examples/libero_env/collection_session.py.

These tests do not need a GPU or a real checkpoint. They cover:
- save_step_activations / save_episode_files write the expected on-disk schema
- env_id slicing on the batch dim is correct
- CollectingPolicy.infer dispatching:
    - rejects requests without magic keys
    - rejects requests with both magic keys
    - __collect__: calls infer_with_intermediates, saves activations, returns actions
    - __finalize_episode__: writes episode files, returns ack
- _batch_single_example handles 1-D state and adds the leading batch dim
- CollectionSession state transitions (start_episode, make_collect_metadata,
  record_step, finalize_episode payload shape)

CollectionSession lives under examples/libero_env/ which is excluded from the
main workspace, but it has zero libero/robosuite imports, so we load it via
importlib for testing.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from openpi.serving.activation_collector import CollectingPolicy
from openpi.serving.activation_collector import save_episode_files
from openpi.serving.activation_collector import save_step_activations


def _load_collection_session_module():
    """Load examples/libero_env/collection_session.py without going through the
    libero workspace (which has its own venv)."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / "examples" / "libero_env" / "collection_session.py"
    spec = importlib.util.spec_from_file_location("_libero_collection_session", src_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


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
    )
    return stub, wrapper


class TestCollectingPolicyMetadata:
    def test_metadata_merges_underlying_and_adds_collection_fields(self, policy_setup, tmp_path) -> None:
        _, wrapper = policy_setup
        meta = wrapper.metadata
        assert meta["underlying"] == "stub"
        assert meta["policy_dir"] == "/fake/policy/dir"
        assert meta["config_name"] == "fake_config"
        assert meta["collection_mode"] == "v1"
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


# ----------------------------------------------- CollectionSession tests


class _RecordingClient:
    """Records every payload that .infer is called with. Returns a canned
    action chunk so callers can iterate further if they want."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, payload: dict) -> dict:
        # Deep-ish copy so subsequent mutations by the caller don't change history.
        self.calls.append(json.loads(json.dumps(payload, default=str)))
        return {"actions": np.zeros((10, 7), dtype=np.float32), "ack": True}


@pytest.fixture
def collection_session_module():
    return _load_collection_session_module()


class TestCollectionSessionStartEpisode:
    def test_resets_state_for_new_episode(self, collection_session_module) -> None:
        client = _RecordingClient()
        session = collection_session_module.CollectionSession(client)

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

    def test_env_id_can_be_overridden(self, collection_session_module) -> None:
        session = collection_session_module.CollectionSession(_RecordingClient())
        session.start_episode("t", task_id=0, episode_id=0, prompt="p", env_id=7)
        meta = session.make_collect_metadata(step=0)
        assert meta["env_id"] == 7


class TestMakeCollectMetadata:
    def test_inference_step_increments_per_call(self, collection_session_module) -> None:
        session = collection_session_module.CollectionSession(_RecordingClient())
        session.start_episode("t", 0, 0, "p")
        m0 = session.make_collect_metadata(step=0)
        m1 = session.make_collect_metadata(step=5)
        m2 = session.make_collect_metadata(step=10)
        assert [m0["inference_step"], m1["inference_step"], m2["inference_step"]] == [0, 1, 2]
        assert [m0["step"], m1["step"], m2["step"]] == [0, 5, 10]

    def test_reward_since_last_inference_resets_after_each_call(self, collection_session_module) -> None:
        session = collection_session_module.CollectionSession(_RecordingClient())
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


class TestRecordStep:
    def test_accumulates_rewards_and_tracks_first_success(self, collection_session_module) -> None:
        client = _RecordingClient()
        session = collection_session_module.CollectionSession(client)
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

    def test_no_success_marks_steps_to_success_as_minus_one(self, collection_session_module) -> None:
        client = _RecordingClient()
        session = collection_session_module.CollectionSession(client)
        session.start_episode("t", 0, 0, "p")
        session.record_step(0, 0.1, done=False)
        session.record_step(1, 0.2, done=False)
        session.finalize_episode()
        finalize = client.calls[-1]["__finalize_episode__"]
        assert finalize["episode_success"] is False
        assert finalize["steps_to_success"] == -1
        assert finalize["total_reward"] == pytest.approx(0.3)


class TestFinalizeEpisode:
    def test_payload_shape_matches_server_expectations(self, collection_session_module) -> None:
        client = _RecordingClient()
        session = collection_session_module.CollectionSession(client)
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
        # Required fields.
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
        session_module,
        task_name: str,
        episode_id: int,
        prompt: str,
        steps_per_inference: int,
        num_inferences: int,
        success_at_step: int | None,
    ) -> None:
        """Mimic eval_task: alternate make_collect_metadata + record_step + finalize_episode."""
        session = session_module.CollectionSession(wrapper)
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

    def test_round_trip_writes_schema_compliant_files(
        self, policy_setup, collection_session_module, tmp_path: pathlib.Path
    ) -> None:
        stub, wrapper = policy_setup
        self._drive_rollout(
            wrapper=wrapper,
            session_module=collection_session_module,
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

    def test_failure_episode_writes_steps_to_success_minus_one(
        self, policy_setup, collection_session_module, tmp_path: pathlib.Path
    ) -> None:
        _, wrapper = policy_setup
        self._drive_rollout(
            wrapper=wrapper,
            session_module=collection_session_module,
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
