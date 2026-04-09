"""
Tests for V2 collected activation data from examples/metaworld/collect_activations_v2.py.

Run after collecting activations:
    ACTIVATIONS_V2_DIR=activations_v2/5000/reach-v3 uv run pytest tests/test_activations_v2.py -v

Or to test all tasks:
    ACTIVATIONS_V2_DIR=activations_v2/5000 uv run pytest tests/test_activations_v2.py -v -k "not cross_env"
"""

import json
import os
import pathlib

import numpy as np
import pytest

ACTIVATIONS_V2_DIR = os.environ.get("ACTIVATIONS_V2_DIR", "activations_v2/5000/reach-v3")
ACTIVATIONS_V2_BASE = os.environ.get("ACTIVATIONS_V2_BASE", "activations_v2/5000")

# V2 collects denoising steps 0, 4, 9
NUM_COLLECTED_DENOISE_STEPS = 3
# V2 collects residual at layers 5, 11
NUM_RESIDUAL_LAYERS = 2
# V2 collects MLP hidden at layer 11 only
NUM_MLP_LAYERS = 1
# V2 collects attention at layers 5, 11
NUM_ATTENTION_LAYERS = 2
# Action Expert has 8 attention heads
NUM_ATTENTION_HEADS = 8
# Hidden dim
HIDDEN_DIM = 1024
# MLP dim
MLP_DIM = 4096
# Action dim
ACTION_DIM = 32
# 18 expert layers total
NUM_EXPERT_LAYERS = 18
# 2 norms per layer (attn + mlp)
NUM_NORMS_PER_LAYER = 2
# NOTE: action_horizon (a.k.a. number of action tokens) is NOT a constant here.
# It varies by training config (32 for pi05_metaworld, 10 for pi05_libero, etc.)
# and is read from the policy config registry via the `expected_action_horizon`
# fixture below. Hardcoding it would either break libero collection coverage or
# create a self-referential check (data discovers its own contract).


@pytest.fixture
def act_dir():
    d = pathlib.Path(ACTIVATIONS_V2_DIR)
    if not d.exists():
        pytest.skip(f"Activations V2 directory not found: {d}")
    return d


@pytest.fixture
def base_dir():
    d = pathlib.Path(ACTIVATIONS_V2_BASE)
    if not d.exists():
        pytest.skip(f"Activations V2 base directory not found: {d}")
    return d


@pytest.fixture
def episode_dirs(act_dir):
    dirs = sorted(act_dir.glob("episode_*"))
    if not dirs:
        pytest.skip("No episode directories found")
    return dirs


@pytest.fixture
def first_episode(episode_dirs):
    return episode_dirs[0]


@pytest.fixture
def step_dirs(first_episode):
    dirs = sorted(first_episode.glob("step_*"))
    assert len(dirs) > 0, "No step directories found"
    return dirs


@pytest.fixture
def first_step(step_dirs):
    return step_dirs[0]


@pytest.fixture
def expected_action_horizon(first_episode):
    """Action horizon from the policy config registered for this episode's
    `config_name`. Source of truth is `openpi.training.config`, NOT the
    activation file under test — deriving the horizon from the data would
    be circular and would silently pass if collection wrote a globally
    wrong horizon (e.g. policy loaded with the wrong config).
    """
    from openpi.training import config as _config

    with open(first_episode / "metadata.json") as f:
        episode_meta = json.load(f)
    config_name = episode_meta.get("config_name")
    if not config_name:
        pytest.skip(f"episode metadata.json has no 'config_name' field at {first_episode}")
    try:
        train_config = _config.get_config(config_name)
    except (ValueError, KeyError) as exc:
        pytest.skip(f"config_name {config_name!r} not registered in openpi.training.config: {exc}")
    if not hasattr(train_config.model, "action_horizon"):
        pytest.skip(f"config {config_name!r} model has no action_horizon attribute")
    return int(train_config.model.action_horizon)


# --- Global file tests ---


class TestGlobalFiles:
    def test_adarms_cond_global_exists(self, base_dir):
        path = base_dir / "adarms_cond_global.npz"
        assert path.exists(), f"Missing global adaRMS conditioning: {path}"

    def test_adarms_cond_global_shape(self, base_dir):
        data = np.load(base_dir / "adarms_cond_global.npz")
        assert "adarms_cond_global" in data
        arr = data["adarms_cond_global"]
        assert arr.ndim == 2, f"Expected 2D, got {arr.ndim}D"
        assert arr.shape[1] == HIDDEN_DIM, f"Expected dim {HIDDEN_DIM}, got {arr.shape[1]}"
        assert arr.dtype == np.float32

    def test_adarms_cond_global_finite(self, base_dir):
        data = np.load(base_dir / "adarms_cond_global.npz")
        assert np.all(np.isfinite(data["adarms_cond_global"]))


# --- Directory structure tests ---


class TestDirectoryStructure:
    def test_episode_dirs_exist(self, episode_dirs):
        assert len(episode_dirs) >= 1

    def test_episode_metadata_exists(self, first_episode):
        assert (first_episode / "metadata.json").exists()

    def test_episode_rewards_exists(self, first_episode):
        assert (first_episode / "rewards.npz").exists()

    def test_step_dirs_exist(self, step_dirs):
        assert len(step_dirs) >= 1

    def test_step_has_all_v2_files(self, first_step):
        expected = [
            "denoising.npz",
            "suffix_residual.npz",
            "suffix_mlp_hidden.npz",
            "attention_weights.npz",
            "adarms_gates.npz",
            "metadata.json",
        ]
        for fname in expected:
            assert (first_step / fname).exists(), f"Missing {fname} in {first_step}"

    def test_no_adarms_cond_per_step(self, first_step):
        """V2 should NOT have per-step adarms_cond (saved globally instead)."""
        assert not (first_step / "adarms_cond.npz").exists(), "V2 should not have per-step adarms_cond.npz"


# --- Episode metadata tests ---


class TestEpisodeMetadata:
    def test_required_fields(self, first_episode):
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        required = [
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
            "collection_version",
            "collected_denoise_steps",
            "collected_residual_layers",
            "collected_mlp_layers",
            "collected_attention_layers",
        ]
        for field in required:
            assert field in meta, f"Missing field: {field}"

    def test_collection_version_is_v2(self, first_episode):
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        assert meta["collection_version"] == "v2"

    def test_collected_layers_match_expected(self, first_episode):
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        assert meta["collected_denoise_steps"] == [0, 4, 9]
        assert meta["collected_residual_layers"] == [5, 11]
        assert meta["collected_mlp_layers"] == [11]
        assert meta["collected_attention_layers"] == [5, 11]

    def test_success_implies_steps_to_success(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            if meta["episode_success"]:
                assert meta["steps_to_success"] >= 0
            else:
                assert meta["steps_to_success"] == -1

    def test_rewards_npz_structure(self, first_episode):
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        data = np.load(first_episode / "rewards.npz")
        assert "per_step_reward" in data
        assert "cumulative_reward" in data
        assert "success_at_step" in data
        assert len(data["per_step_reward"]) == meta["total_env_steps"]


# --- Step metadata tests ---


class TestStepMetadata:
    def test_required_fields(self, first_step):
        with open(first_step / "metadata.json") as f:
            meta = json.load(f)
        required = [
            "task_name",
            "episode_id",
            "env_id",
            "step",
            "inference_step",
            "prompt",
            "cumulative_reward",
            "success_so_far",
            "reward_since_last_inference",
            "proprio_state",
            "object_positions",
            "predicted_actions",
        ]
        for field in required:
            assert field in meta, f"Missing field: {field}"

    def test_proprio_state_format(self, first_step):
        with open(first_step / "metadata.json") as f:
            meta = json.load(f)
        proprio = meta["proprio_state"]
        assert isinstance(proprio, list), f"proprio_state should be list, got {type(proprio)}"
        assert len(proprio) == 4, f"proprio_state should have 4 elements (xyz + gripper), got {len(proprio)}"
        for v in proprio:
            assert isinstance(v, int | float), f"proprio_state values should be numeric, got {type(v)}"

    def test_object_positions_format(self, first_step):
        with open(first_step / "metadata.json") as f:
            meta = json.load(f)
        obj_pos = meta["object_positions"]
        assert isinstance(obj_pos, list), f"object_positions should be list, got {type(obj_pos)}"
        assert len(obj_pos) >= 3, f"object_positions should have >= 3 elements, got {len(obj_pos)}"

    def test_predicted_actions_format(self, first_step):
        with open(first_step / "metadata.json") as f:
            meta = json.load(f)
        actions = meta["predicted_actions"]
        assert isinstance(actions, list), f"predicted_actions should be list, got {type(actions)}"
        assert len(actions) == 4, f"predicted_actions should have 4 elements, got {len(actions)}"

    def test_cumulative_reward_non_decreasing(self, step_dirs):
        rewards = []
        for step_dir in step_dirs:
            with open(step_dir / "metadata.json") as f:
                meta = json.load(f)
            rewards.append(meta["cumulative_reward"])
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i - 1] - 1e-6, (
                f"Cumulative reward decreased: step {i - 1}={rewards[i - 1]}, step {i}={rewards[i]}"
            )


# --- Activation shape tests ---


class TestActivationShapes:
    def test_denoising_shapes(self, first_step, expected_action_horizon):
        data = np.load(first_step / "denoising.npz")
        assert data["all_x_t"].shape == (NUM_COLLECTED_DENOISE_STEPS, expected_action_horizon, ACTION_DIM)
        assert data["all_v_t"].shape == (NUM_COLLECTED_DENOISE_STEPS, expected_action_horizon, ACTION_DIM)

    def test_suffix_residual_shape(self, first_step, expected_action_horizon):
        data = np.load(first_step / "suffix_residual.npz")
        arr = data["all_suffix_residual"]
        assert arr.shape == (NUM_COLLECTED_DENOISE_STEPS, NUM_RESIDUAL_LAYERS, expected_action_horizon, HIDDEN_DIM)

    def test_suffix_mlp_hidden_shape(self, first_step, expected_action_horizon):
        data = np.load(first_step / "suffix_mlp_hidden.npz")
        arr = data["all_suffix_mlp_hidden"]
        assert arr.shape == (NUM_COLLECTED_DENOISE_STEPS, NUM_MLP_LAYERS, expected_action_horizon, MLP_DIM)

    def test_attention_weights_shape(self, first_step, expected_action_horizon):
        data = np.load(first_step / "attention_weights.npz")
        arr = data["all_attention_weights"]
        assert arr.ndim == 5, f"Expected 5D, got {arr.ndim}D"
        assert arr.shape[0] == NUM_COLLECTED_DENOISE_STEPS
        assert arr.shape[1] == NUM_ATTENTION_LAYERS
        assert arr.shape[2] == NUM_ATTENTION_HEADS
        assert arr.shape[3] == expected_action_horizon
        # Last dim is prefix_seq_len (varies, but should be > 0)
        assert arr.shape[4] > 0, "Attention weights last dim (prefix_len) should be > 0"

    def test_adarms_gates_shape(self, first_step):
        data = np.load(first_step / "adarms_gates.npz")
        arr = data["all_adarms_gates"]
        assert arr.ndim == 5, f"Expected 5D, got {arr.ndim}D"
        assert arr.shape[0] == NUM_COLLECTED_DENOISE_STEPS
        assert arr.shape[1] == NUM_EXPERT_LAYERS
        assert arr.shape[2] == NUM_NORMS_PER_LAYER
        # dim 3 is batch (1 per env after slicing)
        assert arr.shape[4] == HIDDEN_DIM

    def test_all_float32(self, first_step):
        for fname in [
            "denoising.npz",
            "suffix_residual.npz",
            "suffix_mlp_hidden.npz",
            "attention_weights.npz",
            "adarms_gates.npz",
        ]:
            data = np.load(first_step / fname)
            for key in data:
                assert data[key].dtype == np.float32, f"{fname}/{key} is {data[key].dtype}"

    def test_no_nan_inf(self, first_step):
        for fname in [
            "denoising.npz",
            "suffix_residual.npz",
            "suffix_mlp_hidden.npz",
            "attention_weights.npz",
            "adarms_gates.npz",
        ]:
            data = np.load(first_step / fname)
            for key in data:
                assert np.all(np.isfinite(data[key])), f"{fname}/{key} has NaN/Inf"


# --- Sanity tests ---


class TestSanityChecks:
    def test_x_t_changes_between_collected_steps(self, first_step):
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        for i in range(1, x_t.shape[0]):
            assert not np.allclose(x_t[i], x_t[i - 1]), f"x_t unchanged between collected steps {i - 1} and {i}"

    def test_x_t_variance_decreases(self, first_step):
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        first_var = np.var(x_t[0])
        last_var = np.var(x_t[-1])
        assert last_var < first_var, f"x_t variance did not decrease: first={first_var:.4f}, last={last_var:.4f}"

    def test_suffix_residual_nonzero_and_varies(self, first_step):
        data = np.load(first_step / "suffix_residual.npz")
        res = data["all_suffix_residual"]
        assert np.any(res != 0), "Suffix residual is all zeros"
        assert not np.allclose(res[0], res[-1]), "Suffix residual same at first and last collected step"

    def test_suffix_mlp_hidden_nonzero(self, first_step):
        data = np.load(first_step / "suffix_mlp_hidden.npz")
        mlp = data["all_suffix_mlp_hidden"]
        assert np.any(mlp != 0), "Suffix MLP hidden is all zeros"

    def test_attention_weights_are_probabilities(self, first_step):
        """Attention weights should sum to ~1 along the key dimension."""
        data = np.load(first_step / "attention_weights.npz")
        attn = data["all_attention_weights"]
        # Sum along last axis (key positions) for each query token
        sums = attn.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=5e-3, err_msg="Attention weights don't sum to 1")

    def test_attention_weights_nonnegative(self, first_step):
        data = np.load(first_step / "attention_weights.npz")
        attn = data["all_attention_weights"]
        assert np.all(attn >= -1e-6), "Attention weights have negative values"

    def test_adarms_gates_nonzero(self, first_step):
        data = np.load(first_step / "adarms_gates.npz")
        gates = data["all_adarms_gates"]
        assert np.any(gates != 0), "adaRMS gates are all zeros"

    def test_cross_env_different_activations(self, act_dir):
        """Different envs should have different activations (different random seeds)."""
        episode_dirs = sorted(act_dir.glob("episode_*"))
        if len(episode_dirs) < 2:
            pytest.skip("Need at least 2 episodes/envs to compare")
        step0_dirs = sorted(episode_dirs[0].glob("step_*"))
        step1_dirs = sorted(episode_dirs[1].glob("step_*"))
        if not step0_dirs or not step1_dirs:
            pytest.skip("No step dirs in one of the episodes")
        data0 = np.load(step0_dirs[0] / "denoising.npz")
        data1 = np.load(step1_dirs[0] / "denoising.npz")
        assert not np.allclose(data0["all_x_t"], data1["all_x_t"]), "Different envs have identical x_t"
