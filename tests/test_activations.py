"""
Tests for collected activation data from examples/metaworld/collect_activations.py.

Run after collecting activations:
    ACTIVATIONS_DIR=activations/5000/reach-v3 uv run pytest tests/test_activations.py -v
"""

import json
import os
import pathlib

import numpy as np
import pytest

ACTIVATIONS_DIR = os.environ.get("ACTIVATIONS_DIR", "activations/5000/reach-v3")


@pytest.fixture
def act_dir():
    d = pathlib.Path(ACTIVATIONS_DIR)
    if not d.exists():
        pytest.skip(f"Activations directory not found: {d}")
    return d


@pytest.fixture
def episode_dirs(act_dir):
    dirs = sorted(act_dir.glob("episode_*"))
    assert len(dirs) > 0, "No episode directories found"
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


# --- Structure tests ---


class TestDirectoryStructure:
    def test_episode_dirs_exist(self, episode_dirs):
        assert len(episode_dirs) >= 1

    def test_episode_metadata_exists(self, first_episode):
        assert (first_episode / "metadata.json").exists()

    def test_episode_rewards_exists(self, first_episode):
        assert (first_episode / "rewards.npz").exists()

    def test_step_dirs_exist(self, step_dirs):
        assert len(step_dirs) >= 1

    def test_step_has_all_files(self, first_step):
        expected = ["denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz", "metadata.json"]
        for fname in expected:
            assert (first_step / fname).exists(), f"Missing {fname} in {first_step}"


# --- Metadata tests ---


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
        ]
        for field in required:
            assert field in meta, f"Missing field: {field}"

    def test_success_implies_steps_to_success(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            if meta["episode_success"]:
                assert meta["steps_to_success"] >= 0
            else:
                assert meta["steps_to_success"] == -1

    def test_rewards_npz_length_matches(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            data = np.load(ep_dir / "rewards.npz")
            assert len(data["per_step_reward"]) == meta["total_env_steps"]
            assert len(data["cumulative_reward"]) == meta["total_env_steps"]
            assert len(data["success_at_step"]) == meta["total_env_steps"]

    def test_rewards_cumulative_matches_total(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            data = np.load(ep_dir / "rewards.npz")
            np.testing.assert_allclose(
                data["cumulative_reward"][-1],
                meta["total_reward"],
                rtol=1e-4,
            )


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
        ]
        for field in required:
            assert field in meta, f"Missing field: {field}"

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
#
# Some dims are fixed by the model architecture (10 denoising steps,
# 4 sampled layers, 1024 hidden, 4096 mlp hidden, 32 action_dim). The
# `action_horizon` dim varies by training config (10 for pi05_libero, 32
# for pi05_metaworld), so we look it up from the policy config registry
# keyed by `config_name` in the episode metadata.json. This is the
# *contract* — deriving it from the data file under test would be circular
# and would silently pass if collection wrote a globally-wrong horizon
# (e.g. policy loaded with the wrong config, or a writer slice bug that
# affects all files in the same way).


@pytest.fixture
def expected_action_horizon(first_episode):
    """Action horizon from the policy config registered for this episode's
    `config_name`. Source of truth is `openpi.training.config`, NOT the
    activation file under test.
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


class TestActivationShapes:
    def test_denoising_shapes(self, first_step, expected_action_horizon):
        data = np.load(first_step / "denoising.npz")
        # (denoising_steps, action_horizon, action_dim)
        assert data["all_x_t"].shape == (10, expected_action_horizon, 32)
        # all_v_t must match all_x_t exactly (paired flow-matching outputs)
        assert data["all_v_t"].shape == data["all_x_t"].shape

    def test_adarms_cond_shape(self, first_step):
        data = np.load(first_step / "adarms_cond.npz")
        assert data["all_adarms_cond"].shape == (10, 1024)

    def test_suffix_residual_shape(self, first_step, expected_action_horizon):
        data = np.load(first_step / "suffix_residual.npz")
        # (denoising_steps, num_layers, action_horizon, hidden_dim)
        assert data["all_suffix_residual"].shape == (10, 4, expected_action_horizon, 1024)

    def test_suffix_mlp_hidden_shape(self, first_step, expected_action_horizon):
        data = np.load(first_step / "suffix_mlp_hidden.npz")
        # (denoising_steps, num_layers, action_horizon, mlp_hidden_dim)
        assert data["all_suffix_mlp_hidden"].shape == (10, 4, expected_action_horizon, 4096)

    def test_all_float32(self, first_step):
        for fname in ["denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz"]:
            data = np.load(first_step / fname)
            for key in data:
                assert data[key].dtype == np.float32, f"{fname}/{key} is {data[key].dtype}"

    def test_no_nan_inf(self, first_step):
        for fname in ["denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz"]:
            data = np.load(first_step / fname)
            for key in data:
                assert np.all(np.isfinite(data[key])), f"{fname}/{key} has NaN/Inf"


# --- Sanity tests ---


class TestSanityChecks:
    def test_x_t_changes_between_steps(self, first_step):
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        for i in range(1, x_t.shape[0]):
            assert not np.allclose(x_t[i], x_t[i - 1]), f"x_t unchanged between steps {i - 1} and {i}"

    def test_x_t_variance_decreases(self, first_step):
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        # Variance of x_t should generally decrease as noise is removed
        first_var = np.var(x_t[0])
        last_var = np.var(x_t[-1])
        assert last_var < first_var, f"x_t variance did not decrease: first={first_var:.4f}, last={last_var:.4f}"

    def test_adarms_cond_varies_across_steps(self, first_step):
        data = np.load(first_step / "adarms_cond.npz")
        cond = data["all_adarms_cond"]
        for i in range(1, cond.shape[0]):
            assert not np.allclose(cond[i], cond[i - 1]), f"adaRMS cond unchanged between steps {i - 1} and {i}"

    def test_suffix_residual_nonzero_and_varies(self, first_step):
        data = np.load(first_step / "suffix_residual.npz")
        res = data["all_suffix_residual"]
        assert np.any(res != 0), "Suffix residual is all zeros"
        assert not np.allclose(res[0], res[-1]), "Suffix residual same at first and last denoising step"

    def test_suffix_mlp_hidden_nonzero_and_varies(self, first_step):
        data = np.load(first_step / "suffix_mlp_hidden.npz")
        mlp = data["all_suffix_mlp_hidden"]
        assert np.any(mlp != 0), "Suffix MLP hidden is all zeros"
        assert not np.allclose(mlp[0], mlp[-1]), "Suffix MLP hidden same at first and last denoising step"

    def test_cross_env_different_activations(self, act_dir):
        """Different envs should have different activations (different random seeds)."""
        episode_dirs = sorted(act_dir.glob("episode_*"))
        if len(episode_dirs) < 2:
            pytest.skip("Need at least 2 episodes/envs to compare")
        # Compare first step of first two envs
        step0_dirs = sorted(episode_dirs[0].glob("step_*"))
        step1_dirs = sorted(episode_dirs[1].glob("step_*"))
        if not step0_dirs or not step1_dirs:
            pytest.skip("No step dirs in one of the episodes")
        data0 = np.load(step0_dirs[0] / "denoising.npz")
        data1 = np.load(step1_dirs[0] / "denoising.npz")
        assert not np.allclose(data0["all_x_t"], data1["all_x_t"]), "Different envs have identical x_t"
