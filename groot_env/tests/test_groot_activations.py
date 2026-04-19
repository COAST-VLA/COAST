"""Tests for collected activation data from `serve.py --collect-activations` (GR00T N1.5).

GR00T-N1.5 analog of `tests/test_activations.py`. Same structure; schema is
aligned with pi0's `sample_actions_with_intermediates` one-for-one where the
architecture allows:

pi0 schema                -> N1.5 schema
----------------------------------------
denoising.npz             -> denoising.npz           (same: all_x_t + all_v_t,
                                                      num_denoising_steps entries each)
adarms_cond.npz           -> backbone_cond.npz        (VL backbone output. N1.5
                                                      cross-attends to a VL sequence
                                                      computed once per infer, so the
                                                      shape is (seq, hidden) — NOT
                                                      pi0's (steps, hidden) pooled
                                                      per-step conditioning.)
suffix_residual.npz       -> dit_hidden_states.npz    (DiT per-layer residual stream,
                                                      shape (steps, layers, seq, hidden))
suffix_mlp_hidden.npz     -> dit_mlp_hidden.npz       (DiT per-layer MLP expanded
                                                      activation, hooked on each
                                                      block.ff.net[2] input — analog
                                                      of pi0's hook on
                                                      mlp.down_proj input.)

Run after collecting activations:
    ACTIVATIONS_DIR=<root>/checkpoint-120000/OpenDrawer \\
        uv run pytest tests/test_groot_activations.py -v

For a bulk walk across the entire dataset root (all tasks / all episodes),
see `tests/test_full_activations.py`.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

ACTIVATIONS_DIR = os.environ.get(
    "ACTIVATIONS_DIR", "/tmp/groot_acts_test/checkpoint-120000/OpenDrawer"
)

# GR00T N1.5's robocasa multitask checkpoint uses 4 denoising steps (default in
# `serve.py`) and a DiT with 16 transformer blocks. Action horizon is 16 and
# action_dim is padded to 32 (GR00TTransform's max_action_dim).
EXPECTED_DENOISING_STEPS = int(os.environ.get("GROOT_DENOISING_STEPS", "4"))
EXPECTED_DIT_LAYERS = int(os.environ.get("GROOT_DIT_LAYERS", "16"))
EXPECTED_ACTION_HORIZON = int(os.environ.get("GROOT_ACTION_HORIZON", "16"))
EXPECTED_ACTION_DIM = int(os.environ.get("GROOT_ACTION_DIM", "32"))


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
        expected = [
            "denoising.npz",
            "backbone_cond.npz",
            "dit_hidden_states.npz",
            "dit_mlp_hidden.npz",
            "metadata.json",
        ]
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
# N1.5 denoising shapes (matches pi0's sample_actions_with_intermediates schema
# one-for-one where the architecture allows):
#   all_x_t: (num_denoising_steps, action_horizon, action_dim)
#   all_v_t: (num_denoising_steps, action_horizon, action_dim)
#   backbone_features: (vl_seq_len, backbone_hidden)                           (one per infer — N1.5
#                                                                               cross-attends to VL seq
#                                                                               once; architecturally
#                                                                               distinct from pi0's pooled
#                                                                               per-step adarms_cond)
#   all_dit_hidden_states: (num_denoising_steps, num_dit_layers, sa_seq_len, dit_hidden)
#   all_dit_mlp_hidden:    (num_denoising_steps, num_dit_layers, sa_seq_len, ff_inner_dim)
#
# vl_seq_len, sa_seq_len, and ff_inner_dim are architecture-dependent but fixed
# per-config, so we probe the first step and require consistency across all
# steps instead of hardcoding.


class TestActivationShapes:
    def test_denoising_shapes(self, first_step):
        data = np.load(first_step / "denoising.npz")
        # (denoising_steps, action_horizon, action_dim) — matches pi0's schema.
        assert data["all_x_t"].shape == (
            EXPECTED_DENOISING_STEPS,
            EXPECTED_ACTION_HORIZON,
            EXPECTED_ACTION_DIM,
        )
        assert data["all_v_t"].shape == (
            EXPECTED_DENOISING_STEPS,
            EXPECTED_ACTION_HORIZON,
            EXPECTED_ACTION_DIM,
        )

    def test_backbone_cond_shape(self, first_step):
        data = np.load(first_step / "backbone_cond.npz")
        arr = data["backbone_features"]
        assert arr.ndim == 2, f"backbone_features should be 2-D, got {arr.shape}"
        # Hidden dim of N1.5's Eagle 2.5 VLM.
        assert arr.shape[-1] in (1536, 2048), (
            f"unexpected backbone hidden dim {arr.shape[-1]}"
        )

    def test_dit_hidden_states_shape(self, first_step):
        data = np.load(first_step / "dit_hidden_states.npz")
        arr = data["all_dit_hidden_states"]
        # (denoising_steps, num_dit_layers, sa_seq_len, dit_hidden) — pi0 analog: suffix_residual.
        assert arr.ndim == 4, f"all_dit_hidden_states should be 4-D, got {arr.shape}"
        assert arr.shape[0] == EXPECTED_DENOISING_STEPS
        assert arr.shape[1] == EXPECTED_DIT_LAYERS, (
            f"expected {EXPECTED_DIT_LAYERS} entries on layer axis, got {arr.shape[1]}"
        )

    def test_dit_mlp_hidden_shape(self, first_step):
        data = np.load(first_step / "dit_mlp_hidden.npz")
        arr = data["all_dit_mlp_hidden"]
        # (denoising_steps, num_dit_layers, sa_seq_len, ff_inner_dim) — pi0 analog: suffix_mlp_hidden.
        assert arr.ndim == 4, f"all_dit_mlp_hidden should be 4-D, got {arr.shape}"
        assert arr.shape[0] == EXPECTED_DENOISING_STEPS
        assert arr.shape[1] == EXPECTED_DIT_LAYERS
        # seq_len should match dit_hidden_states (both taken from the same DiT forward).
        hidden = np.load(first_step / "dit_hidden_states.npz")["all_dit_hidden_states"]
        assert arr.shape[2] == hidden.shape[2], (
            f"mlp seq_len {arr.shape[2]} != hidden seq_len {hidden.shape[2]}"
        )
        # ff_inner_dim should be strictly larger than hidden dim (FeedForward expands
        # hidden -> ff_inner, typically 4x). Sanity-check the expansion.
        assert arr.shape[3] > hidden.shape[3], (
            f"ff_inner_dim {arr.shape[3]} should exceed dit_hidden {hidden.shape[3]}"
        )

    def test_dtypes(self, first_step):
        # Denoising trajectory is saved fp32 (small enough); backbone + DiT are
        # fp16 to keep disk bounded (2x saving vs fp32 on tensors that dominate
        # total size).
        d = np.load(first_step / "denoising.npz")
        for key in d:
            assert d[key].dtype == np.float32, (
                f"denoising/{key} is {d[key].dtype}, expected float32"
            )
        bc = np.load(first_step / "backbone_cond.npz")
        for key in bc:
            assert bc[key].dtype == np.float16, (
                f"backbone_cond/{key} is {bc[key].dtype}, expected float16"
            )
        dh = np.load(first_step / "dit_hidden_states.npz")
        for key in dh:
            assert dh[key].dtype == np.float16, (
                f"dit_hidden_states/{key} is {dh[key].dtype}, expected float16"
            )
        dm = np.load(first_step / "dit_mlp_hidden.npz")
        for key in dm:
            assert dm[key].dtype == np.float16, (
                f"dit_mlp_hidden/{key} is {dm[key].dtype}, expected float16"
            )

    def test_no_nan_inf(self, first_step):
        # Cast fp16 to fp32 before finiteness check to avoid fp16-overflow spurious
        # infinities in intermediate reductions (real +-inf in the stored array
        # would still trip the final ``np.isfinite``).
        for fname in (
            "denoising.npz",
            "backbone_cond.npz",
            "dit_hidden_states.npz",
            "dit_mlp_hidden.npz",
        ):
            data = np.load(first_step / fname)
            for key in data:
                arr = data[key].astype(np.float32)
                assert not np.isnan(arr).any(), f"{fname}/{key} has NaN"
                assert not np.isinf(arr).any(), f"{fname}/{key} has Inf"


# --- Sanity tests ---


class TestSanityChecks:
    def test_x_t_changes_between_steps(self, first_step):
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        for i in range(1, x_t.shape[0]):
            assert not np.allclose(x_t[i], x_t[i - 1]), (
                f"x_t unchanged between steps {i - 1} and {i}"
            )

    def test_x_t_norm_decreases(self, first_step):
        """Flow-matching signature: ||x_t|| drops monotonically as noise becomes action."""
        data = np.load(first_step / "denoising.npz")
        x_t = data["all_x_t"]
        first_norm = float(np.linalg.norm(x_t[0]))
        last_norm = float(np.linalg.norm(x_t[-1]))
        assert last_norm < first_norm, (
            f"x_t norm did not decrease: first={first_norm:.4f}, last={last_norm:.4f}"
        )

    def test_backbone_features_nonzero(self, first_step):
        data = np.load(first_step / "backbone_cond.npz")
        arr = data["backbone_features"]
        assert np.any(arr != 0), "backbone_features is all zeros"

    def test_dit_hidden_states_nonzero_and_varies(self, first_step):
        data = np.load(first_step / "dit_hidden_states.npz")
        h = data["all_dit_hidden_states"].astype(np.float32)
        assert np.any(h != 0), "DiT hidden states are all zeros"
        # Denoising step 0 and last should differ (DiT sees a different x_t each step).
        assert not np.allclose(h[0], h[-1]), (
            "DiT hidden states identical at first and last denoising step"
        )
        # First vs last transformer block output should differ (each layer does
        # non-trivial work on the residual stream).
        assert not np.allclose(h[0, 0], h[0, -1]), (
            "DiT layer 0 and layer -1 outputs identical at denoising step 0"
        )

    def test_dit_mlp_hidden_nonzero_and_varies(self, first_step):
        data = np.load(first_step / "dit_mlp_hidden.npz")
        m = data["all_dit_mlp_hidden"].astype(np.float32)
        assert np.any(m != 0), "DiT MLP hidden is all zeros"
        # Different denoising steps should produce different MLP activations.
        assert not np.allclose(m[0], m[-1]), (
            "DiT MLP hidden identical at first and last denoising step"
        )
        # Different layers should have different MLP activations.
        assert not np.allclose(m[0, 0], m[0, -1]), (
            "DiT MLP layer 0 and layer -1 identical at denoising step 0"
        )

    def test_cross_episode_different_activations(self, act_dir):
        """Different episodes should produce different activations (env randomness)."""
        episode_dirs = sorted(act_dir.glob("episode_*"))
        if len(episode_dirs) < 2:
            pytest.skip("Need at least 2 episodes to compare")
        step0_dirs = sorted(episode_dirs[0].glob("step_*"))
        step1_dirs = sorted(episode_dirs[1].glob("step_*"))
        if not step0_dirs or not step1_dirs:
            pytest.skip("No step dirs in one of the episodes")
        data0 = np.load(step0_dirs[0] / "denoising.npz")
        data1 = np.load(step1_dirs[0] / "denoising.npz")
        assert not np.allclose(data0["all_x_t"], data1["all_x_t"]), (
            "Different episodes have identical x_t"
        )
