"""Unit tests for groot_adapter.py.

These tests avoid loading the GR00T model or needing a GPU: they exercise the
pure-Python translation logic in `GR00TAdapterPolicy` and helpers.

Covers:
- `RobocasaPandaOmronDataConfig.modality_config()` key/dim correctness
- `build_robocasa_state_dict` splits the 16-D concat into the five expected
  GR00T state keys (and rejects wrong dims)
- `build_robocasa_videos` returns three video keys and resizes to 256x256
- `_resize_to_256` is a no-op at 256x256 and upsizes smaller images
- `GR00TAdapterPolicy._squeeze_leading_batch` strips a size-1 batch dim added
  by the CollectingPolicy wrapper (both ndarrays and prompt-list)
- `GR00TAdapterPolicy._action_dict_to_array` concatenates in modality-config
  order and produces the 12-D robocasa action expected by the client
- `GR00TAdapterPolicy.infer` round-trips: openpi obs -> GR00T obs via a stub
  `gr00t_policy.get_action`, and the stub sees the expected nested format
"""

from __future__ import annotations

import numpy as np
import pytest

import groot_adapter


class TestRobocasaDataConfig:
    def test_modality_keys(self):
        cfg = groot_adapter.RobocasaPandaOmronDataConfig()
        mc = cfg.modality_config()
        assert list(mc["video"].modality_keys) == [
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ]
        assert list(mc["state"].modality_keys) == [
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
            "state.base_position",
            "state.base_rotation",
        ]
        assert list(mc["action"].modality_keys) == [
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        ]
        assert list(mc["language"].modality_keys) == [
            "annotation.human.action.task_description"
        ]

    def test_action_horizon(self):
        cfg = groot_adapter.RobocasaPandaOmronDataConfig()
        mc = cfg.modality_config()
        # N1.5 expects 16 action steps (delta_indices 0..15).
        assert list(mc["action"].delta_indices) == list(range(16))
        # Observations are single-step (delta 0).
        assert list(mc["video"].delta_indices) == [0]
        assert list(mc["state"].delta_indices) == [0]
        assert list(mc["language"].delta_indices) == [0]


class TestBuildRobocasaState:
    def test_splits_16d_state(self):
        state = np.arange(16, dtype=np.float32)
        out = groot_adapter.build_robocasa_state_dict({"observation/state": state})
        np.testing.assert_array_equal(
            out["state.end_effector_position_relative"], state[0:3]
        )
        np.testing.assert_array_equal(
            out["state.end_effector_rotation_relative"], state[3:7]
        )
        np.testing.assert_array_equal(out["state.base_position"], state[7:10])
        np.testing.assert_array_equal(out["state.base_rotation"], state[10:14])
        np.testing.assert_array_equal(out["state.gripper_qpos"], state[14:16])

    def test_all_keys_present(self):
        state = np.zeros(16, dtype=np.float32)
        out = groot_adapter.build_robocasa_state_dict({"observation/state": state})
        assert set(out.keys()) == {
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
            "state.base_position",
            "state.base_rotation",
        }

    def test_rejects_wrong_dim(self):
        # 13-D (the misread layout we hit earlier in integration) must raise.
        with pytest.raises(ValueError, match="16-D"):
            groot_adapter.build_robocasa_state_dict(
                {"observation/state": np.zeros(13, dtype=np.float32)}
            )

    def test_dtype_preserved(self):
        state = np.ones(16, dtype=np.float32) * 0.5
        out = groot_adapter.build_robocasa_state_dict({"observation/state": state})
        for v in out.values():
            assert v.dtype == np.float32


class TestBuildRobocasaVideos:
    def test_three_keys_side_agentviews_and_wrist(self):
        obs = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        out = groot_adapter.build_robocasa_videos(obs)
        assert set(out.keys()) == {
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        }

    def test_resized_to_256(self):
        # Client sends 224x224 (pi05's default); adapter must upscale for N1.5.
        obs = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        out = groot_adapter.build_robocasa_videos(obs)
        for arr in out.values():
            assert arr.shape == (256, 256, 3)
            assert arr.dtype == np.uint8

    def test_left_duplicated_as_right(self):
        # The openpi client only sends agentview_left + wrist, so the adapter
        # copies the left view into the right-view slot to satisfy the model's
        # 3-camera input. Verify the left and right ndarrays share values.
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        obs = {"observation/image": img, "observation/wrist_image": np.zeros_like(img)}
        out = groot_adapter.build_robocasa_videos(obs)
        np.testing.assert_array_equal(
            out["video.robot0_agentview_left"], out["video.robot0_agentview_right"]
        )


class TestResizeTo256:
    def test_passthrough_when_already_256(self):
        img = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        out = groot_adapter._resize_to_256(img)
        assert out.shape == (256, 256, 3)
        # Same buffer, no copy.
        np.testing.assert_array_equal(out, img)

    def test_upsizes_224(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        out = groot_adapter._resize_to_256(img)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8

    def test_nonsquare_input(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        out = groot_adapter._resize_to_256(img)
        assert out.shape == (256, 256, 3)


class TestSqueezeLeadingBatch:
    def _policy(self):
        return groot_adapter.GR00TAdapterPolicy(
            gr00t_policy=object(),
            video_builder=groot_adapter.build_robocasa_videos,
            state_builder=groot_adapter.build_robocasa_state_dict,
            action_keys=list(groot_adapter.ROBOCASA_ACTION_KEYS),
        )

    def test_unbatched_passthrough(self):
        obs = {
            "observation/state": np.zeros(16, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "open the drawer",
        }
        out = self._policy()._squeeze_leading_batch(obs)
        assert out["observation/state"].shape == (16,)
        assert out["observation/image"].shape == (224, 224, 3)
        assert out["prompt"] == "open the drawer"

    def test_batched_stripped(self):
        obs = {
            "observation/state": np.zeros((1, 16), dtype=np.float32),
            "observation/image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "prompt": ["open the drawer"],
        }
        out = self._policy()._squeeze_leading_batch(obs)
        assert out["observation/state"].shape == (16,)
        assert out["observation/image"].shape == (224, 224, 3)
        assert out["observation/wrist_image"].shape == (224, 224, 3)
        assert out["prompt"] == "open the drawer"

    def test_missing_state_passthrough(self):
        # Without observation/state, we can't decide batched-ness. Must return
        # the input unchanged, not crash.
        obs = {"foo": np.zeros(3)}
        out = self._policy()._squeeze_leading_batch(obs)
        assert out is obs


class TestActionDictToArray:
    def test_concat_order_matches_convert_action(self):
        """openpi's `robocasa.utils.env_utils.convert_action` splits a 12-D
        action at fixed offsets (0:3, 3:6, 6:7, 7:11, 11:12). The adapter must
        concatenate the GR00T action dict in the SAME order so that the
        client's `convert_action` recovers the original 5 action keys."""
        # Build a one-step action dict with distinct identifiable values per key.
        action_dict = {
            "action.end_effector_position": np.array(
                [[1.0, 1.0, 1.0]], dtype=np.float32
            ),  # 3
            "action.end_effector_rotation": np.array(
                [[2.0, 2.0, 2.0]], dtype=np.float32
            ),  # 3
            "action.gripper_close": np.array([[3.0]], dtype=np.float32),  # 1
            "action.base_motion": np.array(
                [[4.0, 4.0, 4.0, 4.0]], dtype=np.float32
            ),  # 4
            "action.control_mode": np.array([[5.0]], dtype=np.float32),  # 1
        }
        policy = groot_adapter.GR00TAdapterPolicy(
            gr00t_policy=object(),
            video_builder=groot_adapter.build_robocasa_videos,
            state_builder=groot_adapter.build_robocasa_state_dict,
            action_keys=list(groot_adapter.ROBOCASA_ACTION_KEYS),
        )
        arr = policy._action_dict_to_array(action_dict)
        assert arr.shape == (1, 12)
        # Expected layout: [eef_pos(3), eef_rot(3), gripper(1), base(4), control(1)]
        np.testing.assert_array_equal(arr[0, 0:3], [1, 1, 1])
        np.testing.assert_array_equal(arr[0, 3:6], [2, 2, 2])
        np.testing.assert_array_equal(arr[0, 6:7], [3])
        np.testing.assert_array_equal(arr[0, 7:11], [4, 4, 4, 4])
        np.testing.assert_array_equal(arr[0, 11:12], [5])


class TestInferRoundtrip:
    """Stub out the GR00T model, verify the adapter produces the expected
    GR00T-format observation dict and concatenates the action output correctly.
    """

    class _StubGr00tPolicy:
        """Capture calls to `get_action` and return a canned action dict."""

        def __init__(self):
            self.last_obs = None

        def get_action(self, obs):
            self.last_obs = obs
            # Simulate a 16-step action horizon.
            horizon = 16
            return {
                "action.end_effector_position": np.zeros(
                    (horizon, 3), dtype=np.float32
                ),
                "action.end_effector_rotation": np.zeros(
                    (horizon, 3), dtype=np.float32
                ),
                "action.gripper_close": np.zeros((horizon, 1), dtype=np.float32),
                "action.base_motion": np.zeros((horizon, 4), dtype=np.float32),
                "action.control_mode": np.zeros((horizon, 1), dtype=np.float32),
            }

    def test_infer_produces_12d_actions(self):
        stub = self._StubGr00tPolicy()
        policy = groot_adapter.GR00TAdapterPolicy(
            stub,
            video_builder=groot_adapter.build_robocasa_videos,
            state_builder=groot_adapter.build_robocasa_state_dict,
            action_keys=list(groot_adapter.ROBOCASA_ACTION_KEYS),
        )
        obs = {
            "observation/state": np.zeros(16, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "open the drawer",
        }
        result = policy.infer(obs)
        assert "actions" in result
        assert result["actions"].shape == (16, 12)
        assert result["actions"].dtype == np.float32

    def test_infer_passes_groot_format_to_policy(self):
        stub = self._StubGr00tPolicy()
        policy = groot_adapter.GR00TAdapterPolicy(
            stub,
            video_builder=groot_adapter.build_robocasa_videos,
            state_builder=groot_adapter.build_robocasa_state_dict,
            action_keys=list(groot_adapter.ROBOCASA_ACTION_KEYS),
        )
        obs = {
            "observation/state": np.arange(16, dtype=np.float32),
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": "open the drawer",
        }
        policy.infer(obs)
        got = stub.last_obs
        # Video keys — 3 cameras, each (T=1, 256, 256, 3) uint8.
        for k in (
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
            "video.robot0_eye_in_hand",
        ):
            assert got[k].shape == (1, 256, 256, 3)
            assert got[k].dtype == np.uint8
        # State keys — 5 streams, each (T=1, D).
        assert got["state.end_effector_position_relative"].shape == (1, 3)
        assert got["state.end_effector_rotation_relative"].shape == (1, 4)
        assert got["state.gripper_qpos"].shape == (1, 2)
        assert got["state.base_position"].shape == (1, 3)
        assert got["state.base_rotation"].shape == (1, 4)
        # Language — list[str] of length 1 (T=1).
        assert got["annotation.human.action.task_description"] == ["open the drawer"]
