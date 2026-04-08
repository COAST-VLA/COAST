"""CPU-only tests for LiberoInputs / LiberoOutputs in src/openpi/policies/libero_policy.py.

The slice-shape test is a regression test for a real bug where
LiberoOutputs.__call__ used `data["actions"][:, :7]`, which works for
unbatched (action_horizon, action_dim) inputs but silently slices the
wrong axis when called via the batched code path used by
Policy.infer_with_intermediates (where actions are (batch, action_horizon,
action_dim)). The fix is `[..., :7]`, which works for both shapes.
"""

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.policies import libero_policy


class TestLiberoOutputsSlice:
    def test_unbatched_actions_keep_first_seven_dims(self) -> None:
        """The single-example code path passes (action_horizon, action_dim) to the transform."""
        outputs = libero_policy.LiberoOutputs()
        # Model emits 32-dim padded actions; libero env expects 7-dim.
        actions = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
        result = outputs({"actions": actions})
        assert result["actions"].shape == (10, 7)
        # Each row should be the first 7 entries of the corresponding 32-d row.
        np.testing.assert_array_equal(result["actions"], actions[:, :7])

    def test_batched_actions_keep_first_seven_dims(self) -> None:
        """The batched code path (used by infer_with_intermediates) passes
        (batch, action_horizon, action_dim) to the transform. The historical
        bug used `[:, :7]` which silently sliced the action_horizon axis,
        leaving the trailing 32-dim padded actions in place.
        """
        outputs = libero_policy.LiberoOutputs()
        actions = np.arange(2 * 10 * 32, dtype=np.float32).reshape(2, 10, 32)
        result = outputs({"actions": actions})
        assert result["actions"].shape == (
            2,
            10,
            7,
        ), "regression: LiberoOutputs must slice the action_dim axis, not action_horizon. Got shape {}".format(
            result["actions"].shape
        )
        np.testing.assert_array_equal(result["actions"], actions[..., :7])

    def test_single_batch_element_passes_through(self) -> None:
        """Edge case: a (1, action_horizon, action_dim) batch from a single-example collection mode."""
        outputs = libero_policy.LiberoOutputs()
        actions = np.zeros((1, 10, 32), dtype=np.float32)
        result = outputs({"actions": actions})
        assert result["actions"].shape == (1, 10, 7)
