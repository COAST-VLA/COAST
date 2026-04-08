import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.policies import robocasa_policy
from openpi.training import config as _config

CHECKPOINT_DIR = "checkpoints/pi05_pretrain_human300/multitask_learning/75000"


def test_pi05_robocasa_config_registered():
    config = _config.get_config("pi05_robocasa")
    assert config.name == "pi05_robocasa"
    assert config.model.pi05 is True
    assert config.model.max_token_len == 96
    # action_dim is the model's padded dim (32 for pi0/pi0.5).
    assert config.model.action_dim == 32


def test_pi05_robocasa_data_config_create(tmp_path: pathlib.Path):
    config = _config.get_config("pi05_robocasa")
    data_config = config.data.create(tmp_path, config.model)

    # Asset id must be set so policy_config.create_trained_policy can find norm stats.
    assert data_config.asset_id == "robocasa"

    # Robocasa checkpoint uses z-score normalization (no quantile stats).
    assert data_config.use_quantile_norm is False

    # The data transforms must wire RobocasaInputs/Outputs.
    assert len(data_config.data_transforms.inputs) == 1
    assert isinstance(data_config.data_transforms.inputs[0], robocasa_policy.RobocasaInputs)
    assert data_config.data_transforms.inputs[0].action_dim == config.model.action_dim
    assert data_config.data_transforms.inputs[0].model_type == config.model.model_type

    assert len(data_config.data_transforms.outputs) == 1
    assert isinstance(data_config.data_transforms.outputs[0], robocasa_policy.RobocasaOutputs)


def test_make_robocasa_example_shape():
    example = robocasa_policy.make_robocasa_example()
    assert example["observation/state"].shape == (16,)
    assert example["observation/image"].shape == (224, 224, 3)
    assert example["observation/image"].dtype == np.uint8
    assert example["observation/wrist_image"].shape == (224, 224, 3)
    assert example["observation/wrist_image"].dtype == np.uint8
    assert isinstance(example["prompt"], str)


@pytest.mark.parametrize(
    ("model_type", "expected_right_wrist_mask"),
    [
        # Only PI0 masks padded image inputs; PI05 and PI0_FAST leave the mask True.
        (_model.ModelType.PI0, False),
        (_model.ModelType.PI05, True),
        (_model.ModelType.PI0_FAST, True),
    ],
)
def test_robocasa_inputs_transform(model_type, expected_right_wrist_mask):
    transform = robocasa_policy.RobocasaInputs(action_dim=32, model_type=model_type)
    example = robocasa_policy.make_robocasa_example()

    out = transform(example)

    # State padded to action_dim.
    assert out["state"].shape == (32,)
    # Original 16 dims preserved, padding is zero.
    assert np.allclose(out["state"][:16], example["observation/state"])
    assert np.all(out["state"][16:] == 0)

    # Three image slots, padded right wrist with zeros.
    assert set(out["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert out["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert np.all(out["image"]["right_wrist_0_rgb"] == 0)

    # Image masks: real cameras True, padding wrist depends on model type.
    assert bool(out["image_mask"]["base_0_rgb"]) is True
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(out["image_mask"]["right_wrist_0_rgb"]) is expected_right_wrist_mask

    # Prompt is passed through.
    assert out["prompt"] == example["prompt"]


def test_robocasa_inputs_pads_actions_during_training():
    transform = robocasa_policy.RobocasaInputs(action_dim=32, model_type=_model.ModelType.PI0)
    example = robocasa_policy.make_robocasa_example()
    # Action chunk is (horizon, 12) for robocasa.
    example["actions"] = np.random.rand(50, 12).astype(np.float32)

    out = transform(example)
    assert "actions" in out
    assert out["actions"].shape == (50, 32)
    # First 12 dims preserved, rest padded with zeros.
    assert np.allclose(out["actions"][:, :12], example["actions"])
    assert np.all(out["actions"][:, 12:] == 0)


def test_robocasa_inputs_float_chw_image_normalization():
    """Float CHW images should be converted to uint8 HWC."""
    transform = robocasa_policy.RobocasaInputs(action_dim=32, model_type=_model.ModelType.PI0)
    example = {
        "observation/state": np.random.rand(16),
        "observation/image": np.random.rand(3, 224, 224).astype(np.float32),
        "observation/wrist_image": np.random.rand(3, 224, 224).astype(np.float32),
        "prompt": "test",
    }

    out = transform(example)
    assert out["image"]["base_0_rgb"].dtype == np.uint8
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert out["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)


@pytest.mark.parametrize("shape", [(50, 32), (2, 50, 32)])
def test_robocasa_outputs_slices_to_12_dims(shape):
    transform = robocasa_policy.RobocasaOutputs()
    # Model may return unbatched (H, D) or batched (B, H, D) padded actions.
    actions = np.random.rand(*shape).astype(np.float32)

    out = transform({"actions": actions})

    assert out["actions"].shape == shape[:-1] + (12,)
    assert np.allclose(out["actions"], actions[..., :12])


@pytest.mark.manual
def test_pi05_robocasa_create_trained_policy_and_infer():
    """End-to-end inference test against the local pretrained checkpoint.

    Marked ``manual`` because it requires a GPU and the local checkpoint at
    ``CHECKPOINT_DIR``. Run with: ``uv run pytest tests/policies/test_robocasa_policy.py -m manual -v``
    """
    if not pathlib.Path(CHECKPOINT_DIR).exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_DIR}")

    config = _config.get_config("pi05_robocasa")
    policy = _policy_config.create_trained_policy(config, CHECKPOINT_DIR)

    example = robocasa_policy.make_robocasa_example()
    result = policy.infer(example)

    # Robocasa actions are 12-dim, action_horizon comes from the model config.
    assert result["actions"].shape == (config.model.action_horizon, 12)
    assert result["actions"].dtype in (np.float32, np.float64)
