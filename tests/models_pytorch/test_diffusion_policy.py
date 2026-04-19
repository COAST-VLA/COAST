"""Sanity tests for the Diffusion Policy PyTorch model.

Runs on CPU to keep CI cheap. Exercises forward/backward loss, optimizer step, and inference shape.
"""

from __future__ import annotations

import pytest
import torch

from openpi.models.model import Observation
from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig


def _make_observation(batch_size: int = 2, image_size: int = 96, state_dim: int = 8) -> Observation:
    images = {
        key: torch.randn(batch_size, 3, image_size, image_size)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool) for key in images}
    state = torch.randn(batch_size, state_dim)
    return Observation(images=images, image_masks=image_masks, state=state)


def _small_config(**overrides) -> DiffusionPolicyConfig:
    """Tiny config for fast CPU tests."""
    defaults = {
        "action_dim": 4,
        "action_horizon": 8,
        "state_dim": 8,
        "camera_keys": ("base_0_rgb", "left_wrist_0_rgb"),
        "image_size": (64, 64),
        "crop_shape": (56, 56),
        "down_dims": (32, 64),  # tiny UNet
        "diffusion_step_embed_dim": 16,
        "spatial_softmax_num_keypoints": 8,
        "num_train_timesteps": 10,
        "num_inference_steps": 3,
    }
    defaults.update(overrides)
    return DiffusionPolicyConfig(**defaults)


def test_construction():
    cfg = _small_config()
    model = DiffusionPolicy(cfg)
    assert len(model.rgb_encoders) == len(cfg.camera_keys)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


def test_forward_loss_shape():
    cfg = _small_config()
    model = DiffusionPolicy(cfg).eval()
    obs = _make_observation(batch_size=2, image_size=cfg.image_size[0], state_dim=cfg.state_dim)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    loss = model(obs, actions)
    assert loss.shape == (2, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(loss).all()


def test_backward_and_optimizer_step():
    torch.manual_seed(0)
    cfg = _small_config()
    model = DiffusionPolicy(cfg)
    obs = _make_observation(batch_size=2, image_size=cfg.image_size[0], state_dim=cfg.state_dim)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(obs, actions).mean()
    loss.backward()
    # At least some parameters should have non-zero gradients.
    grad_sum = sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in model.parameters())
    assert grad_sum > 0
    optim.step()
    optim.zero_grad(set_to_none=True)


def test_loss_decreases_on_overfit():
    """Overfit a single batch for a few steps; expect loss to decrease meaningfully."""
    torch.manual_seed(0)
    cfg = _small_config()
    model = DiffusionPolicy(cfg)
    obs = _make_observation(batch_size=2, image_size=cfg.image_size[0], state_dim=cfg.state_dim)
    actions = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Fix noise/time so the loss target doesn't change between steps.
    fixed_noise = torch.randn_like(actions)
    fixed_time = torch.tensor([5, 5], dtype=torch.long)
    first_loss = None
    last_loss = None
    for step in range(20):
        loss = model(obs, actions, noise=fixed_noise, time=fixed_time).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    assert last_loss < first_loss * 0.9, f"loss did not decrease: {first_loss=} -> {last_loss=}"


def test_sample_actions_shape():
    cfg = _small_config()
    model = DiffusionPolicy(cfg).eval()
    obs = _make_observation(batch_size=3, image_size=cfg.image_size[0], state_dim=cfg.state_dim)
    actions = model.sample_actions("cpu", obs, num_steps=3)
    assert actions.shape == (3, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all()


def test_sample_actions_deterministic_with_ddim():
    """DDIM is deterministic — same noise + same weights should give identical output."""
    cfg = _small_config(noise_scheduler_type="DDIM")
    model = DiffusionPolicy(cfg).eval()
    obs = _make_observation(batch_size=1, image_size=cfg.image_size[0], state_dim=cfg.state_dim)
    noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    out1 = model.sample_actions("cpu", obs, noise=noise.clone(), num_steps=3)
    out2 = model.sample_actions("cpu", obs, noise=noise.clone(), num_steps=3)
    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize("layout", ["nhwc_float", "nchw_float"])
def test_image_format_robustness(layout):
    """Model should accept NHWC and NCHW float images in [-1, 1]. uint8 is handled upstream in Observation.from_dict."""
    cfg = _small_config()
    model = DiffusionPolicy(cfg).eval()
    b, h, w = 2, cfg.image_size[0], cfg.image_size[1]
    if layout == "nhwc_float":
        images = {
            key: torch.rand(b, h, w, 3) * 2 - 1 for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
    else:
        images = {
            key: torch.rand(b, 3, h, w) * 2 - 1 for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
    obs = Observation(
        images=images,
        image_masks={k: torch.ones(b, dtype=torch.bool) for k in images},
        state=torch.randn(b, cfg.state_dim),
    )
    actions = model.sample_actions("cpu", obs, num_steps=2)
    assert actions.shape == (b, cfg.action_horizon, cfg.action_dim)
