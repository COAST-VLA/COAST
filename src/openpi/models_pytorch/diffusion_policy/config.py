"""Diffusion Policy config — plugs into openpi's TrainConfig / policy loading.

DP is PyTorch-only, so `create()` (JAX path) raises. `load_pytorch` builds the torch model and loads
safetensors, mirroring `BaseModelConfig.load_pytorch` for `PI0Pytorch`.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import safetensors.torch
from typing_extensions import override

from openpi.models import model as _model
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class DiffusionPolicyConfig(_model.BaseModelConfig):
    """Config for the CNN 1D U-Net Diffusion Policy."""

    # --- openpi BaseModelConfig overrides -----------------------------------------
    # Defaults target a single-arm manipulation task (e.g., MetaWorld/LIBERO). Override per-env.
    action_dim: int = 4
    action_horizon: int = 16
    max_token_len: int = 1  # unused; kept for interface compat

    # --- Observation layout -------------------------------------------------------
    # The openpi Observation always carries 3 image keys; we only encode cameras listed here.
    camera_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    # Expected raw state dimension (from the dataset, post-normalize).
    state_dim: int = 8
    # Number of observation history steps. 1 means no history (matches current data pipeline).
    n_obs_steps: int = 1
    # Image size after resize (H, W). Images are resized in the model forward.
    image_size: tuple[int, int] = (96, 96)

    # --- Vision encoder -----------------------------------------------------------
    vision_backbone: str = "resnet18"
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True

    # --- U-Net --------------------------------------------------------------------
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # --- Noise scheduler ----------------------------------------------------------
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    num_inference_steps: int | None = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    def __post_init__(self):
        # camera_keys must be a subset of openpi's canonical image keys — the Observation layout is fixed.
        unknown = set(self.camera_keys) - set(_model.IMAGE_KEYS)
        if unknown:
            raise ValueError(
                f"DiffusionPolicyConfig.camera_keys must be a subset of {_model.IMAGE_KEYS}, got unknown keys: {sorted(unknown)}"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.DIFFUSION_POLICY

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError("DiffusionPolicy is PyTorch-only — use load_pytorch()")

    @override
    def load_pytorch(self, train_config, weight_path: str):
        from openpi.models_pytorch.diffusion_policy.modeling import DiffusionPolicy

        model = DiffusionPolicy(config=self)
        safetensors.torch.load_model(model, weight_path)
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # openpi's Observation always carries all IMAGE_KEYS; DP consumes the subset listed in camera_keys
        # and ignores the rest. Building the spec from IMAGE_KEYS (not camera_keys) keeps the spec faithful
        # to the on-the-wire Observation layout.
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(_model.IMAGE_KEYS, image_spec),
                image_masks=dict.fromkeys(_model.IMAGE_KEYS, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
