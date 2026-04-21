"""DiffusionPolicyConfig — openpi BaseModelConfig for the Transformer-Hybrid Diffusion Policy.

The underlying model is ``DiffusionTransformerHybridImagePolicy`` vendored from
``robocasa-benchmark/diffusion_policy`` (Apache 2.0) under ``.vendored``. The checkpoint we
target lives on HuggingFace at
``robocasa/robocasa365_checkpoints/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/``
and loads bit-exactly into this config when the shape_meta matches.

This config describes the model architecture + the shape_meta (obs / action spec). The
shape_meta's primitive fields (camera keys, state keys, action dim, lang_emb presence)
are carried as explicit tuples rather than a raw dict so tyro can serialize the config.

``action_dim`` / ``action_horizon`` / ``max_token_len`` satisfy ``BaseModelConfig``. The
Transformer architecture ignores ``max_token_len``; we keep it for interface compatibility.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class ImageSpec:
    """Declaration of one RGB camera key that flows through VisualCore(LanguageConditioned)."""

    key: str
    channels: int = 3
    height: int = 256
    width: int = 256


@dataclasses.dataclass(frozen=True)
class LowdimSpec:
    """Declaration of one low-dimensional obs key (state, gripper, etc.)."""

    key: str
    dim: int


@dataclasses.dataclass(frozen=True)
class DiffusionPolicyConfig(_model.BaseModelConfig):
    """Architecture + shape_meta config for the Transformer-Hybrid Diffusion Policy.

    Defaults match the released ``robocasa-benchmark`` pretrain_human300 checkpoint so
    ``load_pytorch(...)`` of that .ckpt populates the model's weights exactly (0 missing /
    0 unexpected keys).
    """

    # --- openpi BaseModelConfig overrides ---
    action_dim: int = 12
    action_horizon: int = 10  # must equal `horizon`
    max_token_len: int = 1  # unused

    # --- shape_meta ---
    # Ordered RGB cameras the model consumes. Each becomes a separate obs_encoder.obs_nets.<key>
    # entry via Robomimic's VisualCore (language-conditioned variant when lang_emb_dim is set).
    images: tuple[ImageSpec, ...] = (
        ImageSpec("robot0_agentview_right_image"),
        ImageSpec("robot0_agentview_left_image"),
        ImageSpec("robot0_eye_in_hand_image"),
    )
    # Ordered low-dim obs keys (non-image state).
    lowdims: tuple[LowdimSpec, ...] = (
        LowdimSpec("robot0_base_to_eef_pos", 3),
        LowdimSpec("robot0_base_to_eef_quat", 4),
        LowdimSpec("robot0_gripper_qpos", 2),
    )
    # Language embedding dim. None disables the language branch and the ResNet18ConvFiLM obs encoder.
    lang_emb_dim: int | None = 768

    # --- diffusion + architecture ---
    horizon: int = 10
    n_obs_steps: int = 2
    n_action_steps: int = 8
    num_inference_steps: int = 100
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    variance_type: str = "fixed_small"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] = (224, 224)
    obs_encoder_group_norm: bool = True
    eval_fixed_crop: bool = True

    n_layer: int = 12
    n_cond_layers: int = 4
    n_head: int = 8
    n_emb: int = 512
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    causal_attn: bool = True
    time_as_cond: bool = True
    obs_as_cond: bool = True

    def __post_init__(self) -> None:
        if self.action_horizon != self.horizon:
            raise ValueError(
                f"DiffusionPolicyConfig requires action_horizon == horizon, got {self.action_horizon} vs {self.horizon}"
            )
        if not self.images and not self.lowdims:
            raise ValueError("DiffusionPolicyConfig requires at least one image or lowdim key")

    def shape_meta_dict(self) -> dict:
        """Build the ``shape_meta`` dict that ``DiffusionTransformerHybridImagePolicy`` consumes."""
        obs: dict = {}
        for img in self.images:
            obs[img.key] = {"shape": [img.channels, img.height, img.width], "type": "rgb"}
        for lowdim in self.lowdims:
            obs[lowdim.key] = {"shape": [lowdim.dim]}
        if self.lang_emb_dim is not None:
            obs["lang_emb"] = {"shape": [self.lang_emb_dim]}
        return {"obs": obs, "action": {"shape": [self.action_dim]}}

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
        model.load_weights(weight_path)
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """openpi Observation spec. Images land in the canonical IMAGE_KEYS slots (padded with
        zero-placeholder masks for any keys that aren't actually consumed by this config)."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        # Infer the flat state dim the data pipeline will feed us (concat of all lowdim entries).
        state_dim = sum(ld.dim for ld in self.lowdims)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images=dict.fromkeys(_model.IMAGE_KEYS, image_spec),
                image_masks=dict.fromkeys(_model.IMAGE_KEYS, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, state_dim], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
