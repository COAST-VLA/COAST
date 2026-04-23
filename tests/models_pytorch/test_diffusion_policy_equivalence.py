"""Equivalence tests between our openpi ``DiffusionPolicy`` wrapper and the vendored policy.

The vendored ``DiffusionTransformerHybridImagePolicy`` contains all the real parameters; our
``DiffusionPolicy`` wrapper adds an obs-format bridge and a few ergonomics. These tests pin
that the bridge layer is numerically transparent — given the same obs and the same weights,
calling through the wrapper yields the same actions as calling the vendored policy directly.

Also: a state_dict-key-equality test that guards against the wrapper introducing any new
buffers/parameters (which would shift checkpoint state_dict paths and break load_weights).
"""

from __future__ import annotations

import os

import pytest
import torch

from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig
from openpi.models_pytorch.diffusion_policy import ImageSpec
from openpi.models_pytorch.diffusion_policy import LowdimSpec
from openpi.models_pytorch.diffusion_policy.vendored import robomimic_extensions  # noqa: F401 — register
from openpi.models_pytorch.diffusion_policy.vendored.normalizer import LinearNormalizer


def _small_cfg(*, with_lang: bool = False) -> DiffusionPolicyConfig:
    return DiffusionPolicyConfig(
        action_dim=4,
        action_horizon=8,
        horizon=8,
        n_obs_steps=1,
        n_action_steps=8,
        num_inference_steps=4,
        num_train_timesteps=10,
        crop_shape=(56, 56),
        images=(ImageSpec("base_0_rgb", 3, 64, 64),),
        lowdims=(LowdimSpec("state", 4),),
        lang_emb_dim=64 if with_lang else None,
        n_layer=2,
        n_head=2,
        n_emb=64,
        n_cond_layers=1,
    )


def _fit_identity_normalizer(model: DiffusionPolicy) -> None:
    """Populate the LinearNormalizer with placeholder stats so sample_actions/compute_loss can run."""
    data: dict[str, torch.Tensor] = {"action": torch.randn(4, 4)}
    for spec in model.config.images:
        data[spec.key] = torch.rand(4, 3, spec.height, spec.width)
    for ld in model.config.lowdims:
        data[ld.key] = torch.randn(4, ld.dim)
    if model.config.lang_emb_dim is not None:
        data["lang_emb"] = torch.randn(4, model.config.lang_emb_dim)
    norm = LinearNormalizer()
    norm.fit(data)
    model._policy.set_normalizer(norm)  # noqa: SLF001


def test_wrapper_state_dict_matches_vendored() -> None:
    """All wrapper parameters live under the vendored ``_policy`` subtree — no extra state.

    This is the property that lets ``load_weights(.ckpt)`` just forward to ``_policy.load_state_dict``:
    there's nothing else for the wrapper to own.
    """
    cfg = _small_cfg()
    model = DiffusionPolicy(cfg)
    wrapper_keys = set(model.state_dict().keys())
    vendored_keys = set(model._policy.state_dict().keys())  # noqa: SLF001
    # Every wrapper key is ``_policy.<vendored_key>`` — no extra state.
    assert wrapper_keys == {f"_policy.{k}" for k in vendored_keys}


@pytest.mark.skipif("CI" in os.environ, reason="runs a forward + DDPM loop; skip to keep CI fast")
def test_wrapper_sample_actions_matches_direct_policy_call() -> None:
    """sample_actions via openpi Observation == direct vendored predict_action with same inputs."""
    from openpi.models.model import Observation

    torch.manual_seed(0)
    cfg = _small_cfg()
    model = DiffusionPolicy(cfg).eval()
    _fit_identity_normalizer(model)

    b = 1
    img = torch.rand(b, 64, 64, 3) * 2 - 1  # NHWC [-1, 1]
    state = torch.randn(b, 4)
    obs = Observation(
        images={"base_0_rgb": img},
        image_masks={"base_0_rgb": torch.ones(b, dtype=torch.bool)},
        state=state,
    )

    # Direct call path: prepare the shape_meta-keyed obs_dict the vendored policy consumes.
    # Match _openpi_obs_to_batch's transformations exactly.
    from openpi.models_pytorch.diffusion_policy.modeling import _to_nchw_01

    img_01 = _to_nchw_01(img)  # NCHW in [0, 1]
    obs_dict = {
        "base_0_rgb": img_01.unsqueeze(1),
        "state": state.unsqueeze(1),
    }

    # RNG control: the vendored DDPM sampler draws random noise via torch.randn (not seeded
    # explicitly). Both call paths go through the same sampler with the same weights; reseed
    # before each call so they take the same random draws.
    torch.manual_seed(123)
    wrapper_actions = model.sample_actions("cpu", obs)

    torch.manual_seed(123)
    direct_result = model._policy.predict_action(obs_dict)  # noqa: SLF001
    direct_actions = direct_result["action_pred"]

    torch.testing.assert_close(wrapper_actions, direct_actions, rtol=1e-6, atol=1e-6)
