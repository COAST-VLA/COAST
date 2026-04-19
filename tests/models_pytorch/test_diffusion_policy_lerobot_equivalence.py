"""Numerical-equivalence tests: our DiffusionPolicy port vs LeRobot's DiffusionModel.

Our ``openpi.models_pytorch.diffusion_policy.DiffusionPolicy`` was ported from
``lerobot.common.policies.diffusion.modeling_diffusion.DiffusionModel`` (Apache 2.0).
These tests guarantee the port is numerically faithful: given identical weights
and inputs, outputs match bit-for-bit (or within tight float tolerances).

Why this matters: the rest of our test suite only checks shape + "loss decreases",
which would not catch a subtle FiLM conditioning bug or a transposed axis. If
these tests fail, the port has drifted from the reference.

The two implementations have structurally identical state_dicts modulo a single
rename (``rgb_encoders`` ours ↔ ``rgb_encoder`` theirs), so we can transplant
weights by just mapping that prefix.
"""

from __future__ import annotations

import pytest
import torch

from openpi.models.model import Observation
from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig

_lerobot_import_error: Exception | None = None
try:
    from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig as _LRDiffusionConfig
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionModel as _LRDiffusionModel
    from lerobot.configs.types import FeatureType as _LRFeatureType
    from lerobot.configs.types import PolicyFeature as _LRPolicyFeature
except Exception as e:  # pragma: no cover - lerobot missing is handled via skip
    _lerobot_import_error = e

pytestmark = pytest.mark.skipif(
    _lerobot_import_error is not None,
    reason=f"lerobot not importable in this environment: {_lerobot_import_error}",
)


# ---- config + model builders --------------------------------------------------------------


def _small_configs():
    """Tiny configs that share every knob between the two implementations.

    Deliberately uses: 2 cameras, small action/state dims, small UNet, few diffusion steps —
    keeps CPU runtime under a few seconds per test.
    """
    common = {
        "action_dim": 4,
        "action_horizon": 8,
        "state_dim": 6,
        "image_size": (64, 64),
        "crop_shape": (56, 56),
        "crop_is_random": False,  # deterministic: center-crop path only
        "vision_backbone": "resnet18",
        "use_group_norm": True,
        "spatial_softmax_num_keypoints": 8,
        "down_dims": (32, 64),
        "kernel_size": 5,
        "n_groups": 8,
        "diffusion_step_embed_dim": 16,
        "use_film_scale_modulation": True,
        "noise_scheduler_type": "DDPM",
        "num_train_timesteps": 10,
        "num_inference_steps": 4,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "clip_sample": True,
        "clip_sample_range": 1.0,
    }
    ours_cfg = DiffusionPolicyConfig(
        camera_keys=("base_0_rgb", "left_wrist_0_rgb"),
        **common,
    )
    lerobot_cfg = _LRDiffusionConfig(
        n_obs_steps=1,
        horizon=common["action_horizon"],
        n_action_steps=common["action_horizon"],
        input_features={
            "observation.state": _LRPolicyFeature(type=_LRFeatureType.STATE, shape=(common["state_dim"],)),
            "observation.images.cam_0": _LRPolicyFeature(type=_LRFeatureType.VISUAL, shape=(3, *common["image_size"])),
            "observation.images.cam_1": _LRPolicyFeature(type=_LRFeatureType.VISUAL, shape=(3, *common["image_size"])),
        },
        output_features={"action": _LRPolicyFeature(type=_LRFeatureType.ACTION, shape=(common["action_dim"],))},
        use_separate_rgb_encoder_per_camera=True,
        crop_shape=common["crop_shape"],
        crop_is_random=common["crop_is_random"],
        use_group_norm=common["use_group_norm"],
        spatial_softmax_num_keypoints=common["spatial_softmax_num_keypoints"],
        down_dims=common["down_dims"],
        kernel_size=common["kernel_size"],
        n_groups=common["n_groups"],
        diffusion_step_embed_dim=common["diffusion_step_embed_dim"],
        use_film_scale_modulation=common["use_film_scale_modulation"],
        noise_scheduler_type=common["noise_scheduler_type"],
        num_train_timesteps=common["num_train_timesteps"],
        num_inference_steps=common["num_inference_steps"],
        beta_start=common["beta_start"],
        beta_end=common["beta_end"],
        beta_schedule=common["beta_schedule"],
        prediction_type=common["prediction_type"],
        clip_sample=common["clip_sample"],
        clip_sample_range=common["clip_sample_range"],
    )
    return ours_cfg, lerobot_cfg


def _build_aligned_pair():
    """Return (ours, theirs) with weights transplanted theirs -> ours via state_dict rename."""
    torch.manual_seed(0)
    ours_cfg, lerobot_cfg = _small_configs()
    ours = DiffusionPolicy(ours_cfg).eval()
    theirs = _LRDiffusionModel(lerobot_cfg).eval()

    # Transplant theirs -> ours: state_dict is identical modulo rgb_encoder(s) prefix.
    remapped = {}
    for k, v in theirs.state_dict().items():
        new_k = "rgb_encoders." + k[len("rgb_encoder.") :] if k.startswith("rgb_encoder.") else k
        remapped[new_k] = v
    missing, unexpected = ours.load_state_dict(remapped, strict=True)
    assert not missing, f"missing keys after transplant: {missing[:5]}"
    assert not unexpected, f"unexpected keys after transplant: {unexpected[:5]}"
    return ours, theirs


def _make_inputs(batch_size: int = 2, *, ours_cfg: DiffusionPolicyConfig, lerobot_cfg):
    """Build matched obs/actions/noise/time pairs for both models.

    Both models expect images in [0, 1] float. Ours accepts NHWC float in [-1, 1] via the
    openpi Observation contract (which _to_nchw_01 converts); LeRobot's model takes NCHW in
    [0, 1] directly. We pick a float-NHWC [-1, 1] tensor for ours and the equivalent NCHW
    [0, 1] tensor for theirs.
    """
    torch.manual_seed(42)
    b = batch_size
    h, w = ours_cfg.image_size
    # NCHW in [0, 1] (LeRobot-native format).
    imgs_01 = {key: torch.rand(b, 3, h, w) for key in ours_cfg.camera_keys}
    # NHWC in [-1, 1] (openpi Observation format): 2 * img - 1 and permute.
    imgs_neg1_1_nhwc = {key: (imgs_01[key] * 2.0 - 1.0).permute(0, 2, 3, 1).contiguous() for key in imgs_01}
    state = torch.randn(b, ours_cfg.state_dim)
    actions = torch.randn(b, ours_cfg.action_horizon, ours_cfg.action_dim)
    noise = torch.randn_like(actions)
    time = torch.randint(0, ours_cfg.num_train_timesteps, (b,), dtype=torch.long)

    # openpi Observation with NHWC [-1, 1] floats + a bogus third camera (model ignores anything outside
    # camera_keys, so value doesn't matter — we still pass all three IMAGE_KEYS because that's the
    # openpi Observation contract).
    our_obs = Observation(
        images={**imgs_neg1_1_nhwc, "right_wrist_0_rgb": imgs_neg1_1_nhwc[ours_cfg.camera_keys[0]]},
        image_masks={
            "base_0_rgb": torch.ones(b, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(b, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(b, dtype=torch.bool),
        },
        state=state,
    )

    # LeRobot batch: "observation.state" (B, n_obs_steps=1, state_dim),
    # "observation.images" (B, n_obs_steps=1, num_cameras, C, H, W).
    stacked_imgs = torch.stack([imgs_01[k] for k in ours_cfg.camera_keys], dim=1)  # (B, N, C, H, W)
    lerobot_batch = {
        "observation.state": state.unsqueeze(1),
        "observation.images": stacked_imgs.unsqueeze(1),  # (B, 1, N, C, H, W)
    }
    return our_obs, lerobot_batch, actions, noise, time, imgs_01


# ---- tests --------------------------------------------------------------------------------


def test_state_dict_structure_matches():
    """278 keys, identical shapes, only rename needed is rgb_encoders <-> rgb_encoder."""
    torch.manual_seed(0)
    ours_cfg, lerobot_cfg = _small_configs()
    ours = DiffusionPolicy(ours_cfg)
    theirs = _LRDiffusionModel(lerobot_cfg)
    ours_keys = {k.replace("rgb_encoders.", "rgb_encoder.", 1) for k in ours.state_dict()}
    theirs_keys = set(theirs.state_dict().keys())
    assert ours_keys == theirs_keys, (
        f"state_dict key mismatch:\n  only ours: {sorted(ours_keys - theirs_keys)[:5]}\n"
        f"  only theirs: {sorted(theirs_keys - ours_keys)[:5]}"
    )
    ours_sd = {k.replace("rgb_encoders.", "rgb_encoder.", 1): v.shape for k, v in ours.state_dict().items()}
    theirs_sd = {k: v.shape for k, v in theirs.state_dict().items()}
    mismatches = [k for k in ours_sd if ours_sd[k] != theirs_sd[k]]
    assert not mismatches, f"shape mismatches: {mismatches[:5]}"


def test_global_conditioning_matches_lerobot():
    """_encode_global_cond (ours) vs _prepare_global_conditioning (theirs) produce identical vectors."""
    ours, theirs = _build_aligned_pair()
    ours_cfg, lerobot_cfg = _small_configs()
    our_obs, lerobot_batch, *_ = _make_inputs(batch_size=2, ours_cfg=ours_cfg, lerobot_cfg=lerobot_cfg)

    with torch.no_grad():
        ours_cond = ours._encode_global_cond(our_obs)  # noqa: SLF001
        theirs_cond = theirs._prepare_global_conditioning(lerobot_batch)  # noqa: SLF001

    assert ours_cond.shape == theirs_cond.shape, f"shape: ours {ours_cond.shape} vs theirs {theirs_cond.shape}"
    torch.testing.assert_close(ours_cond, theirs_cond, rtol=1e-5, atol=1e-5)


def test_unet_forward_matches_lerobot():
    """Given the same noise, timestep, and global conditioning, both U-Nets emit the same output."""
    ours, theirs = _build_aligned_pair()
    ours_cfg, lerobot_cfg = _small_configs()
    b = 2
    global_cond_dim = (
        ours.rgb_encoders[0].feature_dim * len(ours_cfg.camera_keys) + ours_cfg.state_dim
    ) * ours_cfg.n_obs_steps
    torch.manual_seed(7)
    x = torch.randn(b, ours_cfg.action_horizon, ours_cfg.action_dim)
    t = torch.randint(0, ours_cfg.num_train_timesteps, (b,), dtype=torch.long)
    global_cond = torch.randn(b, global_cond_dim)

    with torch.no_grad():
        ours_out = ours.unet(x, t, global_cond=global_cond)
        theirs_out = theirs.unet(x, t, global_cond=global_cond)

    torch.testing.assert_close(ours_out, theirs_out, rtol=1e-5, atol=1e-5)


def test_forward_loss_matches_lerobot():
    """Full add_noise -> unet -> loss pipeline: identical outputs when inputs are aligned."""
    ours, theirs = _build_aligned_pair()
    ours_cfg, lerobot_cfg = _small_configs()
    our_obs, lerobot_batch, actions, noise, time, _ = _make_inputs(
        batch_size=2, ours_cfg=ours_cfg, lerobot_cfg=lerobot_cfg
    )

    with torch.no_grad():
        # Ours: returns per-element MSE loss (B, H, A).
        ours_loss = ours(our_obs, actions, noise=noise, time=time)

        # Theirs: inline the relevant bits of DiffusionModel.compute_loss with the same noise/time.
        global_cond = theirs._prepare_global_conditioning(lerobot_batch)  # noqa: SLF001
        noisy = theirs.noise_scheduler.add_noise(actions, noise, time)
        pred = theirs.unet(noisy, time, global_cond=global_cond)
        theirs_loss = torch.nn.functional.mse_loss(pred, noise, reduction="none")

    torch.testing.assert_close(ours_loss, theirs_loss, rtol=1e-5, atol=1e-5)


def test_sample_actions_matches_lerobot_ddim():
    """DDIM sampling is deterministic — given identical initial noise + timesteps, both samplers emit the same actions."""
    # Use DDIM so the sampler's step() is deterministic.
    torch.manual_seed(0)
    ours_cfg, lerobot_cfg = _small_configs()
    ours_cfg = DiffusionPolicyConfig(
        **{**{k: getattr(ours_cfg, k) for k in ours_cfg.__dataclass_fields__}, "noise_scheduler_type": "DDIM"}
    )
    lerobot_cfg.noise_scheduler_type = "DDIM"

    ours = DiffusionPolicy(ours_cfg).eval()
    theirs = _LRDiffusionModel(lerobot_cfg).eval()
    remapped = {}
    for k, v in theirs.state_dict().items():
        new_k = "rgb_encoders." + k[len("rgb_encoder.") :] if k.startswith("rgb_encoder.") else k
        remapped[new_k] = v
    ours.load_state_dict(remapped, strict=True)

    our_obs, lerobot_batch, _, _, _, _ = _make_inputs(batch_size=1, ours_cfg=ours_cfg, lerobot_cfg=lerobot_cfg)
    torch.manual_seed(123)
    init_noise = torch.randn(1, ours_cfg.action_horizon, ours_cfg.action_dim)

    with torch.no_grad():
        ours_actions = ours.sample_actions(
            "cpu", our_obs, noise=init_noise.clone(), num_steps=ours_cfg.num_inference_steps
        )

        # Replicate LeRobot's conditional_sample with the same initial noise instead of a fresh randn.
        global_cond = theirs._prepare_global_conditioning(lerobot_batch)  # noqa: SLF001
        sample = init_noise.clone()
        theirs.noise_scheduler.set_timesteps(ours_cfg.num_inference_steps)
        for t in theirs.noise_scheduler.timesteps:
            model_out = theirs.unet(sample, torch.full((1,), t, dtype=torch.long), global_cond=global_cond)
            sample = theirs.noise_scheduler.step(model_out, t, sample).prev_sample

    torch.testing.assert_close(ours_actions, sample, rtol=1e-5, atol=1e-5)
