"""Diffusion Policy (CNN 1D U-Net variant).

Core model components adapted from LeRobot's `lerobot/common/policies/diffusion/modeling_diffusion.py`
(Apache 2.0, Columbia ARX / HuggingFace). We strip the LeRobot-specific wrappers and replace them with an
openpi-native module that plugs into `openpi.policies.policy.Policy` via the same `compute_loss` /
`sample_actions` interface used by `pi0_pytorch.PI0Pytorch`.

This is a PyTorch-only model. It ignores language prompts. Uses `n_obs_steps=1` (no observation history)
to avoid changes to the data loader; the obs-history case can be added later.
"""

from __future__ import annotations

from collections.abc import Callable
import itertools
import math

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import einops
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import torchvision


def _make_noise_scheduler(name: str, **kwargs):
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    if name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type: {name}")


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax over 2D feature maps (Finn et al.). Ported from LeRobot DP."""

    def __init__(self, input_shape, num_kp: int | None = None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self._in_w),
            torch.linspace(-1.0, 1.0, self._in_h),
            indexing="xy",
        )
        pos_x = pos_x.reshape(self._in_h * self._in_w, 1)
        pos_y = pos_y.reshape(self._in_h * self._in_w, 1)
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1).float())

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent = root_module
        if parents:
            parent = root_module.get_submodule(".".join(parents))
        src = parent[int(k)] if isinstance(parent, nn.Sequential) else getattr(parent, k)
        tgt = func(src)
        if isinstance(parent, nn.Sequential):
            parent[int(k)] = tgt
        else:
            setattr(parent, k, tgt)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionRgbEncoder(nn.Module):
    """ResNet (GroupNorm) + SpatialSoftmax → feature vector. Ported from LeRobot DP."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        *,
        vision_backbone: str = "resnet18",
        crop_shape: tuple[int, int] | None = None,
        crop_is_random: bool = True,
        use_group_norm: bool = True,
        pretrained_backbone_weights: str | None = None,
        spatial_softmax_num_keypoints: int = 32,
    ):
        super().__init__()
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape) if crop_is_random else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, vision_backbone)(weights=pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            if pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained backbone.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        c, h, w = image_shape
        dummy_hw = crop_shape if crop_shape is not None else (h, w)
        with torch.no_grad():
            dummy = torch.zeros(1, c, *dummy_hw)
            feature_map_shape = self.backbone(dummy).shape[1:]  # (C, H, W)

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_c),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        *,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_c = out_c
        self.conv1 = DiffusionConv1dBlock(in_c, out_c, kernel_size, n_groups=n_groups)
        cond_ch = out_c * 2 if use_film_scale_modulation else out_c
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_ch))
        self.conv2 = DiffusionConv1dBlock(out_c, out_c, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_emb = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_emb[:, : self.out_c]
            bias = cond_emb[:, self.out_c :]
            out = scale * out + bias
        else:
            out = out + cond_emb
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    """1D U-Net with FiLM conditioning. Ported from LeRobot DP."""

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        *,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        in_out = [(action_dim, down_dims[0]), *itertools.pairwise(down_dims)]

        common = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(d_in, d_out, **common),
                        DiffusionConditionalResidualBlock1d(d_out, d_out, **common),
                        nn.Conv1d(d_out, d_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common),
                DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], **common),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for i, (d_out, d_in) in enumerate(reversed(in_out[1:])):
            is_last = i >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(d_in * 2, d_out, **common),
                        DiffusionConditionalResidualBlock1d(d_out, d_out, **common),
                        nn.ConvTranspose1d(d_out, d_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor | None = None) -> Tensor:
        # x: (B, horizon, action_dim) -> (B, action_dim, horizon) for conv1d.
        x = einops.rearrange(x, "b t d -> b d t")
        t_emb = self.diffusion_step_encoder(timestep)
        global_feat = torch.cat([t_emb, global_cond], dim=-1) if global_cond is not None else t_emb

        skips: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feat)
            x = resnet2(x, global_feat)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feat)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feat)
            x = resnet2(x, global_feat)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")


class DiffusionPolicy(nn.Module):
    """Openpi-compatible Diffusion Policy.

    Interface mirrors `PI0Pytorch`:
      - forward(observation, actions) -> per-element MSE loss (same reduction='none' shape as pi0)
      - sample_actions(device, observation, noise=None, num_steps=None) -> (B, horizon, action_dim)

    Observation (from openpi.models.model.Observation):
      images: dict[str, Tensor]   # (B, H, W, C) float in [-1, 1] OR (B, C, H, W)
      image_masks: dict[str, Tensor]  # (B,) bool, True=valid camera; we skip cameras whose mask is False across the batch
      state: Tensor  # (B, state_dim)
      tokenized_prompt / tokenized_prompt_mask: ignored
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Determine which cameras are active; we always use all 3 openpi image keys to match the observation
        # layout, but we allow masking one out (e.g., MetaWorld's right_wrist_0_rgb is a zero placeholder).
        self.camera_keys = tuple(config.camera_keys)  # ordered list, e.g., ("base_0_rgb", "left_wrist_0_rgb")
        num_cameras = len(self.camera_keys)

        # Image encoder. One encoder per camera (paper default).
        image_shape = (3, config.image_size[0], config.image_size[1])
        self.rgb_encoders = nn.ModuleList(
            [
                DiffusionRgbEncoder(
                    image_shape,
                    vision_backbone=config.vision_backbone,
                    crop_shape=config.crop_shape,
                    crop_is_random=config.crop_is_random,
                    use_group_norm=config.use_group_norm,
                    spatial_softmax_num_keypoints=config.spatial_softmax_num_keypoints,
                )
                for _ in range(num_cameras)
            ]
        )
        feature_dim = self.rgb_encoders[0].feature_dim

        global_cond_dim = (feature_dim * num_cameras + config.state_dim) * config.n_obs_steps

        self.unet = DiffusionConditionalUnet1d(
            action_dim=config.action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=tuple(config.down_dims),
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            use_film_scale_modulation=config.use_film_scale_modulation,
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            config.num_inference_steps if config.num_inference_steps is not None else config.num_train_timesteps
        )

    # ---- image handling ------------------------------------------------------------------------

    @staticmethod
    def _to_nchw_01(image: Tensor) -> Tensor:
        """Accepts (B, H, W, C) or (B, C, H, W) in [-1, 1] float (or uint8). Returns (B, C, H, W) in [0, 1]."""
        if image.dtype == torch.uint8:
            if image.shape[1] != 3:  # NHWC uint8
                image = image.permute(0, 3, 1, 2)
            return image.to(torch.float32) / 255.0
        # float path: detect NHWC vs NCHW by channel dim
        if image.ndim == 4 and image.shape[-1] == 3 and image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        # assume [-1, 1]; squash to [0, 1]. Normalize's output range is already centered at 0,
        # but we accept either [-1,1] (from preprocess) or [0,1] (passthrough).
        if image.min() < -1e-3:
            image = image / 2.0 + 0.5
        return image.clamp(0.0, 1.0)

    def _resize(self, image: Tensor) -> Tensor:
        h, w = self.config.image_size
        if image.shape[-2:] != (h, w):
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        return image

    def _encode_global_cond(self, observation) -> Tensor:
        """Build (B, global_cond_dim) vector from images + state. n_obs_steps=1 path."""
        feats = []
        for i, key in enumerate(self.camera_keys):
            img = observation.images[key]
            img = self._to_nchw_01(img)
            img = self._resize(img)
            feats.append(self.rgb_encoders[i](img))
        state = observation.state.to(dtype=feats[0].dtype) if feats else observation.state
        feats.append(state)
        # concat features — corresponds to LeRobot's flatten(start_dim=1) with n_obs_steps=1
        return torch.cat(feats, dim=-1)

    # ---- training ------------------------------------------------------------------------------

    def forward(self, observation, actions: Tensor, noise: Tensor | None = None, time: Tensor | None = None) -> Tensor:
        """Returns per-element MSE loss with shape (B, horizon, action_dim)."""
        global_cond = self._encode_global_cond(observation)
        actions = actions.to(torch.float32)

        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(actions.shape[0],),
                device=actions.device,
            ).long()

        noisy = self.noise_scheduler.add_noise(actions, noise, time)
        pred = self.unet(noisy, time, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

        return F.mse_loss(pred, target, reduction="none")

    # ---- inference -----------------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self, device, observation, *, noise: Tensor | None = None, num_steps: int | None = None
    ) -> Tensor:
        """Full K-step denoising. Returns (B, horizon, action_dim)."""
        global_cond = self._encode_global_cond(observation)
        b = global_cond.shape[0]
        shape = (b, self.config.action_horizon, self.config.action_dim)
        sample = noise if noise is not None else torch.randn(shape, device=device)

        self.noise_scheduler.set_timesteps(num_steps if num_steps is not None else self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_out = self.unet(
                sample,
                torch.full((b,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_out, t, sample).prev_sample
        return sample
