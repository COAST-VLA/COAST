"""robomimic language-conditioning extensions — vendored for DP checkpoint compatibility.

The ``robocasa-benchmark/diffusion_policy`` checkpoint we target expects three robomimic
classes that do not ship with upstream robomimic: ``VisualCoreLanguageConditioned``,
``FiLMLayer``, and ``ResNet18ConvFiLM``. They live in NVlabs' robomimic fork
(`github.com/NVlabs/sage/tree/main/robomimic/robomimic/models`) and are Apache 2.0.

This module ports those three classes into openpi's tree. Importing the module is
sufficient to register ``VisualCoreLanguageConditioned`` with robomimic's encoder-core
registry — ``robomimic.models.obs_core.EncoderCore.__init_subclass__`` auto-registers
every subclass, so simply having the class loaded is enough for
``ObsUtils.initialize_obs_utils_with_config`` to find it when the DP model resolves its
config.

Source attribution:
  VisualCoreLanguageConditioned, ResNet18ConvFiLM, FiLMLayer — ported from NVlabs/sage,
  Apache 2.0, Copyright Stanford AI and Columbia ARX.
"""

from __future__ import annotations

import math

from robomimic.models.base_nets import ConvBase
from robomimic.models.obs_core import VisualCore
import torch
from torch import nn
from torchvision import models as vision_models


class FiLMLayer(ConvBase):
    """Feature-wise Linear Modulation layer that conditions a conv feature map on language."""

    def __init__(self, lang_emb_dim: int, channels: int):
        super().__init__()
        # Linear layer with half outputs for beta and half for gamma.
        self.lang_proj = nn.Linear(lang_emb_dim, channels * 2)
        self.relu = nn.ReLU()

    def output_shape(self, input_shape):
        return input_shape

    def forward(self, x: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        b, c, _h, _w = x.shape
        beta, gamma = torch.split(self.lang_proj(lang_emb).reshape(b, c * 2, 1, 1), [c, c], 1)
        # FiLM paper suggests modulating by (1 + gamma) so activations aren't zeroed out.
        x = (1 + gamma) * x + beta
        return self.relu(x)


class ResNet18ConvFiLM(ConvBase):
    """ResNet18 backbone with FiLM language conditioning between residual blocks.

    State_dict layout matches the robocasa-benchmark DP checkpoint: ``_base_block`` is the
    initial conv+BN stem, ``_conv_blocks`` is the flattened list of residual blocks, and
    ``_film_layers`` are per-block FiLM modulators driven by the language embedding.
    """

    def __init__(
        self,
        input_channel: int = 3,
        pretrained: bool = False,
        input_coord_conv: bool = False,
        lang_emb_dim: int = 768,
    ):
        super().__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        if input_coord_conv:
            # CoordConv2d — not needed for our supported envs; leaving unimplemented. Drop into the
            # regular path instead of silently mis-wiring, since the checkpoint config we target
            # uses input_coord_conv=False.
            raise NotImplementedError("input_coord_conv=True is not supported in the vendored DP port")
        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel

        # Split ResNet into: base block (stem), residual blocks, skipping the final FC / avgpool.
        layers = nn.ModuleList(net.children())
        base_block: list[nn.Module] = []
        conv_blocks: list[nn.Module] = []
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    conv_blocks.append(sub_layer)
            elif len(conv_blocks) == 0:
                base_block.append(layer)

        self._base_block = nn.Sequential(*base_block)
        self._conv_blocks = nn.ModuleList(conv_blocks)

        # One FiLM layer per residual block; channel count is inferred by a dummy forward.
        film_layers = []
        current_channels = self._base_block(torch.rand((1, input_channel, 3, 3))).shape[1]
        for conv in conv_blocks:
            current_channels = conv(torch.rand((1, current_channels, 3, 3))).shape[1]
            film_layers.append(FiLMLayer(lang_emb_dim, current_channels))

        self._film_layers = nn.ModuleList(film_layers)
        self._output_channels = current_channels

    def output_shape(self, input_shape):
        assert len(input_shape) == 3
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [self._output_channels, out_h, out_w]

    def forward(self, inputs: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        x = self._base_block(inputs)
        for conv, film in zip(self._conv_blocks, self._film_layers, strict=True):
            x = conv(x)
            x = film(x, lang_emb)
        return x

    def __repr__(self):
        return f"{type(self).__name__}(input_channel={self._input_channel}, input_coord_conv={self._input_coord_conv})"


class VisualCoreLanguageConditioned(VisualCore):
    """Variant of robomimic's VisualCore that routes a language embedding through the backbone.

    Registered automatically via ``EncoderCore.__init_subclass__`` — importing this module is
    enough for ``ObsUtils.initialize_obs_utils_with_config(...)`` to resolve the
    ``core_class: "VisualCoreLanguageConditioned"`` config entry that the DP checkpoint expects.

    Important: we drop ``self.backbone`` from ``self.nets`` after the parent constructor runs.
    The robocasa-benchmark pretrained checkpoint was saved by a VisualCore variant where
    ``self.nets = [pool, flatten, linear]`` (backbone is *not* inside the ``nn.Sequential``),
    while upstream robomimic prepends backbone to the list. Without this fixup the saved
    state_dict lands an ``nn.Sequential`` layout of ``[SpatialSoftmax, Flatten, Linear]``
    into a module whose ``nets[0]`` is the ResNet backbone, which shape-mismatches.
    """

    def __init__(
        self,
        input_shape,
        backbone_class: str = "ResNet18ConvFiLM",
        pool_class: str = "SpatialSoftmax",
        backbone_kwargs: dict | None = None,
        pool_kwargs: dict | None = None,
        flatten: bool = True,
        feature_dimension: int = 64,
    ):
        super().__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )
        # Rebuild nets without the backbone at index 0. The backbone stays reachable as
        # self.backbone and is invoked explicitly in forward() (with lang_emb).
        self.nets = nn.Sequential(*list(self.nets.children())[1:])

    def forward(self, inputs: torch.Tensor, lang_emb: torch.Tensor | None = None) -> torch.Tensor:
        assert lang_emb is not None, "VisualCoreLanguageConditioned requires a language embedding at forward-time"
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape), (
            f"expected input shape {self.input_shape}, got {tuple(inputs.shape)[-ndim:]}"
        )
        assert self.backbone is not None
        x = self.backbone(inputs, lang_emb)
        x = self.nets(x)
        expected = list(self.output_shape(list(inputs.shape)[1:]))
        actual = list(x.shape)[1:]
        if expected != actual:
            raise ValueError(f"Size mismatch: expected {expected}, got {actual}")
        return x


LANG_EMB_OBS_KEY = "lang_emb"
"""Conventional key used by NVlabs/sage + robocasa-benchmark DP to identify the language embedding
in ``obs_dict``. Our vendored DP policy expects the same key name (the released checkpoint's
shape_meta lists ``lang_emb`` as a low-dim obs)."""


def _register_backbone_classes() -> None:
    """Inject ResNet18ConvFiLM / FiLMLayer into robomimic so string-based class lookups find them.

    ``VisualCore.__init__`` resolves its ``backbone_class`` by calling ``eval(backbone_class)`` in
    ``robomimic.models.obs_core``'s module namespace. That namespace is populated by
    ``from robomimic.models.base_nets import *`` at import time — which, critically, captures the
    names that existed *when obs_core was imported*. We're adding new names afterwards, so the
    star-import snapshot doesn't pick them up. Set the attribute on both modules to cover every
    caller (``base_nets`` for any direct ``base_nets.ResNet18ConvFiLM`` reference, ``obs_core`` for
    the ``eval`` path inside ``VisualCore``).
    """
    from robomimic.models import base_nets as _rm_base_nets
    from robomimic.models import obs_core as _rm_obs_core

    for mod in (_rm_base_nets, _rm_obs_core):
        mod.ResNet18ConvFiLM = ResNet18ConvFiLM
        mod.FiLMLayer = FiLMLayer


def _patch_observation_encoder_for_lang() -> None:
    """Replace ``ObservationEncoder.forward`` with a variant that routes ``lang_emb`` to language-conditioned RGB modules.

    Upstream robomimic 0.3's encoder forward is:

        for k in self.obs_shapes:
            x = obs_dict[k]
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k](x)          # ← plain, no kwargs
            ...

    Language-conditioned ``VisualCoreLanguageConditioned.forward(x, lang_emb)`` requires a second
    argument; without this patch robomimic calls it with only ``x`` and asserts. We monkey-patch
    the upstream so the vendored DP model (and its released checkpoint) can forward.

    Patched behavior: iterate as upstream does, but when the obs_net is a
    ``VisualCoreLanguageConditioned``, pass ``lang_emb=obs_dict["lang_emb"]`` alongside ``x``. We
    deliberately do NOT drop the raw ``lang_emb`` feature from the concatenated output — the
    released robocasa-benchmark checkpoint was trained with ``lang_emb`` participating both as a
    FiLM modulator inside vision and as a separate low-dim feature in the global conditioning
    (``cond_obs_emb.weight`` has shape ``(512, 969) = 3*64 + 9 + 768``). Dropping it would give a
    ``(512, 201)`` projection and silently break checkpoint loading.
    """
    from robomimic.models import obs_nets as _rm_obs_nets
    import robomimic.utils.tensor_utils as _rm_tensor_utils
    import torch

    def forward(self, obs_dict):
        assert self._locked, "ObservationEncoder: @make has not been called yet"
        missing = set(self.obs_shapes.keys()) - set(obs_dict)
        if missing:
            raise AssertionError(
                f"ObservationEncoder: obs_dict missing {sorted(missing)} (has {list(obs_dict.keys())})"
            )

        lang_tensor = obs_dict.get(LANG_EMB_OBS_KEY)

        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            if self.obs_nets[k] is not None:
                module = self.obs_nets[k]
                if isinstance(module, VisualCoreLanguageConditioned):
                    if lang_tensor is None:
                        raise RuntimeError(
                            f"obs_net {k!r} is VisualCoreLanguageConditioned but "
                            f"obs_dict['{LANG_EMB_OBS_KEY}'] is missing"
                        )
                    x = module(x, lang_emb=lang_tensor)
                else:
                    x = module(x)
                if self.activation is not None:
                    x = self.activation(x)
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            x = _rm_tensor_utils.flatten(x, begin_axis=1)
            feats.append(x)

        return torch.cat(feats, dim=-1)

    _rm_obs_nets.ObservationEncoder.forward = forward


_register_backbone_classes()
_patch_observation_encoder_for_lang()
