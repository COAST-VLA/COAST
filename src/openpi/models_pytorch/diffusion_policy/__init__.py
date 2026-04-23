"""Diffusion Policy (Transformer-Hybrid variant) — openpi wrapper.

Public surface:
  - ``DiffusionPolicyConfig``: openpi BaseModelConfig describing the shape_meta + architecture.
  - ``DiffusionPolicy``: the PyTorch model wrapper, built around the vendored
    ``DiffusionTransformerHybridImagePolicy`` from ``robocasa-benchmark/diffusion_policy``.
  - ``ImageSpec`` / ``LowdimSpec``: per-key shape_meta fragments used by the config.

Importing this package triggers registration of vendored robomimic extensions
(``VisualCoreLanguageConditioned`` / ``ResNet18ConvFiLM`` / ``FiLMLayer``) needed for
checkpoint-format compatibility with the released robocasa checkpoint.
"""

from openpi.models_pytorch.diffusion_policy.config import DiffusionPolicyConfig
from openpi.models_pytorch.diffusion_policy.config import ImageSpec
from openpi.models_pytorch.diffusion_policy.config import LowdimSpec
from openpi.models_pytorch.diffusion_policy.modeling import DiffusionPolicy

__all__ = ["DiffusionPolicy", "DiffusionPolicyConfig", "ImageSpec", "LowdimSpec"]
