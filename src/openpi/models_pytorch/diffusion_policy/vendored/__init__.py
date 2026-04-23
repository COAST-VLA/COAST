"""Vendored Diffusion Policy code from ``robocasa-benchmark/diffusion_policy``.

Ported verbatim (Apache License 2.0, Copyright Stanford AI and Columbia ARX) so that the
robocasa-benchmark pretrained checkpoint loads bit-for-bit into openpi — specifically:

    https://huggingface.co/robocasa/robocasa365_checkpoints
    diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300

The upstream repo is https://github.com/robocasa-benchmark/diffusion_policy. We vendor a
flat subset of files (10 from their tree + a small ``robomimic_extensions`` module that
carries NVlabs/sage's ``VisualCoreLanguageConditioned`` / ``ResNet18ConvFiLM`` / ``FiLMLayer``
additions so robomimic can resolve the config's ``core_class: "VisualCoreLanguageConditioned"``
entry) rather than adding a git submodule. Internal ``diffusion_policy.*`` imports were
rewritten to relative imports under this package.

Importing this package triggers ``robomimic_extensions`` which registers its new classes
with robomimic. Everything else is lazy-imported by callers.
"""

from . import robomimic_extensions  # noqa: F401 — import for registration side-effect
