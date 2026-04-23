"""End-to-end: load the robocasa-benchmark DP checkpoint and run inference through our wrapper.

Verifies the vendored Transformer-Hybrid DP port is bit-compatible with the released
pretrain_human300 checkpoint:
  https://huggingface.co/robocasa/robocasa365_checkpoints
  diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/
  checkpoints/epoch=0500-test_mean_score=-1.000.ckpt

The test asserts that:
  - The checkpoint ``.ckpt`` parses and loads with strict state_dict matching (0 missing / 0
    unexpected), proving every tensor in the upstream save lands in our vendored module tree.
  - The wrapper's ``sample_actions`` / ``sample_actions_from_dict`` produces finite actions of
    the right shape on synthetic observations matching the checkpoint's shape_meta.

This is a ``manual`` test because it needs a GPU (CPU forward passes are prohibitively slow for
the 106M-parameter Transformer) and the ~1.6 GB .ckpt cached at ``CHECKPOINT_PATH``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    REPO_ROOT
    / "checkpoints/robocasa_dp/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=0500-test_mean_score=-1.000.ckpt"
)


def _skip_if_missing_prereqs() -> None:
    if not CHECKPOINT_PATH.exists():
        pytest.skip(
            f"RoboCasa DP checkpoint missing at {CHECKPOINT_PATH}. Download with:\n"
            "  hf download robocasa/robocasa365_checkpoints "
            "--include 'diffusion_policy/17.40.09_*/checkpoints/*.ckpt' "
            "--local-dir checkpoints/robocasa_dp"
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.manual
def test_dp_robocasa_loads_ckpt_and_samples_actions():
    """Load the released robocasa DP checkpoint; synthesize a shape_meta-keyed batch; sample actions."""
    _skip_if_missing_prereqs()
    import torch

    from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
    from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig

    # Defaults match the released checkpoint's cfg.policy block exactly — anything different would
    # produce a state_dict shape mismatch at load time.
    cfg = DiffusionPolicyConfig()
    assert cfg.action_dim == 12
    assert cfg.horizon == 10
    assert cfg.n_obs_steps == 2
    assert cfg.lang_emb_dim == 768

    model = DiffusionPolicy(cfg).eval()
    model.load_weights(str(CHECKPOINT_PATH))  # strict=True inside load_weights — will raise on any mismatch
    model = model.cuda()

    # Sanity check: param count matches what the published checkpoint advertises (~106 M).
    param_count = sum(p.numel() for p in model.parameters())
    assert 100e6 < param_count < 112e6, f"unexpected param count {param_count / 1e6:.1f}M"

    # Synthesize a shape_meta-keyed obs dict matching what the vendored policy expects:
    # every value has shape (B, n_obs_steps, ...).
    b, t = 1, cfg.n_obs_steps
    obs_dict: dict[str, torch.Tensor] = {}
    for img_spec in cfg.images:
        obs_dict[img_spec.key] = torch.rand(b, t, 3, img_spec.height, img_spec.width, device="cuda")
    for ld in cfg.lowdims:
        obs_dict[ld.key] = torch.randn(b, t, ld.dim, device="cuda")
    obs_dict["lang_emb"] = torch.randn(b, t, cfg.lang_emb_dim, device="cuda")

    with torch.no_grad():
        actions = model.sample_actions_from_dict(obs_dict)

    assert actions.shape == (b, cfg.horizon, cfg.action_dim), f"got {actions.shape}"
    assert torch.isfinite(actions).all(), "non-finite actions"


@pytest.mark.manual
def test_dp_robocasa_sample_actions_openpi_observation_path():
    """Same as above but through the openpi-Observation path, to cover the training/serving entry point."""
    _skip_if_missing_prereqs()
    import torch

    from openpi.models.model import Observation
    from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
    from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig

    cfg = DiffusionPolicyConfig()
    model = DiffusionPolicy(cfg).eval()
    model.load_weights(str(CHECKPOINT_PATH))
    model = model.cuda()

    b = 1
    # Images in openpi format: NHWC uint8 in [0, 255], converted to float [-1, 1] by
    # Observation.from_dict. We pre-mix to [-1, 1] floats since we construct Observation directly.
    imgs = {spec.key: (torch.rand(b, spec.height, spec.width, 3, device="cuda") * 2 - 1) for spec in cfg.images}
    # state is the flat concat of all lowdims in declared order.
    state_dim_total = sum(ld.dim for ld in cfg.lowdims)
    obs = Observation(
        images=imgs,
        image_masks={k: torch.ones(b, dtype=torch.bool, device="cuda") for k in imgs},
        state=torch.randn(b, state_dim_total, device="cuda"),
    )
    lang_emb = torch.randn(b, cfg.lang_emb_dim, device="cuda")

    with torch.no_grad():
        actions = model.sample_actions("cuda", obs, lang_emb=lang_emb)

    assert actions.shape == (b, cfg.horizon, cfg.action_dim)
    assert torch.isfinite(actions).all()


@pytest.mark.manual
def test_dp_robocasa_strict_load_detects_mismatches():
    """If the config diverges from the checkpoint's architecture, strict load_state_dict must fail.

    This guards against silent regressions in the vendored DP code — any change that alters a
    tensor name or shape should break this test loudly instead of proceeding with an incorrectly
    initialized model.
    """
    _skip_if_missing_prereqs()

    from openpi.models_pytorch.diffusion_policy import DiffusionPolicy
    from openpi.models_pytorch.diffusion_policy import DiffusionPolicyConfig

    # n_emb=256 instead of 512 forces every transformer weight to a different shape.
    cfg = DiffusionPolicyConfig(n_emb=256)
    model = DiffusionPolicy(cfg).eval()
    with pytest.raises(RuntimeError, match=r"size mismatch|Error\(s\) in loading state_dict"):
        model.load_weights(str(CHECKPOINT_PATH))
