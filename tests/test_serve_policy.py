"""Unit tests for scripts/serve_policy.py argument validation.

The serve script gates --collect_activations on --pytorch vs JAX based on the
configured model_type (pi0 / pi0.5 need PyTorch forward hooks; pi0-fast has
no PyTorch port of its autoregressive decode and must use JAX). These tests
pin the guard logic without loading a real checkpoint — ``create_policy`` is
stubbed so the branches are exercised without a GPU.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

# serve_policy.py lives under scripts/ and isn't a package; import it via file path.
sys.modules.pop("serve_policy", None)
_scripts_dir = str(Path(__file__).parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import serve_policy  # noqa: E402

from openpi.models import model as _model  # noqa: E402


def _fake_train_config(model_type: _model.ModelType):
    return types.SimpleNamespace(model=types.SimpleNamespace(model_type=model_type))


def _patch_get_config(monkeypatch, model_type: _model.ModelType) -> None:
    """Patch get_config on the canonical openpi.training.config module.

    serve_policy does ``from openpi.training import config as _config``, so
    binding into the canonical module propagates to the serve_policy reference
    without us having to touch its alias (which ruff flags as private-access).
    """
    from openpi.training import config as _config

    monkeypatch.setattr(_config, "get_config", lambda name: _fake_train_config(model_type))


@pytest.fixture
def stub_policy_creation(monkeypatch):
    """Stub create_policy + CollectingPolicy + WebsocketPolicyServer so main()
    runs through the guard logic without touching any real machinery.

    Returns a list the test can inspect to check what CollectingPolicy was
    called with.
    """
    collecting_calls = []

    class _FakePolicy:
        def __init__(self):
            self.metadata = {"fake": True}

    class _FakeCollecting:
        def __init__(self, **kwargs):
            collecting_calls.append(kwargs)
            self.metadata = {"fake": True}

    class _FakeServer:
        def __init__(self, **kwargs):
            pass

        def serve_forever(self):
            # The real server blocks; main() calls this after the guards, so
            # by intercepting here we skip network I/O but still let the
            # guard logic run.
            pass

    monkeypatch.setattr(serve_policy, "create_policy", lambda args: _FakePolicy())
    monkeypatch.setattr(serve_policy, "CollectingPolicy", _FakeCollecting)
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _FakeServer)
    return collecting_calls


def _args(*, collect_activations: bool, pytorch: bool, config: str = "some_config") -> serve_policy.Args:
    return serve_policy.Args(
        collect_activations=collect_activations,
        pytorch=pytorch,
        output_dir="/tmp/never-used",
        policy=serve_policy.Checkpoint(config=config, dir="/tmp/fake/5000"),
    )


def test_collect_requires_checkpoint_not_default(monkeypatch, stub_policy_creation):
    """--collect_activations with a default policy must raise — no way to
    derive checkpoint_step or config_name from a Default."""
    args = serve_policy.Args(collect_activations=True, pytorch=True)  # default policy
    _patch_get_config(monkeypatch, _model.ModelType.PI05)
    with pytest.raises(ValueError, match="requires --policy=checkpoint"):
        serve_policy.main(args)


def test_collect_pi0_fast_without_pytorch_is_accepted(monkeypatch, stub_policy_creation):
    """pi0-fast is JAX-only, so --collect_activations without --pytorch must
    go through the collection path (not raise)."""
    _patch_get_config(monkeypatch, _model.ModelType.PI0_FAST)
    args = _args(collect_activations=True, pytorch=False, config="pi0_fast_libero")
    serve_policy.main(args)  # must not raise
    assert len(stub_policy_creation) == 1
    assert stub_policy_creation[0]["model_type"] == _model.ModelType.PI0_FAST


def test_collect_pi0_fast_with_pytorch_is_rejected(monkeypatch, stub_policy_creation):
    """pi0-fast + --pytorch is a user error: no PyTorch port of the decode exists.
    Rejecting loudly is better than silently ignoring --pytorch."""
    _patch_get_config(monkeypatch, _model.ModelType.PI0_FAST)
    args = _args(collect_activations=True, pytorch=True, config="pi0_fast_libero")
    with pytest.raises(ValueError, match="--pytorch cannot be combined with a pi0-fast model"):
        serve_policy.main(args)


def test_collect_pi05_without_pytorch_is_rejected(monkeypatch, stub_policy_creation):
    """pi0.5 collection needs PyTorch forward hooks; JAX pi0.5 has no
    sample_actions_with_intermediates."""
    _patch_get_config(monkeypatch, _model.ModelType.PI05)
    args = _args(collect_activations=True, pytorch=False, config="pi05_libero")
    with pytest.raises(ValueError, match="--collect_activations requires --pytorch for pi05"):
        serve_policy.main(args)


def test_collect_pi05_with_pytorch_is_accepted(monkeypatch, stub_policy_creation):
    """pi0.5 with --pytorch + --collect_activations must go through cleanly."""
    _patch_get_config(monkeypatch, _model.ModelType.PI05)
    args = _args(collect_activations=True, pytorch=True, config="pi05_libero")
    serve_policy.main(args)
    assert len(stub_policy_creation) == 1
    assert stub_policy_creation[0]["model_type"] == _model.ModelType.PI05


def test_collect_pi0_without_pytorch_is_rejected(monkeypatch, stub_policy_creation):
    """pi0 (non-pi0.5, non-pi0-fast) also needs PyTorch for forward-hook
    intermediates."""
    _patch_get_config(monkeypatch, _model.ModelType.PI0)
    args = _args(collect_activations=True, pytorch=False, config="pi0_aloha_sim")
    with pytest.raises(ValueError, match="--collect_activations requires --pytorch for pi0"):
        serve_policy.main(args)
