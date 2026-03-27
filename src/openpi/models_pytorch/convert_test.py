"""Tests for PyTorch auto-conversion and hash caching."""

import json
from unittest.mock import patch

import pytest

from openpi.models_pytorch.convert import _compute_checkpoint_hash
from openpi.models_pytorch.convert import ensure_pytorch_checkpoint


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Create a minimal fake JAX checkpoint directory for testing."""
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    metadata = params_dir / "_METADATA"
    metadata.write_text("fake_metadata_content_12345")

    checkpoint_metadata = tmp_path / "_CHECKPOINT_METADATA"
    checkpoint_metadata.write_text(json.dumps({"commit_timestamp_nsecs": 12345}))

    return tmp_path


class TestComputeCheckpointHash:
    def test_deterministic(self, fake_checkpoint):
        h1 = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        h2 = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        assert h1 == h2

    def test_different_config_different_hash(self, fake_checkpoint):
        h1 = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        h2 = _compute_checkpoint_hash(fake_checkpoint, "pi0_aloha_sim")
        assert h1 != h2

    def test_different_metadata_different_hash(self, fake_checkpoint):
        h1 = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        (fake_checkpoint / "params" / "_METADATA").write_text("different_content")
        h2 = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        assert h1 != h2

    def test_fallback_to_checkpoint_metadata(self, tmp_path):
        """When params/_METADATA doesn't exist, falls back to _CHECKPOINT_METADATA."""
        fallback = tmp_path / "_CHECKPOINT_METADATA"
        fallback.write_text("fallback_content")
        h = _compute_checkpoint_hash(tmp_path, "test")
        assert len(h) == 64  # SHA-256 hex digest

    def test_no_metadata_still_works(self, tmp_path):
        """Hash should still work (based on config name alone) if no metadata files exist."""
        h = _compute_checkpoint_hash(tmp_path, "test")
        assert len(h) == 64


class TestEnsurePytorchCheckpoint:
    def test_skips_when_up_to_date(self, fake_checkpoint):
        """Should skip conversion when hash matches."""
        safetensors = fake_checkpoint / "model.safetensors"
        safetensors.write_text("fake_model")
        hash_path = fake_checkpoint / ".pytorch_conversion_hash"
        current_hash = _compute_checkpoint_hash(fake_checkpoint, "pi05_metaworld")
        hash_path.write_text(current_hash)

        # Should not call convert (we mock it to verify)
        with patch("openpi.models_pytorch.convert._import_conversion_module") as mock_import:
            ensure_pytorch_checkpoint(str(fake_checkpoint), "pi05_metaworld")
            mock_import.assert_not_called()

    def test_converts_when_no_safetensors(self, fake_checkpoint):
        """Should trigger conversion when model.safetensors doesn't exist."""
        mock_module = type("MockModule", (), {"convert_pi0_checkpoint": lambda *a: None})()

        with (
            patch("openpi.models_pytorch.convert._import_conversion_module", return_value=mock_module),
            patch("openpi.models_pytorch.convert._config.get_config") as mock_config,
        ):
            mock_config.return_value.model = "fake_model_config"

            # Create model.safetensors as side effect (the real conversion would do this)
            def fake_convert(*args):
                (fake_checkpoint / "model.safetensors").write_text("converted")

            mock_module.convert_pi0_checkpoint = fake_convert

            ensure_pytorch_checkpoint(str(fake_checkpoint), "pi05_metaworld")

            # Hash should now be saved
            hash_path = fake_checkpoint / ".pytorch_conversion_hash"
            assert hash_path.exists()
            assert len(hash_path.read_text().strip()) == 64

    def test_reconverts_when_hash_stale(self, fake_checkpoint):
        """Should reconvert when stored hash doesn't match current."""
        safetensors = fake_checkpoint / "model.safetensors"
        safetensors.write_text("old_model")
        hash_path = fake_checkpoint / ".pytorch_conversion_hash"
        hash_path.write_text("stale_hash_value")

        mock_module = type("MockModule", (), {"convert_pi0_checkpoint": lambda *a: None})()
        convert_called = []

        def fake_convert(*args):
            convert_called.append(True)

        mock_module.convert_pi0_checkpoint = fake_convert

        with (
            patch("openpi.models_pytorch.convert._import_conversion_module", return_value=mock_module),
            patch("openpi.models_pytorch.convert._config.get_config") as mock_config,
        ):
            mock_config.return_value.model = "fake_model_config"
            ensure_pytorch_checkpoint(str(fake_checkpoint), "pi05_metaworld")

        assert len(convert_called) == 1
        # Hash should be updated
        new_hash = hash_path.read_text().strip()
        assert new_hash != "stale_hash_value"
