"""Unit tests for ``openpi.transforms.ComputeLangEmb``.

Marked ``manual`` so CI without GPU + network access can skip the CLIP model download.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.manual
def test_compute_lang_emb_produces_correct_shape_and_cache():
    from openpi.transforms import ComputeLangEmb

    enc = ComputeLangEmb(device="cpu")  # keep CPU to avoid GPU contention in tests
    out1 = enc({"prompt": "pick up the red cube", "state": np.zeros(4, dtype=np.float32)})
    assert "lang_emb" in out1
    assert out1["lang_emb"].dtype == np.float32
    assert out1["lang_emb"].shape == (768,)  # CLIP ViT-L/14 projection_dim
    # prompt is dropped by default (its string leaf would break PyTorch collation).
    assert "prompt" not in out1
    # state and other keys pass through.
    assert "state" in out1

    # Same prompt → identical vector (cache hit, bit-exact).
    out2 = enc({"prompt": "pick up the red cube", "state": np.ones(4, dtype=np.float32)})
    np.testing.assert_array_equal(out1["lang_emb"], out2["lang_emb"])

    # Different prompt → different vector.
    out3 = enc({"prompt": "push the blue button", "state": np.zeros(4, dtype=np.float32)})
    assert not np.array_equal(out1["lang_emb"], out3["lang_emb"])


@pytest.mark.manual
def test_compute_lang_emb_noop_without_prompt():
    from openpi.transforms import ComputeLangEmb

    enc = ComputeLangEmb(device="cpu")
    # Omit prompt → transform is a no-op (doesn't raise, doesn't add lang_emb).
    out = enc({"state": np.zeros(4, dtype=np.float32)})
    assert "lang_emb" not in out
    assert "prompt" not in out


@pytest.mark.manual
def test_compute_lang_emb_accepts_bytes_and_numpy_scalar():
    from openpi.transforms import ComputeLangEmb

    enc = ComputeLangEmb(device="cpu")
    out_bytes = enc({"prompt": b"reach the goal", "state": np.zeros(4, dtype=np.float32)})
    out_str = enc({"prompt": "reach the goal", "state": np.zeros(4, dtype=np.float32)})
    np.testing.assert_array_equal(out_bytes["lang_emb"], out_str["lang_emb"])

    out_nparr = enc({"prompt": np.array("reach the goal"), "state": np.zeros(4, dtype=np.float32)})
    np.testing.assert_array_equal(out_nparr["lang_emb"], out_str["lang_emb"])


@pytest.mark.manual
def test_compute_lang_emb_keep_prompt_option():
    from openpi.transforms import ComputeLangEmb

    enc = ComputeLangEmb(device="cpu", drop_prompt=False)  # drop_prompt is now kwarg-only
    out = enc({"prompt": "close the drawer", "state": np.zeros(4, dtype=np.float32)})
    assert "prompt" in out
    assert out["prompt"] == "close the drawer"
    assert "lang_emb" in out
