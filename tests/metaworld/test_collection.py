"""Tests for metaworld activation collection via ``main.py --collect`` /
``eval_all.py --collect``.

After the unification with libero/robocasa, metaworld no longer has an
in-process collection path: --collect always talks to a policy server started
with --collect_activations. The vectorized rollout sends a list-shaped
__collect__ payload (one entry per env in the batch) so the server can save
one step_dir per env from a single forward pass. See BatchCollectionSession in
the openpi-client package.

Unit tests in this file (no GPU, no server) cover CLI argument handling and
that the script imports cleanly. The end-to-end pipeline (BatchCollectionSession
<-> CollectingPolicy) is exercised in tests/test_activation_collector.py with a
stub underlying policy.

Run unit tests only:
    uv run pytest tests/metaworld/test_collection.py -v -m "not manual"
"""

from __future__ import annotations

from pathlib import Path
import sys

# See tests/metaworld/test_metaworld_envs.py for the rationale: pytest may collect
# examples/{metaworld,libero,robocasa}/main.py in the same process, and the first
# one caches sys.modules['main']. Pop before our imports so examples/metaworld/
# lands on sys.path[0] first.
sys.modules.pop("main", None)
sys.modules.pop("eval_all", None)
_examples_dir = str(Path(__file__).parents[2] / "examples" / "metaworld")
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

import eval_all as mw_eval_all  # noqa: E402
import main as mw_main  # noqa: E402

# ── CLI / Args ───────────────────────────────────────────────────────────────


def test_main_args_defaults_normal_eval():
    args = mw_main.Args()
    # --collect off by default; no --policy.* fields exist anymore (server owns
    # the policy).
    assert args.collect is False
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert not hasattr(args, "policy")
    assert not hasattr(args, "collect_output_dir")


def test_eval_all_args_defaults():
    args = mw_eval_all.Args()
    assert args.collect is False
    assert args.tasks == []
    # Default split is the curated 26-task subset — tasks whose success rate
    # varies meaningfully across training checkpoints (see eval_all.py::SUBSET).
    assert args.split == "subset"
    # --gpus removed; multi-GPU collection is now "start one server per GPU"
    # at the user level, not orchestrated by this client.
    assert not hasattr(args, "gpus")
    assert not hasattr(args, "policy")
    assert not hasattr(args, "collect_output_dir")


def test_main_args_collect_flag_is_bool():
    """--collect is a plain bool toggle now: no policy_dir / config / output_dir
    sub-args to thread through. The server owns all of that."""
    args = mw_main.Args(collect=True)
    assert args.collect is True


# ── Module imports cleanly without GPU/torch ─────────────────────────────────


def test_main_module_loads_without_torch():
    """Normal eval path imports just websocket client + BatchCollectionSession.

    Even with --collect, the client side never touches torch/JAX/openpi —
    everything model-side lives on the server. This pins that contract so a
    future refactor can't accidentally re-introduce a torch import in main.py.
    """
    # main was already imported at module load above; just check no torch leak.
    # (Don't fail on torch already being present in sys.modules from other
    # tests — only fail if main.py itself referenced torch.)
    src = (Path(__file__).parents[2] / "examples" / "metaworld" / "main.py").read_text()
    assert "import torch" not in src
    assert "from openpi.models" not in src
    assert "from openpi.policies" not in src
    assert "from openpi.training" not in src


def test_eval_all_module_loads_without_torch():
    src = (Path(__file__).parents[2] / "examples" / "metaworld" / "eval_all.py").read_text()
    assert "import torch" not in src
    assert "from openpi.models" not in src
    assert "from openpi.policies" not in src
    assert "ensure_pytorch_checkpoint" not in src
