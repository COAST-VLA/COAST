---
paths:
  - "groot_env/**"
  - "third_party/Isaac-GR00T/**"
---

# GR00T Environment Rules

`groot_env/` is an **NVIDIA GR00T N1.5 policy server** in its own Python 3.10 venv at the repo root. Peer to `examples/` (clients) and `scripts/` (pi0 server). Submodule `third_party/Isaac-GR00T` pinned at the `n1.5-release` tag.

## Critical: this is a SERVER, not a client

Clients stay in `examples/robocasa_env/` (etc.) and just point at the server's port. Same WebSocket protocol as `scripts/serve_policy.py`; `groot_env/groot_adapter.py` handles the pi0-protocol ↔ GR00T-nested-dict translation.

## Venv isolation

- Always `cd groot_env` before `uv run`. Python 3.10 here, 3.11 at root — wrong dir picks the wrong interpreter.
- After modifying `packages/openpi-client/`, re-sync: `cd groot_env && uv sync`.

## Commands

```bash
cd groot_env
GIT_LFS_SKIP_SMUDGE=1 uv sync                                    # first-time setup
uv pip install --no-build-isolation flash-attn==2.7.1.post4      # required for N1.5 backbone; not in main deps
export CUDA_VISIBLE_DEVICES=0
uv run python serve.py --port 8000                               # serve
uv run python serve.py --port 8000 --collect_activations --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 --output-dir ../activations/groot_n15-robocasa-activations-v1-15env
```

## Activation Collection

Schema mirrors pi0's `sample_actions_with_intermediates` one-for-one where architecture allows; only per-step `.npz` filenames differ (see `groot_env/README.md`). The adapter's collection path uses a hybrid re-implemented-loop + forward-hooks pattern — identical to `Pi0Pytorch.sample_actions_with_intermediates`. `TestInferEquivalenceRealModel` (`@pytest.mark.manual`) asserts bit-identical output vs upstream `Gr00tPolicy.get_action`; run locally after touching `_get_action_with_intermediates`.
