---
paths:
  - "examples/robocasa_env/**"
---

# RoboCasa Environment Rules

This directory has its own **separate virtual environment** (Python 3.11, but isolated dependencies).

## Critical: Venv Isolation

- Always `cd examples/robocasa_env` before running `uv run` commands
- Never run RoboCasa client code from the root directory — wrong dependencies
- The policy **server** runs from the root venv; only the **client** uses this venv
- After modifying `packages/openpi-client/`, re-sync: `cd examples/robocasa_env && uv sync`

## Commands

```bash
cd examples/robocasa_env
uv sync                                    # first-time setup
uv run python -m robocasa.scripts.setup_macros
uv run python -m robocasa.scripts.download_kitchen_assets  # ~10GB download
MUJOCO_GL=egl uv run python main.py ...   # single task eval
MUJOCO_GL=egl uv run python eval_all.py ...  # full task set eval
```

## Activation Collection

Uses server-client architecture (same protocol as LIBERO). Start a `--collect_activations --pytorch` server from root, then run client with `--collect` flag from this directory. See README.md for details.
