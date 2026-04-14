---
paths:
  - "examples/libero_env/**"
---

# LIBERO Environment Rules

This directory has its own **Python 3.8 virtual environment**, separate from the root venv.

## Critical: Venv Isolation

- Always `cd examples/libero_env` before running `uv run` commands
- Never run LIBERO client code from the root directory — it will use Python 3.11 and fail
- The policy **server** runs from the root venv; only the **client** uses this venv
- After modifying `packages/openpi-client/`, re-sync: `cd examples/libero_env && uv sync`

## Commands

```bash
cd examples/libero_env
uv sync                                    # first-time setup
uv run python setup_libero_config.py       # writes ~/.libero/config.yaml (rerun if repo moves)
MUJOCO_GL=egl uv run python main.py ...   # single task eval
MUJOCO_GL=egl uv run python eval_all.py ...  # full suite eval
```

## Activation Collection

Uses server-client architecture. Start a `--collect_activations --pytorch` server from root, then run client with `--collect` flag from this directory. See README.md for details.
