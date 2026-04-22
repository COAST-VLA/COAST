---
paths:
  - "tests/**"
---

# Test Rules

- Always use `uv run pytest`, never bare `pytest`
- The `manual` marker means GPU-required — these are skipped in CI
- CI default: `uv run pytest --strict-markers -m "not manual"`
- MetaWorld env tests need `MUJOCO_GL=egl`
- Activation validation tests use env vars to point to data:
  - `ACTIVATIONS_DIR` → `tests/test_activations.py` (pi0 / pi0.5 diffusion, `v1` schema) and `groot_env/tests/test_groot_activations.py` (GR00T N1.5, `groot_v1` schema)
  - `ACTIVATIONS_FAST_DIR` → `tests/test_activations_fast.py` (pi0-FAST, `fast_v1` schema)
- Test directories mirror source structure (`tests/models/`, `tests/policies/`, etc.)
