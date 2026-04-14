---
paths:
  - "tests/**"
---

# Test Rules

- Always use `uv run pytest`, never bare `pytest`
- The `manual` marker means GPU-required — these are skipped in CI
- CI default: `uv run pytest --strict-markers -m "not manual"`
- MetaWorld env tests need `MUJOCO_GL=egl`
- Activation validation tests use env vars to point to data: `ACTIVATIONS_DIR`, `ACTIVATIONS_V2_DIR`, `ACTIVATIONS_V2_BASE`
- Test directories mirror source structure (`tests/models/`, `tests/policies/`, etc.)
