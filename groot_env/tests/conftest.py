"""Pytest session bootstrap for groot_env tests.

Adds `groot_env/` to sys.path so test files can `import groot_adapter` and
`import groot_activation_collector` directly. pytest does not add the parent
of the tests/ folder to sys.path automatically — same pattern as
`examples/libero_env/tests/conftest.py` and
`examples/robocasa_env/tests/conftest.py`.
"""

from __future__ import annotations

import pathlib
import sys

_GROOT_ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_GROOT_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_GROOT_ENV_DIR))
