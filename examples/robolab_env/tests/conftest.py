"""Pytest bootstrap for examples/robolab_env tests.

Two side effects, both needed before any test imports ``main``:

1. **Add ``examples/robolab_env`` to ``sys.path``** so tests can
   ``import main`` and ``import tasks`` directly. pytest does not add the
   parent of the ``tests/`` folder to sys.path automatically.

2. **Provide a helper for tests to sanitize ``sys.argv``** before importing
   ``main``. ``main.py`` constructs ``AppLauncher`` at module load time,
   and Omniverse Kit's internal bootstrap (called from ``SimulationApp``)
   re-scans ``sys.argv`` on its own — it is NOT the same parser as
   ``AppLauncher.add_app_launcher_args``. Any pytest-style argument it
   doesn't recognise (e.g. ``-m``, ``-v``, test node ids) is flagged as
   "Ill formed parameter" and segfaults the whole process. So we have to
   strip ``sys.argv`` down to ``[argv[0], "--headless"]`` before the first
   ``import main`` in the session. We do that in the session-scoped
   ``main_module`` fixture rather than here, so CI — which never triggers
   the fixture because it filters ``-m "not manual"`` — sees untouched
   argv.

Tests that need a real Isaac Sim instance are marked ``manual`` and boot
the simulator lazily inside the ``main_module`` fixture — not at module
load. That way the test module remains importable without ``isaacsim``
installed (so CI, which only discovers tests, does not pay the boot cost
or error out on the missing dep).
"""

from __future__ import annotations

import pathlib
import sys

_ROBOLAB_ENV_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_ROBOLAB_ENV_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOLAB_ENV_DIR))
