"""Smoke tests for the separate `examples/libero_env` setup.

These tests intentionally execute the real LIBERO simulation inside
`examples/libero_env/.venv`, since the main openpi test environment does not
carry the LIBERO dependency stack.

Run locally:
    uv run pytest tests/libero/test_libero_env.py -v -m manual
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_ENV_PYTHON = REPO_ROOT / "examples" / "libero_env" / ".venv" / "bin" / "python"
SETUP_SCRIPT = REPO_ROOT / "examples" / "libero_env" / "setup_libero_config.py"


def _require_libero_env() -> None:
    if not LIBERO_ENV_PYTHON.exists():
        pytest.skip("examples/libero_env/.venv/bin/python is missing; sync the dedicated LIBERO env first")


def _run_setup_libero_config(*, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    _require_libero_env()

    return subprocess.run(
        [str(LIBERO_ENV_PYTHON), str(SETUP_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_libero_env_script(script: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    _require_libero_env()

    setup_completed = _run_setup_libero_config()
    if setup_completed.returncode != 0:
        pytest.fail(
            "LIBERO setup script failed\n"
            f"returncode: {setup_completed.returncode}\n"
            f"stdout:\n{setup_completed.stdout}\n"
            f"stderr:\n{setup_completed.stderr}"
        )

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    try:
        completed = subprocess.run(
            [str(LIBERO_ENV_PYTHON), "-c", textwrap.dedent(script)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"LIBERO subprocess timed out after {timeout}s: {exc}")

    if completed.returncode != 0:
        pytest.fail(
            "LIBERO subprocess failed\n"
            f"returncode: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return completed


def _parse_last_json_line(stdout: str) -> dict:
    for line in reversed([line.strip() for line in stdout.splitlines()]):
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    pytest.fail(f"No JSON payload found in subprocess stdout:\n{stdout}")


@pytest.mark.manual
def test_libero_env_reset_step_smoke() -> None:
    """The dedicated LIBERO env can create, reset, set an init state, and step once."""
    completed = _run_libero_env_script(
        """
        import json
        import sys

        sys.path.insert(0, 'examples/libero_env')
        import main

        suite = main.get_task_suite('libero_spatial')
        task = suite.get_task(0)
        env = main.make_env(task, main.LIBERO_ENV_RESOLUTION, 7)
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(0)[0])
            obs2, reward, done, info = env.step(main.LIBERO_DUMMY_ACTION)
            payload = {
                'agentview_shape': list(obs['agentview_image'].shape),
                'agentview_dtype': str(obs['agentview_image'].dtype),
                'wrist_shape': list(obs['robot0_eye_in_hand_image'].shape),
                'wrist_dtype': str(obs['robot0_eye_in_hand_image'].dtype),
                'state_shape': list(main.build_state(obs).shape),
                'state_dtype': str(main.build_state(obs).dtype),
                'reward_type': type(reward).__name__,
                'done': bool(done),
                'info_type': type(info).__name__,
                'next_obs_has_agentview': 'agentview_image' in obs2,
            }
            print(json.dumps(payload))
        finally:
            env.close()
        """
    )
    payload = _parse_last_json_line(completed.stdout)

    assert payload["agentview_shape"] == [256, 256, 3]
    assert payload["agentview_dtype"] == "uint8"
    assert payload["wrist_shape"] == [256, 256, 3]
    assert payload["wrist_dtype"] == "uint8"
    assert payload["state_shape"] == [8]
    assert payload["state_dtype"] == "float32"
    assert payload["reward_type"] == "float"
    assert payload["info_type"] == "dict"
    assert payload["next_obs_has_agentview"] is True


@pytest.mark.manual
def test_libero_eval_task_runs_with_stub_policy_and_writes_video(tmp_path: Path) -> None:
    """The full eval loop runs in the dedicated env with a stub policy and produces a video."""
    output_dir = tmp_path / "libero-output"
    completed = _run_libero_env_script(
        f"""
        import json
        import os
        import pathlib
        import sys

        import numpy as np

        sys.path.insert(0, 'examples/libero_env')
        import main

        class StubPolicy:
            def infer(self, element):
                assert element['observation/image'].shape == (224, 224, 3)
                assert element['observation/wrist_image'].shape == (224, 224, 3)
                assert element['observation/state'].shape == (8,)
                assert isinstance(element['prompt'], str) and element['prompt']
                return {{'actions': np.zeros((1, 7), dtype=np.float32)}}

        args = main.Args(
            num_episodes=1,
            max_steps=1,
            num_steps_wait=1,
            replan_steps=1,
            fps=2,
            seed=7,
        )
        output_dir = pathlib.Path({str(output_dir)!r})
        result = main.eval_task('libero_spatial', 0, StubPolicy(), args, str(output_dir))
        videos = sorted(output_dir.rglob('*.mp4'))
        payload = {{
            'result_keys': sorted(result.keys()),
            'success_rate': result['success_rate'],
            'num_episodes': result['num_episodes'],
            'video_count': len(videos),
            'video_exists': bool(videos and videos[0].exists()),
            'video_size': videos[0].stat().st_size if videos else 0,
        }}
        print(json.dumps(payload))
        """
    )
    payload = _parse_last_json_line(completed.stdout)

    assert payload["result_keys"] == [
        "num_episodes",
        "success_rate",
        "task_description",
        "task_id",
        "task_name",
    ]
    assert payload["success_rate"] in {0.0, 1.0}
    assert payload["num_episodes"] == 1.0
    assert payload["video_count"] == 1
    assert payload["video_exists"] is True
    assert payload["video_size"] > 0


@pytest.mark.manual
def test_setup_libero_config_writes_default_config() -> None:
    """The explicit setup script writes the default ~/.libero config for this checkout."""
    completed = _run_setup_libero_config()
    if completed.returncode != 0:
        pytest.fail(
            "LIBERO setup script failed\n"
            f"returncode: {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    completed = _run_libero_env_script(
        """
        import json
        import pathlib

        config_dir = pathlib.Path.home() / ".libero"
        config_path = config_dir / 'config.yaml'
        payload = {
            'config_exists': config_path.exists(),
            'config_dir_name': config_dir.name,
            'config_text': config_path.read_text(),
        }
        print(json.dumps(payload))
        """
    )
    payload = _parse_last_json_line(completed.stdout)

    assert payload["config_exists"] is True
    assert payload["config_dir_name"] == ".libero"
    assert "benchmark_root:" in payload["config_text"]
    assert "bddl_files:" in payload["config_text"]
    assert "init_states:" in payload["config_text"]
    assert "assets:" in payload["config_text"]
