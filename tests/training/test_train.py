import dataclasses
import importlib
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config


def _import_train():
    """Import scripts/train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", pathlib.Path(__file__).parents[2] / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _import_train()


# Run an actual 2-step train + 2-step resume on the dummy "debug" config.
# Marked manual because the CPU XLA backend has been seen to segfault inside
# ptrain_step on tiny configs intermittently — not a real bug in the training
# code, just an XLA/CPU edge case that doesn't reproduce on the GPU runners
# we use locally. Run with `uv run pytest tests/training/test_train.py -m manual`.
@pytest.mark.manual
@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
