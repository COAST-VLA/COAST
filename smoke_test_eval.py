"""One-episode smoke test of latest.ckpt on a single RoboCasa task."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OMP_NUM_THREADS"] = "1"

# robocasa's env publishes its observation_space as a raw collections.OrderedDict rather than a
# gym.spaces.Dict. AsyncVectorEnv with shared_memory=True can't handle that. For a 1-env smoke
# test we just swap to SyncVectorEnv.
import diffusion_policy.env_runner.robomimic_image_runner as _runner_mod
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv as _SyncVectorEnv

def _SyncVectorEnvWrapper(env_fns, dummy_env_fn=None, **_kwargs):
    return _SyncVectorEnv(env_fns)

_runner_mod.AsyncVectorEnv = _SyncVectorEnvWrapper

# SyncVectorEnv.reset_wait / step_wait call gym's concatenate with the 0.21-era arg order
# (items, out, space). gym 0.26 flipped it to (space, items, out). Rebind the name imported
# *into sync_vector_env* only — do not touch gym internals, since gym's own Dict concatenate
# already uses the modern order.
import diffusion_policy.gym_util.sync_vector_env as _sync_mod
_real_concatenate = _sync_mod.concatenate
def _concatenate_old_order(items, out, space):
    return _real_concatenate(space, items, out)
_sync_mod.concatenate = _concatenate_old_order

from eval_robocasa import eval_task

CHECKPOINT = "/home/kim34/projects/diffusion_policy/checkpoints/latest.ckpt"
OUTPUT_DIR = "/home/kim34/projects/diffusion_policy/smoke_test_output"
TASK = "CloseStandMixerHead"
SPLIT = "pretrain"

if __name__ == "__main__":
    eval_task(
        checkpoint=CHECKPOINT,
        base_output_dir=OUTPUT_DIR,
        device="cuda:0",
        task=TASK,
        num_rollouts=1,
        num_envs=1,
        split=SPLIT,
        overwrite=True,
    )
    print("\n=== smoke test complete ===")
