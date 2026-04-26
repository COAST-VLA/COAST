"""One-episode smoke test of activation collection on a single RoboCasa task.

Mirrors smoke_test_eval.py but calls collect_activations_robocasa.collect_task
so the entire on-disk schema is exercised end-to-end.
"""
from collect_activations_robocasa import collect_task

CHECKPOINT = "/home/kim34/projects/diffusion_policy/checkpoints/latest.ckpt"
ACTIVATIONS_OUTPUT_DIR = "/home/kim34/projects/diffusion_policy/smoke_test_activations"
RUNNER_OUTPUT_DIR = "/home/kim34/projects/diffusion_policy/smoke_test_output"
TASK = "CloseStandMixerHead"
SPLIT = "pretrain"

if __name__ == "__main__":
    collect_task(
        checkpoint=CHECKPOINT,
        activations_output_dir=ACTIVATIONS_OUTPUT_DIR,
        device="cuda:0",
        task=TASK,
        num_rollouts=1,
        num_envs=1,
        split=SPLIT,
        prompt="",
        runner_output_dir=RUNNER_OUTPUT_DIR,
    )
    print("\n=== activation smoke test complete ===")
