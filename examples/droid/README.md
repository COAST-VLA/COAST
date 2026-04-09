# DROID Policies in openpi

We offer instructions for:
- [Running inference for our best $pi_{0.5}$-DROID policy](./README.md#running-droid-inference)
- [Running inference for other pre-trained DROID policies ($\pi_0$, $\pi_0$-FAST, ...)](./README.md#running-roboarena-baseline-policies)
- [Pre-training *generalist* policies on the *full* DROID dataset](./README_train.md#training-on-droid)
- [Fine-tuning expert $\pi_{0.5}$ on your custom DROID dataset](./README_train.md#fine-tuning-on-custom-droid-datasets)

## Running DROID Inference

This example shows how to run the fine-tuned $\pi_{0.5}$-DROID model on the [DROID robot platform](https://github.com/droid-dataset/droid). Based on the [public RoboArena benchmark](https://robo-arena.github.io/leaderboard), this is currently our strongest generalist DROID policy. 


### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi).
2. Start the OpenPI server via the following command:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

You can also run the equivalent command below:

```bash
uv run scripts/serve_policy.py --env=DROID
```

### Step 2: Run the DROID robot

1. Make sure you have the most recent version of the DROID package installed on both the DROID control laptop and the NUC.
2. On the control laptop, activate your DROID conda environment.
3. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
4. Install `tyro`, which we will use for command line parsing: `pip install tyro`.
5. Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.
6. Replace the camera IDs in the `main.py` file with the IDs of your cameras (you can find the camera IDs by running `ZED_Explorer` in the command line, which will open a tool that shows you all connected cameras and their IDs -- you can also use it to make sure that the cameras are well-positioned to see the scene you want the robot to interact with).
7. Run the `main.py` file. Make sure to point the IP and host address to the policy server. (To make sure the server machine is reachable from the DROID laptop, you can run `ping <server_ip>` from the DROID laptop.) Also make sure to specify the external camera to use for the policy (we only input one external camera), choose from ["left", "right"].

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

The script will ask you to enter a free-form language instruction for the robot to follow. Make sure to point the cameras at the scene you want the robot to interact with. You _do not_ need to carefully control camera angle, object positions, etc. The policy is fairly robust in our experience. Happy prompting!

## Optional: Activation Collection

For mech-interp work the droid client can also tell the policy server to save
per-step intermediate activations to disk. This uses the same collection-mode
server and on-disk format as `examples/libero_env` and `examples/metaworld` —
see [`examples/libero_env/README.md`](../libero_env/README.md#activation-collection)
for the wire-level protocol and the directory layout. Activations live entirely
on the **server's** filesystem; the droid laptop never touches them.

Start the collection-mode server on the GPU box (Step 1 above, but with extra
flags):

```bash
# GPU box, main openpi venv
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_droid \
    --policy.dir="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
```

Then run the droid client (Step 2 above) with `--collect`:

```bash
# DROID control laptop
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> \
    --external_camera="left" --collect
```

Each rollout becomes one episode under
`<server_output_dir>/pi05_droid/<task_name>/episode_NNN_env_000/`. The
`task_name` is derived from the language instruction you type at the prompt
(e.g. `"Pick up the red block!"` → `pick-up-the-red-block`); pass
`--task_name <slug>` to override and group multiple phrasings under one
directory. Multiple rollouts of the same instruction in one session are
indexed as `episode_000`, `episode_001`, … automatically.

### Caveat: success is graded out-of-band

`droid.RobotEnv.step()` does not return per-step rewards or done flags, so
the rollout loop records `reward=0, done=False` for every env step. The
final score comes from the prompt that appears after the rollout ends
("Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value
0-100"). The droid client takes that human grade and writes it into the
last entry of `per_step_reward` so the cumulative reward stored in
`rewards.npz` matches `total_reward` (preserves the schema invariant
asserted by `tests/test_activations.py::test_rewards_cumulative_matches_total`).
Scores ≥ 0.5 set `episode_success = True`; the exact float is preserved in
`total_reward`. **Per-step credit assignment is not meaningful for droid
activations** — only episode-level success/total_reward are.

### Notes

- Collection mode requires `--pytorch` on the server. `infer_with_intermediates`
  is implemented for the PyTorch backend only.
- Use the local cache path `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid`,
  not the `gs://` URL — `--pytorch` + `gs://` has a known bug in
  `ensure_pytorch_checkpoint`. To pre-populate the cache, run a normal
  (non-collection) DROID inference first.
- A collection-mode server **rejects** plain inference requests. If you want
  to also run regular eval against the same checkpoint, start a separate
  non-collection server on a different port.
- Ctrl+C-interrupted rollouts are still finalized after you grade them at
  the success prompt, so partial rollouts produce a complete episode
  directory on disk.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach policy server | Make sure the server is running and the IP and port are correct. You can check that the server machine is reachable by running `ping <server_ip>` from the DROID laptop. |
| Cannot find cameras | Make sure the camera IDs are correct and that the cameras are connected to the DROID laptop. Sometimes replugging the cameras can help. You can check all connected cameras by running `ZED_Explore` in the command line. |
| Policy inference is slow / inconsistent | Try using a wired internet connection for the DROID laptop to reduce latency (0.5 - 1 sec latency per chunk is normal). |
| Policy does not perform the task well | In our experiments, the policy could perform simple table top manipulation tasks (pick-and-place) across a wide range of environments, camera positions, and lighting conditions. If the policy does not perform the task well, you can try modifying the scene or object placement to make the task easier. Also make sure that the camera view you are passing to the policy can see all relevant objects in the scene (the policy is only conditioned on a single external camera + wrist camera, make sure you are feeding the desired camera to the policy). Use `ZED_Explore` to check that the camera view you are passing to the policy can see all relevant objects in the scene. Finally, the policy is far from perfect and will fail on more complex manipulation tasks, but it usually makes a decent effort. :) |


## Running Other Policies

We provide configs for running the baseline DROID policies from the [RoboArena](https://robo-arena.github.io/) paper. Simply run the commands below to start inference servers for the respective policies. Then follow the instructions above to run evaluation on the DROID robot.

```
# Train from pi0-FAST, using FAST tokenizer
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# Train from pi0, using flow matching
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# Trained from PaliGemma, using FSQ tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

You can find the inference configs in [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py).
