# DROID

[DROID](https://droid-dataset.github.io/) is a real-robot manipulation dataset / platform (Franka Panda + stereo cameras + wrist cam). Unlike the sim examples, DROID runs on **real hardware**: the policy server runs on a workstation GPU, and `main.py` runs on the DROID control laptop, talking to the server over the network. The DROID control laptop uses its own conda-based env per upstream [DROID setup](https://github.com/droid-dataset/droid); everything else (server, dataset tooling) uses the root openpi venv.

- `main.py` runs one rollout on the real robot. Gets copied to `$DROID_ROOT/scripts/main.py` on the control laptop.
- `convert_droid_data_to_lerobot.py` converts a raw DROID capture to a LeRobot dataset.
- `compute_droid_nonidle_ranges.py` pre-computes non-idle index ranges used during full-dataset training.

## Installation

On a **workstation GPU** (for the policy server), install openpi from the repo root:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

On the **DROID control laptop** (where `main.py` runs), with the DROID conda env activated:

```bash
cd $OPENPI_ROOT/packages/openpi-client && pip install -e .
pip install tyro
```

The control laptop only needs `openpi-client`, not the full openpi install.

## Dataset & Training

We do not re-train DROID by default; evaluate against the upstream checkpoints under `gs://openpi-assets/checkpoints/pi05_droid` / `pi0_fast_droid` / `pi0_droid` / the RoboArena baselines. The rest of this section is for approximating the upstream training pipeline or fine-tuning on a custom dataset.

### Full-DROID training (RLDS)

Full DROID training uses RLDS (LeRobot isn't yet scalable to the full DROID dataset). Install the RLDS dependency group, download the dataset, and train from the repo root:

```bash
# 1. RLDS deps
uv sync --group rlds

# 2. Download DROID v1.0.1 (~1.8 TB; v1.0.1 has the full ~75k language annotations, v1.0.0 only 30k):
gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 <your_download_path>/droid/1.0.1

# 3. Point rlds_data_dir in TrainConfig (src/openpi/training/config.py) at your download path.

# 4. Compute norm stats (~10 min):
uv run --group rlds scripts/compute_norm_stats.py --config-name pi05_full_droid_finetune --max-frames 10_000_000

# 5. Train:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --group rlds scripts/train.py pi05_full_droid_finetune \
    --exp-name=my_experiment --overwrite
```

Compute budget: ~2 days on 8× H100 from pi0 init (100k iters, bs256, ~1 epoch); ~5 days from PaliGemma init (240k iters, ~3 epochs). LoRA hasn't produced usable policies for us here.

**Idle filtering.** The DROID dataset contains many idle timesteps from VR teleop; training benefits from filtering them. The default recipe uses a pre-computed index list pulled from cloud storage. To regenerate it (e.g. for a non-`1.0.1` version), rerun [`compute_droid_nonidle_ranges.py`](compute_droid_nonidle_ranges.py) and pass `filter_dict_path=...` in the train config. The published indices are only valid for `droid/1.0.1`.

### Fine-tuning on a custom DROID dataset (LeRobot)

For smaller custom datasets (<10s of hours), convert to LeRobot first and fine-tune `pi05_droid` on it:

```bash
# 1. Sample: download a tiny 30-demo subset of raw DROID (1.6 GB):
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/IRIS/success/2023-12-04 <your_target_path>

# 2. Language annotations (12 MB) — for your own data you can skip this and supply instructions manually:
gsutil -m cp -r gs://gresearch/robotics/droid_raw/1.0.1/aggregated-annotations-030724.json <your_target_dir>

# 3. Convert to LeRobot (<5 min for 30 demos). Each episode directory must contain recordings/MP4;
#    run droid's svo_to_mp4.py first if yours doesn't.
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir <your_target_path>

# 4. Fine-tune pi05_droid on the converted dataset (config: pi05_droid_finetune in src/openpi/training/config.py):
uv run scripts/train.py pi05_droid_finetune --exp-name=my_experiment --overwrite
```

## Serving the policy

Run the server on a workstation GPU (~RTX 4090 minimum). The default is the `pi05_droid` checkpoint:

```bash
# pi0.5 DROID (default, recommended):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid

# Shorthand:
uv run scripts/serve_policy.py --env=DROID
```

Alternative policies (see [`roboarena_config.py`](../../src/openpi/training/misc/roboarena_config.py)):

```bash
# pi0-FAST DROID (FAST tokenizer):
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# pi0 DROID (flow matching):
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# RoboArena baselines (PaliGemma + binning/FAST/FSQ/diffusion):
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

## Evaluation

On the DROID control laptop:

1. Have the latest DROID package on both the control laptop and the NUC.
2. Activate the DROID conda env.
3. Copy `main.py` from this directory to `$DROID_ROOT/scripts/main.py`.
4. Replace the camera IDs in `main.py` with yours. Run `ZED_Explorer` to enumerate connected cameras and confirm positioning.
5. Run the script, pointing at the policy server. `--external_camera` picks which stereo side feeds the policy; only one external camera is used.

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

The script prompts for a free-form language instruction per rollout. The policy handles a fairly broad range of scenes and camera positions, but failure modes scale with task complexity. Verify the server is reachable with `ping <server_ip>` first if the client hangs on startup.

## Activation collection

DROID supports the same server-side collection protocol as LIBERO / RoboCasa: start the server with `--collect_activations`, run the client with `--collect`, and activations are saved on the **server's** filesystem. Protocol, output layout, schema, and verification are covered in the canonical reference — see **[`docs/activation_collection.md`](../../docs/activation_collection.md)**.

Because DROID runs on real hardware, there is no scripted success signal — `episode_success` and rewards are supplied by the user at the end of each rollout (the `main.py` script prompts for them).

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach policy server | Check the server is running and that `--remote_host` / `--remote_port` are correct. `ping <server_ip>` from the DROID laptop confirms reachability. |
| Cannot find cameras | Verify camera IDs; occasionally replugging helps. Run `ZED_Explorer` to enumerate connected cameras. |
| Policy inference slow / inconsistent | Prefer a wired connection on the DROID laptop. 0.5–1 s latency per action chunk is normal. |
| Policy doesn't perform the task well | The model handles simple tabletop pick-and-place robustly, but harder tasks fail more often. Confirm the feeding camera sees the relevant objects (policy only sees one external + one wrist). Modifying scene/object placement is a valid mitigation. |

## Results — RoboArena

Consider submitting your DROID policies to the [RoboArena benchmark](https://robo-arena.github.io/) for evaluation on diverse real-world tasks and scenes. Questions: [karl.pertsch@gmail.com](mailto:karl.pertsch@gmail.com).
