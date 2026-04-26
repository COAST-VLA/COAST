# Diffusion Policy

This is the Diffusion Policy fork repo for running RoboCasa benchmark experiments.
This fork is based on the original Diffusion Policy code, hosted at [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).

## Recommended system specs
For training we recommend a GPU with at least 24 Gb of memory, but 48 Gb+ is prefered.
For inference we recommend a GPU with at least 8 Gb of memory.

## Installation
```
git clone https://github.com/robocasa-benchmark/diffusion_policy
cd diffusion_policy
pip install -e .
```

## Key files
- Training: [train.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/train.py)
- Evaluation: [eval_robocasa.py](https://github.com/robocasa-benchmark/diffusion_policy/blob/main/eval_robocasa.py)

## Experiment workflow
```
# train model
python train.py \
--config-name=train_diffusion_transformer_bs192 \
task=robocasa/<dataset-soup>

# Evaluate model
python eval_robocasa.py \
--checkpoint <checkpoint-path> \
--task_set <task-set> \
--split <split>

# Report evaluation results
python diffusion_policy/scripts/get_eval_stats.py \
--dir <outputs-dir>
```

## RoboCasa `atomic_seen` task set (18 tasks)

The 18-task `atomic_seen` split is RoboCasa's curated set of atomic skills
with published training data / benchmark results. Horizons below are
RoboCasa's recommended per-task step caps, read via
`robocasa.utils.dataset_registry_utils.get_task_horizon`. `eval_robocasa.py`
uses each task's horizon directly as `env_runner.max_steps` (no 1.5x slack).

| # | Task | Horizon |
|---:|---|---:|
| 1 | CloseBlenderLid | 600 |
| 2 | CloseFridge | 600 |
| 3 | CloseToasterOvenDoor | 300 |
| 4 | CoffeeSetupMug | 400 |
| 5 | NavigateKitchen | 300 |
| 6 | OpenCabinet | 700 |
| 7 | OpenDrawer | 500 |
| 8 | OpenStandMixerHead | 300 |
| 9 | PickPlaceCounterToCabinet | 500 |
| 10 | PickPlaceCounterToStove | 400 |
| 11 | PickPlaceDrawerToCounter | 500 |
| 12 | PickPlaceSinkToCounter | 600 |
| 13 | PickPlaceToasterToCounter | 400 |
| 14 | SlideDishwasherRack | 300 |
| 15 | TurnOffStove | 500 |
| 16 | TurnOnElectricKettle | 300 |
| 17 | TurnOnMicrowave | 300 |
| 18 | TurnOnSinkFaucet | 400 |

For context: `atomic_seen` is 18 of 65 total atomic tasks in
`all_atomic_tasks`; RoboCasa also defines 252 composite tasks
(`all_composite_tasks`), for 317 tasks overall. Horizons across the full
65-atomic-task universe range 200–700 (median 300, mean ≈ 365).

## Activation Collection

Collect per-denoising-step / per-episode activations from a `DiffusionTransformerHybridImagePolicy`
rollout on RoboCasa, in the same on-disk layout as openpi's server-side collector
(see `openpi-metaworld/docs/activation_collection.md`). Downstream mech-interp
tooling written against pi0.5's `v1` tree reads these directly, with one
documented shape deviation (`adarms_cond`, see below).

### Files

- **`collect_activations_robocasa.py`** — argparse-driven collector. Loads the
  checkpoint the same way `eval_robocasa.eval_task` does, attaches forward hooks
  to `TransformerForDiffusion` (per-denoising-step model I/O, encoder memory,
  each `TransformerDecoderLayer` residual stream), then runs a custom rollout
  loop that mirrors `RobomimicImageRunner.run()` but writes `denoising.npz` /
  `adarms_cond.npz` / `suffix_residual.npz` + `metadata.json` per step and
  `metadata.json` + `rewards.npz` per episode. Auto-installs the SyncVectorEnv
  shims from `smoke_test_eval.py` when `--num_envs 1`.
- **`smoke_test_collect_activations.py`** — thin `smoke_test_eval.py`-style
  wrapper pinning the same checkpoint + `CloseStandMixerHead`, writes to
  `./smoke_test_activations/`.

### Output layout

```
<output_root>/<checkpoint_step>/<task_name>/
├── episode_NNN_env_NNN/
│   ├── metadata.json        # task_name, episode_id, env_id, episode_success,
│   │                        # total_reward, steps_to_success, total_env_steps,
│   │                        # total_inference_steps, prompt, checkpoint_dir,
│   │                        # config_name
│   ├── rewards.npz          # per_step_reward, cumulative_reward, success_at_step
│   └── step_NNNN/
│       ├── metadata.json        # step, inference_step, cumulative_reward,
│       │                        # success_so_far, collection_version="dp_v1", ...
│       ├── denoising.npz        # all_x_t, all_v_t
│       ├── adarms_cond.npz      # all_adarms_cond
│       └── suffix_residual.npz  # all_suffix_residual
```

`<checkpoint_step>` is the checkpoint file stem (e.g. `latest`). `step_NNNN` is
the env step at which the inference call was issued (increments by
`n_action_steps` per policy call because of action chunking).

### Per-step array shapes

`D = num_inference_steps`, `L = n_layer` (transformer decoder layers),
`H = horizon`, `A = action_dim`, `T_cond = 1 + n_obs_steps`, `C = n_emb`.
Concrete shapes for the checkpoint at `checkpoints/latest.ckpt`
(D=100, L=12, H=10, A=12, T_cond=3, C=512):

| File | Array keys | Shape | Dtype |
|---|---|---|---|
| `denoising.npz`         | `all_x_t`, `all_v_t`       | `(D, H, A)`       | fp32 |
| `adarms_cond.npz`       | `all_adarms_cond`          | `(D, T_cond, C)`  | fp32 |
| `suffix_residual.npz`   | `all_suffix_residual`      | `(D, L, H, C)`    | fp32 |

Schema identifier: **`dp_v1`** (stamped in per-step `metadata.json` under
`collection_version`). `denoising` and `suffix_residual` shapes are
byte-compatible with pi0.5's `v1` schema. `all_adarms_cond` carries
`(D, T_cond, C)` instead of pi0.5's pooled `(D, C)`: diffusion_policy
conditions via cross-attention tokens (time + obs) rather than AdaLN, so there
is no single pooled per-step vector to match — the full cond token sequence is
stored instead. The FF-inner activation (`suffix_mlp_hidden`, shape
`(D, L, H, 4*C)`) from pi0.5's `v1` is **not** collected here — downstream
conceptor-building and steering only use `suffix_residual`, so capturing the
~4× larger FF tensor was dropped to keep per-sweep output manageable.

### Hook layout

- `policy.model` forward hook → captures input `sample` (→ `all_x_t`) and output
  (→ `all_v_t`) each denoising step.
- `policy.model.encoder` forward hook → captures the cond-token memory
  (→ `all_adarms_cond`) each step. Varies with `t` because `time_emb(t)` is the
  first cond token.
- For each `layer` in `policy.model.decoder.layers`:
  - `layer` forward hook → residual stream output (→ `all_suffix_residual[:, i]`).

Only `DiffusionTransformerHybridImagePolicy` is currently supported;
`DiffusionUnetHybridImagePolicy` raises `NotImplementedError` from
`TransformerActivationCapture.__init__` because its residual shapes vary per
layer due to down/up-sampling (no uniform `(D, L, H, C)` tensor).

### Usage

```bash
# Smoke test (1 env, 1 rollout — SyncVectorEnv shims auto-applied):
python smoke_test_collect_activations.py

# Single task, full run:
python collect_activations_robocasa.py \
    --checkpoint checkpoints/latest.ckpt \
    --activations_output_dir ./activations \
    --task CloseStandMixerHead --split pretrain \
    --num_rollouts 15 --num_envs 5

# Full task set:
python collect_activations_robocasa.py \
    --checkpoint checkpoints/latest.ckpt \
    --activations_output_dir ./activations \
    --task_set atomic_seen --split pretrain \
    --num_rollouts 15 --num_envs 5
```

CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `-c`, `--checkpoint` | required | Path to `*.ckpt`. `Path.stem` becomes `<checkpoint_step>`. |
| `-a`, `--activations_output_dir` | required | Activation tree root. |
| `-s`, `--split` | required | RoboCasa split, e.g. `pretrain`. |
| `-t`, `--task` | — | Single task; mutually exclusive with `-T`. |
| `-T`, `--task_set` | — | One or more `TASK_SET_REGISTRY` keys (e.g. `atomic_seen`). |
| `-n`, `--num_rollouts` | 15 | Rollouts per task. |
| `-e`, `--num_envs` | 5 | Parallel envs. `=1` triggers the SyncVectorEnv shims. |
| `-d`, `--device` | `cuda:0` | |
| `--prompt` | `""` | Optional task instruction stamped into metadata. |
| `--runner_output_dir` | auto | Scratch dir for env_runner videos/etc. Defaults to `<activations>/<ckpt_stem>/<task>/_runner_scratch`. |
```

## Eval sweep (SLURM)

`experiments/robocasa_steering/eval_sweep/submit_sweep.sh` fans out one GPU job
per `.ckpt` in `checkpoints/`, each running all 18 `atomic_seen` tasks via
`collect_activations_robocasa.py` (so the same rollouts produce success rates
**and** the activation tensors downstream conceptor-building uses). An
aggregator job is queued with `--dependency=afterany` and writes
`$ACTIVATIONS_ROOT/results.json` once every eval job terminates.

**Storage:** by default, `ACTIVATIONS_ROOT=/mnt/bird_home/kim34/eval_sweep_results`.
A full sweep produces multi-TB of activations, so the sweep is deliberately
pointed at `bird_home` rather than `/home` (the shared `grasp_home` NFS pool,
chronically ~99% full). Override `ACTIVATIONS_ROOT=/some/other/path` if you
need a different destination.

```bash
# default sweep (30 rollouts × 15 envs per task, split=pretrain)
bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# overrides
CHECKPOINTS_DIR=/path/to/ckpts NUM_ROLLOUTS=10 NUM_ENVS=5 \
    bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# just print what would be submitted
DRY_RUN=1 bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh
```

### Monitoring progress

`experiments/robocasa_steering/eval_sweep/monitor_sweep.sh` prints a one-shot
status report: queue state, recent terminal states from `sacct`, per-job latest
`[i/18] <Task>` tick, per-checkpoint disk usage under `$ACTIVATIONS_ROOT/`,
presence/size of `success_rates.json` + `results.json`, and any traceback /
OOM / `FAILED` signatures in the active logs.

It's an executable shell script — no activation step. From the repo root:

```bash
# one-shot status report
bash experiments/robocasa_steering/eval_sweep/monitor_sweep.sh

# or, since it's chmod +x
./experiments/robocasa_steering/eval_sweep/monitor_sweep.sh

# live-refresh every 30 seconds (Ctrl-C to stop)
watch -n 30 experiments/robocasa_steering/eval_sweep/monitor_sweep.sh

# if the sweep used non-default output/log paths
ACTIVATIONS_ROOT=/scratch/foo SLURM_LOGS_DIR=/scratch/bar \
    bash experiments/robocasa_steering/eval_sweep/monitor_sweep.sh
```

Prints a single snapshot and exits. Run it again anytime, or wrap in `watch`
for a live-refreshing terminal view.

The "per-job latest task tick" section filters to jobs currently in the queue
(falls back to the newest log per checkpoint stem if nothing is queued) so
stale logs from earlier failed submissions don't drown out the live run.

Raw one-liners for ad-hoc peeks:

```bash
# live stdout — all four eval jobs, or a specific checkpoint
tail -F slurm_logs/dp_eval_*.out
tail -F slurm_logs/dp_eval_latest_*.out
tail -F 'slurm_logs/dp_eval_epoch=0300-test_mean_score=-1.000_*.out'

# every task-boundary tick for one log
grep -E "^\[[0-9]+/18\]" slurm_logs/dp_eval_latest_*.out

# final combined file (written by the aggregator job)
cat $ACTIVATIONS_ROOT/results.json
```

Note: `submit_sweep.sh` also prints the checkpoint→jobid map at submission
time (`-> job id: …` per checkpoint, then `Per-checkpoint JIDs:` summary). The
job id is embedded in each log filename as `dp_eval_<ckpt_stem>_<jid>.out`,
so the map is recoverable after the fact.

## Steering pipeline

Conceptor-based steering on top of the activations produced by the eval sweep.
End-to-end: build conceptors → select hyperparameters → run steering eval →
aggregate success rates. See
[`experiments/robocasa_steering/steering_sweep/README.md`](experiments/robocasa_steering/steering_sweep/README.md)
for the full pipeline reference.

**New code added on top of the upstream pi05_libero / pi05_robocasa
implementations:**

| File | Purpose |
|---|---|
| `experiments/robocasa_steering/build_conceptors.py` | Extended with `--per-step-alphas` (multi-alpha per_step support; alpha-aware keys). Backward-compatible with `--per-step-alpha` (singular). |
| `experiments/robocasa_steering/add_per_step_alphas.py` | Post-hoc adds per_step matrices at additional alphas to an existing `conceptors.npz` via eigendecomposition recovery — **no re-read of activation tree** required. Verified bit-equivalent (fp32 noise floor) to a fresh native build. |
| `experiments/robocasa_steering/select_parameters_per_task.py` | Per-task variant of `select_parameters.py` — runs the overlap/quota selection independently for each task in a conceptor file via a `_NpzSubset` view. |
| `experiments/robocasa_steering/build_and_select_all.py` | Driver: walks every checkpoint subdir, runs `build_conceptors.py` then `select_parameters_per_task.py`, writes a single `manifest.json` keyed by `(ckpt_stem, task)`. |
| `experiments/robocasa_steering/steering.py` | Extended with `--skip-baseline`, `--per-step-static-ds K` (pi05-style static-at-ds per_step semantics), and alpha-aware `get_per_step_contrastive(alpha=…)`. Default per_step is **time-varying** — different conceptor at each denoising step via `_PerStepLookup`, more principled for diffusion policy than pi05's static "per_step_K". |
| `experiments/robocasa_steering/steering_sweep/build_recipes.py` | Generates `recipes.json` (one `(layer, α, β)` per `(ckpt, task)`). |
| `experiments/robocasa_steering/steering_sweep/steering_sweep.sbatch` | Per-checkpoint SLURM job: loops tasks, reads recipe (single-value or list schema), invokes `steering.py`. Env knobs: `STRATEGIES`, `SKIP_BASELINE`, `TASKS_OVERRIDE`, `PER_STEP_STATIC_DS`, `ALPHAS_OVERRIDE`, `BETAS_OVERRIDE`. |
| `experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh` | Top-level launcher; one A40 GPU per checkpoint. |

**Quick-start sequence after a completed eval/activation sweep:**
```bash
# 1. Build conceptors + per-(ckpt, task) selections + manifest, all 5 ckpts in parallel:
for d in /mnt/bird_home/kim34/eval_sweep_results/*/; do
    STEM=$(basename "$d")
    mkdir -p experiments/robocasa_steering/conceptors/$STEM
    .venv/bin/python experiments/robocasa_steering/build_conceptors.py \
        --activations-dir "$d" \
        --output-npz       experiments/robocasa_steering/conceptors/$STEM/conceptors.npz &
done
wait
.venv/bin/python experiments/robocasa_steering/build_and_select_all.py \
    --activations-root /mnt/bird_home/kim34/eval_sweep_results \
    --output-dir       experiments/robocasa_steering/conceptors --skip-build

# 2. Generate the recipe table:
.venv/bin/python experiments/robocasa_steering/steering_sweep/build_recipes.py

# 3. Submit the steering sweep (one A40 per ckpt, QoS dj-high caps at 4 concurrent):
bash experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh
```
