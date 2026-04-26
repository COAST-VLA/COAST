# diffusion_policy / RoboCasa 18-task Eval + Activation Sweep

SLURM bundle that, for every checkpoint in a directory:

1. Runs the 18 `atomic_seen` RoboCasa tasks with 30 rollouts × 15 parallel envs
   each via `collect_activations_robocasa.py` (so every rollout also produces
   the `dp_v1`-schema activation tree used downstream by
   `experiments/robocasa_steering/build_conceptors.py`).
2. Writes a per-checkpoint `success_rates.json`.
3. After all per-checkpoint jobs finish, combines everything into a top-level
   `results.json` with a task × checkpoint success-rate table.

One sbatch job per checkpoint = one GPU per checkpoint (jobs run in parallel
on whatever GPU budget the cluster hands you).

## Files

| File | Role |
|---|---|
| `eval_sweep.sbatch` | One sbatch job per checkpoint. 1 GPU, 20 CPUs, 128 GB RAM, 12 h. Loops the 18 `atomic_seen` tasks sequentially, calling `collect_activations_robocasa.py` with `--num_rollouts 30 --num_envs 15` per task. At the end, writes `success_rates.json` at `$ACTIVATIONS_ROOT/<ckpt_stem>/`. |
| `aggregate_results.py` | Two subcommands: `per-checkpoint` (walks one checkpoint's episode metadata, writes per-checkpoint json) and `all` (combines every per-checkpoint file into a top-level `results.json` plus a task × checkpoint table). |
| `aggregate_results.sbatch` | Tiny CPU-only sbatch wrapping `aggregate_results.py all`. |
| `submit_sweep.sh` | Top-level launcher. Globs `CHECKPOINTS_DIR/*.ckpt`, submits one `eval_sweep.sbatch` per checkpoint via `--parsable`, collects the job IDs, then submits the aggregator with `--dependency=afterany:$JID1:$JID2:...` so it fires whether the eval jobs succeed or fail. Supports `DRY_RUN=1` and env-var overrides. |

## Resource sizing rationale

**SLURM `--cpus-per-task=20`, `--mem=128G`**: 15 forked MuJoCo workers need
~15 cores + margin for the main process and SLURM scheduler overhead. Each
worker holds ~1.5 GB of RoboCasa kitchen assets; policy + batched activations
add ~5–10 GB; rounded to 128 GB for safety on heavier tasks like `OpenCabinet`
(horizon 700 with the 1.5× collection-slack cap = 1050 steps).

**Time sizing** (12 h): 18 tasks per checkpoint. Worst-case task: 1050 steps ×
0.4 s/step × ⌈30/15⌉ = 2 chunks ≈ 14 minutes just env-stepping, plus 30 × 100
step-dir writes per chunk for activations. Budget ~25 min/task × 18 = 7.5 h;
bumped to 12 h for slack and scheduler jitter.

**Partition / QoS**: copied from `example_eval.sh` — `dineshj-compute` /
`dj-high`. Edit at the top of `eval_sweep.sbatch` and
`aggregate_results.sbatch` if your cluster uses different names.

## Configuration (env-var overrides)

All knobs live as env vars read by `submit_sweep.sh`; defaults work out of the
box if you just want to sweep `checkpoints/*.ckpt`:

| Env var | Default | Meaning |
|---|---|---|
| `CHECKPOINTS_DIR` | `<repo>/checkpoints` | Directory of `.ckpt` files to sweep. |
| `ACTIVATIONS_ROOT` | `/mnt/bird_home/kim34/eval_sweep_results` | Where activations + `results.json` land. `collect_activations_robocasa.py` writes to `$ACTIVATIONS_ROOT/<ckpt_stem>/<task>/episode_NNN_env_NNN/...`. Default points at `/mnt/bird_home` because a full sweep produces multi-TB of activations and the `grasp_home` (`/home`) pool is chronically near-full. |
| `SPLIT` | `pretrain` | RoboCasa split. |
| `NUM_ROLLOUTS` | `30` | Rollouts per task. |
| `NUM_ENVS` | `15` | Parallel envs per task. |
| `DRY_RUN` | `0` | Set to `1` to print the sbatch invocations without submitting. |

## Usage

```bash
# Defaults (checkpoints/, writes to /mnt/bird_home/kim34/eval_sweep_results/):
bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# Dry-run — prints every sbatch call and the dependency chain, no submissions:
DRY_RUN=1 bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# Point outputs at the shared mount (or wherever you keep activations):
ACTIVATIONS_ROOT=/mnt/kostas-graid/datasets/ksb/activations_dp \
    bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# Smaller smoke sweep (5 rollouts × 5 envs is ~10× faster):
NUM_ROLLOUTS=5 NUM_ENVS=5 \
    bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh

# Different checkpoint set:
CHECKPOINTS_DIR=/path/to/other/checkpoints \
    bash experiments/robocasa_steering/eval_sweep/submit_sweep.sh
```

## Output layout

```
$ACTIVATIONS_ROOT/
├── results.json                        # task × checkpoint success rates (final aggregate)
├── <ckpt_stem_A>/
│   ├── success_rates.json              # per-checkpoint SR breakdown
│   ├── CloseBlenderLid/
│   │   └── episode_NNN_env_NNN/
│   │       ├── metadata.json           # has episode_success → feeds SR computation
│   │       ├── rewards.npz
│   │       └── step_NNNN/
│   │           ├── denoising.npz
│   │           ├── adarms_cond.npz
│   │           ├── suffix_residual.npz
│   │           └── metadata.json
│   ├── CloseFridge/ ...
│   └── ... (18 tasks)
└── <ckpt_stem_B>/ ... (one subtree per checkpoint)
```

`results.json` carries:
- `checkpoints.<stem>.tasks.<task>.{n_episodes, n_success, success_rate}`
- `checkpoints.<stem>.mean_success_rate`
- `per_task_table.<task>.<stem> → success_rate` (wide view)
- `mean_per_checkpoint.<stem> → mean_success_rate`

## Monitoring

`monitor_sweep.sh` prints a one-shot status report: queue state, recent
`sacct` terminal states, per-job latest `[i/18] <Task>` tick (narrowed to
jobs currently in the queue so stale logs don't clutter), per-checkpoint
bytes written under `$ACTIVATIONS_ROOT/`, presence/size of the
`success_rates.json` / `results.json` files, and any traceback / OOM /
`FAILED` signatures in the active logs.

```bash
# one-shot
bash experiments/robocasa_steering/eval_sweep/monitor_sweep.sh

# live refresh every 30s
watch -n 30 experiments/robocasa_steering/eval_sweep/monitor_sweep.sh

# non-default paths (same env-var contract as submit_sweep.sh for ACTIVATIONS_ROOT)
ACTIVATIONS_ROOT=/scratch/foo SLURM_LOGS_DIR=/scratch/bar \
    bash monitor_sweep.sh
```

Raw one-liners for ad-hoc peeks:

```bash
squeue -u $USER                                      # queue state
tail -F slurm_logs/dp_eval_*.out                     # all live eval logs
tail -F 'slurm_logs/dp_eval_<ckpt_stem>_*.out'       # one checkpoint (wildcard on jid)
grep -E "^\[[0-9]+/18\]" slurm_logs/dp_eval_*.out    # every task-boundary tick
cat $ACTIVATIONS_ROOT/<ckpt_stem>/success_rates.json # per-checkpoint SR once done
cat $ACTIVATIONS_ROOT/results.json                   # final aggregate
```

The job id is embedded in each log filename as `dp_eval_<ckpt_stem>_<jid>.out`,
so the checkpoint→jobid mapping from `submit_sweep.sh`'s submission output is
recoverable later via `ls slurm_logs/dp_eval_*.out`.

## The 18 tasks

These are RoboCasa's `atomic_seen` set — the evaluation suite the open-sourced
diffusion_policy checkpoint was trained against. Horizons are RoboCasa's
recommended caps (via `robocasa.utils.dataset_registry_utils.get_task_horizon`);
collection uses `horizon × 1.5` as the env_runner `max_steps`.

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

## Failure-mode notes

- **One eval job fails but others succeed** — the aggregator uses
  `--dependency=afterany` (not `afterok`), so it runs anyway and reports
  `nan` for the affected (task, checkpoint) cells. Re-run a failed
  checkpoint by hand with `sbatch --export=ALL,CHECKPOINT=... eval_sweep.sbatch`
  then re-run `aggregate_results.py all ...` to refresh `results.json`.
- **The aggregator finds a missing `success_rates.json`** — it materializes
  one on demand from the episode metadata files, so partial sweeps still
  produce a useful results table. You'll see a `note: ... computing from
  metadata` line in its log.
- **Filenames like `epoch=0300-test_mean_score=-1.000.ckpt`** round-trip
  through `--export=ALL,CHECKPOINT=...` correctly — sbatch escapes the
  commas it needs to. Verified by `DRY_RUN=1`.
- **Time cap hit** — if a 12 h budget wasn't enough, the job dies after the
  task it was on. `success_rates.json` won't exist for that checkpoint, so
  the aggregator's on-demand fallback reports just the tasks that
  completed. Bump `--time` in `eval_sweep.sbatch` or drop `NUM_ROLLOUTS`.
