# diffusion_policy / RoboCasa Conceptor Steering

In-process adaptation of the pi0.5 LIBERO conceptor-steering bundle
(`openpi-metaworld/experiments/pi05_libero/for_subin/`) for the
`DiffusionTransformerHybridImagePolicy` evaluated on RoboCasa with
`eval_robocasa.py` / `smoke_test_eval.py`.

## TL;DR

```bash
# 0. Pre-req: activations already collected, e.g.
python collect_activations_robocasa.py \
    --checkpoint checkpoints/latest.ckpt \
    --activations_output_dir activations \
    --task_set pretrain50 --split pretrain \
    --num_rollouts 15 --num_envs 5

# 1. Build conceptors.
python experiments/robocasa_steering/build_conceptors.py \
    --activations-dir activations/latest

# 2. Narrow the sweep:
python experiments/robocasa_steering/select_parameters.py \
    --conceptor-npz ~/.cache/diffusion_policy/diffusion_policy_conceptors.npz \
    --output-json   experiments/robocasa_steering/selected_params.json

# 3. Run the steering sweep for one task (in-process):
python experiments/robocasa_steering/steering.py \
    --checkpoint checkpoints/latest.ckpt \
    --conceptor_npz ~/.cache/diffusion_policy/diffusion_policy_conceptors.npz \
    --task CloseFridge --split pretrain \
    --num_rollouts 15 --num_envs 5

# ...or fan out over the default 7-task subset:
CHECKPOINT=checkpoints/latest.ckpt ACTIVATIONS_DIR=activations/latest \
    bash experiments/robocasa_steering/run_steering.sh
```

## What's different from the pi0.5 LIBERO bundle

| Aspect | pi0.5 LIBERO | diffusion_policy / RoboCasa |
|---|---|---|
| Deployment | WebSocket server + subprocess LIBERO client | **In-process** — `env_runner.run(policy)` with forward hooks attached |
| Policy call | `policy.infer_with_steering(obs, steering_hooks=[...])` | Plain `policy.predict_action(obs)` with forward hooks on `policy.model.decoder.layers[i]` |
| Denoising-step counter | Sampler calls `hook.set_denoise_step(t)` at each iteration | `SteeringContext` monkey-patches `policy.conditional_sample` to do the same |
| `D` (denoising steps) | 10 | **100** |
| `L` (captured layers) | 4 — pi0.5 suffix layers `[0, 5, 11, 17]` | **12** — all decoder layers, identity map |
| `C` (hidden dim) | 1024 | **512** (checkpoint uses `n_emb=512`) |
| Per-step build | all 10 steps | **sparse — K evenly-spaced indices** (default K=10). Hook nearest-neighbours at runtime. |
| Single entrypoint vs subprocess per condition | subprocess per condition (client venv) | all conditions in one Python process — policy and env_runner loaded once per task invocation |

Everything else — the conceptor math, the `(layer, alpha, beta, strategy)`
grid, the npz key naming, the `select_parameters.py` quota/overlap rule, the
resume-friendly `summary.json` — is identical to the pi0.5 bundle.

## Files

| File | Role |
|---|---|
| `build_conceptors.py` | Stage 1. Reads `<activations_dir>/<task>/episode_*/step_*/suffix_residual.npz`, mean-pools over the 10 action tokens, builds success / failure / contrastive conceptors on a `(layer × alpha)` grid plus linear-direction vectors. Per-step conceptors at K evenly-spaced ds indices (default 10). Writes `~/.cache/diffusion_policy/diffusion_policy_conceptors.npz`. |
| `select_parameters.py` | Stage 2. Picks best layer by mean quota; keeps alphas whose `Cs`-vs-`Cf` overlap lands in `[0.85, 0.95]`; drops `β=0.5`. Writes a JSON you can wire into the sweep launcher. |
| `steering.py` | Stage 3. Runs one RoboCasa task's full strategy grid end-to-end in a single Python process. No subprocess, no WebSocket. Resume-friendly `summary.json`. |
| `run_steering.sh` | Driver. Runs Stage 1 + Stage 2 once, then loops Stage 3 over the default 7-task subset. Supports `--dry-run`, `--skip-build`, `--skip-select`. |

## Input — what `build_conceptors.py` expects

Produced by `diffusion_policy/collect_activations_robocasa.py`, schema
identifier `dp_v1`:

```
<activations_dir = output_root/checkpoint_stem>/
├── <task>/
│   └── episode_NNN_env_NNN/
│       ├── metadata.json            # episode_success, total_inference_steps, ...
│       └── step_NNNN/
│           └── suffix_residual.npz
│               └── "all_suffix_residual"  (D=100, L=12, H=10, C=512) fp32
```

Mean-pooling `arr[ds, layer_idx]` over the 10 action tokens gives one
`(C=512,)`-vector per inference step per (layer, ds). The build filters to
**mixed-outcome tasks** (≥ 1 success AND ≥ 1 failure) — contrastive
conceptors need both classes.

## The five steering strategies

Same naming as the pi0.5 bundle. All five install a PyTorch forward hook on
`policy.model.decoder.layers[L]` (`nn.TransformerDecoderLayer`). The
`SteeringContext` monkey-patches `policy.conditional_sample` to call
`hook.set_denoise_step(ds_idx)` once per iteration of the DDPM loop, so
`per_step` / `linear_per_step` hooks know which ds they're on.

| Strategy | What it applies | Sweep axes | Role |
|---|---|---|---|
| `linear` | `h' = h + α · v`, `v = unit(mean_s − mean_f)` | `layer × linear_alpha` | ActAdd-style control. |
| `global` | `h' = (1−β) h + β (h @ Cᵀ)`, `C = C_s · (I − C_f)` | `layer × α × β` | Main experiment. |
| `per_step` | Same as `global` but a different `C` at each built ds index (nearest-neighbour at runtime) | `layer × β` | Tests whether the optimal direction drifts through the denoising trajectory. |
| `positive_only` | `C = C_success` only | `layer × α × β` | Ablation — does the NOT term matter? |
| `random` | Random SPD matrix with matched quota | `layer × β` | Control — isolates structure vs. any random rotation. |

## Per-step in a 100-step denoising loop

pi0.5 has D=10, so its `per_step` config stores all 10 conceptors.
Diffusion_policy has D=100 — storing 100× matrices is wasteful. We build at
K evenly-spaced ds indices (default `K=10`, giving `[0, 11, 22, 33, 44, 55,
66, 77, 88, 99]`) and the hook does nearest-neighbour lookup against
`current_step` at runtime. Raise `--per-step-count` if you want a finer
sweep.

## Output layout

```
experiments/robocasa_steering/steering_results/<task>/
├── sweep_args.json               # full CLI args + resolved defaults
├── summary.json                  # condition → success_rate, sorted, resume-friendly
├── _runner_scratch/              # env_runner's output_dir (videos, etc.)
├── baseline/
├── linear_L5_la0.5/
├── global_L5_a1.0_b0.1/
├── per_step_L5_b0.1/
├── posonly_L5_a1.0_b0.1/
└── random_L5_b0.1/
```

`summary.json` merges from disk on every write, so re-running after a crash
skips already-completed conditions.

## Defaults that differ from the pi0.5 bundle

- `--layers 5 8 11` (deeper half of the 12-layer decoder, mirroring pi0.5's
  `[5, 11, 17]` pick of the three deepest captured layers).
- `--alphas 0.5 1.0 2.0` in `steering.py` vs. `0.1 0.5 1.0 2.0 10.0` —
  narrower by default because each condition is a full env_runner rollout,
  which is costlier per condition than LIBERO's episodic subprocess.
- `betas = [0.1, 0.3]` — identical to the pi0.5 bundle; β=0.5 was
  universally harmful in prior experiments.

Override any of these on the CLI.

## Common gotchas

- **`build_conceptors.py` skips every task** — class balance fails. Check
  `metadata.json.episode_success` across your collection; you need at least
  1 success AND 1 failure per task. If the policy is near-perfect on
  one task, increase `--num_rollouts` in `collect_activations_robocasa.py`.
  (Lowering below 1 is not possible — contrastive conceptors require both
  classes; with 1 sample each they're highly under-regularized and mostly
  useful as a smoke-test.)
- **`per_step` and `global` give identical results** — your
  `SteeringContext` didn't patch `conditional_sample`. Check that the
  context manager entered successfully (it raises on the first hook if the
  spec is malformed).
- **`success_rate` is `nan`** — `env_runner.run()` returned without a
  `success_rate/<task>` key. Usually an EGL / MuJoCo rendering issue in the
  workers; run with `--num_envs 1` to trigger the `SyncVectorEnv` shims and
  see the real traceback.
- **Conceptor npz doesn't contain the task** — `steering.py` raises a
  `ValueError` at start-up. Either the task name differs between collection
  and build (check case / exact match) or the class balance filter dropped
  it during the build.

## Integration assumptions

- The checkpoint is a `DiffusionTransformerHybridImagePolicy` (`policy.model
  == TransformerForDiffusion`). The UNet variant is not supported — its
  residual shapes vary per layer and there is no uniform `C`-wide stream.
  `steering.py` reads `hidden_dim` from `policy.model.decoder.layers[0].linear1.in_features`,
  so it adapts to whatever `n_emb` the trained model uses.
- `policy.conditional_sample` has the signature
  `(condition_data, condition_mask, cond=None, generator=None, **kwargs)` —
  matches the transformer variant in this repo. The UNet variant uses
  `local_cond=None, global_cond=None` instead and will error if you swap it
  in without adjusting the patch.
- Conceptor shapes are read directly from the npz (no `HIDDEN_DIM` baked
  into the hook), so if you retrain with a different `n_emb` you only need
  to rebuild the conceptor file.
