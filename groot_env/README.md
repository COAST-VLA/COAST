# GR00T N1.5 Server

An isolated-venv server that serves [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) checkpoints over openpi's WebSocket protocol, so any openpi env client (robocasa, libero, metaworld, droid, …) can target GR00T without client-side changes. N1.5 pins `torch==2.5.1`, which conflicts with the root openpi env — hence the separate venv.

## Installation

From the repo root:

```bash
# 1. Pull the Isaac-GR00T submodule (first time only).
git submodule update --init --recursive

# 2. Create the groot_env venv (uv auto-installs Python 3.10 per pyproject.toml).
cd groot_env
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 3. flash-attn needs torch present at build time, so install it WITHOUT build
#    isolation (matches NVIDIA's official install guide).
uv pip install --no-build-isolation flash-attn==2.7.1.post4

# 4. Download a robocasa365 N1.5 checkpoint (~8 GB including optimizer state).
#    This is the GR00T analog of pi0.5's pi05_pretrain_human300/multitask_learning/75000.
cd ..
uv run hf download robocasa/robocasa365_checkpoints \
    --include "gr00t_n1-5/multitask_learning/checkpoint-120000/*" \
    --local-dir checkpoints/groot_n15
```

Alternative checkpoints under `gr00t_n1-5/` on the same HF repo:

| Path | Recipe |
|---|---|
| `multitask_learning/checkpoint-120000` *(default)* | 365-task multitask pre-training. Reported atomic-seen mean: **43.0%**. Direct analog of pi0.5's `pi05_pretrain_human300/multitask_learning/75000`. |
| `foundation_model_learning/pretraining/checkpoint-80000` | Foundation-model-learning recipe pretraining (65 atomic tasks). |
| `foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` | Same pretraining + post-training on `atomic_seen`. Reported mean: **68.5%**. |
| `lifelong_learning/phase{1..4}/checkpoint-*` | Sequential / lifelong-learning experiments. |

## Serving the policy

From `groot_env/`:

```bash
uv run python serve.py --port 8000
```

`serve.py` defaults to `multitask_learning/checkpoint-120000`; override with `--model-path`. Full flags:

```bash
uv run python serve.py \
    --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \
    --embodiment robocasa \
    --device cuda:0 \
    --port 8000 \
    --denoising-steps 4         # NVIDIA's default; matches their published numbers
```

Metadata reported on connect (`client.get_server_metadata()`):

```json
{
  "backend": "groot_n15",
  "model_path": "…/checkpoint-120000",
  "embodiment": "robocasa",
  "denoising_steps": 4,
  "collection_mode": "groot_v1",  // only when --collect_activations
  "model_type": "groot_n15",      // parallels pi0-side "pi0" / "pi05" / "pi0_fast"
  "checkpoint_step": "checkpoint-120000"
}
```

## Evaluation

Because the WebSocket protocol is identical to `scripts/serve_policy.py`, existing clients target GR00T with **no edits**:

```bash
# In a second terminal, with GR00T serving on :8000
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name OpenDrawer --num_episodes 15

# Or the task-set driver:
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --num_episodes 15 --num_workers 5
```

Videos and `results.json` land under `output/...` exactly as for pi0.5.

## Activation collection

`serve.py --collect_activations` wraps the policy in `groot_activation_collector.CollectingPolicy`, which rejects plain inference requests and only accepts calls carrying `__collect__` or `__finalize_episode__` magic keys. Protocol, output directory layout, and verification are covered in the canonical reference — see **[`../docs/activation_collection.md`](../docs/activation_collection.md)**.

```bash
cd groot_env
export CUDA_VISIBLE_DEVICES=0
uv run python serve.py --port 8000 --collect_activations \
    --output-dir ../groot_n15-robocasa-activations-v1-15env
# …then run any client with --collect (see examples/robocasa_env/README.md).
```

### `groot_v1` schema

Schema mirrors pi0's `sample_actions_with_intermediates` layout one-for-one — hybrid re-implemented denoising loop with forward hooks on the per-layer residuals and MLP expansion. Per-step files in `step_NNNN/`:

| File | Contents | pi0 analog |
|---|---|---|
| `denoising.npz`         | `all_x_t (D, H, A)` fp32, `all_v_t (D, H, A)` fp32 | same |
| `backbone_cond.npz`     | `backbone_features (S, C)` fp16                   | `adarms_cond.npz` (different shape — see note) |
| `dit_hidden_states.npz` | `all_dit_hidden_states (D, L, S, C)` fp16          | `suffix_residual.npz` |
| `dit_mlp_hidden.npz`    | `all_dit_mlp_hidden (D, L, S, F)` fp16             | `suffix_mlp_hidden.npz` |

Where `D = num_denoising_steps`, `L = num_dit_layers`, `H = action_horizon`, `A = padded_action_dim`, `S = state + future + action token count`, `C = dit_hidden_dim`, `F = ff_inner_dim`.

**`backbone_cond.npz` vs pi0's `adarms_cond.npz`.** GR00T uses cross-attention from the DiT to a VL backbone sequence computed **once per inference** — shape `(S, C)`. pi0 uses AdaRMS with a pooled conditioning vector that varies per denoising step — shape `(D, C)`. Everything else matches semantically; shapes reflect the N1.5 DiT's 16 layers + 4 denoising steps vs pi0's 4 suffix layers + 10 denoising steps.

### Verifying activations

```bash
# Deep invariants for ONE task: directory layout, metadata fields, reward
# array/length agreement, dtype/shape strictness, flow-matching norm signature,
# cross-episode variation.
ACTIVATIONS_DIR=../groot_n15-robocasa-activations-v1-15env/checkpoint-120000/OpenDrawer \
    uv run pytest tests/test_groot_activations.py -v

# Sweep every task in a dataset:
for task in ../groot_n15-robocasa-activations-v1-15env/checkpoint-120000/*/; do
    ACTIVATIONS_DIR="$task" uv run pytest tests/test_groot_activations.py -v
done
```

Pre-collected: [`brandonyang/groot_n15-robocasa-activations-v1-15env`](https://huggingface.co/datasets/brandonyang/groot_n15-robocasa-activations-v1-15env) — 7 tasks × 15 episodes on `multitask_learning/checkpoint-120000`.

## Results

Robocasa atomic-seen, 15 ep/task, `multitask_learning/checkpoint-120000`:

| Task | **GR00T N1.5 (here)** | pi0.5 (`examples/robocasa_env/figures/results_75000.json`) | GR00T published |
|---|:-:|:-:|:-:|
| PickPlaceCounterToStove | **60%** | 47% | 63.2% |
| PickPlaceCounterToCabinet | **53%** | 47% | 47.5% |
| OpenDrawer | **53%** | 60% | 81.1% |
| OpenStandMixerHead | **33%** | 67% | — |
| CloseFridge | **27%** | 13% | — |
| TurnOnElectricKettle | **27%** | 13% | — |
| CoffeeSetupMug | **20%** | 27% | 31.0% |
| **Mean (these 7)** | **39%** | 39% | — |

RoboCasa's published N1.5 multitask atomic-seen average is **43.0%** over all 18 tasks.

## Known limitations

1. **Image resolution upsampling.** The pi0-compatible openpi robocasa client sends 224×224 images (pi0.5's training resolution); N1.5's robocasa head expects 256×256 per its modality config, so `groot_adapter._resize_to_256` upscales via `cv2.INTER_LINEAR` before inference. Rendering the env natively at 256×256 would be marginally better but would require either (a) client-side `--resize_size 256` (breaks pi0.5 compat) or (b) an env-wrapper change.
2. **Fixed action horizon, short client replan.** The DiT always produces a 16-step action chunk; the robocasa client uses `replan_steps=5`, so 11 of 16 predicted steps are recomputed every call. Correctness is unaffected — only a perf tax.
3. **Right-view cross-camera fallback.** `build_robocasa_videos` prefers `observation/image2` (agentview_right) when the client sends it. Older clients without `observation/image2` fall back to duplicating `observation/image` as the right view, which degrades the stereo signal N1.5 was trained on but keeps the 3-channel shape contract. The openpi robocasa `main.py` in this repo emits `observation/image2` by default, so this fallback isn't hit in normal use.

## How this integrates (for reference)

- `serve.py` — thin wrapper that loads the checkpoint and starts a `WebsocketPolicyServer`. Mirrors `scripts/serve_policy.py` on the pi0 side.
- `groot_adapter.py` — `GR00TAdapterPolicy(BasePolicy)` translating openpi's flat `observation/*`-keyed client dict to GR00T's nested `{video, state, language}` dict, running `Gr00tPolicy.get_action`, concatenating the per-action-key dict back to a single `(action_horizon, action_dim)` array under `"actions"`. Also houses `_get_action_with_intermediates`, a hybrid re-implementation + hook-based collector matching pi0's `sample_actions_with_intermediates` pattern.
- `groot_activation_collector.py` — `CollectingPolicy` wrapper that dispatches `__collect__` / `__finalize_episode__` magic keys and writes the per-step/per-episode .npz files. Schema matches pi0's semantically; only file names and shapes differ per the architecture.
- `websocket_policy_server.py` — copied verbatim from `src/openpi/serving/` (sha noted in its header) so `groot_env/` can serve without pulling in JAX/flax.
- `tests/test_groot_adapter.py` — unit tests for translation logic + a `@pytest.mark.manual` real-model equivalence test that asserts `_get_action_with_intermediates` is bit-identical to `Gr00tPolicy.get_action`.
- `tests/test_groot_activation_collector.py` — stub-based unit tests for writers + dispatcher, analog of pi0's `tests/test_activation_collector.py`.
- `tests/test_groot_activations.py` — post-collection dataset validator (one task dir at a time, driven by `ACTIVATIONS_DIR`), analog of pi0's `tests/test_activations.py`.
