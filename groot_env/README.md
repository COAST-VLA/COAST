# GR00T N1.5 Server

An isolated-venv server that serves [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) models over openpi's WebSocket protocol, so any openpi env client (robocasa, libero, metaworld, droid, ...) can target GR00T without any client-side changes.

## Setup

From the repo root:

```bash
# 1. Pull the Isaac-GR00T submodule (first time only).
git submodule update --init --recursive

# 2. Create the groot_env venv. uv auto-installs Python 3.10 per pyproject.toml.
cd groot_env
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 3. flash-attn needs torch present at build time, so install it WITHOUT build
#    isolation (matches NVIDIA's official install guide).
uv pip install --no-build-isolation flash-attn==2.7.1.post4

# 4. Download a robocasa365 N1.5 checkpoint (~8 GB incl. optimizer state).
#    This one is the GR00T analog of pi05's pi05_pretrain_human300/multitask_learning/75000.
cd ..
uv run hf download robocasa/robocasa365_checkpoints \
    --include "gr00t_n1-5/multitask_learning/checkpoint-120000/*" \
    --local-dir checkpoints/groot_n15
```

Alternative checkpoints under `gr00t_n1-5/` on the same HF repo:

| Path | Recipe |
|---|---|
| `multitask_learning/checkpoint-120000` *(default)* | 365-task multitask pre-training. Reported atomic-seen mean: **43.0%**. Direct analog of pi05's `pi05_pretrain_human300/multitask_learning/75000`. |
| `foundation_model_learning/pretraining/checkpoint-80000` | Foundation-model-learning recipe pretraining (65 atomic tasks). |
| `foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` | Same pretraining + post-training on the `atomic_seen` task set. Reported mean: **68.5%**. |
| `lifelong_learning/phase{1..4}/checkpoint-*` | Sequential / lifelong-learning experiments. |

## Serving

From `groot_env/`:

```bash
uv run python serve.py --port 8000
```

`serve.py` defaults to the `multitask_learning/checkpoint-120000` you downloaded above; override with `--model-path`. Full flags:

```bash
uv run python serve.py \
    --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \
    --embodiment robocasa \
    --device cuda:0 \
    --port 8000 \
    --denoising-steps 4         # NVIDIA's default; matches their published numbers
```

Collection mode (saves per-step activations to disk):

```bash
uv run python serve.py \
    --collect-activations \
    --output-dir ../groot_n15-robocasa-activations-v1-15env
# …then run any client with --collect (see examples/robocasa_env/README.md).
```

The metadata the server reports on connection (`get_server_metadata()`):

```json
{
  "backend": "groot_n15",
  "model_path": "…/checkpoint-120000",
  "embodiment": "robocasa",
  "denoising_steps": 4,
  "collection_mode": "groot_v1", // only when --collect-activations
  "model_type": "groot_n15",     // parallels pi0-side "pi0" / "pi05" / "pi0_fast"
  "checkpoint_step": "checkpoint-120000"
}
```

---

## Running evaluations

Because the WebSocket protocol is identical to `scripts/serve_policy.py`, existing clients target GR00T with **no edits**:

```bash
# In a second terminal, with GR00T serving on :8000
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name OpenDrawer --num_episodes 15

# Or the task-set driver:
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --num_episodes 15 --num_workers 5
```

That's it. Videos + `results.json` go under `output/...` exactly as for pi05.

---

## Activation collection

`serve.py --collect-activations` wraps the policy in `groot_activation_collector.CollectingPolicy`, which rejects plain inference requests and only accepts calls carrying `__collect__` or `__finalize_episode__` magic keys. The robocasa client's `--collect` flag already emits these (see `openpi_client.collection_session.CollectionSession`).

Collection structure mirrors pi0's `sample_actions_with_intermediates` layout one-for-one — hybrid re-implemented denoising loop with forward hooks on the per-layer residuals and MLP expansion. Output files:

```
<output_dir>/checkpoint-120000/<task_name>/episode_NNN_env_NNN/
├── metadata.json              # task_name, episode_id, episode_success, total_reward, total_inference_steps, prompt, …
├── rewards.npz                # per_step_reward, cumulative_reward, success_at_step
└── step_NNNN/
    ├── metadata.json          # step, inference_step, cumulative_reward, success_so_far
    ├── denoising.npz          # all_x_t (D,H,A) fp32, all_v_t (D,H,A) fp32                (pi0 schema: same)
    ├── backbone_cond.npz      # backbone_features (S,C) fp16                              (pi0 analog: adarms_cond; see note below)
    ├── dit_hidden_states.npz  # all_dit_hidden_states (D,L,S,C) fp16                      (pi0 analog: suffix_residual)
    └── dit_mlp_hidden.npz     # all_dit_mlp_hidden (D,L,S,F) fp16                         (pi0 analog: suffix_mlp_hidden)
```

where `D = num_denoising_steps`, `L = num_dit_layers`, `H = action_horizon`, `A = padded_action_dim`, `S = state+future+action token count`, `C = dit_hidden_dim`, `F = ff_inner_dim`.

**Difference from pi0's schema**: GR00T uses cross-attention from the DiT to a VL backbone sequence that's computed once per inference (not per denoising step), whereas pi0 uses AdaRMS with a pooled conditioning vector that varies per step. So `backbone_cond.npz` has shape `(seq, hidden)` — the VL sequence — while pi0's `adarms_cond.npz` has shape `(num_steps, hidden)`. Everything else (`denoising`, per-layer residuals, MLP hidden) matches pi0's schema semantically, with shapes reflecting the N1.5 DiT's 16 layers + 4 denoising steps vs pi0's 4 suffix layers + 10 denoising steps.

### Verifying activations

Env-var-driven pytest suite (skipped in CI when `ACTIVATIONS_DIR` is unset):

```bash
# Deep invariants for ONE task: directory layout, metadata fields,
# reward array/length agreement, dtype/shape strictness, flow-matching norm
# signature, cross-episode variation.
ACTIVATIONS_DIR=../groot_n15-robocasa-activations-v1-15env/checkpoint-120000/OpenDrawer \
    uv run pytest tests/test_groot_activations.py -v
```

To sweep all tasks in a dataset, loop the same command:

```bash
for task in ../groot_n15-robocasa-activations-v1-15env/checkpoint-120000/*/; do
    ACTIVATIONS_DIR="$task" uv run pytest tests/test_groot_activations.py -v
done
```

---

## Reference: robocasa atomic-seen (15 ep/task, multitask_learning/checkpoint-120000)

| Task | **GR00T N1.5 (here)** | pi05 (`examples/robocasa_env/figures/results_75000.json`) | GR00T published |
|---|:-:|:-:|:-:|
| PickPlaceCounterToStove | **60%** | 47% | 63.2% |
| PickPlaceCounterToCabinet | **53%** | 47% | 47.5% |
| OpenDrawer | **53%** | 60% | 81.1% |
| OpenStandMixerHead | **33%** | 67% | — |
| CloseFridge | **27%** | 13% | — |
| TurnOnElectricKettle | **27%** | 13% | — |
| CoffeeSetupMug | **20%** | 27% | 31.0% |
| **Mean (these 7)** | **39%** | 39% | — |

Robocasa's published N1.5 multitask atomic-seen average is **43.0%** over all 18 tasks.

---

## Known limitations

1. **Image resolution upsampling**. The pi0-compatible openpi robocasa client sends 224×224 images (pi05's training resolution). N1.5's robocasa head expects 256×256 per its modality config, so `groot_adapter._resize_to_256` upscales via `cv2.INTER_LINEAR` before inference. Rendering the env natively at 256×256 would be marginally better but would require either (a) client-side `--resize_size 256` (breaks pi05 compat) or (b) an env-wrapper change.
2. **Fixed action horizon, short client replan**. The DiT always produces a 16-step action chunk; the robocasa client uses `replan_steps=5`, so 11 of 16 predicted steps are recomputed every call. Only a perf tax — correctness is unaffected.
3. **Right-view cross-camera fallback**. `build_robocasa_videos` prefers `observation/image2` (agentview_right) when the client sends it. Older clients without `observation/image2` fall back to duplicating `observation/image` as the right view, which degrades the stereo signal N1.5 was trained on but keeps the 3-channel shape contract. The openpi robocasa `main.py` in this repo emits `observation/image2` by default, so this fallback path isn't hit in normal use.

## How this integrates (for reference)

- `serve.py` — thin wrapper that loads the checkpoint and starts a `WebsocketPolicyServer`. Mirrors `scripts/serve_policy.py` on the pi0 side.
- `groot_adapter.py` — `GR00TAdapterPolicy(BasePolicy)` translating openpi's flat `observation/*`-keyed client dict to GR00T's nested `{video, state, language}` dict, running `Gr00tPolicy.get_action`, concatenating the per-action-key dict back to a single `(action_horizon, action_dim)` array under `"actions"`. Also houses `_get_action_with_intermediates`, a hybrid re-implementation + hook-based collector matching pi0's `sample_actions_with_intermediates` pattern.
- `groot_activation_collector.py` — `CollectingPolicy` wrapper that dispatches `__collect__` / `__finalize_episode__` magic keys and writes the per-step/per-episode .npz files. Schema matches pi0's semantically; only file names and shapes differ per the architecture.
- `websocket_policy_server.py` — copied verbatim from `src/openpi/serving/` (sha noted in its header) so `groot_env/` can serve without pulling in JAX/flax.
- `tests/test_groot_adapter.py` — unit tests for translation logic + a `@pytest.mark.manual` real-model equivalence test that asserts `_get_action_with_intermediates` is bit-identical to `Gr00tPolicy.get_action`.
- `tests/test_groot_activation_collector.py` — stub-based unit tests for writers + dispatcher, analog of pi0's `tests/test_activation_collector.py`.
- `tests/test_groot_activations.py` — post-collection dataset validator (one task dir at a time, driven by `ACTIVATIONS_DIR`), analog of pi0's `tests/test_activations.py`.
