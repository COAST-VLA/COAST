# Implementation Plan: Activation Collection for Pi0.5 Mech Interp (PyTorch)

## Context

We want to collect intermediate activations from the pi0.5 model during MetaWorld evaluation rollouts, as described in `docs/mechanistic_interpretability_plan.md`. We use the **PyTorch implementation** (`src/openpi/models_pytorch/pi0_pytorch.py`) for activation collection because it natively supports `register_forward_hook` for per-layer extraction — no JAX-specific workarounds needed.

**Checkpoint:** `checkpoints/pi05_metaworld/pi05_metaworld_test/5000/`
**Evaluation:** All 45 ML45 train tasks
**Backend:** PyTorch (auto-converted from JAX checkpoint via `--pytorch` flag)
**Approach:** In-process policy loading (no WebSocket server). `register_forward_hook` for per-layer activations.

---

## Why PyTorch Instead of JAX

The JAX implementation had multiple engineering obstacles for activation collection:

| JAX Problem | PyTorch Equivalent |
|---|---|
| `jax.lax.while_loop` blocks side effects | Python `while` loop — just append to a list |
| `nn.scan` hides per-layer outputs | `nn.ModuleList` — iterate or hook any layer |
| Adding methods to NNX classes causes OOM | Adding methods to `nn.Module` is fine |
| `sow()` + NNX bridge compatibility unknown | `register_forward_hook` is a first-class API |
| GPU memory fragmentation after ~30 tasks | PyTorch `torch.cuda.empty_cache()` + per-task processes |
| `module_jit` freezes state, complicates extraction | No JIT by default; `torch.compile` can be disabled |

**Note:** `torch.compile(mode="max-autotune")` is applied to `sample_actions` in `PI0Pytorch.__init__`. For activation collection, we bypass this by calling the uncompiled model methods directly (hooks work on the underlying modules regardless of `torch.compile`).

---

## PyTorch Model Structure (Hook Targets)

### Module Hierarchy

```
PI0Pytorch (src/openpi/models_pytorch/pi0_pytorch.py)
├── paligemma_with_expert: PaliGemmaWithExpertModel
│   ├── paligemma: PaliGemmaForConditionalGeneration
│   │   └── model.language_model.layers: nn.ModuleList[18]  ← PaliGemma transformer
│   │       └── [i]: GemmaDecoderLayer
│   │           ├── input_layernorm: GemmaRMSNorm (no adaRMS)
│   │           ├── self_attn: GemmaAttention
│   │           ├── post_attention_layernorm: GemmaRMSNorm (no adaRMS)
│   │           └── mlp: GemmaMLP
│   │               ├── gate_proj: Linear(2048, 16384)
│   │               ├── up_proj: Linear(2048, 16384)
│   │               └── down_proj: Linear(16384, 2048)
│   │
│   └── gemma_expert: GemmaForCausalLM  ← Action Expert transformer
│       └── model.layers: nn.ModuleList[18]
│           └── [i]: GemmaDecoderLayer
│               ├── input_layernorm: GemmaRMSNorm(cond_dim=1024)  ← adaRMS!
│               ├── self_attn: GemmaAttention
│               ├── post_attention_layernorm: GemmaRMSNorm(cond_dim=1024)  ← adaRMS!
│               └── mlp: GemmaMLP
│                   ├── gate_proj: Linear(1024, 4096)
│                   ├── up_proj: Linear(1024, 4096)
│                   └── down_proj: Linear(4096, 1024)
│
├── action_in_proj: Linear(32, 1024)
├── action_out_proj: Linear(1024, 32)
├── time_mlp_in: Linear(1024, 1024)  (pi05 only)
└── time_mlp_out: Linear(1024, 1024)  (pi05 only)
```

### Hook Paths for Target Activations

| Activation | Hook Target | What to Capture | Shape |
|---|---|---|---|
| Action Expert residual (layer i) | `.gemma_expert.model.layers[i]` | hook output `[0]` | `(B, 32, 1024)` |
| Action Expert MLP hidden (layer i) | `.gemma_expert.model.layers[i].mlp.down_proj` | hook **input** `[0]` | `(B, 32, 4096)` |
| Action Expert adaRMS gate (layer i) | `.gemma_expert.model.layers[i].input_layernorm` | hook output `[1]` (gate) | `(B, 1, 1024)` |
| PaliGemma residual (layer i) | `.paligemma.model.language_model.layers[i]` | hook output `[0]` | `(B, prefix_len, 2048)` |
| PaliGemma MLP hidden (layer i) | `.paligemma.model.language_model.layers[i].mlp.down_proj` | hook **input** `[0]` | `(B, prefix_len, 16384)` |

(All paths prefixed with `model.paligemma_with_expert`)

**MLP hidden via `down_proj` input hook:** `GemmaMLP.forward()` computes `down_proj(act_fn(gate_proj(x)) * up_proj(x))` in a single line. The MLP hidden activation (`act_fn(gate_proj(x)) * up_proj(x)`) is the **input to `down_proj`**. By hooking `down_proj` with `register_forward_hook` and capturing `input[0]`, we get the high-dimensional MLP hidden directly — no monkey-patching or wrapper needed.

### Denoising Loop Structure

```python
# In PI0Pytorch.sample_actions() — 10 denoising steps
x_t = noise                          # (B, 32, 32) — pure noise
time = 1.0
dt = -1.0 / num_steps

while time >= -dt / 2:
    v_t = model.denoise_step(...)    # (B, 32, 32) — velocity prediction
    x_t = x_t + dt * v_t            # Euler step
    time += dt

return x_t                           # denoised actions
```

Each `denoise_step` call:
1. `embed_suffix(state, x_t, timestep)` → suffix embeddings + `adarms_cond`
2. Forward through `paligemma_with_expert` with cached prefix KV → suffix output
3. `action_out_proj(suffix_out[:, -action_horizon:])` → velocity `v_t`

---

## Implementation Plan

### Step 1: Create `sample_actions_with_intermediates()` for PyTorch

**File:** `src/openpi/models_pytorch/pi0_pytorch.py`

Add a method (not standalone — PyTorch `nn.Module` has no NNX inflation issue) that:
- Registers forward hooks on target layers before the denoising loop
- Runs the denoising loop (Python while loop, same as `sample_actions` but without `torch.compile`)
- Collects per-step intermediates from hooks, moves to CPU each step
- Removes hooks after completion
- Returns `(final_actions, intermediates_dict)`

```python
@torch.no_grad()
def sample_actions_with_intermediates(
    self, device, observation, *, noise=None, num_steps=10,
    collect_layers=(0, 5, 11, 17),
) -> tuple[Tensor, dict]:
    """Like sample_actions() but collects per-step intermediates via hooks.

    Does NOT use torch.compile — runs in eager mode for hook compatibility.
    """
    hooks = []
    step_activations = {}

    def make_output_hook(name):
        """Capture module output (for layer residual streams)."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                step_activations[name] = output[0].detach().cpu()
            else:
                step_activations[name] = output.detach().cpu()
        return hook_fn

    def make_input_hook(name):
        """Capture module input (for MLP hidden = input to down_proj)."""
        def hook_fn(module, input, output):
            step_activations[name] = input[0].detach().cpu()
        return hook_fn

    # Hook Action Expert layers
    expert_layers = self.paligemma_with_expert.gemma_expert.model.layers
    for i in collect_layers:
        # Layer residual stream (output of full decoder layer)
        hooks.append(expert_layers[i].register_forward_hook(
            make_output_hook(f"expert_residual_{i}")))
        # MLP hidden: input to down_proj = act_fn(gate_proj(x)) * up_proj(x)
        hooks.append(expert_layers[i].mlp.down_proj.register_forward_hook(
            make_input_hook(f"expert_mlp_hidden_{i}")))

    try:
        # ... prefix pass, then denoising loop (same as sample_actions) ...
        all_x_t, all_v_t, all_adarms_cond = [], [], []
        all_suffix_residual, all_suffix_mlp_hidden = [], []

        for step in range(num_steps):
            step_activations.clear()
            v_t = self.denoise_step(...)

            all_x_t.append(x_t.detach().cpu())
            all_v_t.append(v_t.detach().cpu())
            all_adarms_cond.append(adarms_cond.detach().cpu())

            # Collect hooked activations for this step
            all_suffix_residual.append(
                torch.stack([step_activations[f"expert_residual_{i}"] for i in collect_layers]))
            all_suffix_mlp_hidden.append(
                torch.stack([step_activations[f"expert_mlp_hidden_{i}"] for i in collect_layers]))

            x_t = x_t + dt * v_t
            time += dt
    finally:
        for h in hooks:
            h.remove()

    intermediates = {
        "all_x_t": torch.stack(all_x_t).numpy().astype(np.float32),
        # shape: (num_steps, batch, 32, 32)
        "all_v_t": torch.stack(all_v_t).numpy().astype(np.float32),
        # shape: (num_steps, batch, 32, 32)
        "all_adarms_cond": torch.stack(all_adarms_cond).numpy().astype(np.float32),
        # shape: (num_steps, batch, 1024)
        "all_suffix_residual": torch.stack(all_suffix_residual).numpy().astype(np.float32),
        # shape: (num_steps, num_layers, batch, 32, 1024)
        "all_suffix_mlp_hidden": torch.stack(all_suffix_mlp_hidden).numpy().astype(np.float32),
        # shape: (num_steps, num_layers, batch, 32, 4096)
    }
    return x_t, intermediates
```

### Step 2: Add `infer_with_intermediates()` for PyTorch to `Policy`

**File:** `src/openpi/policies/policy.py`

Add a method that:
- Applies the same input transforms as `infer_batched()`
- Converts to torch tensors
- Calls `model.sample_actions_with_intermediates(device, observation)`
- Applies output transforms to actions
- Returns `(outputs, intermediates)`

### Step 3: Create `scripts/collect_activations.py`

Adapt the collection script from the JAX version:
- Load policy in-process with `create_trained_policy` (auto-detects PyTorch from `model.safetensors`)
- Use `AsyncVectorEnv(context="spawn")` for parallel env stepping
- At each replan point, call `policy.infer_with_intermediates()`
- Save per-env, per-step activations AND metadata to disk
- After each episode completes, write episode-level metadata with outcome labels

**Key difference from JAX version:** No need for per-task subprocess execution. PyTorch's memory management is more predictable. However, still recommended for very long runs (45 tasks) as a safety measure.

#### Metadata to collect

**Episode-level metadata** (written to `episode_*/metadata.json` after episode completes):
```json
{
  "task_name": "reach-v3",
  "episode_id": 0,
  "env_id": 0,
  "episode_success": true,
  "total_reward": 720.89,
  "steps_to_success": 34,
  "total_env_steps": 300,
  "total_inference_steps": 30,
  "prompt": "reach the goal position",
  "checkpoint_dir": "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/",
  "config_name": "pi05_metaworld"
}
```

**Step-level metadata** (written to `step_*/metadata.json` at each inference call):
```json
{
  "task_name": "reach-v3",
  "episode_id": 0,
  "env_id": 0,
  "step": 20,
  "inference_step": 2,
  "prompt": "reach the goal position",
  "cumulative_reward": 45.3,
  "success_so_far": false,
  "reward_since_last_inference": 15.1
}
```

The `cumulative_reward` and `success_so_far` fields enable correlating activations with the reward trajectory *within* an episode — e.g., do activations change when the robot starts accumulating reward? Does the model "know" it's about to succeed?

**Reward trajectory** (written to `episode_*/rewards.npz` after episode completes):
```python
{
  "per_step_reward": np.ndarray,    # (total_env_steps,) — reward at each env step
  "cumulative_reward": np.ndarray,  # (total_env_steps,) — running sum
  "success_at_step": np.ndarray,    # (total_env_steps,) — bool, success flag at each step
}
```

This enables plotting reward curves alongside activation trajectories for individual episodes.

### Step 4: Write tests

**File:** `scripts/test_activations.py`

Test the collected data:
- Directory structure, required files
- Array shapes, dtypes (float32), NaN/Inf checks
- Denoising trajectory sanity (variance decreases, x_t changes between steps)
- Per-layer activation shapes match expectations
- Cross-environment consistency
- Metadata field validation

---

## Activation Shapes Reference

### Per denoising step (10 steps total)

| File | Array | Shape | Description |
|---|---|---|---|
| `denoising.npz` | `all_x_t` | `(10, 32, 32)` | Noisy actions at each step |
| `denoising.npz` | `all_v_t` | `(10, 32, 32)` | Velocity predictions |
| `adarms_cond.npz` | `all_adarms_cond` | `(10, 1024)` | Timestep conditioning input |
| `suffix_residual.npz` | `all_suffix_residual` | `(10, 4, 32, 1024)` | Action Expert post-layer residual at layers 0,5,11,17 |
| `suffix_mlp_hidden.npz` | `all_suffix_mlp_hidden` | `(10, 4, 32, 4096)` | Action Expert MLP hidden (`act_fn(gate) * up`, pre-down_proj) at layers 0,5,11,17 |

### Per prefix pass (computed once)

| File | Array | Shape | Description |
|---|---|---|---|
| `prefix_residual.npz` | `prefix_residual` | `(4, ~968, 2048)` | PaliGemma post-layer residual at layers 0,5,11,17 |

### Storage estimate

- Per inference call: ~5 MB (denoising) + ~5 MB (suffix residual) + ~20 MB (suffix MLP hidden, 4096-d) + ~30 MB (prefix residual) ≈ **60 MB**
- Per task (2 envs, ~15 inference calls): ~900 MB
- 45 tasks total: **~40 GB** (with prefix residual) or **~14 GB** (without prefix)

**Recommendation:** Start without prefix residual to keep storage manageable. Add it for select tasks later.

---

## Storage Format

```
activations/
  {checkpoint_step}/
    {task_name}/
      episode_{episode_id:03d}_env_{env_id:03d}/
        metadata.json              # episode-level: task_name, episode_success, total_reward, steps_to_success, etc.
        rewards.npz                # per-step reward trajectory: per_step_reward, cumulative_reward, success_at_step
        step_{step:04d}/
          denoising.npz            # all_x_t, all_v_t
          adarms_cond.npz          # all_adarms_cond
          suffix_residual.npz      # Action Expert post-layer residual at layers 0,5,11,17
          suffix_mlp_hidden.npz    # Action Expert MLP hidden (pre-down_proj, 4096-d) at layers 0,5,11,17
          prefix_residual.npz      # (optional) PaliGemma residual at layers 0,5,11,17
          metadata.json            # step-level: task_name, cumulative_reward, success_so_far, etc.
```

---

## Verification

1. **Equivalence test:** `sample_actions_with_intermediates()` must produce the same final actions as `sample_actions()` for the same noise input. Compare on reach-v3.

2. **Hook sanity:** All hooked activations should be non-zero, finite, and vary across denoising steps.

3. **Integration test:** Run on reach-v3 with `num_envs=2`, verify files are written, shapes are correct, metadata is valid.

4. **Metadata validation:**
   - Episode metadata has all required fields: `task_name`, `episode_success`, `total_reward`, `steps_to_success`, `total_env_steps`, `total_inference_steps`, `prompt`, `checkpoint_dir`, `config_name`
   - Step metadata has: `task_name`, `episode_id`, `env_id`, `step`, `inference_step`, `prompt`, `cumulative_reward`, `success_so_far`, `reward_since_last_inference`
   - `episode_success=True` → `steps_to_success >= 0`; `episode_success=False` → `steps_to_success == -1`
   - `cumulative_reward` in step metadata is monotonically non-decreasing (MetaWorld rewards are non-negative)
   - `rewards.npz` has arrays of length `total_env_steps` and `cumulative_reward[-1]` matches `total_reward` in episode metadata

5. **Full collection:** Run on all 45 ML45 train tasks, verify storage size is reasonable and success/failure labels match known evaluation results (reach-v3 = 100%, dial-turn-v3 = 0%).

6. **MLP hidden sanity:**
   - MLP hidden should NOT be all zeros
   - Should vary across denoising steps
   - Layer 0 vs layer 17 should have different statistics

7. **GPU memory:** Monitor with `nvidia-smi` during collection. `.detach().cpu()` in hooks should keep GPU memory stable.
