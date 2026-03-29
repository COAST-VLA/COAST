# Pi0.5 Architecture & MetaWorld Inference Pipeline

This document describes the pi0.5 model architecture and how the MetaWorld inference pipeline works end-to-end.

## End-to-End Inference Flow

When you run the two commands (serve + evaluate):

```bash
# Terminal 1: Serve the policy
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=/path/to/your/checkpoint

# Terminal 2: Evaluate
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
```

### Step 1: Policy Server (`scripts/serve_policy.py`)

```
CLI args -> tyro.cli(Args) -> get_config("pi05_metaworld") -> create_trained_policy(config, checkpoint_dir)
```

- Loads the `pi05_metaworld` `TrainConfig` from `src/openpi/training/config.py:840`
- `policy_config.create_trained_policy()`:
  1. Loads model weights from checkpoint (JAX params or PyTorch safetensors)
  2. Creates the transform pipeline (MetaworldInputs -> Normalize -> ResizeImages -> TokenizePrompt -> PadStatesAndActions)
  3. Loads norm stats from the checkpoint's `assets/` directory
  4. Returns a `Policy` object wrapping the model + transforms
- Starts a WebSocket server on port 8000

### Step 2: Client Evaluation (`examples/metaworld/main.py`)

Each step in the environment loop:
1. Collects observations from vectorized MetaWorld envs (camera images + state)
2. Sends observation dict over WebSocket to the policy server
3. Receives an action chunk: `(batch, 32 timesteps, 4 action dims)`
4. Executes `replan_steps=10` actions, then re-queries the model

### Step 3: Inside `Policy.infer()` (`src/openpi/policies/policy.py`)

When batched observations arrive (`state.ndim == 2`), calls `infer_batched()`:

```
Raw obs dict
  -> Unbatch into per-example dicts
  -> Per-example transforms:
      MetaworldInputs: maps observation/image -> base_0_rgb, observation/wrist_image -> left_wrist_0_rgb
      Normalize: quantile normalization to [-1, 1]
      ResizeImages: ensure 224x224
      TokenizePrompt: PaliGemma tokenizer (max 200 tokens for pi0.5)
      PadStatesAndActions: pad state (4,) -> (32,), pad actions to dim 32
  -> Collate back to batch
  -> model.sample_actions() (the denoising loop)
  -> Output transforms:
      Unnormalize: [-1, 1] -> original scale
      MetaworldOutputs: slice actions[..., :4] (only first 4 of 32 dims)
```

---

## Pi0.5 Model Architecture

**File:** `src/openpi/models/pi0.py` (class `Pi0`)

### Components

| Component | Implementation | Dimensions |
|-----------|---------------|------------|
| Vision Encoder | SigLIP-So400m/14 (`src/openpi/models/siglip.py`) | 27 ViT blocks, width 1152, patch 14x14 |
| PaliGemma Expert | Gemma 2B (`src/openpi/models/gemma.py`) | 18 blocks, width 2048, 8 heads, GQA (1 KV head) |
| Action Expert | Gemma 300M | 18 blocks, width 1024, 8 heads, GQA (1 KV head) |
| Embedder | Shared | vocab 257,152 -> 2048-d |

### Dual-Expert Architecture

Both experts share the same attention mechanism but have separate weights:

- **Expert 0 (PaliGemma):** Processes image tokens + language tokens with standard RMSNorm
- **Expert 1 (Action Expert):** Processes action tokens with **adaptive RMSNorm (adaRMS)** conditioned on the flow matching timestep

They share attention: queries/keys/values from both experts are concatenated, so action tokens can attend to image+language tokens (cross-attention happens implicitly).

### Token Flow Through the Model

```
Prefix (bidirectional attention, AR mask = 0):
  +-- Image tokens: 3 cameras x (224/14)^2 = 3 x 256 = 768 tokens -> SigLIP -> (B, 768, 1152) -> projected to 2048-d
  +-- Language tokens: tokenized prompt -> embedded -> (B, <=200, 2048)

Suffix (causal from prefix, AR mask for first action token):
  +-- Action tokens: noisy_actions -> action_in_proj -> (B, 32, 1024)
      + Timestep conditioning via adaRMS (NOT concatenated like in pi0)
```

### Pi0 vs Pi0.5 Key Differences

| Feature | Pi0 | Pi0.5 |
|---------|-----|-------|
| State input | Continuous, separate token in suffix | Discretized into 256 bins, embedded in language tokens |
| Timestep injection | Concatenated with action tokens -> MLP | Injected via **adaptive RMSNorm** in every transformer block |
| max_token_len | 48 | 200 |
| `discrete_state_input` | False | True (by default) |

**Adaptive RMSNorm** (`src/openpi/models/gemma.py:112-131`):
```
Regular RMSNorm:  output = norm(x) * (1 + scale)
Adaptive RMSNorm: output = norm(x) * (1 + scale) + shift,  with gate applied to residual
                  where (scale, shift, gate) = Linear(timestep_embedding)
```

---

## Flow Matching: Training vs Inference

### Training (`Pi0.compute_loss`, `src/openpi/models/pi0.py:189-214`)

Learn a velocity field that transports noise to the action distribution:

```python
noise = N(0, I)                              # shape: (B, 32, 32)
time ~ Beta(1.5, 1) * 0.999 + 0.001         # shape: (B,), range [0.001, 1]
x_t = time * noise + (1 - time) * actions    # linear interpolation
u_t = noise - actions                        # target velocity

v_t = model(x_t, time, observation)          # predicted velocity
loss = MSE(v_t, u_t)                         # per-action-dim, then averaged
```

- Single forward pass per training step (prefix + suffix together)
- Random timestep sampled from Beta(1.5, 1) -- biased toward higher noise levels
- Convention: t=1 is pure noise, t=0 is clean data

### Inference (`Pi0.sample_actions`, `src/openpi/models/pi0.py:217-279`)

Iteratively denoise from pure noise to clean actions:

```python
x_t = N(0, I)           # start with pure noise
dt = -1/10              # 10 denoising steps

# Pre-compute prefix (images + language) KV cache ONCE
_, kv_cache = model.llm([prefix_tokens, None], ...)

# Denoising loop (jax.lax.while_loop for XLA compilation)
for step in range(10):  # t goes 1.0 -> 0.0
    v_t = model(x_t, time, observation, kv_cache=kv_cache)
    x_t = x_t + dt * v_t    # Euler step
    time = time + dt

return x_t  # denoised actions
```

**Critical optimization:** The KV cache for the prefix (image + language tokens) is computed once and reused across all 10 denoising steps. Only the suffix (action tokens) is recomputed each step.

### Summary of Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Timesteps | 1 random t per sample | 10 sequential steps (1.0 -> 0.0) |
| Forward passes | 1 (prefix + suffix together) | 1 prefix + 10 suffix (with KV cache) |
| Input actions | Ground truth + noise interpolation | Iteratively refined from pure noise |
| Objective | Minimize MSE on velocity prediction | Solve the learned ODE via Euler method |
| Data augmentation | Random crop, rotation, color jitter on images | None |

---

## MetaWorld-Specific Configuration

**Config:** `pi05_metaworld` (`src/openpi/training/config.py:840-859`)

```
Model: Pi0Config(pi05=True, action_horizon=32, discrete_state_input=False)
  Note: discrete_state_input=False overrides pi05 default of True
Data: LeRobotMetaworldDataConfig(repo_id="brandonyang/metaworld_ml45")
Training: batch=128, lr=5e-5 -> 5e-6 cosine, 30k steps, EMA=0.999
Base weights: gs://openpi-assets/checkpoints/pi05_base/params
```

**MetaWorld action space:** Only 4 dimensions (3D end-effector position + gripper), padded to 32 for the model.

**Cameras:** corner4 (base camera) + gripperPOV (wrist camera). A third camera slot (right_wrist_0_rgb) is filled with zeros and masked out.

---

## Key Files Reference

| File | What it does |
|------|-------------|
| `src/openpi/models/pi0.py` | Pi0/Pi0.5 model: embed_prefix, embed_suffix, compute_loss, sample_actions |
| `src/openpi/models/gemma.py` | Dual-expert Gemma transformer with RMSNorm/adaRMS, attention, FFN |
| `src/openpi/models/siglip.py` | SigLIP vision encoder (So400m/14 ViT) |
| `src/openpi/models/pi0_config.py` | Pi0Config dataclass with hyperparameters |
| `src/openpi/models/tokenizer.py` | PaliGemma tokenizer with discrete state binning |
| `src/openpi/policies/policy.py` | Policy wrapper: infer(), infer_batched(), transform pipeline |
| `src/openpi/policies/metaworld_lerobot_policy.py` | MetaworldInputs/MetaworldOutputs transforms |
| `src/openpi/training/config.py` | TrainConfig, LeRobotMetaworldDataConfig, pi05_metaworld config |
| `scripts/serve_policy.py` | Policy server CLI |
| `examples/metaworld/main.py` | Single-task evaluation client |
| `examples/metaworld/eval_all.py` | Multi-task ML45 evaluation client |
