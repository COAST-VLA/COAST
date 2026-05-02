"""Download and inspect sample files from both HF datasets."""
import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from huggingface_hub import hf_hub_download, HfApi
import json
import numpy as np

api = HfApi()

# ============================================================
# SMALL DATASET: brandonyang/ml45-activations (2 envs per task)
# ============================================================
print("=" * 80)
print("SMALL DATASET: brandonyang/ml45-activations")
print("=" * 80)

repo = "brandonyang/ml45-activations"

# Download episode metadata
print("\n--- Episode Metadata (assembly-v3, env_000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/metadata.json", repo_type="dataset")
with open(f) as fh:
    meta = json.load(fh)
    print(json.dumps(meta, indent=2))

# Download step metadata
print("\n--- Step Metadata (assembly-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/step_0000/metadata.json", repo_type="dataset")
with open(f) as fh:
    step_meta = json.load(fh)
    print(json.dumps(step_meta, indent=2))

# Download and inspect NPZ files
print("\n--- Denoising NPZ (assembly-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/step_0000/denoising.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

print("\n--- AdaRMS Conditioning NPZ (assembly-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/step_0000/adarms_cond.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

print("\n--- Suffix Residual NPZ (assembly-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/step_0000/suffix_residual.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- Suffix MLP Hidden NPZ (assembly-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/step_0000/suffix_mlp_hidden.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- Rewards NPZ (assembly-v3, env_000) ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/rewards.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    if data[key].dtype == bool:
        print(f"    True count: {data[key].sum()}, first True at: {np.argmax(data[key]) if data[key].any() else 'N/A'}")
    else:
        print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, final={data[key][-1]:.4f}")

# Check a successful task too
print("\n--- Episode Metadata (reach-v3, env_000) - typically successful ---")
f = hf_hub_download(repo, "5000/reach-v3/episode_000_env_000/metadata.json", repo_type="dataset")
with open(f) as fh:
    meta = json.load(fh)
    print(json.dumps(meta, indent=2))

print("\n--- Step Metadata (reach-v3, env_000, step_0000) ---")
f = hf_hub_download(repo, "5000/reach-v3/episode_000_env_000/step_0000/metadata.json", repo_type="dataset")
with open(f) as fh:
    step_meta = json.load(fh)
    print(json.dumps(step_meta, indent=2))

# ============================================================
# Count structure statistics for the SMALL dataset
# ============================================================
print("\n" + "=" * 80)
print("STRUCTURE ANALYSIS: SMALL DATASET")
print("=" * 80)

all_files = list(api.list_repo_tree(repo, repo_type="dataset", recursive=True))
# Count episodes per task, steps per episode
task_episodes = {}
task_steps = {}
for f_obj in all_files:
    if not hasattr(f_obj, 'rfilename'):
        continue
    parts = f_obj.rfilename.split('/')
    if len(parts) >= 3 and parts[0] == '5000':
        task = parts[1]
        episode = parts[2]
        if task not in task_episodes:
            task_episodes[task] = set()
        task_episodes[task].add(episode)
        
        if len(parts) >= 4 and parts[3].startswith('step_'):
            key = (task, episode)
            if key not in task_steps:
                task_steps[key] = set()
            task_steps[key].add(parts[3])

print(f"\nTasks: {len(task_episodes)}")
for task in sorted(task_episodes.keys()):
    eps = sorted(task_episodes[task])
    # Count steps for first episode
    first_ep_key = (task, eps[0])
    n_steps = len(task_steps.get(first_ep_key, set()))
    print(f"  {task}: {len(eps)} episodes, ~{n_steps} inference steps per episode")

# ============================================================
# LARGE DATASET: brandonyang/ml45-activations-15
# ============================================================
print("\n" + "=" * 80)
print("LARGE DATASET: brandonyang/ml45-activations-15")
print("=" * 80)

repo15 = "brandonyang/ml45-activations-15"

# Download episode metadata
print("\n--- Episode Metadata (reach-v3, env_000) ---")
f = hf_hub_download(repo15, "5000/reach-v3/episode_000_env_000/metadata.json", repo_type="dataset")
with open(f) as fh:
    meta = json.load(fh)
    print(json.dumps(meta, indent=2))

# Count episodes for a sample task
print("\n--- Counting episodes for reach-v3 in large dataset ---")
all_files_15 = list(api.list_repo_tree(repo15, repo_type="dataset", path="5000/reach-v3", recursive=False))
episodes_15 = [f_obj for f_obj in all_files_15 if hasattr(f_obj, 'rfilename') and 'episode_' in str(f_obj)]
print(f"  Episodes for reach-v3: {len(all_files_15)} entries")
for e in sorted(str(x) for x in all_files_15)[:20]:
    print(f"    {e}")

# Also check file sizes by downloading a step's worth of data
print("\n--- File sizes for one step (reach-v3/env_000/step_0000) ---")
step_files = ["denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz", "metadata.json"]
total_step_size = 0
for sf in step_files:
    path = f"5000/reach-v3/episode_000_env_000/step_0000/{sf}"
    f = hf_hub_download(repo15, path, repo_type="dataset")
    size = os.path.getsize(f)
    total_step_size += size
    print(f"  {sf}: {size/1e6:.2f} MB")
print(f"  TOTAL per step: {total_step_size/1e6:.2f} MB")

print("\nDone!")
