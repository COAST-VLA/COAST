"""Download and inspect sample NPZ files + large dataset structure."""
import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from huggingface_hub import hf_hub_download, HfApi
import json
import numpy as np

api = HfApi()
repo = "brandonyang/ml45-activations"
repo15 = "brandonyang/ml45-activations-15"

# ============================================================
# NPZ contents from SMALL dataset
# ============================================================
print("=" * 80)
print("NPZ FILE CONTENTS: SMALL DATASET (assembly-v3)")
print("=" * 80)

base = "5000/assembly-v3/episode_000_env_000/step_0000"

print("\n--- denoising.npz ---")
f = hf_hub_download(repo, f"{base}/denoising.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- adarms_cond.npz ---")
f = hf_hub_download(repo, f"{base}/adarms_cond.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- suffix_residual.npz ---")
f = hf_hub_download(repo, f"{base}/suffix_residual.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- suffix_mlp_hidden.npz ---")
f = hf_hub_download(repo, f"{base}/suffix_mlp_hidden.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    print(f"    min={data[key].min():.4f}, max={data[key].max():.4f}, mean={data[key].mean():.4f}")

print("\n--- rewards.npz ---")
f = hf_hub_download(repo, "5000/assembly-v3/episode_000_env_000/rewards.npz", repo_type="dataset")
data = np.load(f)
for key in data.files:
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.dtype == bool:
        print(f"    True count: {arr.sum()}, first True at step: {np.argmax(arr) if arr.any() else 'never'}")
    else:
        print(f"    min={arr.min():.4f}, max={arr.max():.4f}, final={arr[-1]:.4f}")

# ============================================================
# File sizes for one step
# ============================================================
print("\n" + "=" * 80)
print("FILE SIZES PER INFERENCE STEP")
print("=" * 80)

step_files = ["denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz", "metadata.json"]
total = 0
for sf in step_files:
    path = f"5000/reach-v3/episode_000_env_000/step_0000/{sf}"
    f = hf_hub_download(repo, path, repo_type="dataset")
    size = os.path.getsize(f)
    total += size
    print(f"  {sf}: {size/1e6:.2f} MB")
print(f"  TOTAL per step: {total/1e6:.2f} MB (~{total/1e6:.0f} MB)")

# ============================================================
# LARGE DATASET: count episodes per task
# ============================================================
print("\n" + "=" * 80)
print("LARGE DATASET (15 envs): STRUCTURE")
print("=" * 80)

# Get all files for reach-v3 from large dataset
all_files_15 = list(api.list_repo_tree(repo15, repo_type="dataset", recursive=True))

# Count task/episode/step structure
task_episodes_15 = {}
task_steps_15 = {}

for f_obj in all_files_15:
    if not hasattr(f_obj, 'rfilename'):
        continue
    parts = f_obj.rfilename.split('/')
    if len(parts) >= 3 and parts[0] == '5000':
        task = parts[1]
        episode = parts[2]
        if task not in task_episodes_15:
            task_episodes_15[task] = set()
        task_episodes_15[task].add(episode)
        
        if len(parts) >= 4 and parts[3].startswith('step_'):
            key = (task, episode)
            if key not in task_steps_15:
                task_steps_15[key] = set()
            task_steps_15[key].add(parts[3])

print(f"\nTasks: {len(task_episodes_15)}")
print(f"Total files: {len(all_files_15)}")

# Summary per task
for task in sorted(task_episodes_15.keys()):
    eps = sorted(task_episodes_15[task])
    # Count steps for first episode
    first_ep_key = (task, eps[0])
    n_steps = len(task_steps_15.get(first_ep_key, set()))
    print(f"  {task}: {len(eps)} episodes, ~{n_steps} inference steps (env_000)")

# Episode success comparison
print("\n" + "=" * 80)
print("LARGE DATASET: SAMPLE EPISODE METADATA (15 envs, multiple tasks)")
print("=" * 80)

sample_tasks = ["reach-v3", "assembly-v3", "button-press-v3", "pick-place-v3"]
for task in sample_tasks:
    print(f"\n--- {task} ---")
    for env_id in [0, 7, 14]:
        try:
            path = f"5000/{task}/episode_000_env_{env_id:03d}/metadata.json"
            f = hf_hub_download(repo15, path, repo_type="dataset")
            with open(f) as fh:
                meta = json.load(fh)
            print(f"  env_{env_id:03d}: success={meta['episode_success']}, reward={meta['total_reward']:.1f}, "
                  f"steps_to_success={meta['steps_to_success']}, total_steps={meta['total_env_steps']}, "
                  f"inference_steps={meta['total_inference_steps']}")
        except Exception as e:
            print(f"  env_{env_id:03d}: {e}")

print("\nDone!")
