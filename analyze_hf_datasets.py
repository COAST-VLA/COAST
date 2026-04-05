"""Analyze the ml45-activations and ml45-activations-15 HuggingFace datasets."""
import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from huggingface_hub import HfApi, list_repo_tree, hf_hub_download
import json
import tempfile

api = HfApi()

print("=" * 80)
print("DATASET 1: brandonyang/ml45-activations (small, 2 envs per task)")
print("=" * 80)

# List files in the small dataset
print("\n--- Repository files ---")
files_small = list(api.list_repo_tree("brandonyang/ml45-activations", repo_type="dataset", recursive=False))
for f in files_small[:50]:
    print(f"  {f.rfilename if hasattr(f, 'rfilename') else f}")

# Try to get more info
try:
    info = api.dataset_info("brandonyang/ml45-activations")
    print(f"\nDataset card exists: {info.card_data is not None}")
    print(f"Tags: {info.tags}")
    print(f"Last modified: {info.last_modified}")
    if hasattr(info, 'siblings') and info.siblings:
        total_size = sum(s.size for s in info.siblings if s.size)
        print(f"Total size: {total_size / 1e9:.2f} GB")
        print(f"Number of files: {len(info.siblings)}")
        # Show some file examples
        print("\nSample files:")
        for s in info.siblings[:30]:
            size_str = f"{s.size/1e6:.1f} MB" if s.size else "?"
            print(f"  {s.rfilename} ({size_str})")
        if len(info.siblings) > 30:
            print(f"  ... and {len(info.siblings) - 30} more files")
except Exception as e:
    print(f"Error getting dataset info: {e}")

print("\n" + "=" * 80)
print("DATASET 2: brandonyang/ml45-activations-15 (large, 15 envs per task)")
print("=" * 80)

try:
    info15 = api.dataset_info("brandonyang/ml45-activations-15")
    print(f"\nDataset card exists: {info15.card_data is not None}")
    print(f"Tags: {info15.tags}")
    print(f"Last modified: {info15.last_modified}")
    if hasattr(info15, 'siblings') and info15.siblings:
        total_size = sum(s.size for s in info15.siblings if s.size)
        print(f"Total size: {total_size / 1e9:.2f} GB")
        print(f"Number of files: {len(info15.siblings)}")
        print("\nSample files:")
        for s in info15.siblings[:30]:
            size_str = f"{s.size/1e6:.1f} MB" if s.size else "?"
            print(f"  {s.rfilename} ({size_str})")
        if len(info15.siblings) > 30:
            print(f"  ... and {len(info15.siblings) - 30} more files")
except Exception as e:
    print(f"Error getting dataset info: {e}")

# Now let's download a sample metadata.json from the small dataset to understand the structure
print("\n" + "=" * 80)
print("ANALYZING SAMPLE FILES FROM SMALL DATASET")
print("=" * 80)

# Find some metadata.json and .npz files to download
try:
    all_files_small = list(api.list_repo_tree("brandonyang/ml45-activations", repo_type="dataset", recursive=True))
    
    # Categorize files
    metadata_files = [f for f in all_files_small if hasattr(f, 'rfilename') and f.rfilename.endswith('metadata.json')]
    npz_files = [f for f in all_files_small if hasattr(f, 'rfilename') and f.rfilename.endswith('.npz')]
    
    print(f"\nTotal files: {len(all_files_small)}")
    print(f"Metadata JSON files: {len(metadata_files)}")
    print(f"NPZ files: {len(npz_files)}")
    
    # Show directory structure by extracting unique path prefixes
    task_dirs = set()
    episode_dirs = set()
    step_dirs = set()
    for f in all_files_small:
        if not hasattr(f, 'rfilename'):
            continue
        parts = f.rfilename.split('/')
        if len(parts) >= 2:
            task_dirs.add(parts[0] + '/' + parts[1])
        if len(parts) >= 3:
            episode_dirs.add('/'.join(parts[:3]))
        if len(parts) >= 4:
            step_dirs.add('/'.join(parts[:4]))
    
    print(f"\nUnique task directories: {len(task_dirs)}")
    print("Tasks found:")
    for td in sorted(task_dirs)[:50]:
        print(f"  {td}")
    
    print(f"\nUnique episode directories: {len(episode_dirs)}")
    print("Sample episodes:")
    for ed in sorted(episode_dirs)[:10]:
        print(f"  {ed}")
    
    print(f"\nUnique step directories: {len(step_dirs)}")
    print("Sample steps:")
    for sd in sorted(step_dirs)[:10]:
        print(f"  {sd}")
    
except Exception as e:
    print(f"Error listing files: {e}")
    import traceback
    traceback.print_exc()
