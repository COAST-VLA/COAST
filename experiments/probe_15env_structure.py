"""Quick probe: check step counts for 4 tasks in the 15-env dataset."""
import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from huggingface_hub import hf_hub_download, HfApi
import json

api = HfApi()
REPO = "brandonyang/ml45-activations-15"
TASKS = ["reach-v3", "button-press-v3", "drawer-open-v3", "assembly-v3"]

# Check episode metadata for env counts and step counts
for task in TASKS:
    print(f"\n=== {task} ===")
    for env_id in [0, 7, 14]:
        path = f"5000/{task}/episode_000_env_{env_id:03d}/metadata.json"
        try:
            f = hf_hub_download(REPO, path, repo_type="dataset")
            with open(f) as fh:
                meta = json.load(fh)
            print(f"  env_{env_id:03d}: success={meta['episode_success']}, "
                  f"inference_steps={meta['total_inference_steps']}, "
                  f"env_steps={meta['total_env_steps']}, "
                  f"reward={meta['total_reward']:.1f}")
        except Exception as e:
            print(f"  env_{env_id:03d}: NOT FOUND - {e}")

    # Try to find the max step for env_000
    print(f"  Checking step range for env_000...")
    max_step_found = -1
    for step in range(0, 300, 10):
        path = f"5000/{task}/episode_000_env_000/step_{step:04d}/metadata.json"
        try:
            f = hf_hub_download(REPO, path, repo_type="dataset")
            max_step_found = step
        except Exception:
            break
    print(f"  → max step for env_000: step_{max_step_found:04d} ({max_step_found // 10 + 1} inference steps)")

print("\nDone!")
