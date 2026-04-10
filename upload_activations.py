from huggingface_hub import HfApi

api = HfApi()

api.create_repo("ksb21st/robocasa-activations-75000", repo_type="dataset", private=False, exist_ok=True)
print("Repo created/exists")

print("Starting upload of activations/75000 ...")
api.upload_large_folder(
    folder_path="/home/kim34/projects/openpi-metaworld/activations/75000",
    repo_id="ksb21st/robocasa-activations-75000",
    repo_type="dataset",
)
print("Upload complete!")
