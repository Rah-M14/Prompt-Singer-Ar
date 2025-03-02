from huggingface_hub import snapshot_download

repo_id = "Cyanbox/Prompt-Singer"
custom_download_dir = "Hugging"

downloaded_path = snapshot_download(
    repo_id=repo_id,
    local_dir=custom_download_dir
)

print(f"Repository downloaded to: {downloaded_path}")