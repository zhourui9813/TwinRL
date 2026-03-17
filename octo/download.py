from huggingface_hub import snapshot_download

snapshot_download(repo_id="rail-berkeley/octo-small", local_dir="./octo-small")