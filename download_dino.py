from huggingface_hub import snapshot_download
import os

print("Sedang mendownload model DinoV3... Harap tunggu, ini sekitar 1.2GB")
snapshot_download(
    repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
    local_files_only=False,
)
print("Download Selesai!")
