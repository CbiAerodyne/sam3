from huggingface_hub import snapshot_download
import os

# 1. Define the repository and the local directory
repo_id = "facebook/sam3"
local_dir = "checkpoints"

# 2. Create the checkpoints folder if it doesn't exist
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
    print(f"Created directory: {local_dir}")

print(f"Starting download from {repo_id}...")

try:
    # 3. Download the repository
    # This will use your stored token automatically.
    # It will only download the files into the 'checkpoints' folder.
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Ensures actual files are copied to your workspace
    )

    print(f"\nSuccess! Checkpoints are located in: {os.path.abspath(local_dir)}")
    print("Files downloaded:")
    for file in os.listdir(local_dir):
        print(f" - {file}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("\nIf you get a 401 or 403 error, make sure you have:")
    print("1. Accepted the license on https://huggingface.co/facebook/sam3")
    print("2. Logged in using 'python3 -m huggingface_hub login'")