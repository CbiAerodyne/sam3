import os
import huggingface_hub

print("=== debugging huggingface auth ===")
print(f"huggingface_hub version: {huggingface_hub.__version__}")

# 1. Check where the token is stored
print(f"hf home: {os.getenv('HF_HOME')}")
print(f"hf hub cache: {os.getenv('HUGGINGFACE_HUB_CACHE')}")

# 2. Check if token exists
token = None
try:
    # Try modern way
    from huggingface_hub import get_token
    token = get_token()
except ImportError:
    try:
        # Try older way
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except ImportError:
        # Manual fallback
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
             with open(token_path, "r") as f:
                 token = f.read().strip()

if token:
    print(f"Token found: {token[:4]}...{token[-4:]} (length: {len(token)})")
else:
    print("Token NOT found via get_token() or default path.")

# 3. Try to verify whoami
print("\n=== checking whoami ===")
try:
    # whoami might need the token explicitly if not found automatically?
    # verify that whoami is available
    from huggingface_hub import whoami
    try:
        user_info = whoami(token=token)
        print(f"Logged in as: {user_info['name']}")
        print(f"Org memberships: {[org['name'] for org in user_info.get('orgs', [])]}")
        
    except Exception as e:
        print(f"whoami() failed: {e}")
        
except ImportError:
    print("whoami function not available in this version.")

# 4. Try to access the specific model
print("\n=== checking model access (facebook/sam3) ===")
try:
    from huggingface_hub import hf_hub_download
    # Just try to get info or a small file
    # We will try to get config.json as that is what failed for the user
    path = hf_hub_download(repo_id="facebook/sam3", filename="config.json", token=token)
    print(f"Success! Access verified. File downloaded to: {path}")
except Exception as e:
    print(f"Failed to access model: {e}")
