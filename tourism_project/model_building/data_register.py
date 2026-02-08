from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "sagarsb/tourism-prediction"
repo_type = "dataset"

# Setting and retrirving token from local OS env is not working for me even
# when I exported then in zshrc, falling back to local file based tokens
# Note: in the GitHub pipeline, it will always use HF_TOKEN from os env
token = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
