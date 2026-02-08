from huggingface_hub import HfApi
import os

# Setting and retrirving token from local OS env is not working for me even
# when I exported then in zshrc, falling back to local file based tokens
# Note: in the GitHub pipeline, it will always use HF_TOKEN from os env
token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="sagarsb/tourism-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                     # optional: subfolder path inside the repo
)
