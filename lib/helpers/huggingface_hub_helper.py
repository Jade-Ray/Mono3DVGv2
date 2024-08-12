import os
from argparse import Namespace
from pathlib import Path

from huggingface_hub import HfApi


def create_huggingface_hub_repo(cfg: Namespace):
    """
    Handle the repository creation.
    """
    # Retrieve of infer repo_name
    repo_name = cfg.hub_model_id
    if repo_name is None:
        repo_name = Path(cfg.output_dir).absolute().name
    # Create repo and retrieve repo_id
    api = HfApi()
    repo_id = api.create_repo(repo_name, exist_ok=True, token=cfg.hub_token).repo_id

    with open(os.path.join(cfg.output_dir, ".gitignore"), "w+") as gitignore:
        if "step_*" not in gitignore:
            gitignore.write("step_*\n")
        if "epoch_*" not in gitignore:
            gitignore.write("epoch_*\n")
    
    return api, repo_id, cfg.hub_token


def upload_output_folder(
    api: HfApi, 
    repo_id: str, 
    hub_token: str, 
    commit_message: str = '',
    output_dir: str = 'output',
    repo_type: str = 'model',
    **kwargs,
):
    """
    Upload the output folder to the repository.
    """
    api.upload_folder(
        commit_message=commit_message,
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        token=hub_token,
        **kwargs,
    )

