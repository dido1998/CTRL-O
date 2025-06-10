"A script to upload a model to the Hugging Face Hub"

import os
import shutil
import argparse
from huggingface_hub import HfApi, HfFolder, upload_file

def transform_ckpt_path_to_config(ckpt_path):
    # Split the path into directory and filename
    dir_path, filename = os.path.split(ckpt_path)
    
    # Go up one directory level to reach the parent folder
    parent_dir = os.path.dirname(dir_path)
    
    # Replace 'checkpoints' with 'config' in the path
    config_dir = dir_path.replace('checkpoints', 'config')
    
    # Change the filename from .ckpt to .yaml
    config_filename = 'config.yaml'
    
    # Join the new path
    config_path = os.path.join(config_dir, config_filename)
    
    return config_path

def main(model_to_upload, repo_name, access_token, local_dir, config_to_upload):
    # Define the local directory path where the model files will be stored
    if config_to_upload == "auto":
        print("Transforming ckpt path to config path")
        config_to_upload = transform_ckpt_path_to_config(model_to_upload)
    # Create the directory if it does not exist
    repo_dir = os.path.join(local_dir, repo_name)
    os.makedirs(os.path.join(local_dir, repo_name), exist_ok=True)
    model_path = os.path.join(repo_dir, "model.ckpt")
    config_path = os.path.join(repo_dir, "config.yaml")
    # Copy the model file to the local directory
    shutil.copy(model_to_upload, model_path)
    shutil.copy(config_to_upload, config_path)
    # Authenticate and configure the Hugging Face API
    
    HfFolder.save_token(access_token)
    api = HfApi()

    # Create a new repository
    api.create_repo(
        repo_id=repo_name,
        token=access_token,
        private=False,      
        exist_ok=True      
    )
    for file_path, file_name in [(model_path, "model.ckpt"), (config_path, "config.yaml")]:
        print(f"Uploading {file_name}...")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_name,
            token=access_token,
        )
        print(f"Uploaded {file_name} to {repo_name}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file to upload")
    parser.add_argument("--config", default="auto", type=str, help="Path to the config file to upload")
    parser.add_argument("--repo", type=str, required=True, help="Name of the repository to create")
    parser.add_argument("--local-dir", default="./outputs/checkpoints", type=str, help="Dir where the model could be copied to.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.model, args.repo, os.getenv("HUGGINGFACE_HUB_TOKEN"), local_dir=args.local_dir, config_to_upload=args.config)