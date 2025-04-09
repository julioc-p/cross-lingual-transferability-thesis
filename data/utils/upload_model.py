from huggingface_hub import HfApi, upload_folder
import os
import argparse


def upload_model_to_hf(local_path, repo_id, commit_message, token):
    """
    Uploads a local model to Hugging Face Hub, replacing the existing content.

    Args:
        local_path (str): Path to the local model directory
        repo_id (str): Hugging Face repository ID (format: username/repo)
        commit_message (str): Commit message for the upload
        token (str): Hugging Face authentication token
    """
    # Initialize HF API
    api = HfApi(token=token)

    # Create repo if it doesn't exist (will do nothing if exists)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # Upload the entire model directory
    upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )

    print(f"Successfully uploaded model to {repo_id}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Upload a local model to Hugging Face Hub"
    )
    parser.add_argument(
        "--path", required=True, help="Path to the local model directory"
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Hugging Face repository ID (format: username/repo)",
    )
    parser.add_argument(
        "--commit_message",
        default="Uploading fine-tuned model",
        help="Commit message for the upload",
    )

    args = parser.parse_args()

    # Get token from environment variable
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable not found. "
            "Please set your Hugging Face token with: "
            "export HF_TOKEN='your_token_here'"
        )

    # Verify the local model exists
    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Local model path not found: {args.path}")

    # Upload the model
    upload_model_to_hf(
        local_path=args.path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        token=token,
    )


if __name__ == "__main__":
    main()
