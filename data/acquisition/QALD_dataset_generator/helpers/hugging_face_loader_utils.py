import os
from huggingface_hub import HfApi, HfFolder, Repository, login
import pandas as pd
from datasets import Dataset, load_dataset

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
HF_TOKEN = os.getenv("HF_TOKEN", "Replace your token")


def load_to_hugging_face(file_path, repo):
    api = HfApi()
    login(token=HF_TOKEN)
    df = load_dataset(file_path)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo)


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    repo_id = "julioc-p/Question-Sparql"
    file = "qald_challenges.csv"

    load_to_hugging_face(file, repo_id)
