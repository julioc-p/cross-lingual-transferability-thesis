from datasets import load_dataset


def load_data(train_file, eval_file):
    """Loads the training and evaluation datasets."""
    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=eval_file, split="train")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    return train_dataset, eval_dataset
