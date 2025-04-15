from datasets import load_dataset
import argparse


def create_conversation(sample, system_message):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(context=sample["context"]),
            },
            {"role": "user", "content": sample["text_query"]},
            {"role": "assistant", "content": sample["sparql_query"]},
        ]
    }


def process_dataset(language, base_name, output_dir="/netscratch/jperez"):
    system_message = """You are an expert text to SparQL query translator. Users will ask you questions in English and you will generate a SparQL query based on the provided context.
CONTEXT:
{context}"""

    # Load dataset from the hub
    dataset = load_dataset("julioc-p/Question-Sparql", split="train")
    dataset = dataset.filter(
        lambda x: x["language"] == language
        and x["sparql_query"].lower() not in ["out of scope", "none"]
        and "Wikidata" in x["knowledge_graphs"]
        and x["context"] != "None"
    )

    dataset = dataset.shuffle(seed=42).select(range(100000))

    # Convert dataset to OAI messages
    dataset = dataset.map(
        lambda x: create_conversation(x, system_message),
        remove_columns=dataset.features,
        batched=False,
    )

    # First split: separate out the training set (80%) and temporary set (20%)
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)

    # Second split: split the temporary set into test and validation (50% each of the 20%)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

    # Now combine everything into a single DatasetDict
    dataset = {
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }

    print(dataset["train"][345]["messages"])

    # Save files with the base name
    dataset["train"].to_json(
        f"{output_dir}/train_dataset_{base_name}.json", orient="records"
    )
    dataset["test"].to_json(
        f"{output_dir}/test_dataset_{base_name}.json", orient="records"
    )
    dataset["validation"].to_json(
        f"{output_dir}/validation_dataset_{base_name}.json", orient="records"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", type=str, required=True, help="Language to filter (e.g., 'en')"
    )
    parser.add_argument(
        "--base-name",
        type=str,
        required=True,
        help="Base name for output files (e.g., 'en_final')",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/netscratch/jperez", help="Output directory"
    )

    args = parser.parse_args()

    process_dataset(args.language, args.base_name, args.output_dir)
