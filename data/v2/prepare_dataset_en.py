from datasets import load_dataset

system_message = """You are an expert text to SparQL query translator. Users will ask you questions in English and you will generate a SparQL query based on the provided context.
CONTEXT:
{context}"""


def create_conversation(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(schema=sample["context"]),
            },
            {"role": "user", "content": sample["text_query"]},
            {"role": "assistant", "content": sample["sparql_query"]},
        ]
    }


# Load dataset from the hub
dataset = load_dataset("julioc-p/Question-Sparql", split="train")
dataset = dataset.filter(
    lambda x: x["language"] == "en"
    and x["sparql_query"].lower() not in ["out of scope", "none"]
    and "Wikidata" in x["knowledge_graphs"]
)

dataset = dataset.shuffle(seed=42).select(range(100000))

# Convert dataset to OAI messages
dataset = dataset.map(
    create_conversation, remove_columns=dataset.features, batched=False
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

dataset["train"].to_json(
    "/netscratch/jperez/train_dataset_en_final.json", orient="records"
)
dataset["test"].to_json(
    "/netscratch/jperez/test_dataset_en_final.json", orient="records"
)
dataset["validation"].to_json(
    "/netscratch/jperez/validation_dataset_en_final.json", orient="records"
)
