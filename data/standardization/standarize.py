from datasets import load_dataset
import re
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN")
login(token=token)

dataset = load_dataset("julioc-p/Question-Sparql")


def standardize_sparql(query):
    query = query.replace("\n", " ")
    query = re.sub(r"\s+", " ", query).strip()
    return query


dataset["train"] = dataset["train"].map(
    lambda example: {"sparql_query": standardize_sparql(example["sparql_query"])}
)


# print(standardize_sparql(query_example))
print("Original:", dataset["train"][0]["sparql_query"])
print("Standardized:", dataset["train"][0]["sparql_query"])
dataset.push_to_hub(
    "julioc-p/Question-Sparql",
    commit_message="Standardized SPARQL queries",
)
