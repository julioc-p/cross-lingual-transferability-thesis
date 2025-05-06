import re
import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, list_repo_files

SOURCE_REPO_ID = "julioc-p/Question-Sparql"
TARGET_REPO_ID = "julioc-p/Question-Sparql"
SPARQL_COLUMN = "sparql_query"
SPLIT_TO_PROCESS = "validation"

if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN environment variable not set.")
    print("Attempting to proceed, but pushing to hub might fail if not logged in.")


def remove_sparql_prefixes_robust(query_str):
    """
    Removes all leading PREFIX declarations from a SPARQL query string.
    It looks for the main query keywords (SELECT, ASK, etc.) and
    takes the query from that point onwards.
    """
    if not isinstance(query_str, str):
        return query_str

    match = re.search(
        r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", query_str, flags=re.IGNORECASE
    )

    if match:
        main_query_part = query_str[match.start() :]
        return main_query_part.strip()
    else:
        cleaned_query = re.sub(
            r"(PREFIX|prefix)\s+\S+\s*:\s*<[^>]*>\s*",
            "",
            query_str,
            flags=re.IGNORECASE,
        )
        return " ".join(cleaned_query.split()).strip()


def process_batch_sparql(batch):
    """
    Applies robust SPARQL prefix removal to a batch of queries.
    """
    processed_queries = []
    queries_in_batch = batch[SPARQL_COLUMN]

    if not hasattr(process_batch_sparql, "has_debugged_first_query"):
        process_batch_sparql.has_debugged_first_query = False

    for i, query_str in enumerate(queries_in_batch):
        if not process_batch_sparql.has_debugged_first_query and i == 0:
            print(f"\n--- DEBUG (First query in a batch) ---")
            print(f"Original query_str (repr):\n{repr(query_str)}")
            cleaned_debug = remove_sparql_prefixes_robust(query_str)
            print(f"Cleaned query_str by robust func (repr):\n{repr(cleaned_debug)}")
            print(f"--- END DEBUG ---")
            if os.getenv("CI"):
                process_batch_sparql.has_debugged_first_query = True
            elif i == 0:
                process_batch_sparql.has_debugged_first_query = True

        processed_queries.append(remove_sparql_prefixes_robust(query_str))

    batch[SPARQL_COLUMN] = processed_queries
    return batch


print(f"Loading dataset: {SOURCE_REPO_ID}...")
ds = load_dataset(SOURCE_REPO_ID)
print("Original dataset loaded:")
print(ds)

final_ds = DatasetDict()

print("\nProcessing dataset splits...")
if SPLIT_TO_PROCESS not in ds:
    raise ValueError(
        f"Split '{SPLIT_TO_PROCESS}' not found in the dataset {SOURCE_REPO_ID}. Available splits: {list(ds.keys())}"
    )

for split_name, split_dataset in ds.items():
    if split_name == SPLIT_TO_PROCESS:
        print(f"-> Processing split: '{split_name}' to remove SPARQL prefixes...")
        final_ds[split_name] = split_dataset.map(
            process_batch_sparql,
            batched=True,
            num_proc=4,
        )
    else:
        print(f"-> Keeping original split: '{split_name}'")
        final_ds[split_name] = split_dataset

print("Processing complete.")

print("\n--- Verification ---")

if "train" in ds and "train" in final_ds:
    print("\nTrain split (should be unchanged):")
    original_train_example = (
        ds["train"][0][SPARQL_COLUMN] if len(ds["train"]) > 0 else "Train split empty"
    )
    final_train_example = (
        final_ds["train"][0][SPARQL_COLUMN]
        if len(final_ds["train"]) > 0
        else "Train split empty"
    )
    print(f"Original Query Example:\n{original_train_example}")
    print(f"Final Query Example:\n{final_train_example}")
    if original_train_example != final_train_example:
        print("⚠️ WARNING: Train split appears to have changed unexpectedly!")

if SPLIT_TO_PROCESS in ds and SPLIT_TO_PROCESS in final_ds:
    print(f"\n{SPLIT_TO_PROCESS.capitalize()} split (should be processed):")
    if len(ds[SPLIT_TO_PROCESS]) > 0 and len(final_ds[SPLIT_TO_PROCESS]) > 0:
        original_valid_example = ds[SPLIT_TO_PROCESS][0][SPARQL_COLUMN]
        final_valid_example = final_ds[SPLIT_TO_PROCESS][0][SPARQL_COLUMN]
        print(f"Original Query Example:\n{original_valid_example}")
        print(f"Final Query Example (should have no prefixes):\n{final_valid_example}")

        if isinstance(final_valid_example, str) and re.search(
            r"\b(PREFIX)\b", final_valid_example, flags=re.IGNORECASE
        ):
            print(
                f"⚠️ WARNING: Prefixes still found in processed '{SPLIT_TO_PROCESS}' split example!"
            )
            print(f"   Problematic final query: {repr(final_valid_example)}")
        elif (
            original_valid_example == final_valid_example
            and isinstance(original_valid_example, str)
            and re.search(r"\b(PREFIX)\b", original_valid_example, flags=re.IGNORECASE)
        ):
            print(
                f"⚠️ WARNING: {SPLIT_TO_PROCESS} split does not appear to have changed, and original had prefixes."
            )
        else:
            print(
                f"✅ {SPLIT_TO_PROCESS} split appears to be processed correctly (no prefixes found in example or was already clean)."
            )
    else:
        print(f"{SPLIT_TO_PROCESS} split is empty, cannot verify specific example.")


print(f"\nPushing modified dataset dictionary to Hugging Face Hub: {TARGET_REPO_ID}...")
print(
    "🚨 IMPORTANT WARNING: This will overwrite the existing data on the Hub if TARGET_REPO_ID is the same as the source."
)
user_confirmation = input(f"Proceed with pushing to '{TARGET_REPO_ID}'? (yes/no): ")

if user_confirmation.lower() == "yes":
    try:
        final_ds.push_to_hub(TARGET_REPO_ID)
        print("Dataset successfully pushed to Hub.")
        print(f"\nVerifying files in repo {TARGET_REPO_ID}:")
        api = HfApi()
        files = list_repo_files(TARGET_REPO_ID)
        print(files)
    except Exception as e:
        print(f"\nAn error occurred during push_to_hub: {e}")
        print("Please check your HF_TOKEN, permissions, and internet connection.")
else:
    print("Push operation cancelled by user.")

print("\nScript finished.")
