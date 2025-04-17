import json
import requests
import logging
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import (
    HfApi,
    HfFolder,
    notebook_login,
)

YOUR_HF_USERNAME = "julioc-p"
YOUR_DATASET_NAME = "Question-Sparql"
HF_DATASET_ID = f"{YOUR_HF_USERNAME}/{YOUR_DATASET_NAME}"
QALD10_JSON_URL = (
    "https://raw.githubusercontent.com/KGQA/QALD-10/main/data/qald_10/qald_10.json"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def fetch_qald_data(url: str) -> dict | None:
    """Fetches and parses the QALD JSON data from the given URL."""
    logging.info(f"Fetching QALD-10 data from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logging.info("Successfully fetched QALD-10 data.")
        qald_data = response.json()
        return qald_data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching QALD data: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing QALD JSON data: {e}")
        return None


def create_qald_lookup(qald_data: dict) -> dict:
    """Creates a lookup dict mapping SPARQL queries to German questions."""
    lookup = {}
    if not qald_data or "questions" not in qald_data:
        logging.warning("QALD data is missing or invalid. Cannot create lookup.")
        return lookup

    logging.info("Creating SPARQL -> German question lookup table...")
    for question_block in qald_data.get("questions", []):
        sparql_query = question_block.get("query", {}).get("sparql")
        german_question = None
        if sparql_query:
            normalized_sparql = " ".join(sparql_query.split())

            for q_variant in question_block.get("question", []):
                if q_variant.get("language") == "de":
                    german_question = q_variant.get("string")
                    break

            if german_question:
                if normalized_sparql in lookup:
                    pass
                lookup[normalized_sparql] = german_question

    logging.info(f"Created lookup table with {len(lookup)} SPARQL-to-German mappings.")
    return lookup


def add_german_to_split(
    original_split: Dataset, qald_lookup: dict, split_name: str
) -> Dataset:
    """Adds German question rows to a single dataset split."""
    logging.info(f"Processing split: '{split_name}'...")
    new_german_rows = []
    missing_count = 0
    found_count = 0

    required_cols = {"sparql_query", "knowledge_graphs", "text_query", "language"}
    if not required_cols.issubset(original_split.column_names):
        logging.error(
            f"Split '{split_name}' is missing required columns. Expected: {required_cols}, Found: {original_split.column_names}"
        )
        return original_split

    for row in original_split:
        if row.get("language") == "de":
            continue

        sparql_from_ds = row.get("sparql_query", "")
        if not sparql_from_ds:
            continue

        normalized_sparql_ds = " ".join(sparql_from_ds.split())
        german_question = qald_lookup.get(normalized_sparql_ds)

        if german_question:
            found_count += 1
            new_row = {
                "text_query": german_question,
                "language": "de",
                "sparql_query": row["sparql_query"],
                "knowledge_graphs": row["knowledge_graphs"],
            }
            new_german_rows.append(new_row)
        else:
            missing_count += 1

    logging.info(
        f"Split '{split_name}': Found {found_count} German matches, {missing_count} missing in QALD lookup."
    )

    if not new_german_rows:
        logging.info(
            f"No new German rows generated for split '{split_name}'. Returning original."
        )
        return original_split

    try:
        german_ds = Dataset.from_list(new_german_rows, features=original_split.features)

        combined_split = concatenate_datasets([original_split, german_ds])
        logging.info(
            f"Successfully added {len(german_ds)} German rows to split '{split_name}'. New size: {len(combined_split)}"
        )
        return combined_split
    except Exception as e:
        logging.error(
            f"Error creating/concatenating Dataset for split '{split_name}': {e}"
        )
        logging.error("Returning original split due to error.")
        return original_split


if __name__ == "__main__":
    logging.info("--- Starting Dataset Augmentation Script ---")

    qald_data = fetch_qald_data(QALD10_JSON_URL)
    if not qald_data:
        logging.error("Failed to fetch or parse QALD data. Exiting.")
        exit(1)

    qald_lookup = create_qald_lookup(qald_data)
    if not qald_lookup:
        logging.warning(
            "QALD lookup table is empty. Augmentation might not add any data."
        )

    logging.info(f"Loading original dataset: {HF_DATASET_ID}")
    try:
        original_ds = load_dataset(HF_DATASET_ID)
        logging.info("Original dataset loaded successfully.")
        logging.info(f"Original dataset structure:\n{original_ds}")
    except Exception as e:
        logging.error(f"Error loading dataset '{HF_DATASET_ID}': {e}")
        logging.error(
            "Please ensure the dataset exists, you have network access, and necessary permissions."
        )
        exit(1)

    updated_splits = {}
    error_in_processing = False
    for split_name, split_data in original_ds.items():
        updated_split = add_german_to_split(split_data, qald_lookup, split_name)
        if updated_split is split_data and len(qald_lookup) > 0:
            pass

        updated_splits[split_name] = updated_split

    updated_ds = DatasetDict(updated_splits)
    logging.info("Updated dataset structure (with German rows added):")
    logging.info(f"\n{updated_ds}")

    logging.warning(f"--- PREPARING TO PUSH TO HUB: {HF_DATASET_ID} ---")
    logging.warning(
        "!!! This will OVERWRITE the existing dataset on the Hugging Face Hub !!!"
    )
    user_confirmation = (
        input("Do you want to proceed with pushing the updated dataset? (yes/no): ")
        .strip()
        .lower()
    )

    if user_confirmation == "yes":
        logging.info(f"Attempting to push updated dataset to: {HF_DATASET_ID}")
        try:
            updated_ds.push_to_hub(
                repo_id=HF_DATASET_ID,
            )
            logging.info(
                f"Successfully pushed updated dataset to Hugging Face Hub: {HF_DATASET_ID}"
            )
        except Exception as e:
            logging.error(f"Error pushing dataset to Hub: {e}")
            logging.error(
                "Please ensure you are logged in (`huggingface-cli login`) and have write access to the repository."
            )
            logging.error("The updated dataset was NOT uploaded.")
            exit(1)
    else:
        logging.info("Push operation cancelled by user.")
        logging.info("You can save the dataset locally using:")
        logging.info("# updated_ds.save_to_disk('./path_to_save_updated_dataset')")

    logging.info("--- Dataset Augmentation Script Finished ---")
