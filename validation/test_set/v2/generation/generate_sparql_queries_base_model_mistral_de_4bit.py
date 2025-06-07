import os

os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"
HF_TOKEN = os.getenv("HF_TOKEN")
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from tqdm.auto import tqdm
import sys


use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False


compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


output_file = "/netscratch/jperez/output_baseline_mistral_txt_en_v2_test_set.json"


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")


print("Loading dataset...")
try:
    dataset = load_dataset(
        "json",
        data_files="/netscratch/jperez/validation_dataset_en_final.json",
        split="test",
    )
    print(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the JSON file exists and is correctly formatted.")
    sys.exit(1)


if dataset is None or len(dataset) == 0:
    print("Dataset is empty or failed to load. Exiting.")
    sys.exit(1)


def extract_sparql(text):
    try:
        response_part = text.split("[/INST]", 1)[1]
        match = re.search(
            r"```sparql\s*(.*?)\s*```", response_part, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        else:
            print(
                f"Warning: Could not find ```sparql ... ``` block in generated text: {response_part[:200]}..."
            )
            return ""
    except IndexError:
        print(f"Warning: '[/INST]' marker not found in generated text: {text[:200]}...")
        return ""
    except Exception as e:
        print(f"Error during SPARQL extraction: {e}\nText was: {text[:200]}...")
        return ""


output_data = []
start_index = 0
if os.path.exists(output_file):
    print(
        f"Checkpoint file found at '{output_file}'. Attempting to load previous results..."
    )
    try:
        with open(output_file, "r") as f:
            content = f.read()
            if content:
                output_data = json.loads(content)
                if isinstance(output_data, list):
                    start_index = len(output_data)
                    print(
                        f"Successfully loaded {start_index} previous results. Resuming from index {start_index}."
                    )
                else:
                    print(
                        f"Warning: Checkpoint file '{output_file}' did not contain a list. Starting from scratch."
                    )
                    output_data = []
                    start_index = 0
            else:
                print("Checkpoint file is empty. Starting from scratch.")
                output_data = []
                start_index = 0
    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON from checkpoint file '{output_file}': {e}. Starting from scratch."
        )
        output_data = []
        start_index = 0
    except (IOError, OSError) as e:
        print(
            f"Error reading checkpoint file '{output_file}': {e}. Starting from scratch."
        )
        output_data = []
        start_index = 0
else:
    print(f"No checkpoint file found at '{output_file}'. Starting from scratch.")


batch_size = 128
max_new_tokens_generate = 512
total_items = len(dataset)


if start_index >= total_items and total_items > 0:
    print("Dataset already fully processed according to checkpoint file.")
else:
    remaining_items = total_items - start_index
    num_batches_to_process = (remaining_items + batch_size - 1) // batch_size

    print(
        f"Starting batch processing from item index {start_index} ({num_batches_to_process} batches remaining)..."
    )

    batch_start_indices = range(start_index, total_items, batch_size)

    for i in tqdm(
        batch_start_indices, total=num_batches_to_process, desc="Processing Batches"
    ):
        batch_end_index = min(i + batch_size, total_items)
        current_batch_indices = list(range(i, batch_end_index))
        if not current_batch_indices:
            continue
        batch = dataset.select(current_batch_indices)

        batch_prompts = []
        batch_original_data = []
        for example_index_in_batch, original_example in enumerate(batch):
            original_example = original_example["messages"]
            try:
                if (
                    len(original_example) < 3
                    or not isinstance(original_example[0], dict)
                    or "content" not in original_example[0]
                    or not isinstance(original_example[1], dict)
                    or "content" not in original_example[1]
                    or not isinstance(original_example[2], dict)
                    or "content" not in original_example[2]
                ):
                    print(
                        f"Warning: Skipping invalid data format at index {i + example_index_in_batch}: {original_example}"
                    )
                    continue

                context = original_example[0]["content"].split("CONTEXT", 1)[1].strip()
                question = original_example[1]["content"]
                gold_sparql = (
                    original_example[2]["content"]
                    .split("```")[1]
                    .split("sparql")[1]
                    .strip()
                )
                knowledge_graph = "Wikidata"
                prompt_messages = [
                    {
                        "role": "user",
                        "content": f"Write a SparQL query that answers this request: '{question}' from the knowledge graph {knowledge_graph} with the following context: '{context}'. Output only the SPARQL query, enclosed in ```sparql ... ```.",
                    }
                ]
                prompt_string = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                batch_prompts.append(prompt_string)
                batch_original_data.append(
                    {
                        "question_prompt": prompt_messages[0]["content"],
                        "knowledge_graph": knowledge_graph,
                        "gold_sparql": gold_sparql,
                    }
                )
            except Exception as e:
                print(
                    f"Error processing example at index {i + example_index_in_batch}: {e}. Skipping."
                )
                print(f"Problematic example data: {original_example}")
                continue

        if not batch_prompts:
            print(
                f"Warning: All examples in batch starting at index {i} were skipped due to errors."
            )
            try:
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=4)
            except IOError as e:
                print(
                    f"\nError saving checkpoint after skipped batch: {e}. Continuing..."
                )
            continue

        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings
                - max_new_tokens_generate
                - 10,
            ).to(device)
        except Exception as e:
            print(
                f"\nError during tokenization for batch starting at index {i}: {e}. Skipping batch."
            )
            continue

        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens_generate,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except Exception as e:
            print(
                f"\nError during model generation for batch starting at index {i}: {e}. Skipping batch."
            )
            continue

        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if len(generated_texts) != len(batch_original_data):
            print(
                f"Warning: Mismatch between generated texts ({len(generated_texts)}) and original data ({len(batch_original_data)}) for batch starting at {i}. Skipping results for this batch."
            )
        else:
            for j, generated_text in enumerate(generated_texts):
                original_data = batch_original_data[j]
                cleaned_sparql = extract_sparql(generated_text)

                output_data.append(
                    {
                        "question": original_data["question_prompt"],
                        "knowledge_graph": original_data["knowledge_graph"],
                        "gold_sparql": original_data["gold_sparql"],
                        "generated_sparql": cleaned_sparql,
                        "full_generated_text": generated_text,
                    }
                )

        try:
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
        except IOError as e:
            print(f"\nError saving checkpoint: {e}. Continuing...")
        except Exception as e:
            print(f"\nUnexpected error saving checkpoint: {e}. Continuing...")


print(
    f"\nProcessing finished. Saving final {len(output_data)} results to {output_file}..."
)
try:
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print("Final results saved successfully.")
except IOError as e:
    print(f"Error saving final results: {e}.")
except Exception as e:
    print(f"Unexpected error saving final results: {e}.")


print("Processing complete.")
