#!/usr/bin/env python
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
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")
print("Loading and filtering dataset...")
dataset_full = load_dataset("julioc-p/Question-Sparql", split="train")
ds_en = dataset_full.filter(
    lambda x: x["language"] == "en"
    and x["sparql_query"] is not None
    and x["sparql_query"].lower().strip() not in ["out of scope", "none", ""]
    and isinstance(x["knowledge_graphs"], str)
    and "Wikidata" in x["knowledge_graphs"]
)
print(f"Filtered dataset size: {len(ds_en)}")
start_index = 35000
end_index = 38500
if start_index >= len(ds_en):
    print(
        f"Warning: Start index {start_index} is out of bounds for filtered dataset size {len(ds_en)}. Adjusting."
    )
    start_index = 0
if end_index > len(ds_en):
    print(
        f"Warning: End index {end_index} is out of bounds for filtered dataset size {len(ds_en)}. Adjusting."
    )
    end_index = len(ds_en)
if start_index < end_index:
    dataset = ds_en.shuffle(seed=42).select(range(start_index, end_index))
    print(f"Selected {len(dataset)} examples for processing.")
else:
    print("Error: Invalid start/end index range after filtering. No data selected.")
    dataset = None


def extract_sparql(text):
    try:
        return text.split("[/INST]")[1].split("```")[1].split("sparql")[1].strip()
    except:
        return ""


output_data = []
batch_size = 128
max_new_tokens_generate = 512
print(f"Starting batch processing with batch size {batch_size}...")
if dataset is None or len(dataset) == 0:
    print("No data to process. Exiting.")
else:
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch = dataset.select(batch_indices)
        batch_prompts = []
        for example in batch:
            question = example["text_query"]
            knowledge_graph = example["knowledge_graphs"]
            prompt_messages = [
                {
                    "role": "user",
                    "content": f"Write a SparQL query that answers this request: '{question}' from the knowledge graph {knowledge_graph}. Output only the SPARQL query, enclosed in ```sparql ... ```.",
                }
            ]
            prompt_string = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(prompt_string)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings
            - max_new_tokens_generate
            - 10,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens_generate,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for j, generated_text in enumerate(generated_texts):
            original_example = batch[j]
            cleaned_sparql = extract_sparql(generated_text)
            output_data.append(
                {
                    "question": original_example["text_query"],
                    "knowledge_graph": original_example["knowledge_graphs"],
                    "gold_sparql": original_example["sparql_query"],
                    "generated_sparql": cleaned_sparql,
                    "full_generated_text": generated_text,
                }
            )
        print(output_data)
output_file = "/netscratch/jperez/output_baseline_mistral_txt_sparql_4bit_batched.json"
print(f"Saving {len(output_data)} results to {output_file}...")
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)
print("Processing complete.")
