#!/usr/bin/env python3
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

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
base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "julioc-p/mistral_de_txt_sparql_4bit"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="cuda"
)
print("Model loaded.")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
print(
    f"Tokenizer pad token set to: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
)
print("Loading dataset...")
dataset = load_dataset("julioc-p/Question-Sparql", split="validation")
dataset = dataset.filter(lambda x: x["language"] == "de")
print(f"Dataset loaded with {len(dataset)} examples.")
BATCH_SIZE = 256
MAX_NEW_TOKENS = 512
DO_SAMPLE = False
sparql_pattern_strict = re.compile(
    r"""
    (SELECT|ASK|CONSTRUCT|DESCRIBE)
    .*?
    \}
    (
        (?:
            \s*
            (?:
                (?:(?:GROUP|ORDER)\s+BY|HAVING)\s+.+?\s*(?=\s*(?:(?:GROUP|ORDER)\s+BY|HAVING|LIMIT|OFFSET|VALUES|$)) |
                LIMIT\s+\d+ |
                OFFSET\s+\d+ |
                VALUES\s*(?:\{.*?\}|\w+|\(.*?\))
            )
        )*
    )
    """,
    re.DOTALL | re.IGNORECASE | re.VERBOSE,
)


def extract_sparql(text):
    """
    Extracts the first potential SPARQL query block from text, attempting to stop
    cleanly after the last valid SPARQL clause (e.g., LIMIT, OFFSET, ORDER BY).
    """
    code_block_match = re.search(
        r"```(?:sparql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        text_to_search = code_block_match.group(1)
    else:
        text_to_search = text
    match = sparql_pattern_strict.search(text_to_search)
    if match:
        return match.group(0).strip()
    else:
        fallback_match = re.search(
            r"(SELECT|ASK|CONSTRUCT|DESCRIBE).*?\}",
            text_to_search,
            re.DOTALL | re.IGNORECASE,
        )
        if fallback_match:
            return fallback_match.group(0).strip()
    return ""


output_data = []
processed_indices = set()
print(
    f"Starting inference with Batch Size: {BATCH_SIZE}, Max New Tokens: {MAX_NEW_TOKENS}"
)
for i in range(0, len(dataset), BATCH_SIZE):
    batch_indices = list(range(i, min(i + BATCH_SIZE, len(dataset))))
    current_batch_size = len(batch_indices)
    print(
        f"Processing batch {i//BATCH_SIZE + 1}/{ (len(dataset) + BATCH_SIZE - 1)//BATCH_SIZE } (Indices {batch_indices[0]}-{batch_indices[-1]}, Size: {current_batch_size})..."
    )
    batch_examples = [dataset[idx] for idx in batch_indices]
    questions = [ex["text_query"] for ex in batch_examples]
    knowledge_graphs = [ex["knowledge_graphs"] for ex in batch_examples]
    gold_sparqls = [ex["sparql_query"] for ex in batch_examples]
    prompts = []
    for q, kg in zip(questions, knowledge_graphs):
        prompts.append(
            [
                {
                    "role": "user",
                    "content": f"Write a SparQL query that answers this request: '{q}' from the knowledge graph {kg}.",
                },
            ]
        )
    prompt_strings = [
        tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        for conversation in prompts
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompt_strings, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        output_ids_generated = output_ids[:, input_length:]
        generated_texts = tokenizer.batch_decode(
            output_ids_generated, skip_special_tokens=True
        )
        for idx, gen_text, q, kg, gold in zip(
            batch_indices, generated_texts, questions, knowledge_graphs, gold_sparqls
        ):
            cleaned_sparql = extract_sparql(gen_text)
            output_data.append(
                {
                    "question": q,
                    "knowledge_graph": kg,
                    "gold_sparql": gold,
                    "generated_sparql": cleaned_sparql,
                    "raw_answer": gen_text,
                }
            )
        print(output_data)
    except torch.cuda.OutOfMemoryError:
        print(f"\nCUDA OutOfMemoryError encountered on batch {i//BATCH_SIZE + 1}.")
        print("Try reducing BATCH_SIZE. Current BATCH_SIZE:", BATCH_SIZE)
        break
    except Exception as e:
        print(f"\nAn error occurred during batch {i//BATCH_SIZE + 1}: {e}")
        break
print(f"\nFinished processing {len(output_data)} examples.")
output_file = "/netscratch/jperez/output_finetuned_mistral_txt_sparql_4bit_de.json"
print(f"Saving results to {output_file}...")
try:
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print("Results saved successfully.")
except Exception as e:
    print(f"Error saving results to JSON: {e}")
print("Script finished.")
