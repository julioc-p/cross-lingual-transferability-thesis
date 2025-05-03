import os

os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"
HF_TOKEN = os.getenv("HF_TOKEN")

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    MistralForCausalLM,
)
import re

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

base_model = "occiglot/occiglot-7b-eu5-instruct"
model_name = "julioc-p/occiglot-7b-eu5-instruct-txt-sparql-en-Instruct-txt-sparql_4bit"
device = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(42)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
).to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("julioc-p/Question-Sparql", split="validation")


def extract_sparql(text):
    match = re.search(
        r"(SELECT|ASK|CONSTRUCT|DESCRIBE).*?\}", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(0).strip()
    return ""


output_data = []

for example in dataset:
    question = example["text_query"]
    gold_sparql = example["sparql_query"]
    knowledge_graph = example["knowledge_graphs"]

    prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please generate precise SPARQL queries.",
        },
        {
            "role": "user",
            "content": f"Write a SPARQL query that answers this request: '{question}' from the knowledge graph {knowledge_graph}.",
        },
    ]

    tokenized_chat = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(tokenized_chat.to("cuda"), max_new_tokens=100000)

    generated_text = tokenizer.decode(outputs[0][len(tokenized_chat[0]) :])
    cleaned_sparql = extract_sparql(generated_text)
    print(cleaned_sparql)

    output_data.append(
        {
            "question": question,
            "knowledge_graph": knowledge_graph,
            "gold_sparql": gold_sparql,
            "generated_sparql": cleaned_sparql,
            "raw_answer": generated_text,
        }
    )

with open(
    "/netscratch/jperez/output_occiglot_finetuned_model_sparql_4bit.json", "w"
) as f:
    json.dump(output_data, f)
