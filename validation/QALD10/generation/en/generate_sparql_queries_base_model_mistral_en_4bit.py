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

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            "role": "user",
            "content": f"Write a SparQL query that answers this request: '{question}' from the knowledge graph {knowledge_graph}.",
        }
    ]

    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=10000, do_sample=False)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned_sparql = extract_sparql(generated_text)

    output_data.append(
        {
            "question": question,
            "knowledge_graph": knowledge_graph,
            "gold_sparql": gold_sparql,
            "generated_sparql": cleaned_sparql,
        }
    )

with open("/netscratch/jperez/output_baseline_mistral_txt_sparql_4bit.json", "w") as f:
    json.dump(output_data, f)
