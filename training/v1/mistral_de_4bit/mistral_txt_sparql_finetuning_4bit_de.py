#!/usr/bin/env python
import os
import json
import pandas as pds
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login, upload_folder
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from huggingface_hub import login

login(HF_TOKEN)
ds = load_dataset("julioc-p/Question-Sparql", split="train")
ds_de = ds.filter(
    lambda x: x["language"] == "de"
    and x["sparql_query"].lower() not in ["out of scope", "none"]
    and "Wikidata" in x["knowledge_graphs"]
)
ds_de = ds_de.shuffle(seed=42).select(range(35000))
new_model = "/netscratch/jperez/mistralai-sparql-de-Instruct-txt-sparql_4bit"
lora_r = 16
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "/netscratch/jperez/mistral_de_text_sparql"
num_train_epochs = 5
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = True
max_grad_norm = 1.0
learning_rate = 1e-5
weight_decay = 0.05
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.01
group_by_length = True
save_steps = 10000
logging_steps = 250
max_seq_length = None
packing = False
device_map = {"": 0}
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def create_text_row(txt, sparql_query, knowledge_graph):
    messages = [
        {
            "role": "user",
            "content": f"Write a SparQL query that answers this request: '{txt}' from the knowledge graph {knowledge_graph}.",
        },
        {"role": "assistant", "content": sparql_query},
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return formatted_text


def process_jsonl_file(output_file_path):
    with open(output_file_path, "w") as output_jsonl_file:
        for item in ds_de:
            json_object = {
                "text": create_text_row(
                    item["text_query"], item["sparql_query"], item["knowledge_graphs"]
                ),
            }
            output_jsonl_file.write(json.dumps(json_object) + "\n")


process_jsonl_file("/netscratch/jperez/training_dataset_mistral_de.jsonl")
train_dataset = load_dataset(
    "json",
    data_files="/netscratch/jperez/training_dataset_mistral_de.jsonl",
    split="train",
)
messages = [
    {
        "role": "user",
        "content": "Write a SparQL query that answers this question/request: 'Ist Lionel Messi Italiener' from the knowledge graph Wikidata.",
    }
]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
device = "cuda"
model_inputs = encodeds.to(device)
base_model.to(device)
generated_ids = base_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)
messages = [
    {
        "role": "user",
        "content": "Write a SparQL query that answers this request: 'Ist Lionel Messi Italiener' from the knowledge graph Wikidata.",
    },
]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
device = "cuda"
model_inputs = encodeds.to(device)
base_model.to(device)
generated_ids = base_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
upload_folder(
    repo_id="julioc-p/mistral_de_txt_sparql_4bit",
    folder_path="/netscratch/jperez/mistralai-sparql-de-Instruct-txt-sparql_4bit",
    path_in_repo=".",
)
