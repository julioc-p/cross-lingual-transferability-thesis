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
    MistralForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from huggingface_hub import login

login(HF_TOKEN)
ds = load_dataset("julioc-p/Question-Sparql", split="train")
ds_en = ds.filter(
    lambda x: x["language"] == "de"
    and x["sparql_query"].lower() not in ["out of scope", "none"]
    and "Wikidata" in x["knowledge_graphs"]
)
ds_en = ds_en.shuffle(seed=42).select(range(35000))
model_name = "occiglot/occiglot-7b-eu5-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


def create_chat_format(txt, sparql_query, knowledge_graph):
    messages = [
        {
            "role": "system",
            "content": "You are an expert in SPARQL query generation. "
            "Given a user's request and the knowledge graph name, "
            "generate the correct SPARQL query.",
        },
        {
            "role": "user",
            "content": f"Write a SparQL query that answers this request: '{txt}' from the knowledge graph {knowledge_graph}.",
        },
        {"role": "assistant", "content": sparql_query},
    ]
    return messages


def process_jsonl_file(output_file_path):
    with open(output_file_path, "w") as output_jsonl_file:
        for item in ds_en:
            chat_messages = create_chat_format(
                item["text_query"], item["sparql_query"], item["knowledge_graphs"]
            )
            text = tokenizer.apply_chat_template(chat_messages, tokenize=False)
            json_object = {
                "text": text,
            }
            output_jsonl_file.write(json.dumps(json_object) + "\n")


process_jsonl_file("/netscratch/jperez/training_dataset_occiglot_de_4bit.jsonl")
train_dataset = load_dataset(
    "json",
    data_files="/netscratch/jperez/training_dataset_occiglot_de_4bit.jsonl",
    split="train",
)
train_dataset["text"]
new_model = "/netscratch/jperez/occiglot-7b-eu5-instruct-sparql-de-4bit"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "/netscratch/jperez/occiglot-7b-eu5-instruct_de_text_sparql_4bit"
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
model_name = "occiglot/occiglot-7b-eu5-instruct"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
base_model = MistralForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
system_prompt = """You are an expert in SPARQL query generation from german text. Generate the SPARQL query that answers the user's question."""
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": "Write a SPARQL query that answers: 'Ist Lionel Messi Italiener",
    },
]
encodeds = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
generated_ids = base_model.generate(
    encodeds, max_new_tokens=1000, do_sample=True, temperature=0.7, top_p=0.9
)
response = tokenizer.decode(
    generated_ids[0][encodeds.shape[1] :], skip_special_tokens=True
)
print("Generated SPARQL query:\n", response)
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
        "lm_head",
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
upload_folder(
    repo_id="julioc-p/occiglot-7b-eu5-instruct-txt-sparql-en-Instruct-txt-sparql_4bit",
    folder_path="/netscratch/jperez/occiglot-7b-eu5-instruct-sparql-en-Instruct-txt-sparql_4bit",
    commit_message="4bit fine-tuned model",
    path_in_repo=".",
)
system_prompt = """You are an expert in SPARQL query generation from german text. Generate the SPARQL query that answers the user's question."""
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": "Write a SPARQL query that answers this request: 'Ist Lionel Messi Italiener' from the knowledge graph Wikidata",
    },
]
encodeds = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")
generated_ids = base_model.generate(
    encodeds, max_new_tokens=1000, do_sample=True, temperature=0.1, top_p=0.9
)
response = tokenizer.decode(
    generated_ids[0][encodeds.shape[1] :], skip_special_tokens=True
)
print("Generated SPARQL query:\n", response)
upload_folder(
    repo_id="julioc-p/occiglot-7b-eu5-instruct-txt-de-sparql_4bit",
    folder_path="/netscratch/jperez/occiglot-7b-eu5-instruct-sparql-de-4bit",
)
