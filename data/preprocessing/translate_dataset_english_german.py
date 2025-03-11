#!/usr/bin/env python
# coding: utf-8
import os
# Set environment variables
os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"
HF_TOKEN=os.getenv("HF_TOKEN")

import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import bitsandbytes
from huggingface_hub import login
from datasets import load_dataset, Dataset

# os.system("ls /netscratch/jperez/huggingface")

# Authenticate Hugging Face login
login(HF_TOKEN)

# Load dataset
ds = load_dataset("julioc-p/Question-Sparql")
df = ds["train"].to_pandas()
df_en = df[df["language"] == "en"]

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-3-Llama-3.1-70B', trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-70B",
    torch_dtype=torch.float16,
    device_map="cuda",
    load_in_8bit=False,
    load_in_4bit=True,
)

# Add padding token
tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "left"

tokenizer.pad_token_id


def get_prompt(txt):
    return f"""<|im_start|>system
    You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
    <|im_start|>user
    Translate the following text to German and only give me the translation without further characters: '{txt}'<|im_end|>
    <|im_start|>assistant"""

def translate_batch(texts):
    prompts = [get_prompt(txt) for txt in texts]
    tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    with torch.inference_mode():
        generated_ids = model.generate(
            tokenized_inputs.input_ids, 
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=750, 
            temperature=0.8, 
            repetition_penalty=1.1, 
            do_sample=True, 
            eos_token_id=tokenizer.eos_token_id
        )
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return responses

def get_german_text(txt):
    s = txt.split("assistant")
    if len(s) < 2:
        return ""
    txt = s[1].strip()
    if txt[0] == '.':
        txt = txt[1:]
    if txt[0] == "#":
        txt = txt[1:-1]
    if txt.startswith("In German"):
        txt = txt.split('"')
        if len(txt) <= 3:
            return ""
        txt = txt[3]
    return txt

batch_size = 128

checkpoint_file = "/netscratch/jperez/translation_checkpoint.json"

res = []

# Load checkpoint if it exists
if os.path.exists(checkpoint_file):
    df_checkpoint = pd.read_json(checkpoint_file)
    res = df_checkpoint.values.tolist()
    print(f"Resuming from checkpoint with {len(res)} translated rows.")
else:
    res = []


res

start_idx = len(res)

len(res)


def main():
    start_idx = len(res)
    
    for i in range(start_idx, len(df_en), batch_size):
        batch = df_en.iloc[i:i+batch_size]
        translated_texts = translate_batch(batch['text_query'].tolist())
        
        for k, (j, row) in enumerate(batch.iterrows()):
            txt = get_german_text(translated_texts[k])
            if not txt:
                continue
            res.append([txt, "de", row['sparql_query'], row['knowledge_graphs']])
        if i % (batch_size * 10) == 0:
            df_checkpoint = pd.DataFrame(res, columns=["text_query", "language", "sparql_query", "knowledge_graphs"])
            df_checkpoint.to_json(checkpoint_file, orient="records")
            print(f"Checkpoint saved at {i} records.")
            
    df_german = pd.DataFrame(res, columns=["text_query", "language", "sparql_query", "knowledge_graphs"])
    df_combined = pd.concat([df, df_german], ignore_index=True)
    
    new_ds = Dataset.from_pandas(df_combined)
    new_ds.push_to_hub("julioc-p/Question-Sparql")

# +
df_german = pd.DataFrame(res, columns=["text_query", "language", "sparql_query", "knowledge_graphs"])
df_combined = pd.concat([df, df_german], ignore_index=True)

new_ds = Dataset.from_pandas(df_combined)
new_ds.push_to_hub("julioc-p/Question-Sparql")
# -

if __name__ == "__main__":
    main()
