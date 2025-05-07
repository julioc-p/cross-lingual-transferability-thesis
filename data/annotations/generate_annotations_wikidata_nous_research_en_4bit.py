import os
os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"
HF_TOKEN = os.getenv("HF_TOKEN")
import json
import torch
import pandas as pd
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset, Dataset

login(HF_TOKEN)

use_4bit = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model_name = "NousResearch/Hermes-3-Llama-3.1-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

dataset = load_dataset("julioc-p/Question-Sparql")
df = dataset["train"].to_pandas()

df_filtered = df[
    (df["language"] == "en") &
    (df["sparql_query"].notna()) &
    (df["knowledge_graphs"].str.contains("Wikidata", case=False, na=False))
].copy()

def build_prompt(question, sparql):
    system_prompt = (
        "You are a helpful assistant that extracts Wikidata entities and properties from SPARQL queries.\n"
        "Output a valid JSON dictionary with no trailing characters. Format:\n"
        "{\n"
        '  "entities": {"ENTITY": "QID"},\n'
        '  "relationships": {"RELATION": "PID"}\n'
        "}\n"
        "Only use what you see in the SPARQL, no inferred knowledge."
    )
    examples = [
        {
            "question": "Did The Seventh Victim's Canadian director executive produce The Little Hut",
            "sparql": "ASK WHERE { wd:Q31212 wdt:P57 ?x0 . wd:Q1218719 wdt:P1431 ?x0 . ?x0 wdt:P27 wd:Q16 }",
            "answer": """{
  "entities": {
    "The Seventh Victim": "Q31212",
    "The Little Hut": "Q1218719",
    "Canada": "Q16"
  },
  "relationships": {
    "director": "P57",
    "executive producer": "P1431",
    "country of citizenship": "P27"
  }
}"""
        }
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": "user", "content": f"Question: {ex['question']}\nSPARQL: {ex['sparql']}"}
            for ex in examples
        ],
        {"role": "assistant", "content": examples[0]['answer']},
        {"role": "user", "content": f"Question: {question}\nSPARQL: {sparql}"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def extract_entities_batch(prompts):
    """
    Process a list of prompts at once and return a list of extracted JSON dictionaries (or None on failure).
    """
    tokenized_batch = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokenized_batch.input_ids.to(model.device)
    attention_mask = tokenized_batch.attention_mask.to(model.device)

    input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    results = []
    for i, prompt_len in enumerate(input_lengths):
        generated_tokens = outputs[i][prompt_len:]
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        try:
            start_idx = decoded_text.find('{')
            end_idx = decoded_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                results.append(None)
            else:
                json_str = ' '.join(decoded_text[start_idx:end_idx].split())
                results.append(json.loads(json_str))
        except (json.JSONDecodeError, AttributeError):
            results.append(None)
    return results

batch_size = 32
checkpoint_path = "/netscratch/jperez/wikidata_extraction_checkpoint.csv"
results = []
checkpoint_interval = 30000

if os.path.exists(checkpoint_path):
    checkpoint_df = pd.read_csv(checkpoint_path)
    results = checkpoint_df.to_dict('records')
    print(f"Resuming from checkpoint with {len(results)} processed rows")

next_checkpoint = ((len(results) // checkpoint_interval) + 1) * checkpoint_interval
for i in range(len(results), len(df_filtered), batch_size):
    batch = df_filtered.iloc[i:i+batch_size]
    prompts = []

    for _, row in batch.iterrows():
        prompt = build_prompt(row["text_query"], row["sparql_query"])
        prompts.append(prompt)

    batch_contexts = extract_entities_batch(prompts)

    batch_results = []
    for (_, row), context in zip(batch.iterrows(), batch_contexts):
        result = {
            **row.to_dict(),
            "context": json.dumps(context) if context is not None else None
        }
        batch_results.append(result)
    results.extend(batch_results)
    if len(results) >= next_checkpoint:
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"Processed {len(results)}/{len(df_filtered)} rows. Checkpoint saved.")
        next_checkpoint += checkpoint_interval


if results:
    results_df = pd.DataFrame(results)[['text_query', 'sparql_query', 'context']]


    df.drop(columns=['context'], inplace=True, errors='ignore')
    df_merged = pd.merge(
        df,
        results_df,
        on=['text_query', 'sparql_query'],
        how='left'
    )

    df_merged['context'].fillna("None", inplace=True)
    df = df_merged

else:
    df['context'] = "None"


print("Updating Hugging Face dataset...")
ds = load_dataset("julioc-p/Question-Sparql")

ds['train'] = ds['train'].remove_columns(['context']) if 'context' in ds['train'].column_names else ds['train']
ds['train'] = ds['train'].add_column("context", df['context'].tolist())

if 'context' not in ds['validation'].column_names:
    filler = ["None"] * len(ds["validation"])
    ds['validation'] = ds['validation'].add_column("context", filler)
else:
    pass

ds.push_to_hub("julioc-p/Question-Sparql", token=HF_TOKEN)
print("Dataset update complete.")
