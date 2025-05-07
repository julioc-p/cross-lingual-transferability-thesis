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
    (df["language"] == "de") &
    (df["sparql_query"].notna()) &
    (df["knowledge_graphs"].str.contains("Wikidata", case=False, na=False))
].copy()

def build_prompt(question, sparql):
    system_prompt = (
        "Du bist ein hilfreicher Assistent, der Wikidata-Entitäten und -Eigenschaften aus SPARQL-Abfragen extrahiert.\n"
        "Gib ein gültiges JSON-Wörterbuch ohne abschließende Zeichen aus. Format:\n"
        "{\n"
        '  "entitäten": {"ENTITÄT": "QID"},\n'
        '  "beziehungen": {"BEZIEHUNG": "PID"}\n'
        "}\n"
        "Verwende nur das, was du in der SPARQL-Abfrage siehst – kein zusätzliches Wissen."
    )

    examples = [
        {
            "question": "Hat der kanadische Regisseur von The Seventh Victim auch The Little Hut als Executive Producer betreut?",
            "sparql": "ASK WHERE { wd:Q31212 wdt:P57 ?x0 . wd:Q1218719 wdt:P1431 ?x0 . ?x0 wdt:P27 wd:Q16 }",
            "answer": """{
      "entitäten": {
        "The Seventh Victim": "Q31212",
        "The Little Hut": "Q1218719",
        "Kanada": "Q16"
      },
      "beziehungen": {
        "Regisseur": "P57",
        "Executive Producer": "P1431",
        "Staatsangehörigkeit": "P27"
      }
    }"""
        },
        {
            "question": "Hat die United States Army einen Ehepartner eines Charakters angestellt?",
            "sparql": "ASK WHERE { ?x0 wdt:P26 ?x1 . ?x1 wdt:P31 wd:Q95074 . FILTER ( ?x0 != ?x1 ) . ?x0 wdt:P108 wd:Q9212 }",
            "answer": """{
      "entitäten": {
        "United States Army": "Q9212",
        "Figur": "Q95074"
      },
      "beziehungen": {
        "Ehepartner": "P26",
        "Arbeitgeber": "P108",
        "Instanz von": "P31"
      }
    }"""
        },
        {
            "question": "Hat der männliche Schauspieler des Herzogs von Mantua Adua Veroni geheiratet?",
            "sparql": "ASK WHERE { ?x0 wdt:P453 wd:Q5815108 . ?x0 wdt:P21 wd:Q6581097 . ?x0 wdt:P26 wd:Q108650588 . FILTER ( ?x0 != wd:Q108650588 ) }",
            "answer": """{
      "entitäten": {
        "Adua Veroni": "Q108650588",
        "Herzog von Mantua": "Q5815108",
        "männlich": "Q6581097"
      },
      "beziehungen": {
        "geheiratet": "P26",
        "Geschlecht": "P21",
        "Rolle": "P453"
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
        print(decoded_text)

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
checkpoint_path = "/netscratch/jperez/wikidata_extraction_checkpoint_de.csv"
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
    print(results)

    if len(results) >= next_checkpoint:
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"Processed {len(results)}/{len(df_filtered)} rows. Checkpoint saved.")
        next_checkpoint += checkpoint_interval


df["context"] = "None"
for result in results:
    idx = df[
        (df["text_query"] == result["text_query"]) &
        (df["sparql_query"] == result["sparql_query"])
    ].index
    if not idx.empty:
        df.loc[idx, "context"] = result["context"]

ds = load_dataset("julioc-p/Question-Sparql")
ds['train'] = ds['train'].add_column("context", df['context'])
ds.push_to_hub("julioc-p/Question-Sparql", token=HF_TOKEN)

