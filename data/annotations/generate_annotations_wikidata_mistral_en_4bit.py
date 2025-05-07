import os
os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"

import pandas as pd
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, quantization_config=bnb_config,).to(device)

data = [
    {
        "text_query": "Did United States Army employ a spouse of a character",
        "language": "en",
        "sparql_query": "ASK WHERE { ?x0 wdt:P26 ?x1 . ?x1 wdt:P31 wd:Q95074 . FILTER ( ?x0 != ?x1 ) . ?x0 wdt:P108 wd:Q9212 }",
        "knowledge_graphs": "Wikidata"
    },
    {
        "text_query": "Did Duke of Mantua's male actor marry Adua Veroni",
        "language": "en",
        "sparql_query": "ASK WHERE { ?x0 wdt:P453 wd:Q5815108 . ?x0 wdt:P21 wd:Q6581097 . ?x0 wdt:P26 wd:Q108650588 . FILTER ( ?x0 != wd:Q108650588 ) }",
        "knowledge_graphs": "Wikidata"
    },
]

df = pd.DataFrame(data)

def build_prompt(question, sparql):
    return f"""<s>[INST] You are a helpful assistant. Given a natural language question and a SPARQL query, extract all referenced entities and properties as Wikidata IDs. Output a dictionary like:

{{
  "entities": {{
    "ENTITY": "QID"
  }},
  "relationships": {{
    "RELATION": "PID"
  }}
}}

Use only what you see in the SPARQL, not inferred knowledge.

Examples:

Question: Did The Seventh Victim's Canadian director executive produce The Little Hut
SPARQL: ASK WHERE {{ wd:Q31212 wdt:P57 ?x0 . wd:Q1218719 wdt:P1431 ?x0 . ?x0 wdt:P27 wd:Q16 }}
Answer: {{
  "entities": {{
    "The Seventh Victim": "Q31212",
    "The Little Hut": "Q1218719",
    "Canada": "Q16"
  }},
  "relationships": {{
    "director": "P57",
    "executive producer": "P1431",
    "country of citizenship": "P27"
  }}
}}

Question: Did Resurrecting the Champ's male cinematographer's Canadian spouse's spouse marry Leslie Hope
SPARQL: ASK WHERE {{ ?x0 wdt:P26 ?x1 . ?x0 wdt:P26 wd:Q239150 . ?x1 wdt:P27 wd:Q16 . ?x1 wdt:P26 ?x2 . wd:Q155485 wdt:P344 ?x2 . ?x2 wdt:P21 wd:Q6581097 . FILTER ( ?x0 != ?x1 ) . FILTER ( ?x0 != wd:Q239150 ) . FILTER ( ?x1 != ?x2 ) }}
Answer: {{
  "entities": {{
    "Leslie Hope": "Q239150",
    "Resurrecting the Champ": "Q155485",
    "Canada": "Q16",
    "male": "Q6581097"
  }},
  "relationships": {{
    "spouse": "P26",
    "cinematographer": "P344",
    "country of citizenship": "P27",
    "sex or gender": "P21"
  }}
}}

Question: Did Strait-Jacket's costume designer write All the King's Horses
SPARQL: ASK WHERE {{ wd:Q1217802 wdt:P2515 ?x0 . wd:Q1653551 wdt:P58 ?x0 }}
Answer: {{
  "entities": {{
    "Strait-Jacket": "Q1217802",
    "All the King's Horses": "Q1653551"
  }},
  "relationships": {{
    "costume designer": "P2515",
    "screenwriter": "P58"
  }}
}}

Now answer this:

Question: {question}
SPARQL: {sparql}
Answer:
[/INST]"""

def infer_dict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    match = re.search(r'\{[\s\S]*?\}', response)
    if match:
        try:
            match = match.group(0).replace("\n", "")
            parsed = json.loads(match)
            return parsed
        except json.JSONDecodeError:
            return {"error": "json_parse_error", "raw": match}
    else:
        return {"error": "no_dict_found", "raw": response}

for example in data:
    print(infer_dict(build_prompt(example["text_query"], example["sparql_query"])))

batch_size = 2
checkpoint_path = "/netscratch/jperez/annotated_dataset.csv"
results = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    for idx, row in batch.iterrows():
        if row["knowledge_graphs"].lower() != "wikidata":
            continue

        prompt = build_prompt(row["text_query"], row["sparql_query"])
        context = infer_dict(prompt)

        results.append({
            **row,
            "context": json.dumps(context, ensure_ascii=False)
        })

    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved with {len(results)} rows")

print(results)

print("✅ All done!")

