import os

os.environ["HF_HUB_CACHE"] = "/netscratch/jperez/huggingface"
os.environ["HF_HOME"] = "/netscratch/jperez/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/netscratch/jperez/huggingface"
HF_TOKEN = os.getenv("HF_TOKEN")
import json
import torch
import pandas as pd
import re

if not HF_TOKEN:
    raise ValueError(
        "Hugging Face token (HF_TOKEN) not found in environment variables. "
        "Llama 3.3 requires authentication."
    )

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from datasets import load_dataset, Dataset

try:
    login(token=HF_TOKEN)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    exit(1)


use_4bit = True
bnb_config = None
compute_dtype = torch.bfloat16

if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    print("Using 4-bit quantization (nf4).")
else:
    print("Using full precision (or model's default).")


model_name = "meta-llama/Llama-3.3-70B-Instruct"

print(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token.")

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
)
print("Model loaded successfully.")

dataset = load_dataset("julioc-p/Question-Sparql")
df = dataset["train"].to_pandas()

df_filtered = df[
    (df["language"] == "de")
    & (df["sparql_query"].notna())
    & (df["knowledge_graphs"].str.contains("Wikidata", case=False, na=False))
].copy()
print(f"Filtered DataFrame shape: {df_filtered.shape}")


def build_prompt(question, sparql):
    """Builds the prompt string using the tokenizer's chat template for Llama 3.3 Instruct."""
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
}""",
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
}""",
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
}""",
        },
    ]

    messages = [{"role": "system", "content": system_prompt}]

    for ex in examples:
        messages.append(
            {
                "role": "user",
                "content": f"Question: {ex['question']}\nSPARQL: {ex['sparql']}",
            }
        )
        messages.append({"role": "assistant", "content": ex["answer"].strip()})

    messages.append(
        {"role": "user", "content": f"Question: {question}\nSPARQL: {sparql}"}
    )

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return None


def extract_entities_batch(prompts):
    """
    Process a list of prompts at once and return a list of extracted JSON dictionaries (or None on failure).
    """
    if not prompts:
        return []

    tokenizer.padding_side = "left"
    tokenized_batch = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    )

    input_ids = tokenized_batch.input_ids.to(model.device)
    attention_mask = tokenized_batch.attention_mask.to(model.device)

    input_seq_len = input_ids.shape[1]

    print(
        f"Running generation for batch size: {len(prompts)}, input sequence length: {input_seq_len}"
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("Generation complete for batch.")

    results = []
    for i in range(len(prompts)):
        generated_tokens = outputs[i][input_seq_len:]

        decoded_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        try:
            start_idx = decoded_text.find("{")
            end_idx = decoded_text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = decoded_text[start_idx : end_idx + 1]
                json_str = re.sub(r"^```json\s*", "", json_str, flags=re.IGNORECASE)
                json_str = re.sub(r"\s*```$", "", json_str)
                json_str = json_str.strip()
                parsed_json = json.loads(json_str)
                results.append(parsed_json)
                print(f"Successfully parsed JSON for batch item {i}")
            else:
                print(f"Could not find valid JSON boundaries for batch item {i}")
                results.append(None)
        except json.JSONDecodeError as e:
            print(
                f"JSON Decode Error for batch item {i}: {e}\nContent (first 100 chars): {decoded_text[:100]}"
            )
            results.append(None)
        except Exception as e:
            print(
                f"Unexpected error processing output for batch item {i}: {e}\nContent (first 100 chars): {decoded_text[:100]}"
            )
            results.append(None)

    return results


batch_size = 32
checkpoint_path = "/netscratch/jperez/wikidata_extraction_checkpoint_de_llama33.csv"
results = []
checkpoint_interval = 5000

if os.path.exists(checkpoint_path):
    try:
        checkpoint_df = pd.read_csv(checkpoint_path)
        results = checkpoint_df.to_dict("records")
        print(f"Resuming from checkpoint with {len(results)} processed rows.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        results = []

start_index = len(results)
print(f"Starting processing from index {start_index}")

next_checkpoint = ((start_index // checkpoint_interval) + 1) * checkpoint_interval

for i in range(start_index, len(df_filtered), batch_size):
    batch_end = min(i + batch_size, len(df_filtered))
    print(f"\nProcessing batch: rows {i} to {batch_end-1}")
    batch_df = df_filtered.iloc[i:batch_end]
    prompts = []

    for _, row in batch_df.iterrows():
        prompt = build_prompt(row["text_query"], row["sparql_query"])
        if prompt:
            prompts.append(prompt)
        else:
            prompts.append(None)

    valid_prompts = [p for p in prompts if p is not None]
    prompt_indices = [idx for idx, p in enumerate(prompts) if p is not None]

    if not valid_prompts:
        print("Skipping batch - no valid prompts generated.")
        batch_contexts = [None] * len(batch_df)
    else:
        extracted_contexts = extract_entities_batch(valid_prompts)

        batch_contexts = [None] * len(batch_df)
        for original_idx, context in zip(prompt_indices, extracted_contexts):
            batch_contexts[original_idx] = context

    batch_results_list = []
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        context = batch_contexts[idx]
        result_dict = {
            **row.to_dict(),
            "context": (
                json.dumps(context, ensure_ascii=False) if context is not None else None
            ),
        }
        batch_results_list.append(result_dict)

    results.extend(batch_results_list)
    print(f"Processed {len(results)}/{len(df_filtered)} total rows.")

    if len(results) >= next_checkpoint:
        try:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved at {len(results)} rows.")
            next_checkpoint += checkpoint_interval
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


try:
    final_df = pd.DataFrame(results)
    final_df.to_csv(checkpoint_path.replace(".csv", "_final.csv"), index=False)
    print(f"Final results saved to {checkpoint_path.replace('.csv', '_final.csv')}")
except Exception as e:
    print(f"Error saving final results: {e}")


print("\nMerging results back into the original dataset structure...")
result_map = {
    (res["text_query"], res["sparql_query"]): res["context"] for res in results
}


def get_context(row):
    return result_map.get((row["text_query"], row["sparql_query"]), None)


original_dataset = load_dataset("julioc-p/Question-Sparql")
original_df = original_dataset["train"].to_pandas()

if "context" not in original_df.columns:
    original_df["context"] = None

original_df["context"] = original_df.apply(get_context, axis=1)

print("Sample of updated DataFrame with context:")
print(original_df[original_df["context"].notna()].head())

try:
    updated_dataset = Dataset.from_pandas(original_df)
    original_dataset["train"] = updated_dataset

    print("Pushing updated dataset to Hugging Face Hub...")
    original_dataset.push_to_hub("julioc-p/Question-Sparql", token=HF_TOKEN)
    print("Dataset successfully pushed to Hub.")
except Exception as e:
    print(f"Error updating or pushing dataset to Hub: {e}")

print("Script finished.")
