#!/usr/bin/env python
import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig
from huggingface_hub import login

train_file = "/netscratch/jperez/train_dataset_en_final.json"
eval_file = "/netscratch/jperez/validation_dataset_en_final.json"
output_dir = "/netscratch/jperez/mistral_txt_sparql_en_v2"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
HF_TOKEN = os.environ.get("HF_TOKEN")
max_seq_length = 3072
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("HF_TOKEN environment variable not set. Skipping Hugging Face Hub login.")
print("Loading datasets...")
train_dataset = load_dataset("json", data_files=train_file, split="train")
eval_dataset = load_dataset("json", data_files=eval_file, split="train")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
print("Setting up model and tokenizer...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    print("Adding pad token")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
model, tokenizer = setup_chat_format(model, tokenizer)
peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
print("Setting up BLEU metric...")
bleu_metric = evaluate.load("bleu")


def compute_metrics(eval_preds):
    """
    Computes BLEU score for sequence generation tasks.
    Args:
        eval_preds (EvalPrediction): A tuple containing predictions and label_ids.
                                     Predictions are the generated token IDs.
                                     Label_ids are the ground truth token IDs.
    Returns:
        dict: A dictionary containing the BLEU score.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels_for_bleu = [[label.strip()] for label in decoded_labels]
    result = bleu_metric.compute(
        predictions=decoded_preds, references=decoded_labels_for_bleu
    )
    return {"bleu": result["bleu"]}


print("Configuring Training Arguments...")
args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=2,
    predict_with_generate=True,
    neftune_noise_alpha=5,
)
early_stopping_patience = 3
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stopping_patience,
)
print("Initializing SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)
print("Starting training...")
trainer.train()
print("Saving final model...")
final_save_path = os.path.join(output_dir, "final_model")
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Final best model saved to {final_save_path}")
print("Training finished.")
