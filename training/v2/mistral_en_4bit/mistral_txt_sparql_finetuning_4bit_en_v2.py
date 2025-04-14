from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from huggingface_hub import login
import os
from peft import LoraConfig
from trl import SFTTrainer

# Load jsonl data from disk
dataset = load_dataset(
    "json", data_files="/netscratch/jperez/train_dataset_en_final.json", split="train"
)

HF_TOKEN = os.environ.get("HF_TOKEN")
login(
    token=HF_TOKEN,
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

model, tokenizer = setup_chat_format(model, tokenizer)

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir="/netscratch/jperez/mistral_txt_sparql_en_v2",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=100,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="tensorboard",
)

max_seq_length = 3072

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
)

trainer.train()

# save model
trainer.save_model("/netscratch/jperez/mistral_txt_sparql_en_v2")
