import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, EarlyStoppingCallback
from trl import SFTConfig


def get_bnb_config():
    """Returns the BitsAndBytesConfig."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def get_peft_config():
    """Returns the LoraConfig."""
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.01,
        r=32,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def get_sft_config(output_dir, max_seq_length):
    """Returns the SFTConfig."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=2e-4,
        eval_accumulation_steps=1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        fp16=True,
        max_grad_norm=0.3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
        report_to="tensorboard",
        max_seq_length=max_seq_length,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        batch_eval_metrics=True,
    )


def get_early_stopping_callback():
    """Returns the EarlyStoppingCallback."""
    return EarlyStoppingCallback(
        early_stopping_patience=3,
    )
