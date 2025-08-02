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


def get_peft_config(lora_args):
    """Returns the LoraConfig."""
    return LoraConfig(
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        r=lora_args.r,
        bias=lora_args.bias,
        target_modules=(
            lora_args.target_modules.split(",")
            if lora_args.target_modules != "all-linear"
            else lora_args.target_modules
        ),
        task_type=lora_args.task_type,
    )


def get_sft_config(sft_config: SFTConfig, max_seq_length: int):
    """Updates and returns the SFTConfig."""

    # Set attributes that are derived from other arguments or are fixed
    sft_config.max_seq_length = max_seq_length
    sft_config.packing = False  # Or whatever fixed value you need
    sft_config.dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token": False,
    }
    sft_config.batch_eval_metrics = True

    # The other arguments (learning_rate, epochs, etc.) are already set
    # by HfArgumentParser from the command line or their defaults in SFTConfig.
    # We don't need to redeclare them here.

    return sft_config


def get_early_stopping_callback(early_stopping_patience):
    """Returns the EarlyStoppingCallback."""
    return EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
    )
