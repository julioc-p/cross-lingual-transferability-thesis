#!/usr/bin/env python
import os
from transformers import HfArgumentParser
from trl import SFTConfig
from huggingface_hub import login

from arguments import (
    CustomTrainingArguments,
    ModelArguments,
    DataArguments,
    LoraArguments,
    TrainingArgumentsWithDefaults,
)
from config import (
    get_bnb_config,
    get_peft_config,
    get_sft_config,
    get_early_stopping_callback,
)
from data import load_data
from metrics import compute_metrics_fn
from model_utils import load_model_and_tokenizer
from trainer import train_model, save_model, test_inference


def main():
    """Main function to run the training script."""
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            LoraArguments,
            TrainingArgumentsWithDefaults,
            CustomTrainingArguments,
        )
    )
    (model_args, data_args, lora_args, sft_config, custom_args) = (
        parser.parse_args_into_dataclasses()
    )

    # Environment and login
    os.environ["HF_HUB_CACHE"] = model_args.hf_hub_cache
    os.environ["HF_HOME"] = model_args.hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = model_args.hf_hub_cache
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print(
            "HF_TOKEN environment variable not set. " "Skipping Hugging Face Hub login."
        )

    # Load data
    train_dataset, eval_dataset = load_data(data_args.train_file, data_args.eval_file)

    # Get configs
    bnb_config = get_bnb_config()
    peft_config = get_peft_config(lora_args)
    sft_config.max_seq_length = 1024
    sft_config.packing = False
    sft_config.dataset_kwargs = {
        "add_special_tokens": False,
        "append_concat_token": False,
    }
    sft_config.batch_eval_metrics = True

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_args.model_name_or_path, bnb_config
    )

    # Get metrics and callbacks
    compute_metrics = compute_metrics_fn(tokenizer)
    early_stopping_callback = get_early_stopping_callback(
        custom_args.early_stopping_patience
    )

    # Train the model
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        peft_config,
        sft_config,
        compute_metrics,
        early_stopping_callback,
    )

    # Save the model
    save_model(trainer, tokenizer, sft_config.output_dir, model_args.repo_name)

    # Test inference
    test_inference(trainer.model, tokenizer)

    print("Training finished.")


if __name__ == "__main__":
    main()
