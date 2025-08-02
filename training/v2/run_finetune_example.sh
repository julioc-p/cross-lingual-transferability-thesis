#!/bin/bash


python finetune.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --train_file "/path/to/your/train_dataset.json" \
    --eval_file "/path/to/your/validation_dataset.json" \
    --output_dir "/path/to/your/output_dir" \
    --repo_name "your-hf-username/your-model-name" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4