import os
from trl import SFTTrainer


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    peft_config,
    sft_config,
    compute_metrics,
    early_stopping_callback,
):
    """Initializes and runs the SFTTrainer."""
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        processing_class=tokenizer,
    )
    print("Starting training...")
    trainer.train()
    return trainer


def save_model(trainer, tokenizer, output_dir, repo_name):
    """Saves the final model and pushes to Hugging Face Hub."""
    print("Saving final model...")
    final_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"Final best model saved to {final_save_path}")
    if repo_name:
        print(f"Pushing model and tokenizer to Hugging Face Hub: {repo_name}")
        trainer.model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
        print("Push to hub complete.")


def test_inference(model, tokenizer):
    """Performs a test inference."""
    print("Performing test inference...")
    example = [
        {
            "content": (
                "You are an expert text to SparQL query translator. Users will ask "
                "you questions in English and you will generate a SparQL query based "
                "on the provided context.\nCONTEXT:\n"
                '{"entities": {"Reese Witherspoon": "Q44063", "Julia Roberts": '
                '"Q40523"}, "relationships": {"influenced by": "P737"}}'
            ),
            "role": "system",
        },
        {
            "content": "Was Reese Witherspoon influenced by Julia Roberts?",
            "role": "user",
        },
    ]
    encodeds = tokenizer.apply_chat_template(example, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    model.to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print("Test inference result:")
    print(decoded[0])
