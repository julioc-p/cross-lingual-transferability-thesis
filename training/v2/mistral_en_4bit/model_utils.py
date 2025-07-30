import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format


def load_model_and_tokenizer(model_id, bnb_config):
    """Loads the model and tokenizer."""
    print("Setting up model and tokenizer...")
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
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tokenizer.chat_template = None
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer
