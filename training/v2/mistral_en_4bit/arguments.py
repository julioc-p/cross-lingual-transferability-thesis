from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "huggingface.co/models"
        }
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in "
            "AutoModelForCausalLM#from_pretrained."
        },
    )
    hf_hub_cache: Optional[str] = field(
        default="/netscratch/jperez/huggingface",
        metadata={"help": "Hugging Face Hub cache directory."},
    )
    repo_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository to push to the Hub."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    train_file: str = field(
        metadata={"help": "The input training data file (a json file)."}
    )
    eval_file: str = field(
        metadata={"help": "An optional input evaluation data file (a json file)."}
    )


@dataclass
class LoraArguments:
    """
    Arguments pertaining to LoRA configuration.
    """

    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.01, metadata={"help": "Lora dropout."})
    r: int = field(default=64, metadata={"help": "Lora R dimension."})
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"},
    )
    target_modules: str = field(
        default="all-linear",
        metadata={"help": "Comma separated list of module names to apply LoRA to."},
    )
    task_type: str = field(default="CAUSAL_LM", metadata={"help": "Task type."})
