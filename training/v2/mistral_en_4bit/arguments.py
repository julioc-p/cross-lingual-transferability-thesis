from dataclasses import dataclass, field
from typing import Optional
from trl import SFTConfig


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


@dataclass
class CustomTrainingArguments:
    """
    Custom training arguments
    """

    early_stopping_patience: int = field(
        default=3, metadata={"help": "Patience for early stopping."}
    )


@dataclass
class TrainingArgumentsWithDefaults(SFTConfig):
    """
    SFTConfig with custom default values.
    """

    output_dir: str = field(
        default="results", metadata={"help": "The output directory."}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "Number of updates steps to accumulate."}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to use."}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=500, metadata={"help": "Log every X updates steps."}
    )
    learning_rate: float = field(
        default=2e-4, metadata={"help": "The initial learning rate."}
    )
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use 16-bit precision."}
    )
    max_grad_norm: float = field(default=0.3, metadata={"help": "Max gradient norm."})
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio."}
    )
    lr_scheduler_type: str = field(
        default="constant", metadata={"help": "The scheduler type to use."}
    )
    # --- Other fixed settings ---
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="bleu")
    greater_is_better: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    gradient_checkpointing: bool = field(default=True)
    report_to: str = field(default="tensorboard")
    push_to_hub: bool = field(default=False)
