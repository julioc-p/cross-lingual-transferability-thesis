import numpy as np
import torch
import evaluate
from transformers.trainer_utils import EvalPrediction


def compute_metrics_fn(tokenizer):
    """Returns a function to compute metrics for the SFTTrainer."""
    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_preds: EvalPrediction, compute_result: bool = False):
        """
        Batch‐wise BLEU for SFTTrainer:
        - Moves logits off GPU
        - Argmax→IDs if needed
        - Decodes in memory‐efficient chunks
        - Only returns on last batch (batch_eval_metrics=True)
        """
        preds, labels = eval_preds
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        elif isinstance(preds, tuple):
            preds = preds[0].detach().cpu().numpy()
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        chunk_size = 128
        for i in range(0, preds.shape[0], chunk_size):
            batch_preds = preds[i : i + chunk_size]
            batch_labels = labels[i : i + chunk_size]
            decoded_preds = tokenizer.batch_decode(
                batch_preds, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(
                batch_labels, skip_special_tokens=True
            )
            decoded_preds = [p.strip() for p in decoded_preds]
            references = [[label.strip()] for label in decoded_labels]
            bleu_metric.add_batch(predictions=decoded_preds, references=references)
        if not compute_result:
            return {}
        result = bleu_metric.compute()
        return {"bleu": result["bleu"]}

    return compute_metrics
