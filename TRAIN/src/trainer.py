"""
Standardized training and evaluation loop utilities for CrossVQA experiments.
"""

import os
import json
import torch
from tqdm.auto import tqdm
from .metrics import evaluate_predictions

def train_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler=None,
    device="cuda",
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
):
    """Executes one epoch of training with gradient accumulation."""
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        accumulated_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += accumulated_loss * gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{accumulated_loss * gradient_accumulation_steps:.4f}"})
            accumulated_loss = 0.0

    return total_loss / len(dataloader)


def run_evaluation(
    model,
    processor,
    eval_dataset,
    device="cuda",
    max_new_tokens: int = 32,
    metric_type: str = "both",
    max_eval_samples: int = None,
):
    """
    Evaluates greedy autoregressive generation against ground-truth answers.
    """
    model.eval()
    predictions = []
    ground_truths = []

    count = 0
    pbar = tqdm(eval_dataset, desc="Evaluating")

    for sample in pbar:
        if max_eval_samples and count >= max_eval_samples:
            break

        from .collator import CrossVQADataCollator
        collator = CrossVQADataCollator(processor, is_train=False)
        batch = collator([sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode generated answer (excluding input prompt tokens)
        input_len = batch["input_ids"].shape[1]
        ans_ids = generated_ids[0][input_len:]
        pred_text = processor.tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

        predictions.append(pred_text)
        ground_truths.append(sample.get("answers", [sample.get("canonical_answer", "")]))
        count += 1

    metrics = evaluate_predictions(predictions, ground_truths, metric_type=metric_type)
    return metrics, predictions
