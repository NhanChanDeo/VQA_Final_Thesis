"""
Standardized training and evaluation loop utilities for CrossVQA experiments.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from .collator import CrossVQADataCollator
from .metrics import evaluate_predictions

NON_MODEL_BATCH_KEYS = {"domain_ids", "sample_weights"}


def _prepare_model_inputs(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Filters out metadata keys and moves tensor inputs onto target device."""
    model_inputs = {}
    for key, val in batch.items():
        if key in NON_MODEL_BATCH_KEYS:
            continue
        model_inputs[key] = val.to(device) if isinstance(val, torch.Tensor) else val
    return model_inputs


def train_epoch(
    model,
    dataloader,
    optimizer,
    lr_scheduler=None,
    device: str = "cuda",
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    max_steps: Optional[int] = None,
) -> float:
    """Executes one training epoch (or up to max_steps) with gradient accumulation."""
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    optimizer_steps = 0
    steps_taken = 0

    try:
        total_len = len(dataloader)
    except TypeError:
        total_len = None

    pbar = tqdm(dataloader, desc="Training", total=max_steps or total_len)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        if max_steps is not None and steps_taken >= max_steps:
            break

        model_inputs = _prepare_model_inputs(batch, device)
        outputs = model(**model_inputs)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        accumulated_loss += float(loss.item())
        steps_taken += 1

        is_last_step = (max_steps is not None and steps_taken >= max_steps) or (
            total_len is not None and (step + 1) == total_len
        )
        if (step + 1) % gradient_accumulation_steps == 0 or is_last_step:
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            step_loss = accumulated_loss * gradient_accumulation_steps
            total_loss += step_loss
            optimizer_steps += 1
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})
            accumulated_loss = 0.0

    # Flush trailing gradients for IterableDataset streams with unknown length
    if accumulated_loss > 0.0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss * gradient_accumulation_steps
        optimizer_steps += 1

    return total_loss / max(optimizer_steps, 1)


def run_evaluation(
    model,
    processor,
    eval_dataset,
    device: str = "cuda",
    max_new_tokens: int = 32,
    metric_type: str = "both",
    max_eval_samples: Optional[int] = None,
    model_type: str = "qwen2_vl",
    include_ocr: bool = False,
    max_ocr_tokens: int = 50,
    eval_batch_size: int = 4,
    collator: Optional[CrossVQADataCollator] = None,
    return_details: bool = False,
) -> Tuple[Dict[str, float], List[Any]]:
    """
    Evaluates batched greedy autoregressive generation against ground-truth answers.
    """
    model.eval()
    if collator is None:
        collator = CrossVQADataCollator(
            processor,
            model_type=model_type,
            include_ocr=include_ocr,
            max_ocr_tokens=max_ocr_tokens,
            is_train=False,
        )

    predictions: List[str] = []
    ground_truths: List[List[str]] = []
    details: List[Dict[str, Any]] = []

    sample_buffer: List[Dict[str, Any]] = []
    count = 0

    def _eval_chunk(chunk: List[Dict[str, Any]]) -> None:
        batch = collator(chunk)
        model_inputs = _prepare_model_inputs(batch, device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = model_inputs["input_ids"].shape[1]
        tokenizer = getattr(processor, "tokenizer", processor)
        gen_tokens = generated_ids[:, input_len:]
        decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for sample, pred_raw in zip(chunk, decoded_batch):
            pred_text = pred_raw.strip()
            gt_list = sample.get("answers") or [sample.get("canonical_answer", "")]
            predictions.append(pred_text)
            ground_truths.append(gt_list)
            if return_details:
                details.append(
                    {
                        "sample_id": sample.get("sample_id", ""),
                        "domain_type": sample.get("domain_type", ""),
                        "question": sample.get("question", ""),
                        "prediction": pred_text,
                        "answers": gt_list,
                    }
                )

    pbar = tqdm(eval_dataset, desc="Evaluating", total=max_eval_samples)
    for sample in pbar:
        if max_eval_samples is not None and count >= max_eval_samples:
            break
        sample_buffer.append(sample)
        count += 1
        if len(sample_buffer) >= max(eval_batch_size, 1):
            _eval_chunk(sample_buffer)
            sample_buffer = []

    if sample_buffer:
        _eval_chunk(sample_buffer)

    metrics = evaluate_predictions(predictions, ground_truths, metric_type=metric_type)
    if return_details:
        return metrics, details
    return metrics, predictions
