"""
Evaluation metrics implementation for CrossVQA:
- ANLS (Average Normalized Levenshtein Similarity) for DocVQA
- VQA Accuracy (10-annotator soft consensus) for TextVQA
"""

import re
from typing import List, Union

def _normalize_answer(s: str) -> str:
    """Standard lowercase and whitespace strip."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes basic edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Computes ANLS (Average Normalized Levenshtein Similarity) for DocVQA.
    Formula: max_{a in GT} (1 - NL(pred, a)) if NL < threshold else 0.0
    """
    pred_clean = _normalize_answer(prediction)
    if not ground_truths:
        return 0.0

    max_sim = 0.0
    for gt in ground_truths:
        gt_clean = _normalize_answer(gt)
        if not pred_clean and not gt_clean:
            sim = 1.0
        elif not pred_clean or not gt_clean:
            sim = 0.0
        else:
            dist = levenshtein_distance(pred_clean, gt_clean)
            max_len = max(len(pred_clean), len(gt_clean))
            nld = dist / max_len
            sim = (1.0 - nld) if nld < threshold else 0.0
        if sim > max_sim:
            max_sim = sim
    return max_sim


def compute_vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    Computes standard VQA Accuracy for TextVQA:
    min(sum(I(pred == a)) / 3.0, 1.0) across 10 annotator answers.
    """
    pred_clean = _normalize_answer(prediction)
    if not ground_truths:
        return 0.0

    match_count = 0
    for gt in ground_truths:
        if pred_clean == _normalize_answer(gt):
            match_count += 1

    return min(match_count / 3.0, 1.0)


def evaluate_predictions(predictions: List[str], targets: List[List[str]], metric_type: str = "both"):
    """
    Evaluates a batch or dataset of predictions against ground-truth answer lists.
    """
    n = len(predictions)
    if n == 0:
        return {}

    results = {}
    if metric_type in ["anls", "both"]:
        anls_scores = [compute_anls(p, g) for p, g in zip(predictions, targets)]
        results["anls"] = sum(anls_scores) / n

    if metric_type in ["vqa_accuracy", "both"]:
        vqa_scores = [compute_vqa_accuracy(p, g) for p, g in zip(predictions, targets)]
        results["vqa_accuracy"] = sum(vqa_scores) / n

    return results
