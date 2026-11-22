"""
Evaluation metrics implementation and run-card serialization for CrossVQA:
- ANLS (Average Normalized Levenshtein Similarity) for DocVQA
- VQA Accuracy (10-annotator soft consensus) for TextVQA
- Transfer Gain (TG) and Forgetting Rate (FR)
- JSONL prediction logger and standardized Run Card exporter
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


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


def evaluate_predictions(predictions: List[str], targets: List[List[str]], metric_type: str = "both") -> Dict[str, float]:
    """Evaluates a batch or dataset of predictions against ground-truth answer lists."""
    n = len(predictions)
    if n == 0:
        return {}

    results: Dict[str, float] = {}
    if metric_type in ["anls", "both"]:
        anls_scores = [compute_anls(p, g) for p, g in zip(predictions, targets)]
        results["anls"] = sum(anls_scores) / n

    if metric_type in ["vqa_accuracy", "both"]:
        vqa_scores = [compute_vqa_accuracy(p, g) for p, g in zip(predictions, targets)]
        results["vqa_accuracy"] = sum(vqa_scores) / n

    return results


def compute_transfer_gain(score_model: float, score_b1: float) -> float:
    """Transfer Gain: TG = Score_Proposed - Score_TargetOnly."""
    return float(score_model) - float(score_b1)


def compute_forgetting_rate(anls_b2: float, anls_model: float) -> float:
    """Forgetting Rate: FR = ANLS_SourceOnly(Doc) - ANLS_Model(Doc)."""
    return float(anls_b2) - float(anls_model)


def save_predictions_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """Writes per-sample predictions enriched with per-sample ANLS and VQA Accuracy."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            pred = rec.get("prediction", "")
            answers = rec.get("answers", [])
            enriched = dict(rec)
            enriched["anls"] = compute_anls(pred, answers)
            enriched["vqa_accuracy"] = compute_vqa_accuracy(pred, answers)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def save_run_card(
    output_dir: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes standardized val_metrics.json and run_card.yaml into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_card = {
        "experiment": config.get("experiment", {}),
        "data": config.get("data", {}),
        "model": config.get("model", {}),
        "training": config.get("training", {}),
        "evaluation_results": metrics,
        "meta": extra_meta or {},
    }
    (out / "run_card.yaml").write_text(
        yaml.safe_dump(run_card, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out / "run_card.yaml"
