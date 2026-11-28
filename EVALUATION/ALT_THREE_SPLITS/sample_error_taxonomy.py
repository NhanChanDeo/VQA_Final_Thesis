"""
Deterministic Stratified Error Sampler for Phase D 7-Class Failure Taxonomy (E1–E7).

Exports N failures per domain (default 100 DocVQA + 100 TextVQA = 200 total)
stratified between zero-score (hard fail) and partial-score (near-miss) predictions,
pre-formatted for manual annotation into E1–E7 classes.

Usage:
    python EVALUATION/ALT_THREE_SPLITS/sample_error_taxonomy.py \
        --model smolvlm_500m --namespace ebs64 --exp B4 --seed 42 --eval_split test
"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from evaluate_statistical_suite import resolve_run_dir

ERROR_TAXONOMY_CLASSES = {
    "E1": "Optical / Recognition blurring artifacts",
    "E2": "Scene-object context reasoning confusion",
    "E3": "Spatial layout / tabular grid traversal errors",
    "E4": "Formatting / casing / lexical normalization mismatch",
    "E5": "Upstream OCR ceiling omissions",
    "E6": "Ground-truth ambiguity / conflicting annotations",
    "E7": "Token truncation / spatial downsampling limit",
}


def sample_failures_for_domain(
    jsonl_path: Path,
    domain: str,
    n_samples: int = 100,
    threshold: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministically samples up to n_samples failures stratified across zero vs. partial errors."""
    metric_key = "anls" if domain == "docvqa" else "vqa_accuracy"
    hard_fails: List[Dict[str, Any]] = []
    partial_fails: List[Dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            score = float(rec.get(metric_key, 0.0))
            if score >= threshold:
                continue
            row = {
                "domain": domain,
                "sample_id": str(rec.get("sample_id", "")),
                "question": str(rec.get("question", "")),
                "prediction": str(rec.get("prediction", "")),
                "ground_truths": " | ".join(str(a) for a in (rec.get("answers") or [])),
                "anls": round(float(rec.get("anls", 0.0)), 4),
                "vqa_accuracy": round(float(rec.get("vqa_accuracy", 0.0)), 4),
                "error_tier": "zero_score" if score == 0.0 else "partial_below_threshold",
                "error_class_E1_to_E7": "",
                "notes": "",
            }
            if score == 0.0:
                hard_fails.append(row)
            else:
                partial_fails.append(row)

    total_fail = len(hard_fails) + len(partial_fails)
    if total_fail == 0 or n_samples <= 0:
        return []

    rng = random.Random(int(seed))
    rng.shuffle(hard_fails)
    rng.shuffle(partial_fails)

    target = min(int(n_samples), total_fail)
    # Proportional stratified allocation with at least 1 partial fail if available
    n_partial = round(target * (len(partial_fails) / total_fail))
    if partial_fails and n_partial == 0 and target > 1:
        n_partial = 1
    n_hard = min(len(hard_fails), target - n_partial)
    n_partial = min(len(partial_fails), target - n_hard)
    # Top up if one bucket exhausted
    selected = hard_fails[:n_hard] + partial_fails[:n_partial]
    if len(selected) < target:
        rem = hard_fails[n_hard:] + partial_fails[n_partial:]
        selected.extend(rem[: target - len(selected)])
    return selected


def export_error_worksheet(
    doc_jsonl: Optional[Path],
    text_jsonl: Optional[Path],
    out_csv: Path,
    out_jsonl: Optional[Path] = None,
    n_per_domain: int = 100,
    threshold: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if doc_jsonl is not None and doc_jsonl.exists():
        rows.extend(sample_failures_for_domain(doc_jsonl, "docvqa", n_per_domain, threshold, seed))
    if text_jsonl is not None and text_jsonl.exists():
        rows.extend(sample_failures_for_domain(text_jsonl, "textvqa", n_per_domain, threshold, seed + 1))

    for i, row in enumerate(rows, start=1):
        row["taxonomy_id"] = f"ERR_{i:03d}"

    fieldnames = [
        "taxonomy_id",
        "domain",
        "sample_id",
        "question",
        "prediction",
        "ground_truths",
        "anls",
        "vqa_accuracy",
        "error_tier",
        "error_class_E1_to_E7",
        "notes",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample stratified failures for 7-class E1–E7 error taxonomy.")
    parser.add_argument("--runs_root", default="TRAIN/ALT_THREE_SPLITS/runs")
    parser.add_argument("--model", default="smolvlm_500m")
    parser.add_argument("--namespace", default="ebs64_v2")
    parser.add_argument("--exp", default="B4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot_pct", type=int, default=100)
    parser.add_argument("--eval_split", default="test", choices=["validation", "test"])
    parser.add_argument("--n_per_domain", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--out_csv", default="EVALUATION/ALT_THREE_SPLITS/results/error_taxonomy_200.csv")
    parser.add_argument("--out_jsonl", default="EVALUATION/ALT_THREE_SPLITS/results/error_taxonomy_200.jsonl")
    args = parser.parse_args()

    run_dir = resolve_run_dir(
        Path(args.runs_root), args.model, args.namespace, args.exp, args.seed, args.few_shot_pct
    )
    doc_p = run_dir / f"preds_{args.eval_split}_docvqa.jsonl"
    tv_p = run_dir / f"preds_{args.eval_split}_textvqa.jsonl"
    rows = export_error_worksheet(
        doc_jsonl=doc_p,
        text_jsonl=tv_p,
        out_csv=Path(args.out_csv),
        out_jsonl=Path(args.out_jsonl) if args.out_jsonl else None,
        n_per_domain=args.n_per_domain,
        threshold=args.threshold,
        seed=args.sample_seed,
    )
    print(f"Exported {len(rows)} stratified error samples from {run_dir} -> {args.out_csv}")
    print("Classes:", ERROR_TAXONOMY_CLASSES)


if __name__ == "__main__":
    main()
