"""
Shared configuration, seeding, Hugging Face token resolution, and dual-domain
evaluation helpers for TRAIN/ALT_THREE_SPLITS/scripts/.
"""

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import yaml

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def ensure_repo_root() -> Path:
    """Finds repository root containing TRAIN/src and ensures it is on sys.path."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "TRAIN" / "src" / "__init__.py").exists():
            os.chdir(candidate)
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Could not locate repository root containing TRAIN/src/__init__.py")


def load_yaml(rel_path: str) -> Dict[str, Any]:
    with Path(rel_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    return cli_token or os.environ.get("HF_TOKEN") or None


def evaluate_both_domains(
    model,
    processor,
    model_type: str,
    eval_split: str,
    run_dir: Path,
    device: str,
    include_ocr: bool = False,
    max_new_tokens: int = 32,
    eval_batch_size: int = 4,
    max_eval_samples: Optional[int] = None,
    local_data_dir: Optional[str] = None,
    split_regime: str = "alt_three_splits",
    manifest_path: Optional[str] = None,
) -> Dict[str, float]:
    """Runs batched greedy generation across both DocVQA and TextVQA splits."""
    from TRAIN.src.dataset import load_crossvqa_dataset
    from TRAIN.src.metrics import save_predictions_jsonl
    from TRAIN.src.trainer import run_evaluation

    summary: Dict[str, float] = {}
    for ds_name in ("docvqa", "textvqa"):
        ds = load_crossvqa_dataset(
            dataset_name=ds_name,
            split=eval_split,
            split_regime=split_regime,
            manifest_path=manifest_path,
            streaming=True,
            local_data_dir=local_data_dir,
        )
        metrics, details = run_evaluation(
            model=model,
            processor=processor,
            eval_dataset=ds,
            device=device,
            max_new_tokens=max_new_tokens,
            metric_type="both",
            max_eval_samples=max_eval_samples,
            model_type=model_type,
            include_ocr=include_ocr,
            eval_batch_size=eval_batch_size,
            return_details=True,
        )
        save_predictions_jsonl(details, str(run_dir / f"preds_{eval_split}_{ds_name}.jsonl"))
        summary[f"{ds_name}_anls"] = round(float(metrics.get("anls", 0.0)), 6)
        summary[f"{ds_name}_vqa_accuracy"] = round(float(metrics.get("vqa_accuracy", 0.0)), 6)

    doc_anls = summary["docvqa_anls"]
    tv_acc = summary["textvqa_vqa_accuracy"]
    summary["harmonic_mean"] = round(
        (2.0 * doc_anls * tv_acc) / max(doc_anls + tv_acc, 1e-8), 6
    )
    return summary
