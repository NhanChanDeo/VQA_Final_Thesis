"""
TRAIN/src: Shared core modules for CrossVQA model training and evaluation.
Supports Qwen/Qwen2-VL-2B and HuggingFaceTB/SmolVLM-Base across both
ALT_THREE_SPLITS and VALIDATION_AS_TEST split regimes.
"""

from .metrics import compute_anls, compute_vqa_accuracy, evaluate_predictions
from .models import build_vlm_model_and_processor
from .dataset import load_crossvqa_dataset
from .collator import CrossVQADataCollator

__all__ = [
    "build_vlm_model_and_processor",
    "load_crossvqa_dataset",
    "CrossVQADataCollator",
    "compute_anls",
    "compute_vqa_accuracy",
    "evaluate_predictions",
]
