"""
TRAIN/src: Shared core modules for CrossVQA model training and evaluation.
Supports Qwen/Qwen2-VL-2B, HuggingFaceTB/SmolVLM-Base, and SmolVLM-500M-Base
across both ALT_THREE_SPLITS and VALIDATION_AS_TEST split regimes.
"""

from .adapters import (
    CrossVQADualAdapterModel,
    DualLoRAContext,
    DualLoRALinear,
    ProjectorAlignmentAdapter,
    VisualSoftDomainRouter,
)
from .augmentations import CurriculumPseudoDomainTransform
from .collator import CrossVQADataCollator
from .dataset import (
    InterleavedCrossVQADataset,
    PairedDomainDataset,
    decode_image_payload,
    load_crossvqa_dataset,
)
from .losses import (
    CrossDomainInfoNCELoss,
    FeatureQueue,
    MultiKernelRBFMMDLoss,
    compute_proxy_a_distance,
    compute_rkhs_mmd,
)
from .metrics import (
    compute_anls,
    compute_forgetting_rate,
    compute_transfer_gain,
    compute_vqa_accuracy,
    evaluate_predictions,
    save_predictions_jsonl,
    save_run_card,
)
from .models import SUPPORTED_MODELS, build_vlm_model_and_processor
from .trainer import (
    HARDWARE_PROFILES,
    CheckpointManager,
    CrossVQATrainer,
    resolve_hardware_profile,
    run_evaluation,
    train_epoch,
)

__all__ = [
    "SUPPORTED_MODELS",
    "HARDWARE_PROFILES",
    "build_vlm_model_and_processor",
    "load_crossvqa_dataset",
    "decode_image_payload",
    "InterleavedCrossVQADataset",
    "PairedDomainDataset",
    "CrossVQADataCollator",
    "DualLoRAContext",
    "DualLoRALinear",
    "ProjectorAlignmentAdapter",
    "VisualSoftDomainRouter",
    "CrossVQADualAdapterModel",
    "FeatureQueue",
    "MultiKernelRBFMMDLoss",
    "CrossDomainInfoNCELoss",
    "compute_rkhs_mmd",
    "compute_proxy_a_distance",
    "CurriculumPseudoDomainTransform",
    "compute_anls",
    "compute_vqa_accuracy",
    "evaluate_predictions",
    "compute_transfer_gain",
    "compute_forgetting_rate",
    "save_predictions_jsonl",
    "save_run_card",
    "resolve_hardware_profile",
    "CheckpointManager",
    "CrossVQATrainer",
    "train_epoch",
    "run_evaluation",
]
