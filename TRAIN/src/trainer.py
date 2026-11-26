"""
Standardized training, checkpoint resilience, hardware profiling, and evaluation
utilities for CrossVQA baselines (Z0, B1–B5) and proposed methods (M1–M3).
"""

import inspect
import io
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .collator import CrossVQADataCollator
from .metrics import evaluate_predictions, save_predictions_jsonl

NON_MODEL_BATCH_KEYS = {"domain_ids", "sample_weights"}

HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "l4_24gb": {
        "name": "l4_24gb",
        "dtype": "bf16",
        "load_in_4bit": True,
        "attn_implementation": "auto",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "eval_batch_size": 16,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 1024,
        "smolvlm_longest_edge": 2048,
        "optim": "paged_adamw_8bit",
    },
    "a100_40gb": {
        "name": "a100_40gb",
        "dtype": "bf16",
        "load_in_4bit": True,
        "attn_implementation": "auto",
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "eval_batch_size": 16,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1536 * 1024,
        "smolvlm_longest_edge": 2048,
        "optim": "paged_adamw_8bit",
    },
    "g4_96gb": {
        "name": "g4_96gb",
        "dtype": "bf16",
        "load_in_4bit": False,
        "attn_implementation": "auto",
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "eval_batch_size": 32,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 2048 * 1024,
        "smolvlm_longest_edge": 2048,
        "optim": "adamw",
    },
    "t4_16gb": {
        "name": "t4_16gb",
        "dtype": "fp16",
        "load_in_4bit": True,
        "attn_implementation": "sdpa",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
        "eval_batch_size": 8,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 1024,
        "smolvlm_longest_edge": 2048,
        "optim": "paged_adamw_8bit",
    },
    "cpu": {
        "name": "cpu",
        "dtype": "fp32",
        "load_in_4bit": False,
        "attn_implementation": "sdpa",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 2,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 512 * 512,
        "smolvlm_longest_edge": 512,
        "optim": "adamw",
    },
}


def resolve_hardware_profile(
    requested: str = "auto",
    custom_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Resolves hardware training profile by detected VRAM / compute capability:
    L4 (24GB, primary), A100 (40/80GB, second), G4 (96GB Blackwell), T4 (16GB fallback), or CPU.
    """
    merged = {k: dict(v) for k, v in HARDWARE_PROFILES.items()}
    if custom_profiles:
        for k, v in custom_profiles.items():
            if not isinstance(v, dict):
                continue
            if k in merged:
                merged[k].update(v)
            else:
                merged[k] = dict(v)

    req = requested.lower()
    if req != "auto":
        if req not in merged:
            raise ValueError(f"Unknown hardware profile '{requested}'. Available: {list(merged.keys())}")
        return merged[req]

    if not torch.cuda.is_available():
        return merged["cpu"]

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / float(1024 ** 3)
    sm_major = torch.cuda.get_device_capability(0)[0]
    if vram_gb >= 85.0 and sm_major >= 10:
        return merged["g4_96gb"]
    if vram_gb >= 36.0 and sm_major >= 8:
        return merged["a100_40gb"]
    if vram_gb >= 20.0 and sm_major >= 8:
        return merged["l4_24gb"]
    return merged["t4_16gb"]


class CheckpointManager:
    """
    Persists adapter/router checkpoints locally, mirrors to personal Google Drive
    when mounted, and syncs to a private Hugging Face Hub repository.
    """

    def __init__(
        self,
        local_ckpt_dir: str,
        hf_repo_id: Optional[str] = "gamusa/crossvqa-checkpoints",
        hf_path_in_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        gdrive_backup_dir: Optional[str] = None,
        private: bool = True,
    ) -> None:
        self.local_ckpt_dir = Path(local_ckpt_dir)
        self.local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id
        self.hf_path_in_repo = (hf_path_in_repo or "default").strip("/")
        self.hf_token = hf_token
        self.gdrive_backup_dir = Path(gdrive_backup_dir) if gdrive_backup_dir else None
        self.private = bool(private)

    def save(
        self,
        model,
        tag: str = "latest",
        trainer_state: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        target_dir = self.local_ckpt_dir / tag
        target_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(target_dir))
        else:
            torch.save(model.state_dict(), target_dir / "model_state.pt")

        if trainer_state is not None:
            (target_dir / "trainer_state.json").write_text(
                json.dumps(trainer_state, indent=2), encoding="utf-8"
            )

        if resume_state is not None and tag == "latest":
            torch.save(resume_state, target_dir / "resume_state.pt")

        if self.gdrive_backup_dir is not None:
            drive_target = self.gdrive_backup_dir / tag
            drive_target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target_dir, drive_target, dirs_exist_ok=True)

        if self.hf_repo_id and self.hf_token:
            from huggingface_hub import HfApi

            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.hf_repo_id, repo_type="model", private=self.private, exist_ok=True)
            upload_kwargs: Dict[str, Any] = {
                "folder_path": str(target_dir),
                "repo_id": self.hf_repo_id,
                "path_in_repo": f"{self.hf_path_in_repo}/{tag}",
                "repo_type": "model",
            }
            if tag == "latest":
                upload_kwargs["delete_patterns"] = "*"
            api.upload_folder(**upload_kwargs)

        return target_dir

    @staticmethod
    def load_resume_state(ckpt_dir: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Loads resume_state.pt from a checkpoint directory if present, else returns None."""
        if not ckpt_dir:
            return None
        state_path = Path(ckpt_dir) / "resume_state.pt"
        if not state_path.exists():
            return None
        return torch.load(state_path, map_location="cpu", weights_only=False)

    def restore_checkpoint(self, tag: str = "latest") -> Optional[Path]:
        local_candidate = self.local_ckpt_dir / tag
        if local_candidate.exists() and any(local_candidate.iterdir()):
            return local_candidate

        if self.hf_repo_id and self.hf_token:
            try:
                from huggingface_hub import snapshot_download

                prefix = f"{self.hf_path_in_repo}/{tag}"
                dl_root = Path(
                    snapshot_download(
                        repo_id=self.hf_repo_id,
                        repo_type="model",
                        allow_patterns=f"{prefix}/**",
                        token=self.hf_token,
                    )
                )
                sub = dl_root / prefix
                if sub.exists() and any(sub.iterdir()):
                    local_candidate.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(sub, local_candidate, dirs_exist_ok=True)
                    return local_candidate
            except Exception as exc:
                print(f"Notice: Could not restore '{tag}' from HF Hub ({exc}); checking Drive fallback...")

        if self.gdrive_backup_dir is not None:
            drive_candidate = self.gdrive_backup_dir / tag
            if drive_candidate.exists() and any(drive_candidate.iterdir()):
                local_candidate.mkdir(parents=True, exist_ok=True)
                shutil.copytree(drive_candidate, local_candidate, dirs_exist_ok=True)
                return local_candidate

        return None


def _prepare_model_inputs(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Filters out metadata keys and moves tensor inputs onto target device."""
    non_blocking = isinstance(device, str) and device.startswith("cuda")
    model_inputs = {}
    for key, val in batch.items():
        if key in NON_MODEL_BATCH_KEYS:
            continue
        model_inputs[key] = (
            val.to(device, non_blocking=non_blocking) if isinstance(val, torch.Tensor) else val
        )
    return model_inputs


def _compute_weighted_causal_lm_loss(
    outputs,
    labels: Optional[torch.Tensor],
    sample_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if sample_weights is None or labels is None or not hasattr(outputs, "logits"):
        return outputs.loss
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid_mask = (shift_labels != -100).float()
    per_sample = (per_tok * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
    weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).mean()


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


class _IndexedEvalDataset(Dataset):
    """Map-style dataset wrapper that tracks original sample indices across resolution bucketing."""

    def __init__(self, dataset: Any, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, Any]]:
        orig_idx = self.indices[idx]
        return orig_idx, self.dataset[orig_idx]


class _EvalCollateWrapper:
    """Picklable DataLoader collate wrapper returning model tensor batch plus lightweight metadata."""

    def __init__(self, collator: CrossVQADataCollator) -> None:
        self.collator = collator

    def __call__(self, items: List[Tuple[int, Dict[str, Any]]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        samples = [sample for _, sample in items]
        batch = self.collator(samples)
        meta = [
            {
                "_orig_idx": orig_idx,
                "sample_id": sample.get("sample_id", ""),
                "image_id": sample.get("image_id", ""),
                "domain_type": sample.get("domain_type", ""),
                "question": sample.get("question", ""),
                "answers": sample.get("answers") or [sample.get("canonical_answer", "")],
            }
            for orig_idx, sample in items
        ]
        return batch, meta


def _image_size_from_payload(payload: Any) -> int:
    if isinstance(payload, Image.Image):
        return int(payload.width) * int(payload.height)
    if isinstance(payload, dict):
        if payload.get("bytes") is not None:
            with Image.open(io.BytesIO(payload["bytes"])) as im:
                return int(im.width) * int(im.height)
        if payload.get("path"):
            with Image.open(payload["path"]) as im:
                return int(im.width) * int(im.height)
    if isinstance(payload, bytes):
        with Image.open(io.BytesIO(payload)) as im:
            return int(im.width) * int(im.height)
    return 0


def _bucket_indices_by_resolution(dataset: Any, count: int) -> List[int]:
    """
    Returns dataset indices [0..count-1] sorted by image pixel area (ascending)
    using metadata columns or lazy image header parsing (no full pixel decode).
    """
    indices = list(range(count))
    if count <= 1:
        return indices
    try:
        col_names = list(getattr(dataset, "column_names", []) or [])
        for w_col, h_col in (("image_width", "image_height"), ("width", "height")):
            if w_col in col_names and h_col in col_names:
                sub = dataset.select(indices) if hasattr(dataset, "select") else dataset
                widths = sub[w_col][:count]
                heights = sub[h_col][:count]
                areas = [int(w) * int(h) for w, h in zip(widths, heights)]
                return sorted(indices, key=lambda i: areas[i])

        if hasattr(dataset, "cast_column") and "image" in getattr(dataset, "features", {}):
            from datasets import Image as HFImage

            raw_ds = dataset.select(indices) if hasattr(dataset, "select") else dataset
            raw_ds = raw_ds.cast_column("image", HFImage(decode=False))
            areas = [_image_size_from_payload(p) for p in raw_ds["image"]]
            return sorted(indices, key=lambda i: areas[i])

        if isinstance(dataset, (list, tuple)):
            areas = [_image_size_from_payload(dataset[i].get("image")) for i in indices]
            return sorted(indices, key=lambda i: areas[i])
    except Exception:
        pass
    return indices


from contextlib import contextmanager


@contextmanager
def _temporary_use_cache(model: Any, enabled: bool):
    """Temporarily sets `use_cache` on model/backbone configs (including `text_config`) and restores prior values."""
    saved: List[Tuple[Any, bool]] = []
    seen_ids = set()

    def _visit_cfg(cfg: Any) -> None:
        if cfg is None or id(cfg) in seen_ids:
            return
        seen_ids.add(id(cfg))
        if hasattr(cfg, "use_cache"):
            saved.append((cfg, bool(cfg.use_cache)))
            cfg.use_cache = enabled
        _visit_cfg(getattr(cfg, "text_config", None))

    candidates = [model, getattr(model, "backbone", None), getattr(model, "base_model", None)]
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    for obj in candidates:
        if obj is not None:
            _visit_cfg(getattr(obj, "config", None))

    try:
        yield
    finally:
        for cfg, prev in saved:
            cfg.use_cache = prev


def run_evaluation(
    model,
    processor,
    eval_dataset,
    device: str = "cuda",
    max_new_tokens: int = 32,
    metric_type: str = "both",
    max_eval_samples: Optional[int] = None,
    model_type: str = "smolvlm",
    include_ocr: bool = False,
    max_ocr_tokens: int = 50,
    eval_batch_size: int = 4,
    collator: Optional[CrossVQADataCollator] = None,
    return_details: bool = False,
    num_workers: int = 0,
    bucket_by_resolution: bool = True,
    on_batch_evaluated: Optional[Callable[[List[Dict[str, Any]], Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, float], List[Any]]:
    """
    Evaluates batched greedy autoregressive generation against ground-truth answers.
    Uses DataLoader prefetching, resolution bucketing, and torch.inference_mode() when
    given a map-style dataset while preserving original sample ordering.
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
    orig_order: List[int] = []
    tokenizer = getattr(processor, "tokenizer", processor)

    def _run_batch(batch: Dict[str, Any], meta_list: List[Dict[str, Any]]) -> None:
        model_inputs = _prepare_model_inputs(batch, device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        if on_batch_evaluated is not None:
            on_batch_evaluated(meta_list, model_inputs)

        input_len = model_inputs["input_ids"].shape[1]
        gen_tokens = generated_ids[:, input_len:]
        decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for meta, pred_raw in zip(meta_list, decoded_batch):
            pred_text = pred_raw.strip()
            gt_list = meta["answers"]
            orig_order.append(meta["_orig_idx"])
            predictions.append(pred_text)
            ground_truths.append(gt_list)
            if return_details:
                details.append(
                    {
                        "sample_id": meta["sample_id"],
                        "domain_type": meta["domain_type"],
                        "question": meta["question"],
                        "prediction": pred_text,
                        "answers": gt_list,
                    }
                )

    is_map_style = (
        hasattr(eval_dataset, "__len__")
        and hasattr(eval_dataset, "__getitem__")
        and not isinstance(eval_dataset, IterableDataset)
    )
    bs = max(eval_batch_size, 1)

    with _temporary_use_cache(model, True):
        if is_map_style:
            total_avail = len(eval_dataset)
            eval_count = min(total_avail, max_eval_samples) if max_eval_samples is not None else total_avail
            indices = _bucket_indices_by_resolution(eval_dataset, eval_count) if bucket_by_resolution else list(range(eval_count))
            wrapped_ds = _IndexedEvalDataset(eval_dataset, indices)
            eff_workers = 0 if isinstance(eval_dataset, (list, tuple)) else max(int(num_workers), 0)
            dl_kwargs: Dict[str, Any] = {
                "batch_size": bs,
                "shuffle": False,
                "num_workers": eff_workers,
                "collate_fn": _EvalCollateWrapper(collator),
                "pin_memory": isinstance(device, str) and device.startswith("cuda"),
            }
            if eff_workers > 0:
                dl_kwargs["prefetch_factor"] = 2
                dl_kwargs["persistent_workers"] = False

            loader = DataLoader(wrapped_ds, **dl_kwargs)
            for batch, meta_list in tqdm(loader, desc="Evaluating", total=len(loader)):
                _run_batch(batch, meta_list)

            if bucket_by_resolution and len(orig_order) > 1:
                restore_perm = sorted(range(len(orig_order)), key=lambda i: orig_order[i])
                predictions = [predictions[i] for i in restore_perm]
                ground_truths = [ground_truths[i] for i in restore_perm]
                if return_details:
                    details = [details[i] for i in restore_perm]
        else:
            sample_buffer: List[Tuple[int, Dict[str, Any]]] = []
            count = 0
            collate_wrapper = _EvalCollateWrapper(collator)
            pbar = tqdm(eval_dataset, desc="Evaluating", total=max_eval_samples)
            for sample in pbar:
                if max_eval_samples is not None and count >= max_eval_samples:
                    break
                sample_buffer.append((count, sample))
                count += 1
                if len(sample_buffer) >= bs:
                    batch, meta_list = collate_wrapper(sample_buffer)
                    _run_batch(batch, meta_list)
                    sample_buffer = []

            if sample_buffer:
                batch, meta_list = collate_wrapper(sample_buffer)
                _run_batch(batch, meta_list)

    metrics = evaluate_predictions(predictions, ground_truths, metric_type=metric_type)
    if return_details:
        return metrics, details
    return metrics, predictions


class _DeterministicEpochSubset(Dataset):
    """Map-style dataset view over a deterministic per-epoch index slice for O(1) resume."""

    def __init__(self, dataset: Any, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass


def _move_optimizer_state_to_device(optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor) and k != "step":
                    st[k] = v.to(device=p.device)


def _validate_resume_fingerprint(
    saved_fp: Optional[Dict[str, Any]],
    current_fp: Optional[Dict[str, Any]],
    saved_stage: Optional[str],
    current_stage: str,
    allow_mismatch: bool = False,
) -> None:
    if allow_mismatch:
        return
    if saved_stage and saved_stage != current_stage:
        raise ValueError(
            f"Resume stage mismatch: checkpoint is for stage '{saved_stage}', "
            f"but current trainer is running stage '{current_stage}'."
        )
    if not saved_fp or not current_fp:
        return
    for key in ("effective_batch_size", "model", "exp"):
        if key in saved_fp and key in current_fp and saved_fp[key] != current_fp[key]:
            raise ValueError(
                f"Resume config mismatch on '{key}': checkpoint has {saved_fp[key]!r}, "
                f"current run has {current_fp[key]!r}. Use a separate checkpoint_namespace "
                "or pass --allow_mismatched_resume."
            )


class CrossVQATrainer:
    """
    End-to-end trainer orchestrating Z0/B1–B5 and M1–M3 optimization,
    composite alignment + router losses, curriculum scheduling, JSONL logging,
    validation checkpoint selection, and HF Hub + Google Drive checkpoint sync.
    """

    def __init__(
        self,
        model,
        processor,
        run_dir: str,
        model_type: str = "smolvlm",
        device: Optional[str] = None,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        optim_name: str = "paged_adamw_8bit",
        include_ocr: bool = False,
        use_soft_weights: bool = False,
        align_loss_fn=None,
        lambda_align: float = 0.0,
        lambda_gate: float = 0.0,
        curriculum_transform=None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.gradient_accumulation_steps = int(max(gradient_accumulation_steps, 1))
        self.max_grad_norm = float(max_grad_norm)
        self.optim_name = optim_name
        self.include_ocr = include_ocr
        self.use_soft_weights = use_soft_weights
        self.align_loss_fn = align_loss_fn
        self.lambda_align = float(lambda_align)
        self.lambda_gate = float(lambda_gate)
        self.curriculum_transform = curriculum_transform
        self.checkpoint_manager = checkpoint_manager
        self.log_path = self.run_dir / "train_log.jsonl"

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optim_name == "paged_adamw_8bit" and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb

                return bnb.optim.PagedAdamW8bit(params, lr=self.lr, weight_decay=self.weight_decay)
            except Exception:
                pass
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _append_log(self, entry: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _save_checkpoint(
        self,
        tag: str,
        trainer_state: Dict[str, Any],
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.checkpoint_manager is None:
            return
        sig_params = inspect.signature(self.checkpoint_manager.save).parameters
        if "resume_state" in sig_params:
            self.checkpoint_manager.save(
                self.model,
                tag=tag,
                trainer_state=trainer_state,
                resume_state=resume_state,
            )
        else:
            target_dir = self.checkpoint_manager.save(
                self.model,
                tag=tag,
                trainer_state=trainer_state,
            )
            if resume_state is not None and tag == "latest" and target_dir is not None:
                torch.save(resume_state, Path(target_dir) / "resume_state.pt")

    def fit(
        self,
        train_dataset,
        epochs: int = 2,
        batch_size: int = 4,
        max_steps_per_epoch: Optional[int] = None,
        eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        monitor_metric: str = "selection_score",
        forced_domain_id: Optional[int] = None,
        curriculum_start: float = 0.2,
        curriculum_end: float = 1.0,
        save_every_steps: Optional[int] = None,
        estimated_batches_per_epoch: Optional[int] = None,
        seed: int = 42,
        resume_from: Optional[str] = None,
        stage_name: str = "main",
        config_fingerprint: Optional[Dict[str, Any]] = None,
        save_optimizer_state: bool = True,
        allow_mismatched_resume: bool = False,
    ) -> Dict[str, Any]:
        collator = CrossVQADataCollator(
            self.processor,
            model_type=self.model_type,
            include_ocr=self.include_ocr,
            is_train=True,
            use_soft_weights=self.use_soft_weights,
            image_transform=self.curriculum_transform,
        )
        is_map_style = (
            hasattr(train_dataset, "__len__")
            and hasattr(train_dataset, "__getitem__")
            and not isinstance(train_dataset, IterableDataset)
        )
        eff_workers = (
            min(2, os.cpu_count() or 1)
            if (is_map_style and torch.cuda.is_available() and sys.platform != "win32")
            else 0
        )

        try:
            batches_per_epoch = math.ceil(len(train_dataset) / max(1, int(batch_size)))
        except TypeError:
            batches_per_epoch = estimated_batches_per_epoch or max_steps_per_epoch or 1000
        if max_steps_per_epoch is not None:
            batches_per_epoch = min(batches_per_epoch, max_steps_per_epoch)

        total_opt_steps = max(
            1, math.ceil((batches_per_epoch * epochs) / self.gradient_accumulation_steps)
        )
        warmup_steps = int(total_opt_steps * self.warmup_ratio)

        optimizer = self._build_optimizer()
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps
        )

        best_score = -1.0
        best_metrics: Dict[str, float] = {}
        global_opt_step = 0
        total_micro_steps = max(1, batches_per_epoch * epochs)
        micro_step_idx = 0
        start_epoch = 1
        resume_skip_samples = 0

        if resume_from:
            r_state = CheckpointManager.load_resume_state(resume_from)
            if r_state is None:
                print(
                    f"Notice: No resume_state.pt found in '{resume_from}'; "
                    "falling back to weight-only warm start from Epoch 1."
                )
            else:
                _validate_resume_fingerprint(
                    saved_fp=r_state.get("config_fingerprint"),
                    current_fp=config_fingerprint,
                    saved_stage=r_state.get("stage"),
                    current_stage=stage_name,
                    allow_mismatch=allow_mismatched_resume,
                )
                optimizer.load_state_dict(r_state["optimizer_state_dict"])
                _move_optimizer_state_to_device(optimizer)
                scheduler.load_state_dict(r_state["scheduler_state_dict"])
                if self.align_loss_fn is not None and hasattr(self.align_loss_fn, "load_queue_state"):
                    self.align_loss_fn.load_queue_state(r_state.get("align_queue_state"))
                best_score = float(r_state.get("best_score", -1.0))
                best_metrics = dict(r_state.get("best_metrics", {}))
                global_opt_step = int(r_state.get("opt_step", 0))
                micro_step_idx = int(r_state.get("micro_step_idx", 0))
                saved_epoch = int(r_state.get("epoch", 1))
                epoch_completed = bool(r_state.get("epoch_completed", False))
                if epoch_completed:
                    start_epoch = saved_epoch + 1
                    resume_skip_samples = 0
                else:
                    start_epoch = saved_epoch
                    resume_skip_samples = int(r_state.get("samples_seen_in_epoch", 0))
                _restore_rng_state(r_state.get("rng_state"))
                print(
                    f"Resumed stage '{stage_name}' from epoch={start_epoch}/{epochs}, "
                    f"opt_step={global_opt_step}, skip_samples={resume_skip_samples}, "
                    f"lr={scheduler.get_last_lr()[0]:.6g}"
                )
                if start_epoch > epochs:
                    print(f"Stage '{stage_name}' already completed ({saved_epoch}/{epochs} epochs). Skipping.")
                    return {
                        "epochs": epochs,
                        "opt_steps": global_opt_step,
                        "elapsed_sec": 0.0,
                        "best_metrics": best_metrics,
                        "skipped_already_completed": True,
                    }

        def _build_resume_payload(ep: int, samples_seen: int, ep_done: bool) -> Optional[Dict[str, Any]]:
            if not save_optimizer_state:
                return None
            payload: Dict[str, Any] = {
                "stage": stage_name,
                "epoch": ep,
                "epoch_completed": ep_done,
                "samples_seen_in_epoch": samples_seen,
                "opt_step": global_opt_step,
                "micro_step_idx": micro_step_idx,
                "best_score": best_score,
                "best_metrics": dict(best_metrics),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": _capture_rng_state(),
                "config_fingerprint": dict(config_fingerprint or {}),
            }
            if self.align_loss_fn is not None and hasattr(self.align_loss_fn, "get_queue_state"):
                payload["align_queue_state"] = self.align_loss_fn.get_queue_state()
            return payload

        start_time = time.time()
        for epoch in range(start_epoch, epochs + 1):
            skip_here = resume_skip_samples if epoch == start_epoch else 0
            samples_seen_in_epoch = skip_here
            epoch_steps = (skip_here // max(1, int(batch_size))) if skip_here > 0 else 0

            dl_kwargs: Dict[str, Any] = {
                "batch_size": batch_size,
                "shuffle": False,
                "collate_fn": collator,
                "pin_memory": torch.cuda.is_available(),
                "num_workers": eff_workers,
            }
            if eff_workers > 0:
                dl_kwargs["prefetch_factor"] = 2

            if is_map_style:
                perm_gen = torch.Generator()
                perm_gen.manual_seed(int(seed) + (epoch - 1) * 1009)
                perm = torch.randperm(len(train_dataset), generator=perm_gen).tolist()
                epoch_ds = _DeterministicEpochSubset(train_dataset, perm[skip_here:])
                dataloader = DataLoader(epoch_ds, **dl_kwargs)
            else:
                manual_skip = False
                if hasattr(train_dataset, "set_epoch"):
                    try:
                        train_dataset.set_epoch(epoch - 1, skip_samples=skip_here)
                    except TypeError:
                        train_dataset.set_epoch(epoch - 1)
                        manual_skip = skip_here > 0
                elif skip_here > 0:
                    manual_skip = True
                if manual_skip:
                    dataloader = DataLoader(itertools.islice(train_dataset, skip_here, None), **dl_kwargs)
                else:
                    dataloader = DataLoader(train_dataset, **dl_kwargs)

            self.model.train()
            optimizer.zero_grad()
            acc_total, acc_gen, acc_align, acc_gate, acc_router_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            micro_in_window = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{epochs}",
                total=max_steps_per_epoch or batches_per_epoch,
                initial=epoch_steps,
            )
            for batch in pbar:
                if max_steps_per_epoch is not None and epoch_steps >= max_steps_per_epoch:
                    break

                if self.curriculum_transform is not None:
                    sev = self.curriculum_transform.linear_schedule(
                        micro_step_idx, total_micro_steps, start=curriculum_start, end=curriculum_end
                    )
                    self.curriculum_transform.set_severity(sev)

                domain_ids = batch.get("domain_ids")
                sample_weights = batch.get("sample_weights")
                model_inputs = _prepare_model_inputs(batch, self.device)
                batch_n = int(model_inputs["input_ids"].shape[0]) if "input_ids" in model_inputs else int(batch_size)

                if forced_domain_id is not None and domain_ids is not None:
                    forced_tensor = torch.full_like(domain_ids, int(forced_domain_id))
                    outputs = self.model(**model_inputs, forced_domain_ids=forced_tensor)
                else:
                    outputs = self.model(**model_inputs)

                loss_gen = _compute_weighted_causal_lm_loss(
                    outputs, model_inputs.get("labels"), sample_weights
                )
                loss_align = loss_gen.new_tensor(0.0)
                loss_gate = loss_gen.new_tensor(0.0)
                router_acc = 0.0

                routing = getattr(self.model, "last_routing_outputs", None)
                if routing and domain_ids is not None:
                    d_ids = domain_ids.to(device=loss_gen.device, dtype=torch.long)
                    gate_logits = routing.get("gate_logits")
                    pooled_vis = routing.get("pooled_vis")
                    if gate_logits is not None and self.lambda_gate > 0.0:
                        loss_gate = F.cross_entropy(gate_logits.float(), d_ids)
                        router_acc = float((gate_logits.argmax(dim=-1) == d_ids).float().mean().item())
                    if self.align_loss_fn is not None and pooled_vis is not None and self.lambda_align > 0.0:
                        loss_align = self.align_loss_fn(pooled_vis, d_ids)

                loss_total = loss_gen + self.lambda_align * loss_align + self.lambda_gate * loss_gate
                (loss_total / self.gradient_accumulation_steps).backward()

                acc_total += float(loss_total.item())
                acc_gen += float(loss_gen.item())
                acc_align += float(loss_align.item())
                acc_gate += float(loss_gate.item())
                acc_router_acc += router_acc
                micro_in_window += 1
                epoch_steps += 1
                micro_step_idx += 1
                samples_seen_in_epoch += batch_n

                hit_boundary = micro_in_window >= self.gradient_accumulation_steps
                hit_end = max_steps_per_epoch is not None and epoch_steps >= max_steps_per_epoch
                if hit_boundary or hit_end:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_opt_step += 1

                    gpu_mem_gb = (
                        torch.cuda.max_memory_allocated() / float(1024 ** 3)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    log_entry = {
                        "epoch": epoch,
                        "opt_step": global_opt_step,
                        "loss_total": acc_total / micro_in_window,
                        "loss_gen": acc_gen / micro_in_window,
                        "loss_align": acc_align / micro_in_window,
                        "loss_gate": acc_gate / micro_in_window,
                        "router_acc": acc_router_acc / micro_in_window,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "gpu_mem_gb": round(gpu_mem_gb, 3),
                    }
                    self._append_log(log_entry)
                    pbar.set_postfix({"loss": f"{log_entry['loss_total']:.4f}"})
                    acc_total = acc_gen = acc_align = acc_gate = acc_router_acc = 0.0
                    micro_in_window = 0

                    if (
                        self.checkpoint_manager is not None
                        and save_every_steps is not None
                        and global_opt_step % save_every_steps == 0
                    ):
                        self._save_checkpoint(
                            tag="latest",
                            trainer_state={"epoch": epoch, "opt_step": global_opt_step},
                            resume_state=_build_resume_payload(
                                ep=epoch,
                                samples_seen=samples_seen_in_epoch,
                                ep_done=False,
                            ),
                        )

            # Flush remainder at end of epoch
            if micro_in_window > 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_opt_step += 1

            # Per-epoch validation & checkpointing
            val_metrics = eval_fn() if eval_fn is not None else {}
            if self.checkpoint_manager is not None:
                state = {"epoch": epoch, "opt_step": global_opt_step, "val_metrics": val_metrics}
                score = float(val_metrics.get(monitor_metric, 0.0))
                is_best = (not val_metrics) or (score >= best_score)
                if is_best:
                    best_score = score
                    best_metrics = dict(val_metrics)
                self._save_checkpoint(
                    tag="latest",
                    trainer_state=state,
                    resume_state=_build_resume_payload(
                        ep=epoch,
                        samples_seen=0,
                        ep_done=True,
                    ),
                )
                if is_best:
                    self._save_checkpoint(
                        tag="best",
                        trainer_state=state,
                        resume_state=None,
                    )

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "epochs": epochs,
            "opt_steps": global_opt_step,
            "elapsed_sec": elapsed_sec,
            "best_metrics": best_metrics,
        }
