"""
Standardized training, checkpoint resilience, hardware profiling, and evaluation
utilities for CrossVQA baselines (Z0, B1–B5) and proposed methods (M1–M3).
"""

import io
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "eval_batch_size": 8,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 1024,
        "optim": "paged_adamw_8bit",
    },
    "a100_40gb": {
        "name": "a100_40gb",
        "dtype": "bf16",
        "load_in_4bit": True,
        "attn_implementation": "auto",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "eval_batch_size": 16,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1536 * 1024,
        "optim": "paged_adamw_8bit",
    },
    "g4_96gb": {
        "name": "g4_96gb",
        "dtype": "bf16",
        "load_in_4bit": False,
        "attn_implementation": "auto",
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 32,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 2048 * 1024,
        "optim": "adamw",
    },
    "t4_16gb": {
        "name": "t4_16gb",
        "dtype": "fp16",
        "load_in_4bit": True,
        "attn_implementation": "sdpa",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "eval_batch_size": 4,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1024 * 1024,
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

        if self.gdrive_backup_dir is not None:
            drive_target = self.gdrive_backup_dir / tag
            drive_target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target_dir, drive_target, dirs_exist_ok=True)

        if self.hf_repo_id and self.hf_token:
            from huggingface_hub import HfApi

            api = HfApi(token=self.hf_token)
            api.create_repo(repo_id=self.hf_repo_id, repo_type="model", private=self.private, exist_ok=True)
            api.upload_folder(
                folder_path=str(target_dir),
                repo_id=self.hf_repo_id,
                path_in_repo=f"{self.hf_path_in_repo}/{tag}",
                repo_type="model",
            )

        return target_dir

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
        model_type: str = "qwen2_vl",
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
    ) -> Dict[str, Any]:
        collator = CrossVQADataCollator(
            self.processor,
            model_type=self.model_type,
            include_ocr=self.include_ocr,
            is_train=True,
            use_soft_weights=self.use_soft_weights,
            image_transform=self.curriculum_transform,
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

        try:
            batches_per_epoch = len(dataloader)
        except TypeError:
            batches_per_epoch = max_steps_per_epoch or 1000
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

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            acc_total, acc_gen, acc_align, acc_gate, acc_router_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            micro_in_window = 0
            epoch_steps = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{epochs}",
                total=max_steps_per_epoch,
            )
            for step, batch in enumerate(pbar):
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

                hit_boundary = (step + 1) % self.gradient_accumulation_steps == 0
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
                        self.checkpoint_manager.save(
                            self.model,
                            tag="latest",
                            trainer_state={"epoch": epoch, "opt_step": global_opt_step},
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
                self.checkpoint_manager.save(self.model, tag="latest", trainer_state=state)
                score = float(val_metrics.get(monitor_metric, 0.0))
                if not val_metrics or score >= best_score:
                    best_score = score
                    best_metrics = dict(val_metrics)
                    self.checkpoint_manager.save(self.model, tag="best", trainer_state=state)

        elapsed_sec = round(time.time() - start_time, 2)
        return {
            "epochs": epochs,
            "opt_steps": global_opt_step,
            "elapsed_sec": elapsed_sec,
            "best_metrics": best_metrics,
        }
