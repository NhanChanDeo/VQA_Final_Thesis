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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


def resolve_checkpoint_paths(
    runs_root: str,
    model: str,
    exp_dir: str,
    seed: int,
    checkpoint_namespace: Optional[str] = None,
    few_shot_pct: int = 100,
    few_shot_tag: Optional[str] = None,
) -> Tuple[Path, str]:
    """Computes isolated local run directory and HF Hub / Drive subpath for a given namespace."""
    ns = (checkpoint_namespace or "").strip().strip("/")
    if int(few_shot_pct) < 100:
        leaf = f"{few_shot_tag or exp_dir}_pct_{int(few_shot_pct):02d}/seed_{int(seed)}"
        rel_parts = (model, ns, "few_shot", leaf) if ns else (model, "few_shot", leaf)
        hf_subpath = (
            f"alt_three_splits/{ns}/{model}/few_shot/{leaf}"
            if ns
            else f"alt_three_splits/{model}/few_shot/{leaf}"
        )
    else:
        leaf = f"{exp_dir}/seed_{int(seed)}"
        rel_parts = (model, ns, leaf) if ns else (model, leaf)
        hf_subpath = (
            f"alt_three_splits/{ns}/{model}/{leaf}"
            if ns
            else f"alt_three_splits/{model}/{leaf}"
        )
    run_rel = Path(runs_root).joinpath(*rel_parts)
    return run_rel, hf_subpath


def apply_training_overrides(
    hw: Dict[str, Any],
    tr_cfg: Dict[str, Any],
    args: Any,
) -> Tuple[int, int, int, float]:
    """Resolves per-device batch size, grad accum, eval batch size, and LR with optional CLI overrides."""
    batch_size = (
        int(args.batch_size)
        if getattr(args, "batch_size", None) is not None
        else int(hw["per_device_train_batch_size"])
    )
    grad_accum = (
        int(args.gradient_accumulation_steps)
        if getattr(args, "gradient_accumulation_steps", None) is not None
        else int(hw["gradient_accumulation_steps"])
    )
    eval_bs = (
        int(args.eval_batch_size)
        if getattr(args, "eval_batch_size", None) is not None
        else int(hw["eval_batch_size"])
    )
    lr = (
        float(args.learning_rate)
        if getattr(args, "learning_rate", None) is not None
        else float(tr_cfg["learning_rate"])
    )
    return batch_size, grad_accum, eval_bs, lr


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
    num_workers: Optional[int] = None,
    bucket_by_resolution: bool = True,
    embedding_out: Optional[Dict[str, Any]] = None,
    domain_gap_samples: int = 0,
    dedup_by_image_id: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    """Runs batched greedy generation across both DocVQA and TextVQA splits."""
    from TRAIN.src.adapters import _pool_visual_tokens, _resolve_projector_module
    from TRAIN.src.dataset import load_crossvqa_dataset
    from TRAIN.src.metrics import save_predictions_jsonl
    from TRAIN.src.trainer import run_evaluation

    use_streaming = local_data_dir is None
    if num_workers is None:
        eff_workers = (
            min(2, os.cpu_count() or 1)
            if (not use_streaming and torch.cuda.is_available() and sys.platform != "win32")
            else 0
        )
    else:
        eff_workers = max(int(num_workers), 0)

    collect_embs = embedding_out is not None and domain_gap_samples > 0
    proj_mod = _resolve_projector_module(model)[1] if collect_embs else None

    summary: Dict[str, float] = {}
    for ds_name in ("docvqa", "textvqa"):
        ds = load_crossvqa_dataset(
            dataset_name=ds_name,
            split=eval_split,
            split_regime=split_regime,
            manifest_path=manifest_path,
            streaming=use_streaming,
            local_data_dir=local_data_dir,
        )

        captured: Dict[str, torch.Tensor] = {}
        collected: Dict[Any, np.ndarray] = {}
        hook_handle = None
        on_batch = None

        if collect_embs and proj_mod is not None:
            def _hook(mod, inp, out):
                if (dedup_by_image_id or len(collected) < domain_gap_samples) and "vis" not in captured:
                    vis_t = out[0] if isinstance(out, (tuple, list)) else out
                    captured["vis"] = vis_t.detach()

            def _on_batch(meta_list, model_inputs):
                vis = captured.pop("vis", None)
                if vis is None:
                    return
                if not dedup_by_image_id and len(collected) >= domain_gap_samples:
                    return
                if dedup_by_image_id:
                    needed = [
                        (i, str(m.get("image_id") or m.get("sample_id") or m["_orig_idx"]))
                        for i, m in enumerate(meta_list)
                        if str(m.get("image_id") or m.get("sample_id") or m["_orig_idx"]) not in collected
                    ]
                else:
                    needed = [
                        (i, m["_orig_idx"])
                        for i, m in enumerate(meta_list)
                        if m["_orig_idx"] < domain_gap_samples and m["_orig_idx"] not in collected
                    ]
                if not needed:
                    return
                pooled = _pool_visual_tokens(
                    vis,
                    batch_size=len(meta_list),
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    pixel_attention_mask=model_inputs.get("pixel_attention_mask"),
                    pixel_values=model_inputs.get("pixel_values"),
                )
                pooled_norm = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()
                seen_in_batch = set()
                for batch_idx, key in needed:
                    if key in seen_in_batch:
                        continue
                    seen_in_batch.add(key)
                    collected[key] = pooled_norm[batch_idx]

            hook_handle = proj_mod.register_forward_hook(_hook)
            on_batch = _on_batch

        try:
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
                num_workers=eff_workers,
                bucket_by_resolution=bucket_by_resolution,
                on_batch_evaluated=on_batch,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if collect_embs and embedding_out is not None:
            if collected:
                keys = sorted(collected.keys())
                if dedup_by_image_id:
                    rng = random.Random(int(seed))
                    rng.shuffle(keys)
                keys = keys[: int(domain_gap_samples)]
                embedding_out[ds_name] = {
                    "embs": np.vstack([collected[k] for k in keys]),
                    "keys": [str(k) for k in keys],
                }
            else:
                embedding_out[ds_name] = {"embs": np.zeros((0, 1), dtype=np.float32), "keys": []}

        save_predictions_jsonl(details, str(run_dir / f"preds_{eval_split}_{ds_name}.jsonl"))
        summary[f"{ds_name}_anls"] = round(float(metrics.get("anls", 0.0)), 6)
        summary[f"{ds_name}_vqa_accuracy"] = round(float(metrics.get("vqa_accuracy", 0.0)), 6)

    doc_anls = summary["docvqa_anls"]
    tv_acc = summary["textvqa_vqa_accuracy"]
    summary["harmonic_mean"] = round(
        (2.0 * doc_anls * tv_acc) / max(doc_anls + tv_acc, 1e-8), 6
    )
    return summary
