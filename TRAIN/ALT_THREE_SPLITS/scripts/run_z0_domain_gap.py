"""
Phase A (Tasks A8–A10) & Phase C Zero-Shot Anchor (Z0) Runner:
- Evaluates frozen base VLM across NEW_validation and/or NEW_test
- Extracts pooled visual-projector embeddings on N samples per domain (deduplicated by image_id)
- Computes Phase A8 2D UMAP projection + plot (saving image_id in umap_coords.parquet)
- Computes Phase A9 RKHS MMD & 5-fold Linear Proxy A-distance on balanced domains
- Non-destructively merges existing val_metrics.json when running partial modes

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/run_z0_domain_gap.py --model smolvlm_500m
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from _common import (
    ensure_repo_root,
    evaluate_both_domains,
    get_git_sha,
    load_yaml,
    set_seed,
)


class _ProjectorCaptured(Exception):
    """Sentinel raised inside the projector hook to skip downstream LLM layers."""


def _resolve_z0_eval_batch_size(hw: Dict[str, Any], load_in_4bit: bool, requested: Optional[int]) -> int:
    if requested is not None:
        return max(int(requested), 1)
    base_bs = int(hw["eval_batch_size"])
    if not load_in_4bit and hw.get("name") in ("l4_24gb", "a100_40gb", "g4_96gb"):
        return base_bs * 2
    return base_bs


def _resolve_num_workers(requested: Optional[int], local_data_dir: Optional[str]) -> int:
    if requested is not None:
        return max(int(requested), 0)
    if local_data_dir and torch.cuda.is_available() and sys.platform != "win32":
        return min(2, os.cpu_count() or 1)
    return 0


def _select_unique_image_indices(ds: Any, max_samples: int, seed: int = 42) -> Tuple[List[int], List[str]]:
    """Selects deterministic shuffled indices with at most one row per unique image_id."""
    if isinstance(ds, (list, tuple)):
        seen: Dict[str, int] = {}
        for i, item in enumerate(ds):
            key = str(item.get("image_id") or item.get("sample_id") or i)
            if key not in seen:
                seen[key] = i
    else:
        col_names = list(getattr(ds, "column_names", []) or [])
        if "image_id" in col_names:
            raw_ids = ds["image_id"]
        elif "sample_id" in col_names:
            raw_ids = ds["sample_id"]
        else:
            raw_ids = [str(i) for i in range(len(ds))]
        seen = {}
        for i, raw in enumerate(raw_ids):
            key = str(raw if raw is not None else i)
            if key not in seen:
                seen[key] = i

    keys = sorted(seen.keys())
    random.Random(int(seed)).shuffle(keys)
    keys = keys[: int(max_samples)]
    return [seen[k] for k in keys], keys


def extract_domain_embeddings(
    model,
    processor,
    model_type: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    batch_size: int,
    device: str,
    local_data_dir: Optional[str] = None,
    num_workers: int = 0,
    eval_dataset: Optional[Any] = None,
    bucket_by_resolution: bool = True,
    dedup_by_image_id: bool = True,
    seed: int = 42,
    return_keys: bool = False,
):
    """
    Extracts L2-normalized pooled visual projector embeddings using an early-exit
    forward hook so the downstream LLM layers are skipped entirely.
    """
    from TRAIN.src.adapters import _pool_visual_tokens, _resolve_projector_module
    from TRAIN.src.collator import CrossVQADataCollator
    from TRAIN.src.dataset import load_crossvqa_dataset
    from TRAIN.src.trainer import (
        _EvalCollateWrapper,
        _IndexedEvalDataset,
        _bucket_indices_by_resolution,
        _prepare_model_inputs,
    )

    _, proj_mod = _resolve_projector_module(model)
    captured: Dict[str, torch.Tensor] = {}

    def _hook(mod, inp, out):
        vis_t = out[0] if isinstance(out, (tuple, list)) else out
        captured["vis"] = vis_t.detach()
        raise _ProjectorCaptured()

    handle = proj_mod.register_forward_hook(_hook)
    collator = CrossVQADataCollator(processor, model_type=model_type, include_ocr=False, is_train=False)
    use_streaming = local_data_dir is None
    if eval_dataset is not None:
        ds = eval_dataset
    else:
        ds = load_crossvqa_dataset(
            dataset_name=dataset_name,
            split=split,
            split_regime="alt_three_splits",
            streaming=use_streaming,
            local_data_dir=local_data_dir,
        )

    embs: List[np.ndarray] = []
    keys_out: List[str] = []
    model.eval()
    from tqdm.auto import tqdm

    def _forward_capture(batch: Dict[str, Any], cur_bs: int) -> np.ndarray:
        captured.clear()
        inputs = _prepare_model_inputs(batch, device)
        with torch.inference_mode():
            try:
                _ = model(**inputs)
            except _ProjectorCaptured:
                pass
        if "vis" not in captured:
            raise RuntimeError("Projector hook did not capture visual features before forward completion.")
        pooled = _pool_visual_tokens(
            captured.pop("vis"),
            batch_size=cur_bs,
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            pixel_values=inputs.get("pixel_values"),
        )
        pooled_norm = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return pooled_norm.cpu().numpy()

    is_map_style = (
        hasattr(ds, "__len__")
        and hasattr(ds, "__getitem__")
        and not isinstance(ds, IterableDataset)
    )
    bs = max(int(batch_size), 1)

    try:
        if is_map_style:
            if dedup_by_image_id:
                chosen_indices, chosen_keys = _select_unique_image_indices(ds, max_samples=max_samples, seed=seed)
                if bucket_by_resolution and len(chosen_indices) > 1:
                    if hasattr(ds, "select"):
                        sub_ds = ds.select(chosen_indices)
                    else:
                        sub_ds = [ds[i] for i in chosen_indices]
                    rel_order = _bucket_indices_by_resolution(sub_ds, len(chosen_indices))
                    ordered_indices = [chosen_indices[r] for r in rel_order]
                else:
                    ordered_indices = chosen_indices
                key_by_orig = dict(zip(chosen_indices, chosen_keys))
            else:
                target_n = min(len(ds), max_samples)
                chosen_indices = list(range(target_n))
                ordered_indices = (
                    _bucket_indices_by_resolution(ds, target_n)
                    if bucket_by_resolution
                    else chosen_indices
                )
                key_by_orig = {i: str(i) for i in chosen_indices}

            indexed_ds = _IndexedEvalDataset(ds, ordered_indices)
            eff_workers = 0 if isinstance(ds, (list, tuple)) else max(int(num_workers), 0)
            dl_kwargs: Dict[str, Any] = {
                "batch_size": bs,
                "shuffle": False,
                "num_workers": eff_workers,
                "collate_fn": _EvalCollateWrapper(collator),
                "pin_memory": isinstance(device, str) and device.startswith("cuda"),
            }
            if eff_workers > 0:
                dl_kwargs["prefetch_factor"] = 2
            loader = DataLoader(indexed_ds, **dl_kwargs)
            indexed_vecs: Dict[int, np.ndarray] = {}
            for batch, meta_list in tqdm(loader, desc=f"Extracting {dataset_name} ({split}) embeddings", total=len(loader)):
                batch_vecs = _forward_capture(batch, len(meta_list))
                for row_i, m in enumerate(meta_list):
                    indexed_vecs[int(m["_orig_idx"])] = batch_vecs[row_i]
            ordered_present = [idx for idx in chosen_indices if idx in indexed_vecs]
            arr = (
                np.stack([indexed_vecs[idx] for idx in ordered_present], axis=0)
                if ordered_present
                else np.zeros((0, 1), dtype=np.float32)
            )
            keys_out = [key_by_orig[idx] for idx in ordered_present]
            return (arr, keys_out) if return_keys else arr
        else:
            buf = []
            seen_stream = set()
            for sample in tqdm(ds, desc=f"Extracting {dataset_name} ({split}) embeddings", total=max_samples):
                key = str(sample.get("image_id") or sample.get("sample_id") or len(keys_out) + len(buf))
                if dedup_by_image_id:
                    if key in seen_stream:
                        continue
                    seen_stream.add(key)
                buf.append(sample)
                keys_out.append(key)
                if len(keys_out) >= max_samples:
                    batch = collator(buf)
                    embs.append(_forward_capture(batch, len(buf)))
                    buf = []
                    break
                if len(buf) >= bs:
                    batch = collator(buf)
                    embs.append(_forward_capture(batch, len(buf)))
                    buf = []
            if buf:
                batch = collator(buf)
                embs.append(_forward_capture(batch, len(buf)))
    finally:
        handle.remove()

    arr = np.vstack(embs) if embs else np.zeros((0, 1), dtype=np.float32)
    return (arr, keys_out) if return_keys else arr


def run_umap_and_plot(
    doc_embs: np.ndarray,
    scene_embs: np.ndarray,
    run_dir: Path,
    model_key: str,
    doc_keys: Optional[List[str]] = None,
    scene_keys: Optional[List[str]] = None,
    n_neighbors: int = 30,
) -> None:
    import matplotlib.pyplot as plt

    x = np.vstack([doc_embs, scene_embs])
    labels = ["document_text"] * len(doc_embs) + ["scene_text"] * len(scene_embs)
    image_ids = list(doc_keys or [f"doc_{i}" for i in range(len(doc_embs))]) + list(
        scene_keys or [f"scene_{i}" for i in range(len(scene_embs))]
    )

    try:
        import umap

        eff_neighbors = min(int(n_neighbors), max(2, len(x) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=eff_neighbors, random_state=42, metric="cosine")
        coords = reducer.fit_transform(x)
        method_name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(x)
        method_name = "PCA (umap-learn unavailable)"

    df_coords = pl.DataFrame(
        {
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "domain_type": labels,
            "image_id": image_ids,
        }
    )
    df_coords.write_parquet(run_dir / "umap_coords.parquet")

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    doc_mask = np.array([l == "document_text" for l in labels])
    ax.scatter(coords[doc_mask, 0], coords[doc_mask, 1], s=10, alpha=0.55, label=f"DocVQA (n={len(doc_embs)})")
    ax.scatter(coords[~doc_mask, 0], coords[~doc_mask, 1], s=10, alpha=0.55, label=f"TextVQA (n={len(scene_embs)})")
    ax.set_title(f"{model_key} Multimodal Visual-Projector Latent Space ({method_name})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(run_dir / "umap_domain_gap.png")
    plt.close(fig)


def main() -> None:
    ensure_repo_root()
    from TRAIN.src.losses import compute_proxy_a_distance, compute_rkhs_mmd
    from TRAIN.src.metrics import save_run_card
    from TRAIN.src.models import build_vlm_model_and_processor
    from TRAIN.src.trainer import resolve_hardware_profile

    parser = argparse.ArgumentParser(description="Run Z0 zero-shot + Phase A8/A9 domain gap diagnostics.")
    parser.add_argument("--model", default="smolvlm_500m", choices=["smolvlm_500m", "smolvlm_base", "qwen2_vl_2b"])
    parser.add_argument("--hardware_profile", default="auto", choices=["auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb", "cpu"])
    parser.add_argument("--eval_splits", nargs="+", default=["validation"], help="Splits to evaluate VQA zero-shot on.")
    parser.add_argument("--domain_gap_only", action="store_true", help="Skip VQA text generation and run only Phase A8/A9 domain gap.")
    parser.add_argument("--domain_gap_split", default="validation", choices=["validation", "test", "train"])
    parser.add_argument("--domain_gap_samples", type=int, default=794)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--no_dedup_by_image_id", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--local_data_dir", default=None)
    parser.add_argument("--include_ocr", action="store_true")
    parser.add_argument("--skip_domain_gap", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true", help="Force 4-bit quantization for Z0 (off by default for faster pure-bf16/fp16 inference)")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_bucket_by_resolution", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    bb_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/backbones.yaml")
    base_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/baselines_B1_B5.yaml")
    hw = resolve_hardware_profile(args.hardware_profile, bb_cfg.get("hardware_profiles"))

    model_info = bb_cfg["backbones"][args.model]
    model_type = model_info["model_type"]
    suffix = "_ocr" if args.include_ocr else ""
    run_dir = Path(base_cfg["runs_root"]) / args.model / f"Z0{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    load_in_4bit = bool(args.load_in_4bit and model_info.get("allow_4bit", True))
    eval_batch_size = _resolve_z0_eval_batch_size(hw, load_in_4bit=load_in_4bit, requested=args.eval_batch_size)
    num_workers = _resolve_num_workers(args.num_workers, args.local_data_dir)
    bucket_by_resolution = not args.no_bucket_by_resolution
    dedup_by_image_id = not args.no_dedup_by_image_id
    active_eval_splits = (
        []
        if args.domain_gap_only
        else [s for s in args.eval_splits if s.lower() not in ("none", "skip", "")]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = build_vlm_model_and_processor(
        model_name_or_key=args.model,
        load_in_4bit=load_in_4bit,
        use_lora=False,
        architecture_mode="none",
        min_pixels=hw["min_pixels"],
        max_pixels=hw["max_pixels"],
        smolvlm_longest_edge=hw.get("smolvlm_longest_edge", 2048),
        gradient_checkpointing=False,
        attn_implementation=hw["attn_implementation"],
    )

    # Non-destructive merge: preserve existing splits / domain_gap from earlier runs
    all_results: Dict[str, Any] = {"splits": {}, "domain_gap": {}}
    existing_metrics_path = run_dir / "val_metrics.json"
    if existing_metrics_path.exists():
        try:
            loaded_prev = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded_prev, dict):
                all_results["splits"].update(loaded_prev.get("splits", {}))
                if isinstance(loaded_prev.get("domain_gap"), dict):
                    all_results["domain_gap"].update(loaded_prev["domain_gap"])
        except Exception:
            pass

    val_embeddings: Dict[str, Any] = {}
    for sp in active_eval_splits:
        print(f"\n=== Running Z0 Zero-Shot Evaluation on '{sp}' (bs={eval_batch_size}, 4bit={load_in_4bit}, workers={num_workers}, bucket={bucket_by_resolution}) ===")
        collect_here = (sp == args.domain_gap_split) and (not args.skip_domain_gap) and (not args.include_ocr)
        res = evaluate_both_domains(
            model=model,
            processor=processor,
            model_type=model_type,
            eval_split=sp,
            run_dir=run_dir,
            device=device,
            include_ocr=args.include_ocr,
            max_new_tokens=base_cfg["training"]["max_new_tokens"],
            eval_batch_size=eval_batch_size,
            max_eval_samples=args.max_eval_samples,
            local_data_dir=args.local_data_dir,
            num_workers=num_workers,
            bucket_by_resolution=bucket_by_resolution,
            embedding_out=val_embeddings if collect_here else None,
            domain_gap_samples=args.domain_gap_samples if collect_here else 0,
            dedup_by_image_id=dedup_by_image_id,
            seed=args.seed,
        )
        all_results["splits"][sp] = res

    fused_stage_b = False
    if not args.skip_domain_gap:
        can_reuse_val = (
            args.domain_gap_split in active_eval_splits
            and not args.include_ocr
            and "docvqa" in val_embeddings
            and "textvqa" in val_embeddings
            and len(val_embeddings["docvqa"]["embs"]) >= 10
            and len(val_embeddings["textvqa"]["embs"]) >= 10
            and (args.max_eval_samples is None or args.max_eval_samples >= args.domain_gap_samples)
        )
        if can_reuse_val:
            fused_stage_b = True
            doc_embs = val_embeddings["docvqa"]["embs"]
            doc_keys = val_embeddings["docvqa"]["keys"]
            scene_embs = val_embeddings["textvqa"]["embs"]
            scene_keys = val_embeddings["textvqa"]["keys"]
            print(f"\n=== Phase A8/A9: Reusing {len(doc_embs)} DocVQA + {len(scene_embs)} TextVQA visual embeddings captured during '{args.domain_gap_split}' pass ===")
        else:
            print(f"\n=== Computing Phase A8 UMAP & A9 MMD / Proxy A-distance on '{args.domain_gap_split}' (N={args.domain_gap_samples}/domain, dedup={dedup_by_image_id}, early-exit LLM skip) ===")
            doc_embs, doc_keys = extract_domain_embeddings(
                model, processor, model_type, "docvqa", args.domain_gap_split,
                args.domain_gap_samples, eval_batch_size, device, args.local_data_dir,
                num_workers=num_workers, bucket_by_resolution=bucket_by_resolution,
                dedup_by_image_id=dedup_by_image_id, seed=args.seed, return_keys=True,
            )
            scene_embs, scene_keys = extract_domain_embeddings(
                model, processor, model_type, "textvqa", args.domain_gap_split,
                args.domain_gap_samples, eval_batch_size, device, args.local_data_dir,
                num_workers=num_workers, bucket_by_resolution=bucket_by_resolution,
                dedup_by_image_id=dedup_by_image_id, seed=args.seed, return_keys=True,
            )

        # Balance class sizes before Proxy A-distance & MMD
        n_bal = min(len(doc_embs), len(scene_embs), int(args.domain_gap_samples))
        if n_bal >= 10:
            doc_embs, doc_keys = doc_embs[:n_bal], doc_keys[:n_bal]
            scene_embs, scene_keys = scene_embs[:n_bal], scene_keys[:n_bal]
            mmd_val = compute_rkhs_mmd(doc_embs, scene_embs, device=device)
            pad_stats = compute_proxy_a_distance(doc_embs, scene_embs, n_splits=5, seed=args.seed)
            run_umap_and_plot(
                doc_embs, scene_embs, run_dir, args.model,
                doc_keys=doc_keys, scene_keys=scene_keys, n_neighbors=args.umap_neighbors,
            )
            all_results["domain_gap"] = {
                "n_doc": int(len(doc_embs)),
                "n_scene": int(len(scene_embs)),
                "dedup_by_image_id": dedup_by_image_id,
                "rkhs_mmd": round(mmd_val, 6),
                **{k: round(float(v), 6) for k, v in pad_stats.items()},
            }
            (run_dir / "domain_gap_metrics.json").write_text(
                json.dumps(all_results["domain_gap"], indent=2), encoding="utf-8"
            )

    card_cfg = {
        "experiment": {"id": f"Z0_{args.model}{suffix}", "exp": "Z0", "seed": args.seed, "git_sha": get_git_sha()},
        "data": {"regime": "alt_three_splits", "revision": "three-splits", "include_ocr": args.include_ocr},
        "model": {"backbone": model_info["repo_id"], "architecture_mode": "frozen_zero_shot", "load_in_4bit": load_in_4bit},
        "training": {
            "hardware_profile": hw["name"],
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "bucket_by_resolution": bucket_by_resolution,
            "dedup_by_image_id": dedup_by_image_id,
            "fused_stage_b": fused_stage_b,
        },
    }
    save_run_card(str(run_dir), card_cfg, all_results)
    print(f"\nSaved Z0 + Domain Gap artifacts to {run_dir}")


if __name__ == "__main__":
    main()
