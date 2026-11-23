"""
Phase A (Tasks A8–A10) & Phase C Zero-Shot Anchor (Z0) Runner:
- Evaluates frozen base VLM across NEW_validation and NEW_test
- Extracts pooled visual-projector embeddings on N samples per domain
- Computes Phase A8 2D UMAP projection + plot
- Computes Phase A9 RKHS MMD & 5-fold Linear Proxy A-distance

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/run_z0_domain_gap.py --model qwen2_vl_2b
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import polars as pl
import torch

from _common import (
    ensure_repo_root,
    evaluate_both_domains,
    get_git_sha,
    load_yaml,
    set_seed,
)


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
) -> np.ndarray:
    """Extracts L2-normalized pooled visual projector embeddings using a forward hook."""
    from TRAIN.src.adapters import _pool_visual_tokens, _resolve_projector_module
    from TRAIN.src.collator import CrossVQADataCollator
    from TRAIN.src.dataset import load_crossvqa_dataset
    from TRAIN.src.trainer import _prepare_model_inputs

    _, proj_mod = _resolve_projector_module(model)
    captured = {}

    def _hook(mod, inp, out):
        captured["vis"] = out

    handle = proj_mod.register_forward_hook(_hook)
    collator = CrossVQADataCollator(processor, model_type=model_type, include_ocr=False, is_train=False)
    ds = load_crossvqa_dataset(
        dataset_name=dataset_name,
        split=split,
        split_regime="alt_three_splits",
        streaming=True,
        local_data_dir=local_data_dir,
    )

    embs: List[np.ndarray] = []
    buf = []
    count = 0
    model.eval()
    from tqdm.auto import tqdm

    try:
        for sample in tqdm(ds, desc=f"Extracting {dataset_name} embeddings", total=max_samples):
            if count >= max_samples:
                break
            buf.append(sample)
            count += 1
            if len(buf) >= batch_size or count >= max_samples:
                batch = collator(buf)
                inputs = _prepare_model_inputs(batch, device)
                with torch.no_grad():
                    _ = model(**inputs)
                pooled = _pool_visual_tokens(
                    captured["vis"],
                    batch_size=len(buf),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )
                pooled_norm = torch.nn.functional.normalize(pooled.float(), dim=-1)
                embs.append(pooled_norm.cpu().numpy())
                buf = []
    finally:
        handle.remove()

    return np.vstack(embs) if embs else np.zeros((0, 1), dtype=np.float32)


def run_umap_and_plot(doc_embs: np.ndarray, scene_embs: np.ndarray, run_dir: Path, model_key: str) -> None:
    import matplotlib.pyplot as plt

    x = np.vstack([doc_embs, scene_embs])
    labels = ["document_text"] * len(doc_embs) + ["scene_text"] * len(scene_embs)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
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
        }
    )
    df_coords.write_parquet(run_dir / "umap_coords.parquet")

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    doc_mask = np.array([l == "document_text" for l in labels])
    ax.scatter(coords[doc_mask, 0], coords[doc_mask, 1], s=10, alpha=0.55, label="DocVQA (document_text)")
    ax.scatter(coords[~doc_mask, 0], coords[~doc_mask, 1], s=10, alpha=0.55, label="TextVQA (scene_text)")
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
    parser.add_argument("--model", default="qwen2_vl_2b", choices=["qwen2_vl_2b", "smolvlm_base", "smolvlm_500m"])
    parser.add_argument("--hardware_profile", default="auto", choices=["auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb", "cpu"])
    parser.add_argument("--eval_splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--domain_gap_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--local_data_dir", default=None)
    parser.add_argument("--include_ocr", action="store_true")
    parser.add_argument("--skip_domain_gap", action="store_true")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = build_vlm_model_and_processor(
        model_name_or_key=args.model,
        load_in_4bit=bool(hw.get("load_in_4bit", base_cfg["training"]["load_in_4bit"])),
        use_lora=False,
        architecture_mode="none",
        min_pixels=hw["min_pixels"],
        max_pixels=hw["max_pixels"],
        gradient_checkpointing=False,
        attn_implementation=hw["attn_implementation"],
    )

    all_results = {"splits": {}, "domain_gap": {}}
    for sp in args.eval_splits:
        print(f"\n=== Running Z0 Zero-Shot Evaluation on '{sp}' ===")
        res = evaluate_both_domains(
            model=model,
            processor=processor,
            model_type=model_type,
            eval_split=sp,
            run_dir=run_dir,
            device=device,
            include_ocr=args.include_ocr,
            max_new_tokens=base_cfg["training"]["max_new_tokens"],
            eval_batch_size=hw["eval_batch_size"],
            max_eval_samples=args.max_eval_samples,
            local_data_dir=args.local_data_dir,
        )
        all_results["splits"][sp] = res

    if not args.skip_domain_gap:
        print(f"\n=== Computing Phase A8 UMAP & A9 MMD / Proxy A-distance (N={args.domain_gap_samples}/domain) ===")
        doc_embs = extract_domain_embeddings(
            model, processor, model_type, "docvqa", "validation",
            args.domain_gap_samples, hw["eval_batch_size"], device, args.local_data_dir
        )
        scene_embs = extract_domain_embeddings(
            model, processor, model_type, "textvqa", "validation",
            args.domain_gap_samples, hw["eval_batch_size"], device, args.local_data_dir
        )
        if len(doc_embs) >= 10 and len(scene_embs) >= 10:
            mmd_val = compute_rkhs_mmd(doc_embs, scene_embs)
            pad_stats = compute_proxy_a_distance(doc_embs, scene_embs, n_splits=5, seed=args.seed)
            run_umap_and_plot(doc_embs, scene_embs, run_dir, args.model)
            all_results["domain_gap"] = {
                "n_doc": int(len(doc_embs)),
                "n_scene": int(len(scene_embs)),
                "rkhs_mmd": round(mmd_val, 6),
                **{k: round(float(v), 6) for k, v in pad_stats.items()},
            }
            (run_dir / "domain_gap_metrics.json").write_text(
                json.dumps(all_results["domain_gap"], indent=2), encoding="utf-8"
            )

    card_cfg = {
        "experiment": {"id": f"Z0_{args.model}{suffix}", "exp": "Z0", "seed": args.seed, "git_sha": get_git_sha()},
        "data": {"regime": "alt_three_splits", "revision": "three-splits", "include_ocr": args.include_ocr},
        "model": {"backbone": model_info["repo_id"], "architecture_mode": "frozen_zero_shot"},
        "training": {"hardware_profile": hw["name"]},
    }
    save_run_card(str(run_dir), card_cfg, all_results)
    print(f"\nSaved Z0 + Domain Gap artifacts to {run_dir}")


if __name__ == "__main__":
    main()
