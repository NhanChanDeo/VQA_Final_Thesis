"""
Unified CLI runner for CrossVQA Baselines B1–B5 under ALT_THREE_SPLITS:
- B1: Target-only (TextVQA)
- B2: Source-only (DocVQA)
- B3: Joint training (DocVQA + TextVQA interleaved)
- B4: Sequential (DocVQA -> TextVQA; initializes from B2 adapter)
- B5: Reverse sequential (TextVQA -> DocVQA; initializes from B1 adapter)

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py --exp B1 --model qwen2_vl_2b --seed 42
"""

import argparse
from pathlib import Path
import torch

from _common import (
    ensure_repo_root,
    evaluate_both_domains,
    get_git_sha,
    load_yaml,
    resolve_hf_token,
    set_seed,
)

EXP_FOLDER_MAP = {
    "B1": "B1_target_only",
    "B2": "B2_source_only",
    "B3": "B3_joint",
    "B4": "B4_seq_doc2text",
    "B5": "B5_seq_text2doc",
}


def main() -> None:
    ensure_repo_root()
    from TRAIN.src.dataset import InterleavedCrossVQADataset, load_crossvqa_dataset
    from TRAIN.src.metrics import save_run_card
    from TRAIN.src.models import build_vlm_model_and_processor
    from TRAIN.src.trainer import CheckpointManager, CrossVQATrainer, resolve_hardware_profile

    parser = argparse.ArgumentParser(description="Run CrossVQA baseline experiments B1–B5.")
    parser.add_argument("--exp", required=True, choices=["B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--model", default="smolvlm_500m", choices=["smolvlm_500m", "smolvlm_base", "qwen2_vl_2b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware_profile", default="auto", choices=["auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb", "cpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--few_shot_pct", type=int, default=100, choices=[1, 5, 10, 25, 100])
    parser.add_argument("--include_ocr", action="store_true")
    parser.add_argument("--use_soft_weights", action="store_true")
    parser.add_argument("--local_data_dir", default=None)
    parser.add_argument("--stage1_adapter_path", default=None)
    parser.add_argument("--hf_ckpt_repo", default="gamusa/crossvqa-checkpoints")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--gdrive_backup_dir", default=None)
    parser.add_argument("--resume_from_hub", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=250)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    bb_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/backbones.yaml")
    base_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/baselines_B1_B5.yaml")
    fs_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/few_shot.yaml")
    hw = resolve_hardware_profile(args.hardware_profile, bb_cfg.get("hardware_profiles"))

    model_info = bb_cfg["backbones"][args.model]
    model_type = model_info["model_type"]
    tr_cfg = base_cfg["training"]
    epochs = args.epochs if args.epochs is not None else int(tr_cfg["epochs"])

    ocr_tag = "_ocr" if args.include_ocr else ""
    if args.few_shot_pct < 100:
        run_rel = Path(base_cfg["runs_root"]) / args.model / "few_shot" / f"{args.exp}{ocr_tag}_pct_{args.few_shot_pct:02d}" / f"seed_{args.seed}"
        hf_subpath = f"alt_three_splits/{args.model}/few_shot/{args.exp}{ocr_tag}_pct_{args.few_shot_pct:02d}/seed_{args.seed}"
    else:
        exp_dir = EXP_FOLDER_MAP[args.exp] + ocr_tag
        run_rel = Path(base_cfg["runs_root"]) / args.model / exp_dir / f"seed_{args.seed}"
        hf_subpath = f"alt_three_splits/{args.model}/{exp_dir}/seed_{args.seed}"
    run_rel.mkdir(parents=True, exist_ok=True)

    hf_token = resolve_hf_token(args.hf_token)
    gdrive_run_dir = str(Path(args.gdrive_backup_dir) / hf_subpath) if args.gdrive_backup_dir else None
    ckpt_mgr = CheckpointManager(
        local_ckpt_dir=str(run_rel / "checkpoints"),
        hf_repo_id=args.hf_ckpt_repo,
        hf_path_in_repo=hf_subpath,
        hf_token=hf_token,
        gdrive_backup_dir=gdrive_run_dir,
        private=bool(base_cfg["checkpoints"].get("hf_private", True)),
    )

    # Resolve Stage-1 checkpoint for B4 / B5 or --resume_from_hub
    adapter_ckpt = args.stage1_adapter_path
    if args.resume_from_hub and adapter_ckpt is None:
        restored = ckpt_mgr.restore_checkpoint("latest")
        if restored is not None:
            adapter_ckpt = str(restored)

    if adapter_ckpt is None and args.exp in ("B4", "B5"):
        stage1_folder = ("B2_source_only" if args.exp == "B4" else "B1_target_only") + ocr_tag
        stage1_hf_path = f"alt_three_splits/{args.model}/{stage1_folder}/seed_{args.seed}"
        stage1_local = Path(base_cfg["runs_root"]) / args.model / stage1_folder / f"seed_{args.seed}" / "checkpoints"
        stage1_mgr = CheckpointManager(
            local_ckpt_dir=str(stage1_local),
            hf_repo_id=args.hf_ckpt_repo,
            hf_path_in_repo=stage1_hf_path,
            hf_token=hf_token,
            gdrive_backup_dir=str(Path(args.gdrive_backup_dir) / stage1_hf_path) if args.gdrive_backup_dir else None,
            private=True,
        )
        restored_s1 = stage1_mgr.restore_checkpoint("best")
        if restored_s1 is None:
            raise FileNotFoundError(
                f"{args.exp} requires Stage-1 checkpoint from {stage1_folder} (seed {args.seed}), "
                "or pass --stage1_adapter_path explicitly."
            )
        adapter_ckpt = str(restored_s1)

    use_4bit = bool(hw.get("load_in_4bit", tr_cfg["load_in_4bit"]) and model_info.get("allow_4bit", True))
    model, processor = build_vlm_model_and_processor(
        model_name_or_key=args.model,
        load_in_4bit=use_4bit,
        use_lora=True,
        lora_r=tr_cfg["lora_r"],
        lora_alpha=tr_cfg["lora_alpha"],
        lora_dropout=tr_cfg["lora_dropout"],
        min_pixels=hw["min_pixels"],
        max_pixels=hw["max_pixels"],
        smolvlm_longest_edge=hw.get("smolvlm_longest_edge", 2048),
        gradient_checkpointing=True,
        attn_implementation=hw["attn_implementation"],
        adapter_checkpoint=adapter_ckpt,
        architecture_mode="standard_lora",
        target_modules=model_info.get("target_modules"),
        exclude_modules=model_info.get("exclude_lora_modules"),
    )

    few_shot_manifest = fs_cfg["few_shot_manifest_path"]
    use_streaming = args.local_data_dir is None
    if args.exp in ("B1", "B4"):
        train_ds = load_crossvqa_dataset(
            "textvqa", "train", "alt_three_splits",
            streaming=use_streaming, local_data_dir=args.local_data_dir,
            few_shot_pct=args.few_shot_pct, few_shot_manifest_path=few_shot_manifest,
        )
    elif args.exp in ("B2", "B5"):
        train_ds = load_crossvqa_dataset(
            "docvqa", "train", "alt_three_splits",
            streaming=use_streaming, local_data_dir=args.local_data_dir,
        )
    else:  # B3 Joint
        doc_ds = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=use_streaming, local_data_dir=args.local_data_dir)
        tv_ds = load_crossvqa_dataset(
            "textvqa", "train", "alt_three_splits",
            streaming=use_streaming, local_data_dir=args.local_data_dir,
            few_shot_pct=args.few_shot_pct, few_shot_manifest_path=few_shot_manifest,
        )
        train_ds = InterleavedCrossVQADataset(doc_ds, tv_ds, doc_prob=0.5, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    monitor_key = base_cfg["experiments"][args.exp]["monitor_metric"]

    def _eval_val():
        res = evaluate_both_domains(
            model=model,
            processor=processor,
            model_type=model_type,
            eval_split="validation",
            run_dir=run_rel,
            device=device,
            include_ocr=args.include_ocr,
            max_new_tokens=tr_cfg["max_new_tokens"],
            eval_batch_size=hw["eval_batch_size"],
            max_eval_samples=args.max_eval_samples,
            local_data_dir=args.local_data_dir,
        )
        res["selection_score"] = res[monitor_key]
        return res

    trainer = CrossVQATrainer(
        model=model,
        processor=processor,
        run_dir=str(run_rel),
        model_type=model_type,
        device=device,
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        gradient_accumulation_steps=hw["gradient_accumulation_steps"],
        max_grad_norm=tr_cfg["max_grad_norm"],
        optim_name=hw["optim"],
        include_ocr=args.include_ocr,
        use_soft_weights=args.use_soft_weights,
        checkpoint_manager=ckpt_mgr,
    )

    save_every = args.save_every_steps if args.save_every_steps and args.save_every_steps > 0 else None
    summary = trainer.fit(
        train_dataset=train_ds,
        epochs=epochs,
        batch_size=hw["per_device_train_batch_size"],
        max_steps_per_epoch=args.max_train_steps,
        eval_fn=_eval_val,
        monitor_metric="selection_score",
        save_every_steps=save_every,
        seed=args.seed,
    )

    card_cfg = {
        "experiment": {"id": f"{args.exp}_{args.model}{ocr_tag}_seed{args.seed}", "exp": args.exp, "seed": args.seed, "git_sha": get_git_sha()},
        "data": {"regime": "alt_three_splits", "revision": "three-splits", "few_shot_pct": args.few_shot_pct, "include_ocr": args.include_ocr},
        "model": {"backbone": model_info["repo_id"], "architecture_mode": "standard_lora", "lora_r": tr_cfg["lora_r"], "lora_alpha": tr_cfg["lora_alpha"]},
        "training": {"hardware_profile": hw["name"], "epochs": epochs, "lr": tr_cfg["learning_rate"], "elapsed_sec": summary["elapsed_sec"]},
    }
    save_run_card(str(run_rel), card_cfg, summary["best_metrics"], extra_meta={"stage1_adapter_used": adapter_ckpt})
    print(f"\nCompleted {args.exp} ({args.model}, seed={args.seed}). Artifacts saved to {run_rel}")


if __name__ == "__main__":
    main()
