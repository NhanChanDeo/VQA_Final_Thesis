"""
Unified CLI runner for Proposed CrossVQA Architectures M1–M3 under ALT_THREE_SPLITS:
- M1: Dual Domain-Specific LoRA (A_doc, A_scene) + Visual Soft Gating Router
- M2: M1 + Cross-Domain Feature Alignment (Multi-Kernel RBF MMD default / InfoNCE ablation)
- M3: M2 + Curriculum Pseudo-Domain Homography & Shadow Transform

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/run_crossvqa.py --exp M2 --model qwen2_vl_2b --seed 42
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

METHOD_FOLDER_MAP = {
    "M1": "M1_dual_lora",
    "M2": "M2_dual_lora_align",
    "M3": "M3_curriculum",
}


def main() -> None:
    ensure_repo_root()
    from TRAIN.src.augmentations import CurriculumPseudoDomainTransform
    from TRAIN.src.dataset import PairedDomainDataset, load_crossvqa_dataset
    from TRAIN.src.losses import CrossDomainInfoNCELoss, MultiKernelRBFMMDLoss
    from TRAIN.src.metrics import save_run_card
    from TRAIN.src.models import build_vlm_model_and_processor
    from TRAIN.src.trainer import CheckpointManager, CrossVQATrainer, resolve_hardware_profile

    parser = argparse.ArgumentParser(description="Run Proposed CrossVQA M1–M3 experiments.")
    parser.add_argument("--exp", required=True, choices=["M1", "M2", "M3"])
    parser.add_argument("--model", default="qwen2_vl_2b", choices=["qwen2_vl_2b", "smolvlm_base", "smolvlm_500m"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware_profile", default="auto", choices=["auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb", "cpu"])
    parser.add_argument("--align_type", default=None, choices=["none", "mmd", "infonce"])
    parser.add_argument("--lambda_align", type=float, default=None)
    parser.add_argument("--lambda_gate", type=float, default=None)
    parser.add_argument("--few_shot_pct", type=int, default=100, choices=[1, 5, 10, 25, 100])
    parser.add_argument("--include_ocr", action="store_true")
    parser.add_argument("--use_soft_weights", action="store_true")
    parser.add_argument("--local_data_dir", default=None)
    parser.add_argument("--hf_ckpt_repo", default="gamusa/crossvqa-checkpoints")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--gdrive_backup_dir", default=None)
    parser.add_argument("--resume_from_hub", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true", help="Reuse cached M_stage1_doc adapter if available")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    bb_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/backbones.yaml")
    m_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/crossvqa_M1_M3.yaml")
    fs_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/few_shot.yaml")
    hw = resolve_hardware_profile(args.hardware_profile, bb_cfg.get("hardware_profiles"))

    model_info = bb_cfg["backbones"][args.model]
    model_type = model_info["model_type"]
    tr_cfg = m_cfg["training"]
    method_cfg = m_cfg["methods"][args.exp]

    align_type = args.align_type or method_cfg.get("align_type", "none")
    lambda_align = args.lambda_align if args.lambda_align is not None else float(method_cfg.get("lambda_align", 0.0))
    lambda_gate = args.lambda_gate if args.lambda_gate is not None else float(method_cfg.get("lambda_gate", 0.1))

    ocr_tag = "_ocr" if args.include_ocr else ""
    align_tag = f"_{align_type}" if (args.exp in ("M2", "M3") and align_type == "infonce") else ""
    if args.few_shot_pct < 100:
        run_rel = (
            Path(m_cfg["runs_root"]) / args.model / "few_shot"
            / f"{args.exp}{align_tag}{ocr_tag}_pct_{args.few_shot_pct:02d}" / f"seed_{args.seed}"
        )
        hf_subpath = f"alt_three_splits/{args.model}/few_shot/{args.exp}{align_tag}{ocr_tag}_pct_{args.few_shot_pct:02d}/seed_{args.seed}"
    else:
        exp_dir = METHOD_FOLDER_MAP[args.exp] + align_tag + ocr_tag
        run_rel = Path(m_cfg["runs_root"]) / args.model / exp_dir / f"seed_{args.seed}"
        hf_subpath = f"alt_three_splits/{args.model}/{exp_dir}/seed_{args.seed}"
    run_rel.mkdir(parents=True, exist_ok=True)

    hf_token = resolve_hf_token(args.hf_token)
    gdrive_run = str(Path(args.gdrive_backup_dir) / hf_subpath) if args.gdrive_backup_dir else None
    ckpt_mgr = CheckpointManager(
        local_ckpt_dir=str(run_rel / "checkpoints"),
        hf_repo_id=args.hf_ckpt_repo,
        hf_path_in_repo=hf_subpath,
        hf_token=hf_token,
        gdrive_backup_dir=gdrive_run,
        private=bool(m_cfg["checkpoints"].get("hf_private", True)),
    )

    stage1_hf_subpath = f"alt_three_splits/{args.model}/M_stage1_doc{ocr_tag}/seed_{args.seed}"
    stage1_mgr = CheckpointManager(
        local_ckpt_dir=str(Path(m_cfg["runs_root"]) / args.model / f"M_stage1_doc{ocr_tag}" / f"seed_{args.seed}" / "checkpoints"),
        hf_repo_id=args.hf_ckpt_repo,
        hf_path_in_repo=stage1_hf_subpath,
        hf_token=hf_token,
        gdrive_backup_dir=str(Path(args.gdrive_backup_dir) / stage1_hf_subpath) if args.gdrive_backup_dir else None,
        private=True,
    )

    resume_ckpt = None
    if args.resume_from_hub:
        restored = ckpt_mgr.restore_checkpoint("latest")
        if restored is not None:
            resume_ckpt = str(restored)

    model, processor = build_vlm_model_and_processor(
        model_name_or_key=args.model,
        load_in_4bit=bool(hw.get("load_in_4bit", tr_cfg["load_in_4bit"])),
        use_lora=True,
        lora_r=tr_cfg["lora_r"],
        lora_alpha=tr_cfg["lora_alpha"],
        lora_dropout=tr_cfg["lora_dropout"],
        min_pixels=hw["min_pixels"],
        max_pixels=hw["max_pixels"],
        gradient_checkpointing=True,
        attn_implementation=hw["attn_implementation"],
        adapter_checkpoint=resume_ckpt,
        architecture_mode="dual_lora",
        router_tau=float(method_cfg.get("router_tau", 1.0)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    few_shot_manifest = fs_cfg["few_shot_manifest_path"]

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
        res["selection_score"] = res["harmonic_mean"]
        return res

    # Stage 1: DocVQA Source Adaptation (A_doc + Router)
    stage1_restored = stage1_mgr.restore_checkpoint("best") if (args.skip_stage1 or args.resume_from_hub) else None
    if stage1_restored is not None and resume_ckpt is None:
        print(f"Loading cached Stage-1 DocVQA Dual-LoRA from {stage1_restored}")
        model.load_pretrained(str(stage1_restored))
    elif resume_ckpt is None:
        print(f"\n=== [{args.exp}] Stage 1: DocVQA Source Adaptation (A_doc + Router) ===")
        model.set_trainable_stage("doc_only")
        model.print_trainable_parameters()
        doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=True, local_data_dir=args.local_data_dir)
        s1_trainer = CrossVQATrainer(
            model=model,
            processor=processor,
            run_dir=str(run_rel / "stage1_doc"),
            model_type=model_type,
            device=device,
            lr=tr_cfg["learning_rate"],
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
            gradient_accumulation_steps=hw["gradient_accumulation_steps"],
            max_grad_norm=tr_cfg["max_grad_norm"],
            optim_name=hw["optim"],
            include_ocr=args.include_ocr,
            lambda_gate=lambda_gate,
            checkpoint_manager=stage1_mgr,
        )
        s1_trainer.fit(
            train_dataset=doc_train,
            epochs=int(method_cfg.get("stage1_doc_epochs", 1)),
            batch_size=hw["per_device_train_batch_size"],
            max_steps_per_epoch=args.max_train_steps,
            forced_domain_id=0,
        )

    # Alignment loss initialization (M2 / M3)
    align_loss_fn = None
    if align_type == "mmd" and lambda_align > 0.0:
        align_loss_fn = MultiKernelRBFMMDLoss(
            sigmas=tuple(method_cfg.get("mmd_sigmas", [0.5, 1.0, 2.0, 4.0, 8.0])),
            queue_size=int(method_cfg.get("queue_size", 256)),
        )
    elif align_type == "infonce" and lambda_align > 0.0:
        align_loss_fn = CrossDomainInfoNCELoss(queue_size=int(method_cfg.get("queue_size", 256)))

    # Stage 2 (M3 Only): Curriculum Pseudo-Domain Homography Bridge
    if args.exp == "M3":
        print("\n=== [M3] Stage 2: Curriculum Pseudo-Domain Homography Bridge ===")
        model.set_trainable_stage("scene_only_frozen_doc")
        doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=True, local_data_dir=args.local_data_dir)
        tv_train = load_crossvqa_dataset(
            "textvqa", "train", "alt_three_splits",
            streaming=True, local_data_dir=args.local_data_dir,
            few_shot_pct=args.few_shot_pct, few_shot_manifest_path=few_shot_manifest,
        )
        paired_curr = PairedDomainDataset(doc_train, tv_train, seed=args.seed + 100)
        curr_transform = CurriculumPseudoDomainTransform(
            severity=float(method_cfg.get("curriculum_severity_start", 0.2)),
            seed=args.seed,
        )
        s2_trainer = CrossVQATrainer(
            model=model,
            processor=processor,
            run_dir=str(run_rel / "stage2_curriculum"),
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
            align_loss_fn=align_loss_fn,
            lambda_align=lambda_align,
            lambda_gate=lambda_gate,
            curriculum_transform=curr_transform,
            checkpoint_manager=ckpt_mgr,
        )
        s2_trainer.fit(
            train_dataset=paired_curr,
            epochs=int(method_cfg.get("stage2_curriculum_epochs", 1)),
            batch_size=max(2, hw["per_device_train_batch_size"]),
            max_steps_per_epoch=args.max_train_steps,
            curriculum_start=float(method_cfg.get("curriculum_severity_start", 0.2)),
            curriculum_end=float(method_cfg.get("curriculum_severity_end", 1.0)),
        )

    # Final Transfer Stage: Target TextVQA Adaptation (with frozen A_doc + Paired Alignment)
    freeze_doc = bool(method_cfg.get("freeze_doc_in_stage2", method_cfg.get("freeze_doc_in_stage3", True)))
    model.set_trainable_stage("scene_only_frozen_doc" if freeze_doc else "joint")
    model.print_trainable_parameters()

    doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=True, local_data_dir=args.local_data_dir)
    tv_train = load_crossvqa_dataset(
        "textvqa", "train", "alt_three_splits",
        streaming=True, local_data_dir=args.local_data_dir,
        few_shot_pct=args.few_shot_pct, few_shot_manifest_path=few_shot_manifest,
    )
    paired_ds = PairedDomainDataset(doc_train, tv_train, seed=args.seed)

    transfer_epochs = int(
        method_cfg.get("stage3_transfer_epochs" if args.exp == "M3" else "stage2_transfer_epochs", 2)
    )
    final_trainer = CrossVQATrainer(
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
        align_loss_fn=align_loss_fn,
        lambda_align=lambda_align,
        lambda_gate=lambda_gate,
        checkpoint_manager=ckpt_mgr,
    )
    summary = final_trainer.fit(
        train_dataset=paired_ds,
        epochs=transfer_epochs,
        batch_size=max(2, hw["per_device_train_batch_size"]),
        max_steps_per_epoch=args.max_train_steps,
        eval_fn=_eval_val,
        monitor_metric="selection_score",
    )

    card_cfg = {
        "experiment": {"id": f"{args.exp}_{args.model}{align_tag}{ocr_tag}_seed{args.seed}", "exp": args.exp, "seed": args.seed, "git_sha": get_git_sha()},
        "data": {"regime": "alt_three_splits", "revision": "three-splits", "few_shot_pct": args.few_shot_pct, "include_ocr": args.include_ocr},
        "model": {"backbone": model_info["repo_id"], "architecture_mode": "dual_lora", "lora_r": tr_cfg["lora_r"], "lora_alpha": tr_cfg["lora_alpha"], "align_type": align_type, "lambda_align": lambda_align, "lambda_gate": lambda_gate},
        "training": {"hardware_profile": hw["name"], "epochs": transfer_epochs, "lr": tr_cfg["learning_rate"], "elapsed_sec": summary["elapsed_sec"]},
    }
    save_run_card(str(run_rel), card_cfg, summary["best_metrics"])
    print(f"\nCompleted {args.exp} ({args.model}, seed={args.seed}). Artifacts saved to {run_rel}")


if __name__ == "__main__":
    main()
