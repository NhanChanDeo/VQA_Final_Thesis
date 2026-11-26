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
    apply_training_overrides,
    ensure_repo_root,
    evaluate_both_domains,
    get_git_sha,
    load_yaml,
    resolve_checkpoint_paths,
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
    parser.add_argument("--model", default="smolvlm_500m", choices=["smolvlm_500m", "smolvlm_base", "qwen2_vl_2b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware_profile", default="auto")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--checkpoint_namespace", default=None)
    parser.add_argument("--align_type", default=None, choices=["none", "mmd", "infonce"])
    parser.add_argument("--align_adapter_rank", type=int, default=None)
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
    parser.add_argument("--allow_mismatched_resume", action="store_true")
    parser.add_argument("--no_save_optimizer_state", action="store_true")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--no_group_by_resolution", action="store_false", dest="group_by_resolution", default=True)
    parser.add_argument("--skip_stage1", action="store_true", help="Reuse cached M_stage1_doc adapter if available")
    parser.add_argument("--save_every_steps", type=int, default=None)
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
    batch_size, grad_accum, eval_batch_size, lr = apply_training_overrides(hw, tr_cfg, args)
    effective_batch_size = batch_size * grad_accum
    ckpt_ns = (
        args.checkpoint_namespace
        if args.checkpoint_namespace is not None
        else m_cfg.get("checkpoints", {}).get("checkpoint_namespace", "")
    )
    save_every_cfg = int(m_cfg.get("checkpoints", {}).get("save_every_steps", 64))
    save_every_arg = args.save_every_steps if args.save_every_steps is not None else save_every_cfg

    align_type = args.align_type or method_cfg.get("align_type", "none")
    lambda_align = args.lambda_align if args.lambda_align is not None else float(method_cfg.get("lambda_align", 0.0))
    lambda_gate = args.lambda_gate if args.lambda_gate is not None else float(method_cfg.get("lambda_gate", 0.1))
    align_adapter_rank = (
        args.align_adapter_rank
        if args.align_adapter_rank is not None
        else int(method_cfg.get("align_adapter_rank", 64 if (args.exp in ("M2", "M3") and lambda_align > 0.0) else 0))
    )

    ocr_tag = "_ocr" if args.include_ocr else ""
    align_tag = f"_{align_type}" if (args.exp in ("M2", "M3") and align_type == "infonce") else ""
    exp_dir = METHOD_FOLDER_MAP[args.exp] + align_tag + ocr_tag
    run_rel, hf_subpath = resolve_checkpoint_paths(
        runs_root=m_cfg["runs_root"],
        model=args.model,
        exp_dir=exp_dir,
        seed=args.seed,
        checkpoint_namespace=ckpt_ns,
        few_shot_pct=args.few_shot_pct,
        few_shot_tag=f"{args.exp}{align_tag}{ocr_tag}",
    )
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

    stage1_rel, stage1_hf_subpath = resolve_checkpoint_paths(
        runs_root=m_cfg["runs_root"],
        model=args.model,
        exp_dir=f"M_stage1_doc{ocr_tag}",
        seed=args.seed,
        checkpoint_namespace=ckpt_ns,
    )
    stage1_mgr = CheckpointManager(
        local_ckpt_dir=str(stage1_rel / "checkpoints"),
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
    resumed_state = CheckpointManager.load_resume_state(resume_ckpt) if resume_ckpt else None
    resumed_stage = resumed_state.get("stage") if resumed_state else None
    final_stage_name = "stage3_transfer" if args.exp == "M3" else "stage2_transfer"

    use_4bit = bool(hw.get("load_in_4bit", tr_cfg["load_in_4bit"]) and model_info.get("allow_4bit", True))
    use_grad_ckpt = (
        bool(args.gradient_checkpointing)
        if args.gradient_checkpointing is not None
        else (True if use_4bit else bool(hw.get("gradient_checkpointing", True)))
    )
    optim_name = "paged_adamw_8bit" if use_4bit else str(hw.get("optim", "adamw_torch_fused"))
    model, processor = build_vlm_model_and_processor(
        model_name_or_key=args.model,
        load_in_4bit=use_4bit,
        use_lora=True,
        lora_r=tr_cfg["lora_r"],
        lora_alpha=tr_cfg["lora_alpha"],
        lora_dropout=tr_cfg["lora_dropout"],
        min_pixels=hw["min_pixels"],
        max_pixels=hw["max_pixels"],
        smolvlm_longest_edge=hw.get("smolvlm_longest_edge", 1536),
        gradient_checkpointing=use_grad_ckpt,
        attn_implementation=hw["attn_implementation"],
        adapter_checkpoint=resume_ckpt,
        architecture_mode="dual_lora",
        target_modules=model_info.get("target_modules"),
        exclude_modules=model_info.get("exclude_lora_modules"),
        router_tau=float(method_cfg.get("router_tau", 1.0)),
        align_adapter_rank=align_adapter_rank,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    few_shot_manifest = fs_cfg["few_shot_manifest_path"]
    use_streaming = args.local_data_dir is None
    save_every = save_every_arg if save_every_arg and save_every_arg > 0 else None
    base_fingerprint = {
        "exp": args.exp,
        "model": args.model,
        "seed": args.seed,
        "effective_batch_size": effective_batch_size,
        "lr": lr,
        "lora_r": tr_cfg["lora_r"],
        "checkpoint_namespace": ckpt_ns,
    }

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
            eval_batch_size=eval_batch_size,
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
        s1_resume_dir = None
        if args.resume_from_hub:
            s1_latest = stage1_mgr.restore_checkpoint("latest")
            if s1_latest is not None:
                print(f"Resuming Stage-1 DocVQA Dual-LoRA from {s1_latest}")
                model.load_pretrained(str(s1_latest))
                s1_resume_dir = str(s1_latest)
        print(f"\n=== [{args.exp}] Stage 1: DocVQA Source Adaptation (A_doc + Router) ===")
        model.set_trainable_stage("doc_only")
        model.print_trainable_parameters()
        doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=use_streaming, local_data_dir=args.local_data_dir)
        s1_trainer = CrossVQATrainer(
            model=model,
            processor=processor,
            run_dir=str(run_rel / "stage1_doc"),
            model_type=model_type,
            device=device,
            lr=lr,
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=tr_cfg["max_grad_norm"],
            optim_name=optim_name,
            include_ocr=args.include_ocr,
            lambda_gate=lambda_gate,
            checkpoint_manager=stage1_mgr,
        )
        s1_trainer.fit(
            train_dataset=doc_train,
            epochs=int(method_cfg.get("stage1_doc_epochs", 1)),
            batch_size=batch_size,
            max_steps_per_epoch=args.max_train_steps,
            forced_domain_id=0,
            save_every_steps=save_every,
            seed=args.seed,
            resume_from=s1_resume_dir,
            stage_name="stage1_doc",
            config_fingerprint=base_fingerprint,
            save_optimizer_state=not args.no_save_optimizer_state,
            allow_mismatched_resume=args.allow_mismatched_resume,
            group_by_resolution=args.group_by_resolution,
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
    if args.exp == "M3" and resumed_stage != final_stage_name:
        print("\n=== [M3] Stage 2: Curriculum Pseudo-Domain Homography Bridge ===")
        model.set_trainable_stage("scene_only_frozen_doc")
        doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=use_streaming, local_data_dir=args.local_data_dir)
        tv_train = load_crossvqa_dataset(
            "textvqa", "train", "alt_three_splits",
            streaming=use_streaming, local_data_dir=args.local_data_dir,
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
            lr=lr,
            weight_decay=tr_cfg["weight_decay"],
            warmup_ratio=tr_cfg["warmup_ratio"],
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=tr_cfg["max_grad_norm"],
            optim_name=optim_name,
            include_ocr=args.include_ocr,
            use_soft_weights=args.use_soft_weights,
            align_loss_fn=align_loss_fn,
            lambda_align=lambda_align,
            lambda_gate=lambda_gate,
            curriculum_transform=curr_transform,
            checkpoint_manager=ckpt_mgr,
        )
        s2_resume = resume_ckpt if (resume_ckpt and resumed_stage == "stage2_curriculum") else None
        s2_trainer.fit(
            train_dataset=paired_curr,
            epochs=int(method_cfg.get("stage2_curriculum_epochs", 1)),
            batch_size=max(2, batch_size),
            max_steps_per_epoch=args.max_train_steps,
            curriculum_start=float(method_cfg.get("curriculum_severity_start", 0.2)),
            curriculum_end=float(method_cfg.get("curriculum_severity_end", 1.0)),
            save_every_steps=save_every,
            seed=args.seed + 100,
            resume_from=s2_resume,
            stage_name="stage2_curriculum",
            config_fingerprint=base_fingerprint,
            save_optimizer_state=not args.no_save_optimizer_state,
            allow_mismatched_resume=args.allow_mismatched_resume,
            group_by_resolution=args.group_by_resolution,
        )

    # Final Transfer Stage: Target TextVQA Adaptation (with frozen A_doc + Paired Alignment)
    freeze_doc = bool(method_cfg.get("freeze_doc_in_stage2", method_cfg.get("freeze_doc_in_stage3", True)))
    model.set_trainable_stage("scene_only_frozen_doc" if freeze_doc else "joint")
    model.print_trainable_parameters()

    doc_train = load_crossvqa_dataset("docvqa", "train", "alt_three_splits", streaming=use_streaming, local_data_dir=args.local_data_dir)
    tv_train = load_crossvqa_dataset(
        "textvqa", "train", "alt_three_splits",
        streaming=use_streaming, local_data_dir=args.local_data_dir,
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
        lr=lr,
        weight_decay=tr_cfg["weight_decay"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=tr_cfg["max_grad_norm"],
        optim_name=optim_name,
        include_ocr=args.include_ocr,
        use_soft_weights=args.use_soft_weights,
        align_loss_fn=align_loss_fn,
        lambda_align=lambda_align,
        lambda_gate=lambda_gate,
        checkpoint_manager=ckpt_mgr,
    )
    final_resume = resume_ckpt if (resume_ckpt and (resumed_stage is None or resumed_stage == final_stage_name)) else None
    summary = final_trainer.fit(
        train_dataset=paired_ds,
        epochs=transfer_epochs,
        batch_size=max(2, batch_size),
        max_steps_per_epoch=args.max_train_steps,
        eval_fn=_eval_val,
        monitor_metric="selection_score",
        save_every_steps=save_every,
        seed=args.seed,
        resume_from=final_resume,
        stage_name=final_stage_name,
        config_fingerprint=base_fingerprint,
        save_optimizer_state=not args.no_save_optimizer_state,
        allow_mismatched_resume=args.allow_mismatched_resume,
        group_by_resolution=args.group_by_resolution,
    )

    card_cfg = {
        "experiment": {"id": f"{args.exp}_{args.model}{align_tag}{ocr_tag}_seed{args.seed}", "exp": args.exp, "seed": args.seed, "git_sha": get_git_sha()},
        "data": {"regime": "alt_three_splits", "revision": "three-splits", "few_shot_pct": args.few_shot_pct, "include_ocr": args.include_ocr},
        "model": {"backbone": model_info["repo_id"], "architecture_mode": "dual_lora", "lora_r": tr_cfg["lora_r"], "lora_alpha": tr_cfg["lora_alpha"], "align_type": align_type, "align_adapter_rank": align_adapter_rank, "lambda_align": lambda_align, "lambda_gate": lambda_gate},
        "training": {
            "hardware_profile": hw["name"],
            "checkpoint_namespace": ckpt_ns,
            "per_device_train_batch_size": max(2, batch_size),
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": max(2, batch_size) * grad_accum,
            "epochs": transfer_epochs,
            "lr": lr,
            "elapsed_sec": summary["elapsed_sec"],
        },
    }
    save_run_card(str(run_rel), card_cfg, summary["best_metrics"], extra_meta={"resumed_from": final_resume})
    print(f"\nCompleted {args.exp} ({args.model}, seed={args.seed}). Artifacts saved to {run_rel}")


if __name__ == "__main__":
    main()
