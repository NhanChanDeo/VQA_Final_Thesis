"""
Evaluates Z0 (frozen) and trained LoRA checkpoints (B1–B5, M1–M3) on a target split
(default: held-out `test` split under ALT_THREE_SPLITS), writing per-sample
`preds_{eval_split}_{docvqa,textvqa}.jsonl` and `{eval_split}_metrics.json`.

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/eval_test_checkpoints.py \
        --model smolvlm_500m --seed 42 --exps Z0 B1 B2 B3 B4 \
        --local_data_dir /content/data
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from _common import (
    ensure_repo_root,
    evaluate_both_domains,
    load_yaml,
    resolve_checkpoint_paths,
    resolve_hf_token,
    set_seed,
)

EXP_FOLDER_MAP: Dict[str, Tuple[str, str]] = {
    "Z0": ("Z0", "none"),
    "B1": ("B1_target_only", "standard_lora"),
    "B2": ("B2_source_only", "standard_lora"),
    "B3": ("B3_joint", "standard_lora"),
    "B4": ("B4_seq_doc2text", "standard_lora"),
    "B5": ("B5_seq_text2doc", "standard_lora"),
    "M1": ("M1_dual_lora", "dual_lora"),
    "M2": ("M2_dual_lora_align", "dual_lora"),
    "M3": ("M3_curriculum", "dual_lora"),
}


def resolve_eval_targets(
    exps: List[str],
    few_shot_pcts: List[int],
    runs_root: str,
    model: str,
    seed: int,
    checkpoint_namespace: str,
    include_ocr: bool = False,
    align_type: str = "mmd",
) -> List[Dict[str, Any]]:
    """Builds the ordered list of (exp, pct, run_rel, hf_subpath, arch_mode) evaluation targets."""
    ocr_tag = "_ocr" if include_ocr else ""
    targets: List[Dict[str, Any]] = []
    seen = set()

    for exp in exps:
        if exp not in EXP_FOLDER_MAP:
            raise ValueError(f"Unsupported experiment '{exp}'. Valid: {list(EXP_FOLDER_MAP.keys())}")
        base_folder, arch_mode = EXP_FOLDER_MAP[exp]

        if exp == "Z0":
            run_rel = Path(runs_root) / model / f"Z0{ocr_tag}"
            key = ("Z0", 100, str(run_rel))
            if key not in seen:
                seen.add(key)
                targets.append(
                    {
                        "exp": "Z0",
                        "few_shot_pct": 100,
                        "arch_mode": "none",
                        "run_rel": run_rel,
                        "hf_subpath": None,
                    }
                )
            continue

        align_tag = f"_{align_type}" if (exp in ("M2", "M3") and align_type == "infonce") else ""
        exp_dir = f"{base_folder}{align_tag}{ocr_tag}"
        pct_list = [100] if exp in ("B2", "B3", "B5") else few_shot_pcts
        for pct in pct_list:
            run_rel, hf_subpath = resolve_checkpoint_paths(
                runs_root=runs_root,
                model=model,
                exp_dir=exp_dir,
                seed=seed,
                checkpoint_namespace=checkpoint_namespace,
                few_shot_pct=int(pct),
                few_shot_tag=f"{exp}{align_tag}{ocr_tag}",
            )
            key = (exp, int(pct), str(run_rel))
            if key not in seen:
                seen.add(key)
                targets.append(
                    {
                        "exp": exp,
                        "few_shot_pct": int(pct),
                        "arch_mode": arch_mode,
                        "run_rel": run_rel,
                        "hf_subpath": hf_subpath,
                    }
                )
    return targets


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Z0 and trained checkpoints on held-out split.")
    parser.add_argument("--model", default="smolvlm_500m", choices=["smolvlm_500m", "smolvlm_base", "qwen2_vl_2b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exps", nargs="+", default=["Z0", "B1", "B2", "B3", "B4"])
    parser.add_argument("--few_shot_pcts", nargs="+", type=int, default=[100], choices=[1, 5, 10, 25, 100])
    parser.add_argument("--eval_split", default="test", choices=["validation", "test"])
    parser.add_argument("--checkpoint_namespace", default=None)
    parser.add_argument("--checkpoint_tag", default="best", choices=["best", "latest"])
    parser.add_argument("--hardware_profile", default="auto")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--include_ocr", action="store_true")
    parser.add_argument("--align_type", default="mmd", choices=["none", "mmd", "infonce"])
    parser.add_argument("--local_data_dir", default=None)
    parser.add_argument("--hf_ckpt_repo", default="gamusa/crossvqa-checkpoints")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--gdrive_backup_dir", default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no_bucket_by_resolution", action="store_true")
    parser.add_argument("--skip_missing", action="store_true", help="Skip targets without an available checkpoint")
    return parser.parse_args(argv)


def main() -> None:
    ensure_repo_root()
    from TRAIN.src.models import build_vlm_model_and_processor
    from TRAIN.src.trainer import CheckpointManager, resolve_hardware_profile

    args = parse_args()
    set_seed(args.seed)

    bb_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/backbones.yaml")
    base_cfg = load_yaml("TRAIN/ALT_THREE_SPLITS/configs/baselines_B1_B5.yaml")
    hw = resolve_hardware_profile(args.hardware_profile, bb_cfg.get("hardware_profiles"))

    model_info = bb_cfg["backbones"][args.model]
    model_type = model_info["model_type"]
    ckpt_ns = (
        args.checkpoint_namespace
        if args.checkpoint_namespace is not None
        else base_cfg.get("checkpoints", {}).get("checkpoint_namespace", "")
    )
    hf_token = resolve_hf_token(args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_bs = int(args.eval_batch_size) if args.eval_batch_size is not None else int(hw["eval_batch_size"])

    targets = resolve_eval_targets(
        exps=args.exps,
        few_shot_pcts=args.few_shot_pcts,
        runs_root=base_cfg["runs_root"],
        model=args.model,
        seed=args.seed,
        checkpoint_namespace=ckpt_ns,
        include_ocr=args.include_ocr,
        align_type=args.align_type,
    )

    all_summaries: List[Dict[str, Any]] = []
    for tgt in targets:
        exp = tgt["exp"]
        pct = tgt["few_shot_pct"]
        arch_mode = tgt["arch_mode"]
        run_rel: Path = tgt["run_rel"]
        hf_subpath: Optional[str] = tgt["hf_subpath"]
        run_rel.mkdir(parents=True, exist_ok=True)

        adapter_ckpt: Optional[str] = None
        if arch_mode != "none":
            gdrive_run = str(Path(args.gdrive_backup_dir) / hf_subpath) if args.gdrive_backup_dir and hf_subpath else None
            ckpt_mgr = CheckpointManager(
                local_ckpt_dir=str(run_rel / "checkpoints"),
                hf_repo_id=args.hf_ckpt_repo,
                hf_path_in_repo=hf_subpath,
                hf_token=hf_token,
                gdrive_backup_dir=gdrive_run,
                private=bool(base_cfg["checkpoints"].get("hf_private", True)),
            )
            restored = ckpt_mgr.restore_checkpoint(args.checkpoint_tag)
            if restored is None:
                msg = f"Checkpoint '{args.checkpoint_tag}' not found for {exp} (pct={pct}) at {hf_subpath}"
                if args.skip_missing:
                    print(f"SKIP: {msg}")
                    continue
                raise FileNotFoundError(msg)
            adapter_ckpt = str(restored)

        use_4bit = False if arch_mode == "none" else bool(
            hw.get("load_in_4bit", base_cfg["training"]["load_in_4bit"]) and model_info.get("allow_4bit", True)
        )
        cur_bs = eval_bs * 2 if (arch_mode == "none" and not use_4bit and args.eval_batch_size is None and hw.get("name") in ("l4_24gb", "a100_40gb", "g4_96gb")) else eval_bs

        print(f"\n=== Evaluating {exp} (pct={pct}%, split={args.eval_split}, bs={cur_bs}, ckpt={adapter_ckpt or 'frozen'}) ===")
        model, processor = build_vlm_model_and_processor(
            model_name_or_key=args.model,
            load_in_4bit=use_4bit,
            use_lora=False,
            architecture_mode=arch_mode,
            min_pixels=hw["min_pixels"],
            max_pixels=hw["max_pixels"],
            smolvlm_longest_edge=hw.get("smolvlm_longest_edge", 1536),
            gradient_checkpointing=False,
            attn_implementation=hw["attn_implementation"],
            adapter_checkpoint=adapter_ckpt,
            target_modules=model_info.get("target_modules"),
            exclude_modules=model_info.get("exclude_lora_modules"),
        )
        try:
            metrics = evaluate_both_domains(
                model=model,
                processor=processor,
                model_type=model_type,
                eval_split=args.eval_split,
                run_dir=run_rel,
                device=device,
                include_ocr=args.include_ocr,
                max_new_tokens=base_cfg["training"]["max_new_tokens"],
                eval_batch_size=cur_bs,
                max_eval_samples=args.max_eval_samples,
                local_data_dir=args.local_data_dir,
                bucket_by_resolution=not args.no_bucket_by_resolution,
                seed=args.seed,
            )
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        metrics_path = run_rel / f"{args.eval_split}_metrics.json"
        payload = {
            "exp": exp,
            "model": args.model,
            "seed": args.seed,
            "few_shot_pct": pct,
            "eval_split": args.eval_split,
            "checkpoint_tag": args.checkpoint_tag if arch_mode != "none" else "frozen",
            "metrics": metrics,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {metrics_path}: {metrics}")
        all_summaries.append(payload)

    summary_out = Path(base_cfg["runs_root"]) / args.model / ckpt_ns / f"{args.eval_split}_eval_summary_seed{args.seed}.json"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nCompleted {len(all_summaries)} {args.eval_split} evaluations. Summary: {summary_out}")


if __name__ == "__main__":
    main()
