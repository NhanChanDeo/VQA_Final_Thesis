# CrossVQA Training Regime: `ALT_THREE_SPLITS`

> **Target HF Dataset Revision:** `gamusa/docvqa-canonical@three-splits` & `gamusa/textvqa-canonical@three-splits`  
> **Locked Backbone Priority:** `Qwen/Qwen2-VL-2B` (Primary) $\rightarrow$ `HuggingFaceTB/SmolVLM-Base` (Secondary 1) $\rightarrow$ `HuggingFaceTB/SmolVLM-500M-Base` (Secondary 2 / Pilot)  
> **Preferred GPU Profile:** Colab **L4 (24 GB, `bf16`)** with automatic fallback to **T4 (16 GB, `fp16`)**.

---

## Directory Layout

* `configs/` — Locked YAML configurations (`backbones.yaml`, `baselines_B1_B5.yaml`, `crossvqa_M1_M3.yaml`, `few_shot.yaml`).
* `scripts/` — CLI runners (`generate_few_shot_splits.py`, `run_z0_domain_gap.py`, `run_baseline.py`, `run_crossvqa.py`, `build_training_notebooks.py`).
* `notebooks/` — Self-contained Google Colab GPU notebooks (`00_...` through `05_...`) with masked `getpass` token inputs, private HF Hub sync (`gamusa/crossvqa-checkpoints`), and Google Drive backup mirroring.
* `runs/` — Git-tracked lightweight metrics, JSONL step logs, UMAP figures, and `run_card.yaml` files (model weights are excluded via `.gitignore`).

## Quick CLI Reference

```bash
# 1. Zero-shot Z0 + Phase A8 UMAP + Phase A9 MMD/Proxy A-distance
#    Defaults to pure bf16/fp16 (no 4-bit overhead), Stage A/B validation embedding fusion,
#    local Arrow multi-worker prefetch, resolution bucketing, and GPU RKHS MMD.
#    Optional overrides: --load_in_4bit, --eval_batch_size N, --num_workers N, --no_bucket_by_resolution
python TRAIN/ALT_THREE_SPLITS/scripts/run_z0_domain_gap.py --model qwen2_vl_2b

# 2. Baselines B1–B5 (OCR-free Track A default; pass --include_ocr for Track B ablation)
python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py --exp B1 --model qwen2_vl_2b --seed 42
python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py --exp B4 --model qwen2_vl_2b --seed 42 --stage1_exp B2_source_only

# 3. Proposed CrossVQA M1–M3
python TRAIN/ALT_THREE_SPLITS/scripts/run_crossvqa.py --exp M2 --model qwen2_vl_2b --seed 42
```
