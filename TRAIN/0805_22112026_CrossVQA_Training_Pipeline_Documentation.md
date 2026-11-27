# CrossVQA Phase C — Training Pipeline Architecture & Technical Rationale

> **Document Type:** End-to-End Training & Evaluation Pipeline Specification  
> **Directory Scope:** `TRAIN/src/`, `TRAIN/ALT_THREE_SPLITS/`, and `EVALUATION/ALT_THREE_SPLITS/`  
> **Locked Canonical Datasets:** `gamusa/docvqa-canonical@three-splits` & `gamusa/textvqa-canonical@three-splits`  
> **Revision Note (2026-11-27):** Synced hardware profiles with live `ebs64` configs, documented the `collator.py` EOS-safe padding fix (`attention_mask == 0`), added Phase C/D `test` evaluation & statistical scripts, and defined the **2-Week Active Scope (`Option A`: `Z0, B1–B4, Few-Shot`, with `M1–M3` retained as Implemented Architectural Prototype / Future Work)**.

---

## 1. Executive Architecture & Directory Layout

The training and evaluation codebase separates **regime-agnostic core ML modules** (`TRAIN/src/`) from **split-specific configurations, CLI runners, Colab notebooks, and experiment logs** (`TRAIN/ALT_THREE_SPLITS/`, `EVALUATION/ALT_THREE_SPLITS/`).

```text
DATA_PIPELINE/ALT_THREE_SPLITS/splits/
├── crossvqa_split_manifest_pooled.parquet   # 84,414-row zero-leakage pooled split manifest
└── textvqa_few_shot_manifest.parquet        # Nested image_id-grouped 1/5/10/25/100% subsets

TRAIN/
├── 0805_22112026_CrossVQA_Training_Pipeline_Documentation.md
├── VLM_Backbones_Technical_Reference.md
├── src/                                     # Shared core library (regime-agnostic)
│   ├── __init__.py
│   ├── dataset.py                           # Local NVMe Parquet / HF streaming + B3 & M1-M3 samplers
│   ├── collator.py                          # Image placeholders, answer-only label masking, EOS-safe padding
│   ├── models.py                            # Locked backbone loader, selective 4-bit NF4, PEFT/Dual-LoRA
│   ├── adapters.py                          # Prototype M1: DualLoRALinear + VisualSoftDomainRouter
│   ├── losses.py                            # Prototype M2 & Phase A9: RBF MMD, InfoNCE, FIFO Queue, Proxy A-dist
│   ├── augmentations.py                     # Prototype M3: CurriculumPseudoDomainTransform
│   ├── trainer.py                           # Hardware profiles (L4/T4), CheckpointManager, CrossVQATrainer
│   └── metrics.py                           # ANLS, VQA Accuracy, TG, FR, JSONL predictions, Run Cards
├── tests/
│   └── test_pipeline_smoke.py               # Offline CPU verification suite (all unit + smoke tests)
└── ALT_THREE_SPLITS/
    ├── README.md
    ├── configs/                             # Locked YAML specifications
    │   ├── backbones.yaml
    │   ├── baselines_B1_B5.yaml
    │   ├── crossvqa_M1_M3.yaml
    │   └── few_shot.yaml
    ├── scripts/                             # Reproducible CLI scripts
    │   ├── _common.py
    │   ├── generate_few_shot_splits.py
    │   ├── run_z0_domain_gap.py
    │   ├── run_baseline.py
    │   ├── run_crossvqa.py                  # Prototype M1-M3 runner (deferred in 2-week plan)
    │   ├── eval_test_checkpoints.py         # Held-out NEW_test evaluation across saved checkpoints
    │   └── build_training_notebooks.py
    ├── notebooks/                           # Colab GPU notebooks (00..05 + split 01_B1/01_B2)
    └── runs/                                # Git-tracked metrics, JSONL logs & UMAP plots (weights gitignored)

EVALUATION/ALT_THREE_SPLITS/
├── evaluate_statistical_suite.py            # Offline 95% Bootstrap CIs, paired permutation p-values, LaTeX tables
└── sample_error_taxonomy.py                 # Stratified 200-sample E1-E7 error taxonomy worksheet builder
```

### Key Architectural Principles
1. **Relative Path Discipline:** Configs, scripts, and notebooks resolve paths relative to the repository root (`DATA_PIPELINE/...`, `TRAIN/...`, `EVALUATION/...`).
2. **Ephemeral Colab Resilience:** Parquet shards (`train`, `validation`, `test`) stage directly to Colab NVMe (`/content/data/`), while adapters + `resume_state.pt` sync every `save_every_steps=64` optimizer steps (and at epoch end) to private HF Hub (`gamusa/crossvqa-checkpoints`, namespace `ebs64`) and Google Drive (`/content/drive/MyDrive/CrossVQA_Checkpoints/ALT_THREE_SPLITS`).
3. **Zero Token Exposure:** Colab notebooks read `HF_TOKEN` and optional `GH_TOKEN` via `google.colab.userdata` or masked `getpass` prompts; Drive log mirroring excludes `checkpoints/` and binary weight extensions.

---

## 2. Locked Backbone Hierarchy & Hardware Profiles

### 2.1 Backbone Priority Order (`configs/backbones.yaml` & `src/models.py`)

| Priority | Key | Hugging Face Repo ID | Architecture | 2-Week Role |
| :---: | :--- | :--- | :--- | :--- |
| **1 (Primary)** | `smolvlm_500m` | `HuggingFaceTB/SmolVLM-500M-Base` | `Idefics3` (12L ViT + Pixel Shuffle + 32L Llama) | **Active 2-Week Training Backbone (`Z0, B1–B4, Few-Shot`):** Pure `bf16`/`fp16` LoRA (`allow_4bit=false`), non-saturated `Z0` floor (`0.0059` DocVQA ANLS / `0.1963` TextVQA Acc). |
| **2 (Secondary 1)** | `smolvlm_base` | `HuggingFaceTB/SmolVLM-Base` | `Idefics3` (27L ViT + Pixel Shuffle + 24L SmolLM2) | **Deferred (2-Week Cut):** Intra-family 2.2B scale control (`allow_4bit=true`). |
| **3 (Secondary 2)** | `qwen2_vl_2b` | `Qwen/Qwen2-VL-2B` | 32L ViT + 2×2 Merger + 28L Qwen2 (M-RoPE) | **Saturation Reference (`Z0` Complete):** High zero-shot OCR (`0.9458` DocVQA ANLS / `0.8153` TextVQA Acc) establishes capacity ceiling; fine-tuning deferred. |

* **Contamination Hygiene:** All three checkpoints are **Base** (non-`-Instruct`) checkpoints.

### 2.2 Selective 4-Bit NF4 Quantization & Vision Module Parity
* `smolvlm_500m` sets `allow_4bit: false` and trains in pure `bfloat16` (SM $\ge 8.0$) or `float16` (T4 SM $7.5$).
* `smolvlm_base` and `qwen2_vl_2b` enable 4-bit NF4 QLoRA on the LLM decoder only, exempting `["vision_model", "connector"]` (SmolVLM) or `["visual", "merger"]` (Qwen2-VL).
* Standard LoRA (`r=16, lora_alpha=32, lora_dropout=0.05`) targets `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` while excluding `["vision_model", "visual"]`.

### 2.3 Hardware Profiles & `ebs64` Hyperparameters (`backbones.yaml`, `baselines_B1_B5.yaml`, `trainer.py`)
`resolve_hardware_profile("auto")` selects from `preferred_order: ["l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"]`, targeting **effective batch size = 64** (`checkpoint_namespace: "ebs64"`), `learning_rate = 4.0e-4`, cosine schedule with `warmup_ratio = 0.05`, `weight_decay = 0.01`, `max_grad_norm = 1.0`, `max_new_tokens = 32`, `epochs = 2`, and `smolvlm_longest_edge = 1536`:

| Profile | Trigger (VRAM / SM) | Dtype / 4-bit | Grad Ckpt | Train BS $\times$ Accum (Eff=64) | Eval BS | `max_pixels` (Qwen2-VL) | Optimizer |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`l4_24gb`** *(Primary)* | $\ge 20\text{ GB}$, $\text{SM} \ge 8$ | `bf16` / NF4* | `True` | `16 × 4` | `16` | `1,310,720` | `adamw_torch_fused` |
| **`a100_40gb`** | $\ge 36\text{ GB}$, $\text{SM} \ge 8$ | `bf16` / NF4* | `False` | `16 × 4` | `16` | `1,572,864` | `adamw_torch_fused` |
| **`g4_96gb`** | $\ge 85\text{ GB}$, $\text{SM} \ge 10$ | `bf16` / `False` | `False` | `16 × 4` | `32` | `2,097,152` | `adamw_torch_fused` |
| **`t4_16gb`** *(Fallback)* | $< 20\text{ GB}$ | `fp16` / NF4* | `True` | `8 × 8` | `8` | `1,048,576` | `paged_adamw_8bit` |

*\*Note: `smolvlm_500m` overrides `load_in_4bit` to `False` (`allow_4bit: false`). On `Z0` inference (`run_z0_domain_gap.py`), `load_in_4bit=False` is the default and `eval_batch_size` auto-doubles on L4/A100/G4.*

---

## 3. Multimodal Data Processing & Collator Guarantees (`dataset.py`, `collator.py`)

### 3.1 Dual-Source Ingestion & Resolution Bucketing
* **NVMe-First Loading:** `load_crossvqa_dataset()` reads `data/{split}-*.parquet` from `--local_data_dir` (Colab `/content/data/{docvqa,textvqa}`) with Arrow memory mapping and falls back to HF streaming otherwise.
* **Experiment Samplers:** `InterleavedCrossVQADataset` (`B3 Joint`, `doc_prob=0.5`) and `PairedDomainDataset` (`M1–M3` prototype). Map-style datasets use deterministic mega-batch resolution bucketing (`group_by_resolution=True`) aligned to `batch_size` boundaries for exact O(1) mid-epoch resume.

### 3.2 Prompt Conditioning, Label Masking & EOS Preservation (`collator.py`)
* **Track A (Primary — OCR-Free):** Default (`--include_ocr` omitted): `<image>\nQuestion: {question}\nAnswer:` (or Qwen2-VL chat template).
* **Critical EOS-Safe Padding Mask (Fixed 2026-11-27):**
  - On `HuggingFaceTB/SmolVLM-500M-Base`, **`pad_token_id == eos_token_id == 2` (`<|im_end|>`)**.
  - Masking via `labels[labels == pad_id] = -100` previously erased the target `<|im_end|>` token on every sample, causing runaway greedy generation and metric collapse (`B1 textvqa_vqa_acc = 0.0`).
  - `CrossVQADataCollator` now masks trailing padding strictly via **`labels[inputs["attention_mask"] == 0] = -100`** (falling back to `pad_id` only if `attention_mask` is absent and `pad_id != eos_id`), preserving `<|im_end|>` supervision on every example (verified by `test_collator_preserves_eos_when_pad_equals_eos`).
* **Evaluation:** Uses `padding_side="left"`, `use_cache=True`, `do_sample=False`, `max_new_tokens=32`, slicing generated tokens from `input_ids.shape[1]`.

---

## 4. Experimental Matrix, Prototype Architecture & 2-Week Active Scope

### 4.1 Active 2-Week Empirical Matrix (`Z0`, `B1–B4`, Few-Shot)
* **Z0 (`run_z0_domain_gap.py` — DONE):** Frozen zero-shot benchmark + UMAP (`A8`) + RKHS MMD and 5-fold Linear Proxy $\mathcal{A}$-distance (`A9`) testing **H1**.
* **B1 (Target-Only — TextVQA `train`, 2 ep):** Reference target specialization bound (**H2, H4**).
* **B2 (Source-Only — DocVQA `train`, 2 ep):** Tests OOD transfer `Doc -> Text` (**H2**) and provides `B2/best` for `B4` + $\text{ANLS}_{\text{B2}}$ for Forgetting Rate (**H3**).
* **B4 (Sequential `Doc -> Text`, 2 ep):** Resumes from `B2/best` and fine-tunes on TextVQA; directly quantifies Catastrophic Forgetting ($\text{FR} = \text{ANLS}_{\text{B2}} - \text{ANLS}_{\text{B4}}$) (**H3**).
* **B3 (Joint Interleaved LoRA, 1 ep $\approx 70.8\text{k}$ samples):** Tests whether joint multi-domain training mitigates sequential forgetting ($\text{FR}_{\text{B3}} < \text{FR}_{\text{B4}}$) vs. single-domain negative transfer (**H5, Revised**).
* **Few-Shot Scaling (`1%, 5%, 10%, 25%` on `B1, B4`):** Evaluates whether relative transfer gain ($\text{TG}$) peaks under data scarcity (**H4**).

### 4.2 Implemented Architectural Prototype (`M1–M3` — Deferred in 2-Week Plan)
The full Dual-LoRA + Feature Alignment + Curriculum Homography architecture remains implemented and tested (`test_pipeline_smoke.py`) in `TRAIN/src/{adapters,losses,augmentations}.py` and [`run_crossvqa.py`](ALT_THREE_SPLITS/scripts/run_crossvqa.py) as **Future Research / Architectural Extension**:
* **M1 (`adapters.py`):** `DualLoRALinear` ($\mathcal{A}_{\text{doc}}, \mathcal{A}_{\text{scene}}$) + `VisualSoftDomainRouter` with frozen $\mathcal{A}_{\text{doc}}$ during Stage 2.
* **M2 (`losses.py`):** Zero-initialized `ProjectorAlignmentAdapter` (`rank=64`) + `MultiKernelRBFMMDLoss` / `CrossDomainInfoNCELoss` with FIFO `FeatureQueue` (`size=256`).
* **M3 (`augmentations.py`):** `CurriculumPseudoDomainTransform` (3D perspective homography, directional shadow, optical blur/JPEG severity ramp $0.2 \rightarrow 1.0$).

### 4.3 Hypothesis Traceability & 2-Week Active Execution Scope

| Hypothesis | Active 2-Week Verification Artifact | Est. L4 Cost (`smolvlm_500m`, `seed=42`) | Disposition |
| :---: | :--- | :---: | :---: |
| **H1 (RQ1)** | `Z0` UMAP, RKHS MMD (`0.919`), Proxy $\mathcal{A}$-dist (`1.992` / `2.000`) | **0h** | **DONE** |
| **H2 (RQ2)** | `B2` (DocVQA-only) vs. `Z0` and `B1` (TextVQA-only) | **~23.5h** | **ACTIVE (P0)** |
| **H3 (RQ2, RQ3)** | `B4` (Sequential `Doc -> Text`) Forgetting Rate $\text{FR}$ vs. `B2` | **~10.5h** | **ACTIVE (P0)** |
| **H4 (RQ2)** | Few-Shot `1%, 5%, 10%, 25%` learning curves (`B1` vs. `B4`) | **~13.4h** | **ACTIVE (P0)** |
| **H5 (RQ3, Revised)** | `B3` (Joint 1 ep) forgetting & negative-transfer audit vs. `B4, B1, B2` | **~11.5h** | **ACTIVE (P0)** |
| **Phase D (`test`)** | `eval_test_checkpoints.py` + `evaluate_statistical_suite.py` + `sample_error_taxonomy.py` | **~3.0h GPU + Offline CPU** | **ACTIVE (P0)** |
| *Extensions* | `M1–M3`, `B5`, 3-seed sweeps (`123, 999`), `smolvlm_base`/`qwen2_vl_2b` fine-tuning, Track B OCR, `VALIDATION_AS_TEST` | **0h** | **CUT / FUTURE WORK** |

---

## 5. Checkpointing, Evaluation & Colab Workflow (`trainer.py`, `EVALUATION/`)

* **Triple-Redundant Checkpointing & O(1) Resume:** Saves `latest/` (`resume_state.pt` with optimizer, scheduler, RNG, `samples_seen_in_epoch`, and config fingerprint) every `64` optimizer steps and `best/` at epoch validation. Cross-GPU resume (e.g., T4 `PagedAdamW8bit` $\leftrightarrow$ L4 `adamw_torch_fused`) safely resets incompatible optimizer momentum while keeping weights, scheduler, and dataset position.
* **Colab Notebooks (`TRAIN/ALT_THREE_SPLITS/notebooks/`):**
  - `00_Z0_ZeroShot_and_DomainGap.ipynb` (Complete)
  - `01_B1_TextVQA_Only_LoRA.ipynb` & `01_B2_DocVQA_Only_LoRA.ipynb` (Split single-domain runners with scoped Cell 5 loss/metric inspection)
  - `02_B3_Joint_MultiDomain_LoRA.ipynb`, `03_B4_B5_Sequential_Transfer.ipynb`, `05_FewShot_Scaling_Matrix.ipynb`
  - `04_M1_M3_CrossVQA_Proposed.ipynb` (Deferred prototype)
* **Phase D Offline Evaluation (`EVALUATION/ALT_THREE_SPLITS/`):**
  - [`eval_test_checkpoints.py`](ALT_THREE_SPLITS/scripts/eval_test_checkpoints.py) runs held-out `NEW_test` evaluation on `Z0` and saved `best` LoRA adapters.
  - [`evaluate_statistical_suite.py`](../EVALUATION/ALT_THREE_SPLITS/evaluate_statistical_suite.py) computes ANLS, VQA Accuracy, Harmonic Mean, $\text{TG}$, $\text{FR}$, **95% Bootstrap CIs ($B=1,000$)**, and **paired permutation $p$-values**, exporting Markdown/LaTeX tables and learning-curve plots.
  - [`sample_error_taxonomy.py`](../EVALUATION/ALT_THREE_SPLITS/sample_error_taxonomy.py) exports 200 stratified failures (`100` DocVQA + `100` TextVQA) for `E1–E7` manual error analysis.
