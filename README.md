# CrossVQA: Cross-Domain Knowledge Transfer from Document Images to Natural Scene Images for Vision-Language Models

[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/gamusa)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Undergraduate Graduation Thesis**  
> **Institution:** Posts and Telecommunications Institute of Technology ([PTIT](https://portal.ptit.edu.vn/)), Hanoi, Vietnam  
> **Faculty:** Faculty of Information Technology  
> **Author:** Tung Tran  
> **Detailed Execution Plans:** See [`PLAN/`](PLAN/) (Available in [English](PLAN/12-08-00_2026-10-09_(EN)_CrossVQA_Lossless_Branch_Inspection_and_Thesis_Execution_Plan.md) and [Vietnamese](PLAN/11-30-00_2026-10-09_CrossVQA_Lossless_Branch_Inspection_and_Thesis_Execution_Plan.md))

---

## 1. Research Overview

This graduation research investigates **Cross-Domain Knowledge Transfer** for text-centric Vision-Language Models (VLMs), aiming to bridge the multimodal representational gap between two distinct visual-textual domains:

1. **Source Domain — Document Images ([DocVQA](https://huggingface.co/datasets/gamusa/docvqa-canonical)):** Structured, planar 2D document scans with dense tabular layouts, high aspect ratios, high resolution ($1,700 \times 2,200 \rightarrow 7,000 \times 9,000$ px), and formal semantics.
2. **Target Domain — Natural Scene Images ([TextVQA](https://huggingface.co/datasets/gamusa/textvqa-canonical)):** Unstructured 3D scene photographs characterized by arbitrary perspective distortions, extreme illumination variances, non-uniform orientations, artistic scripts, and complex scene-object relationships.

### Core Research Questions (RQs)
* **RQ1 (Domain Gap Characterization):** How does the representational divergence between 2D documents and 3D scenes manifest in multimodal latent feature space (measured via MMD and Proxy $\mathcal{A}$-distance)?
* **RQ2 (Knowledge Transfer Mechanism):** How can reading comprehension representations be effectively transferred from structured documents to scene text without deteriorating visual-object grounding?
* **RQ3 (Catastrophic Forgetting Mitigation):** How can parameter-efficient architectures (e.g., Domain-Specific Adapters) prevent catastrophic forgetting on the source document domain during transfer?

---

## 2. Canonical Benchmark Datasets (Hugging Face Hub)

Both foundational datasets have been thoroughly standardized, cured of generational compression loss, and officially published on the Hugging Face Hub in sharded Parquet format with native streaming and fast page indexing.

### 2.1 Repository Links
* **DocVQA Canonical:** [`https://huggingface.co/datasets/gamusa/docvqa-canonical`](https://huggingface.co/datasets/gamusa/docvqa-canonical)
* **TextVQA Canonical:** [`https://huggingface.co/datasets/gamusa/textvqa-canonical`](https://huggingface.co/datasets/gamusa/textvqa-canonical)

### 2.2 Dual-Branch Architecture

To ensure experimental reproducibility while providing full historical archiving, both repositories maintain a synchronized **Dual-Branch Architecture**:

| Repository Branch | Purpose & Evaluation Role | TextVQA Scale | DocVQA Scale | Total Size |
| :--- | :--- | :--- | :--- | :--- |
| **`main`** *(Recommended)* | **Curated Experimental Benchmark:** Contains only clean, fully labelled `train` and `validation` splits. The blind `test` split (which permanently withheld labels as `answers = []`) was intentionally pruned to eliminate dead-weight download overhead (saving 2.01 GB) and prevent split leakage. | 16 shards (6.64 GB)<br>39,602 samples | 57 shards (8.35 GB)<br>44,812 samples | **14.99 GB**<br>(84,414 samples) |
| **`lossless`** | **Full Archival Benchmark:** Preserves all 3 splits (`train`, `validation`, `test`) without pruning. Maintained if generating blind predictions for external challenge evaluation servers. | 19 shards (7.54 GB)<br>45,336 samples | 64 shards (9.46 GB)<br>50,000 samples | **17.00 GB**<br>(95,336 samples) |

### 2.3 Key Technical Innovations
1. **True Lossless Visual Preservation:**
   * **DocVQA:** Preserves **100% original PNG byte streams** from high-resolution document scans, completely eliminating lossy JPEG compression ringing artifacts around small characters and table lines.
   * **TextVQA:** Retains original JPEG bytes (bit-for-bit) for upright images; applies `ImageOps.exif_transpose` and maximum quality encoding (`quality=100, subsampling=0` 4:4:4) for rotated images to maintain exact pixel-level alignment with Rosetta OCR bounding boxes.
2. **PyArrow Streaming Sharding:**
   * Shards partitioned by sample budget: **DocVQA has 50 train shards (800 samples/shard, ~150–350 MB)**; **TextVQA has 14 train shards (2,500 samples/shard, ~350–500 MB)**.
   * Built with `write_page_index=True`, enabling sub-second random seeking and streaming in DuckDB, PyArrow, and Hugging Face Dataset Viewer without RAM saturation.
3. **Unified 15-Column Canonical Schema:**
   Eliminates all schema mismatches across both datasets, providing a unified downstream ingestion interface:

```text
├── sample_id        : string (Unique identifier, e.g., 'textvqa_34602', 'docvqa_49153')
├── domain_type      : string ('scene_text' | 'document_text')
├── question_id      : string
├── question         : string
├── image            : struct {'bytes': binary, 'path': null} (Decodable via PIL)
├── image_format     : string ('PNG' | 'JPEG')
├── image_width      : int32
├── image_height     : int32
├── answers          : list<string> (Full list of ground-truth annotations)
├── canonical_answer : string (Normalized majority-vote answer)
├── ocr_tokens       : list<string> (Rosetta OCR for TextVQA, MS Azure for DocVQA)
├── ocr_boxes        : list<list<float32>> ([ymin, xmin, ymax, xmax] normalized to [0, 1000])
├── source_dataset   : string ('textvqa' | 'docvqa')
├── original_split   : string ('train' | 'validation' | 'test')
└── metadata         : string (Serialized JSON with dataset-specific auxiliary fields)
```

---

## 3. Quickstart: Streaming DataLoader

You can stream and inspect samples directly from Hugging Face Hub without downloading the full 15 GB archive:

### Installation
```bash
pip install datasets pillow pyarrow
```

### Python Streaming Snippet
```python
import io
from PIL import Image
from datasets import load_dataset

# Stream from curated 'main' branch (default)
repo_id = "gamusa/docvqa-canonical"
dataset = load_dataset(repo_id, revision="main", split="validation", streaming=True)

for sample in dataset:
    # Decode raw bytes into PIL Image
    image = Image.open(io.BytesIO(sample["image"]["bytes"]))
    
    print(f"Sample ID        : {sample['sample_id']}")
    print(f"Domain Type      : {sample['domain_type']}")
    print(f"Question         : {sample['question']}")
    print(f"Canonical Answer : {sample['canonical_answer']}")
    print(f"Image Format     : {sample['image_format']} ({image.size})")
    print(f"OCR Word Count   : {len(sample['ocr_tokens'])}")
    break
```

---

## 4. Repository Layout

```text
├── DATASETS SCHEMA NORMALIZATION/  # Normalization pipelines, lossless encoding & sharding (Steps 6–10)
│   ├── Step10_Mirror_Lossless_To_Main_Without_Test.ipynb # Server-side mirror and test pruning
│   ├── Step9_Normalize_CrossVQA_Parquet_lossless_branch.ipynb # Lossless streaming generator
│   ├── canonical_schema.py         # 15-column PyArrow & Hugging Face schema definitions
│   ├── normalize_docvqa.py         # DocVQA normalization logic
│   └── normalize_textvqa.py        # TextVQA normalization logic
├── DATASETS UPLOAD/                # Initial raw ingestion & HF repository setup (Steps 1–5)
├── PLAN/                           # Thesis premise, academic feedback & detailed 14-week plans
│   ├── 12-08-00_..._(EN)_CrossVQA_..._Plan.md # Master execution plan (English)
│   ├── 11-30-00_..._CrossVQA_..._Plan.md      # Master execution plan (Vietnamese)
│   ├── ThesisPremise.md            # Foundational thesis topic statement
│   └── Papers_List.md              # Academic literature review references
├── ANALYSIS/                       # Phase A: EDA, MMD, Proxy A-distance & UMAP [Under Development]
├── DATA_PIPELINE/                  # Phase B: Grouped leak-free splitters, AnyRes Tiling & DataLoaders [Under Development]
├── TRAIN/                          # Phase C: Baselines (Z0, B1-B5) & Proposed CrossVQA LoRA Architectures [Under Development]
├── EVALUATION/                     # Phase D: Benchmark official val, 95% Bootstrap CI & 7-class Error Taxonomy [Under Development]
├── DEMO/                           # Phase E: Interactive Gradio Web App & FastAPI REST service [Under Development]
└── README.md                       # Project documentation & overview
```

---

## 5. Project Roadmap & Milestone Tracker

- [x] **Milestone 0 (Data Collection):** Fetched raw DocVQA and TextVQA source assets.
- [x] **Milestone 1 (Schema Normalization):** Designed unified 15-column Canonical Schema.
- [x] **Milestone 2 (Lossless Engineering):** Formulated True Lossless PNG for DocVQA and Native JPEG bit-for-bit for TextVQA.
- [x] **Milestone 3 (Streaming Sharding & Publishing):** Exported Parquet shards with page index to Hugging Face Hub branch `lossless` (17.00 GB).
- [x] **Milestone 4 (Benchmark Modernization):** Executed Step 10 server-side mirror to branch `main` with unlabelled test split pruning (14.99 GB).
- [ ] **Phase A (Exploratory Data Analysis):** Quantify multimodal domain discrepancy (MMD, Proxy $\mathcal{A}$-distance) and empirical OCR coverage ceilings.
- [ ] **Phase B (Data Pipeline Engineering):** Build grouped leak-free splitters (`image_id` / `ucsf_document_id`) and Dynamic High-Res Tiling (AnyRes).
- [ ] **Phase C (Cross-Domain Training):** Implement Baselines (Z0, B1–B5) and proposed CrossVQA architecture (Domain-Specific LoRA + Feature Alignment Loss).
- [ ] **Phase D (Evaluation & Analysis):** Benchmark on frozen official validation splits, compute 95% bootstrap confidence intervals, and formalize 7-class failure taxonomy.
- [ ] **Phase E (Deployment & Thesis Manuscript):** Package interactive Gradio web demo on Hugging Face Spaces and finalize thesis dissertation chapters.

---

## 6. Citation & Academic Attribution

If you utilize the CrossVQA canonical datasets or execution framework in your research, please cite:

```bibtex
@misc{crossvqa2026,
  author       = {Tung Tran},
  title        = {CrossVQA: Cross-Domain Knowledge Transfer from Document Images to Natural Scene Images for Vision-Language Models},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/gamusa}},
  note         = {Graduation Thesis, Posts and Telecommunications Institute of Technology (PTIT)}
}
```