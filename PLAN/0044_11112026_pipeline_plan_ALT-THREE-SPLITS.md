# CrossVQA Execution Plan: De Novo Pooled 3-Way Splitting Pipeline (ALT-THREE-SPLITS)

> **Document Type:** Production Pipeline Specification & Execution Plan  
> **Phase:** Phase B -- Alternative Partitioning Experiment (`DATA_PIPELINE/ALT_THREE_SPLITS/`)  
> **Target Canonical Repositories:** [`gamusa/docvqa-canonical`](https://huggingface.co/datasets/gamusa/docvqa-canonical) & [`gamusa/textvqa-canonical`](https://huggingface.co/datasets/gamusa/textvqa-canonical)  
> **Primary Execution Runtime:** Google Colab VM (High-speed network & NVMe scratch storage)  
> **Local Audit Runtime:** Local Workstation (100% Offline verification)  
> **Date Generated:** 2026-11-11 00:44  

---

## 1. Executive Summary & Objective

In the baseline CrossVQA protocol, official `validation` was strictly preserved as `NEW_test` to maintain 100% academic comparability with published literature. However, this inherited 523 overlapping multi-page parent documents between train and validation from Mathew et al. (2021).

The **`ALT_THREE_SPLITS`** pipeline pools the original `train` and `validation` splits (84,414 queries total) into **one unified corpus**, removes the historical partition boundaries, and executes a **De Novo 2-Stage GroupShuffleSplit** with an integrated **Visual Similarity Gate**. 

### Primary Goals:
1. **Total Eradication of Historical Document Leakage:** $0$ shared parent documents and $0$ shared images across all 3 partitions.
2. **Strict Statistical Power Floors:**
   - `NEW_test`: $\text{MoE}_{95\%} \le \pm 1.45\%$ ($N \approx 4,600 - 4,900$ queries)
   - `NEW_validation`: $\text{MoE}_{95\%} \le \pm 1.75\%$ ($N \approx 3,200 - 3,500$ queries)
   - `NEW_train`: Retains $\approx 81\%$ of total volume ($N \approx 32,000 - 36,500$ queries)
3. **Dual Verification Architecture:** Pure metadata audit first, followed by a cross-partition perceptual hashing audit (`pHash`, $d_H \le 3$).

---

## 2. End-to-End Pipeline Architecture

```
                                  [ Google Colab VM ]
                                           │
        ┌──────────────────────────────────┴──────────────────────────────────┐
        │                                                                     │
        ▼                                                                     ▼
 [PHASE 1: Ingestion & Metadata]                           [PHASE 2: 2-Stage Group Split]
  • Fast HF Transfer to /content/data/                      • Global group aggregation (no split col)
  • Polars metadata projection (drop raw pixels)            • Outlier pinning (|z| > 3.0 mega-docs)
  • Cache: textvqa/docvqa_polars_meta.parquet               • Stage 1: Carve NEW_test (11%, MoE <= 1.45%)
                                                            • Stage 2: Carve NEW_val  (8%,  MoE <= 1.75%)
                                                            • Remainder: NEW_train    (~81%)
                                                                          │
                                                                          ▼
                                                           [PHASE 3: Visual Similarity Gate]
                                                            • Compute 64-bit pHash on images in /content
                                                            • Audit cross-split pairs (Train ∩ Test/Val)
                                                            • If duplicates: DSU merge -> re-split
                                                            • If clean (dH > 3): Sign-off candidate
                                                                          │
                                                                          ▼
                                                           [PHASE 4: Artifact Export & Git Push]
                                                            • Export crossvqa_split_manifest_pooled.parquet
                                                            • Export group aggregate tables & figures
                                                            • Push lightweight artifacts (~500 KB) to GitHub
        │                                                                     │
        └──────────────────────────────────┬──────────────────────────────────┘
                                           │  git pull
                                           ▼
                              [ Local Machine / Workstation ]
                                           │
        ┌──────────────────────────────────┴──────────────────────────────────┐
        │                                                                     │
        ▼                                                                     ▼
 [PHASE 5: Offline Integrity Certification]                [PHASE 6: Optional HF Hub Branch]
  • Run Check_Leakage_Pooled(local).ipynb                   • Materialize & push physical shards
  • 100% offline verification (0 bytes downloaded)            to Hugging Face branch `three-splits`
  • Verify 0 shared documents across all 3 splits           • (Only when splits are 100% finalized)
  • Formal thesis audit table generated
```

---

## 3. Detailed Phase Execution Protocols

### Phase 1: High-Speed Ingestion & Polars Projection (Colab VM)
* **Notebook:** `DATA_PIPELINE/ALT_THREE_SPLITS/Aggregate_and_NEW_Splits_Pooled.ipynb` (Cells 1–3)
* **Storage Location:** `/content/data/` (Ephemeral NVMe scratch disk)
* **Actions:**
  1. Set `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"`.
  2. Snapshot-download Parquet shards from `gamusa/textvqa-canonical` and `gamusa/docvqa-canonical` directly to `/content/data/` ($\approx 60-90\text{ seconds}$ total across 15 GB).
  3. Execute `scan_minimal_metadata_polars()`: Polars projects only the essential indexing columns (`sample_id`, `domain_type`, `split`, `question_id`, `image_id`, `ocr_tokens`, `metadata`) and completely skips decoding binary image bytes.
  4. Write lightweight metadata cache to `DATA_PIPELINE/cache/` ($\approx 10\text{ MB}$).

---

### Phase 2: Statistical 2-Stage Group-First Partitioning (Colab VM)
* **Notebook:** `Aggregate_and_NEW_Splits_Pooled.ipynb` (Cells 4–7)
* **Decision Parameters (Cell 1):**
  ```python
  TARGET_TEST_SHARE = 0.11   # 11.0% (N ≈ 4,600 - 4,900 | MoE <= +-1.45%)
  TARGET_VAL_SHARE  = 0.08   #  8.0% (N ≈ 3,200 - 3,500 | MoE <= +-1.75%)
  TARGET_TRAIN_SHARE= 0.81   # 81.0% (N ≈ 32,000 - 36,500)
  Z_THRESHOLD       = 3.0    # Pin |z| > 3.0 mega-documents to NEW_train
  RANDOM_SEED       = 42     # Deterministic cluster seed
  ```
* **Step-by-step Algorithm:**
  1. **Global Aggregation (Cell 5):** Group by parent identifier **alone** (`ucsf_document_id` for DocVQA, `image_id` for TextVQA) across all 84,414 queries. Remove `split` from grouping columns so documents spanning original train and validation become a single atomic group.
  2. **Outlier Mega-Document Pinning:** Compute mean and standard deviation of `question_count` and `image_count`. Pin all groups where $|z| > 3.0$ (e.g., 66 DocVQA mega-documents with up to 700 questions) strictly to `NEW_train`.
  3. **Stage 1 Split (Test Partition):** Run `GroupShuffleSplit` on the inlier pool to carve out $\approx 11.0\%$ of queries into `NEW_test`.
  4. **Stage 2 Split (Validation Partition):** Run `GroupShuffleSplit` on the remaining inliers to carve out $\approx 8.0\%$ of total queries into `NEW_validation`.
  5. **Train Partition:** Assign the remainder of inliers plus all pinned mega-documents to `NEW_train`.
  6. **Mathematical Assertions:** Assert that all 6 pairwise group and image leakage intersections equal strictly $0$.

---

### Phase 3: Visual Similarity Audit Gate (Colab VM)
* **Objective:** Verify that no cross-partition near-duplicates exist due to camera burst shots (TextVQA) or repeated template scans (DocVQA).
* **Why Colab:** The 15 GB of raw images are already decompressed on Colab's NVMe scratch disk `/content/data/`.
* **Protocol:**
  1. Compute 64-bit `pHash` for all images using `imagehash` ($\approx 2\text{ minutes}$ across 36,599 images).
  2. Scan exclusively for **cross-partition pairs** with Hamming Distance $d_H \le 3$:
     - $\text{NEW\_train} \cap \text{NEW\_test}$
     - $\text{NEW\_train} \cap \text{NEW\_validation}$
  3. **Branch Decision:**
     - **Clean ($d_H > 3$ for all cross-pairs):** Split is mathematically verified. Proceed to Phase 4.
     - **Flagged ($d_H \le 3$ detected):** Pass duplicate pairs to NetworkX Graph Disjoint Set Union (DSU) to construct `super_group_key`, update group table, and re-run Phase 2 (takes 2 seconds).

---

### Phase 4: Artifact Export & GitHub Push (Colab VM)
* **Notebook:** `Aggregate_and_NEW_Splits_Pooled.ipynb` (Cells 8–9)
* **Export Destination:** `DATA_PIPELINE/ALT_THREE_SPLITS/splits/`
  - `crossvqa_split_manifest_pooled.parquet` ($\approx 150\text{ KB}$)
  - `docvqa_document_aggregates_pooled.parquet` & `.csv` ($\approx 120\text{ KB}$)
  - `textvqa_image_aggregates_pooled.parquet` & `.csv` ($\approx 250\text{ KB}$)
  - `DATA_PIPELINE/ALT_THREE_SPLITS/figures/polars_final_partition_summary.png`
* **GitHub Sync (Cell 9):**
  - Commit and push **only** the lightweight manifests and figures ($\approx 500\text{ KB}$ total).
  - *No heavy image shards are pushed to Git history.*

---

### Phase 5: 100% Offline Integrity Certification (Local Workstation)
* **Notebook:** `DATA_PIPELINE/ALT_THREE_SPLITS/Check_Leakage_Pooled(local).ipynb`
* **Prerequisite:** Pull the generated manifests from GitHub:
  ```powershell
  git pull origin main
  ```
* **Execution:**
  - Runs in $< 3\text{ seconds}$ locally without internet or image downloads.
  - Verifies the **10-Point Audit Matrix**:
    1. TextVQA image disjointness (Train vs Val = 0)
    2. DocVQA document disjointness (Train vs Val = 0)
    3. TextVQA image disjointness (All 3 pairs = 0)
    4. DocVQA page disjointness (All 3 pairs = 0)
    5. Sample & Question ID global uniqueness ($84,414$ unique IDs)
    6. Exact total query conservation ($84,414$ queries)
    7. 3-Way split cardinality integrity ($N_{\text{train}} + N_{\text{val}} + N_{\text{test}} = 84,414$)
    8. Mega-document pinning compliance ($|z| > 3.0$ strictly in `NEW_train`)
    9. Margin of Error target compliance ($\text{MoE}_{\text{test}} \le \pm 1.45\%$, $\text{MoE}_{\text{val}} \le \pm 1.75\%$)
    10. Complete eradication of historical overlap ($0$ shared parent documents between test and train)

---

### Phase 6: (Optional) Materialize Physical Shards to Hugging Face
* **When to run:** Only after Phase 5 produces a 10/10 passed certification.
* **Execution (on Colab):**
  - Stream records from `/content/data/` indexed by `crossvqa_split_manifest_pooled.parquet`.
  - Export sharded Parquet files ($\approx 300 - 500\text{ MB}$ each) for `data/train-*`, `data/validation-*`, and `data/test-*`.
  - Push directly to the clean Hugging Face branch [`three-splits`](https://huggingface.co/datasets/gamusa/docvqa-canonical/tree/three-splits) using `huggingface_hub`.

---

## 4. Colab VM Environment & Resource Best Practices

1. **Keep Raw Parquets on Ephemeral Scratch (`/content`):**  
   Never mount Google Drive to store raw Parquet shards (`/content/drive/MyDrive/...`). Google Drive FUSE filesystem has high I/O latency and will exhaust your Google Drive storage quota.
2. **RAM Protection:**  
   Always use Polars projection scanning (`pl.scan_parquet().select(...)`). By never decoding binary image byte strings during the splitting phase, memory usage on Colab peaks at $< 1.5\text{ GB}$ RAM (well within Colab's 12 GB free tier limit).
3. **Deterministic Reproducibility:**  
   Because seeds and outlier cutoffs are locked (`RANDOM_SEED = 42`, `Z_THRESHOLD = 3.0`), the pipeline is 100% deterministic and reproducible across any runtime.

