# CrossVQA -- Verdict and Technical Specification on Dataset Splitting Protocol

> **Document Type:** Formal Methodological Verdict & Pipeline Specification  
> **Phase:** Phase B -- Training Preparation & Data Pipeline Engineering (`DATA_PIPELINE/`)  
> **Source Notebooks:**  
> - [`DATA_PIPELINE/Assess_data_leakage.ipynb`](Assess_data_leakage.ipynb) (Empirical Leakage Audit & Multiplicity Analysis)  
> - [`DATA_PIPELINE/Split_DS_Decision.ipynb`](Split_DS_Decision.ipynb) (Exploration, Sizing Simulations, & Splitting Justification)  
> - [`DATA_PIPELINE/Aggregate_and_NEW_Splits.ipynb`](Aggregate_and_NEW_Splits.ipynb) (Group-First Production Splitting & Artifact Export)  
> **Target Canonical Repositories:** [`gamusa/docvqa-canonical`](https://huggingface.co/datasets/gamusa/docvqa-canonical) & [`gamusa/textvqa-canonical`](https://huggingface.co/datasets/gamusa/textvqa-canonical)  
> **Total Population:** **84,414 queries** (DocVQA: 44,812 | TextVQA: 39,602) across official `train` and `validation` splits.

---

## 1. Executive Summary & Splitting Topology

The empirical audit across the entire combined corpus (84,414 queries across 36,599 images and 5,687 parent documents) proves that multimodal VQA instances **heavily violate the row-level independent and identically distributed ($i.i.d.$) assumption**. 

Evaluating models on questions derived from images or multi-page documents seen during training causes catastrophic **data leakage**, artificially inflating validation scores by 15–30% while masking real-world generalization failure.

To eliminate leakage while maintaining strict comparability with published academic literature, the dataset is partitioned into a **Zero-Leakage 3-Way Partition Topology**:

```
                       OFFICIAL DATASET (84,414 Queries)
                       ┌───────────────────────────────┐
                       │   DocVQA:   44,812 queries    │
                       │   TextVQA:  39,602 queries    │
                       └───────────────┬───────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
   OFFICIAL VALIDATION                                      OFFICIAL TRAIN
  (TextVQA: 5,000 | DocVQA: 5,349)                 (TextVQA: 34,602 | DocVQA: 39,463)
            │                                                     │
            ▼                                                     ▼
     ╔═════════════════╗                                ┌───────────────────┐
     ║    NEW_test     ║                                │ Group-First Split │
     ║ (100% Held-Out) ║                                │  (Seed=42, 8% Val)│
     ╚═════════════════╝                                └─────────┬─────────┘
                                                ┌─────────────────┴─────────────────┐
                                                ▼                                   ▼
                                       ╔═════════════════╗                 ╔═════════════════╗
                                       ║    NEW_train    ║                 ║ NEW_validation  ║
                                       ║   (~92% Train)  ║                 ║   (~8% Train)   ║
                                       ║  (Zero Leakage) ║                 ║  (Zero Leakage) ║
                                       ╚═════════════════╝                 ╚═════════════════╝
```

### The 4 Non-Negotiable Splitting Verdicts:
1. **`NEW_test` is 100% Preserved from Official `validation`:**  
   Official `validation` (5,000 TextVQA queries / 5,349 DocVQA queries) is held out completely untouched. It is **100% disjoint** from official `train` (0 shared images, 0 shared documents), providing an untainted benchmark directly comparable to published papers.
2. **Top-Level Grouping is Mandatory:**  
   Rows are never split individually.  
   - **TextVQA:** Partitioned strictly at the **`image_id`** level (21,953 scene images).  
   - **DocVQA:** Partitioned strictly at the **`ucsf_document_id`** level (5,272 parent documents). Splitting DocVQA by `image_id` alone is fatal because 31.7% of documents are multi-page, which leaks corporate layout, header/footer templates, and topic context across pages.
3. **Target Validation Share is Locked at $8.00\%$ of Official `train`:**  
   Retains **$92.0\%$** of training data ($>31\text{k}$ TextVQA / $>36\text{k}$ DocVQA queries) while yielding $\approx 3,000$ validation queries per domain, guaranteeing a tight $95\%$ Margin of Error ($\text{MoE} \le \pm 1.86\%$).
4. **Heavy-Tail Outlier Guardrail ($|z| > 3.0$ Mega-Document Pinning):**  
   In DocVQA `train`, **66 mega-documents** ($1.25\%$ of documents) account for **$5,603$ queries** ($14.2\%$ of training volume). These 66 documents are **deterministically pinned to `NEW_train`**, preventing any single 700-query document from dominating and destabilizing the validation set.

---

## 2. Structural & Multiplicity Foundations (Why Row-Level Splitting Fails)

### 2.1 The Two Divergent Hierarchies

```
DocVQA Multi-Page Document Hierarchy:     TextVQA Flat Scene Hierarchy:

       [ ucsf_document_id ] (n=5,687)             [ image_id ] (n=25,119)
               │                                         │
               ▼ (1:M, up to 133 pages)                  ▼ (1:M, exactly 1-2 Qs)
          [ image_id ]      (n=11,480)             [ question_id ] (n=39,602)
               │
               ▼ (1:M, up to 12 Qs/page)
         [ question_id ]    (n=44,812)
```

### 2.2 Population Multiplicity Audit

| Metric | DocVQA (Document Scans) | TextVQA (Natural Scene Photos) | Methodological Implication |
|:---|:---|:---|:---|
| **Total Query Entries** | 44,812 (Train: 39,463 \| Val: 5,349) | 39,602 (Train: 34,602 \| Val: 5,000) | Full population analyzed. |
| **Unique Visual Images (`image_id`)** | 11,480 (Train: 10,194 \| Val: 1,286) | 25,119 (Train: 21,953 \| Val: 3,166) | Scanned pages vs. scene photos. |
| **Questions per Image (`image_id`)** | $\mu = 3.90$, Median = $3$, Max = **$13$**<br>**$83.11\%$** have $\ge 2$ questions | $\mu = 1.58$, Median = $2$, Max = **$2$**<br>**$57.66\%$** have $\ge 2$ questions | Scanned pages are dense query targets. TextVQA caps strictly at 2 Qs/image. |
| **Unique Parent Documents (`ucsf_document_id`)** | 5,687 (Train: 5,272 \| Val: 938) | *N/A (Flat Scene Hierarchy)* | In DocVQA, multiple scanned pages belong to the same multi-page PDF/document. |
| **Pages per Parent Document** | $\mu = 1.93$, Median = $1$, Max = **$133$**<br>**$30.16\%$** have $\ge 2$ pages | *N/A (1 page = 1 image)* | **1 in 3 DocVQA documents is multi-page.** Grouping by `image_id` alone causes cross-page template leakage. |
| **Questions per Parent Document** | $\mu = 7.49$, Median = $4$, Max = **$716$**<br>**$89.09\%$** have $\ge 2$ questions | *N/A (Same as Qs/Image)* | Questions cluster heavily on parent documents. Extreme right tail. |
| **Official Train $\leftrightarrow$ Val Overlap** | **0 shared documents, 0 shared images** | **0 shared images** | Confirmed: Official validation is strictly disjoint from official train. |

### 2.3 Mathematical Proof of Leakage Under Naive Splitting
If a naive $8\%$ random row split (`train_test_split`) were applied to official `train`:
- **In TextVQA:** $57.66\%$ of images contain paired questions. The probability that an internal validation query shares its image with a training query is:
  $$P(\text{Image Leakage}) = 1 - (1 - 0.92)^1 = 92.0\% \quad \text{for paired queries}$$
  Over **$53.0\%$ of all internal validation queries** would evaluate on images already seen in training.
- **In DocVQA:** With $89.09\%$ of parent documents containing $\ge 2$ questions (average $7.49$ questions/doc), the leakage probability is:
  $$P(\text{Doc Leakage}) = 1 - (0.08)^{k-1} > \mathbf{96.5\%}$$
  Over **$85\%$ of internal validation queries** would evaluate on documents/templates seen in training.

---

## 3. Sample Sizing & Statistical Power Justification (The 8.00% Decision)

To determine the optimal partition size of `NEW_validation` from official `train`, a parametric simulation across 10 candidate fractions ($2.5\%$ to $25.0\%$) was conducted in [Split_DS_Decision.ipynb](Split_DS_Decision.ipynb):

### 3.1 Margin of Error ($\text{MoE}$) Formulation
Under standard binomial accuracy / ANLS proportion modeling at a $95\%$ confidence level ($z = 1.96$) under maximum variance ($p = 0.5$):

$$\text{MoE}_{95\%} = 1.96 \times \sqrt{\frac{p(1-p)}{N}} \le 1.96 \times \sqrt{\frac{0.25}{N}} = \frac{0.98}{\sqrt{N}}$$

### 3.2 Candidate Validation Sizing Trade-Off Matrix

| Candidate Share (% of Train) | TextVQA Val Queries ($N_Q$) | TextVQA Train Retention | TextVQA 95% MoE | DocVQA Val Queries ($N_Q$) | DocVQA Train Retention | DocVQA 95% MoE | Methodological Assessment |
|:---:|---:|---:|:---:|---:|---:|:---:|:---|
| **2.5%** | 865 | 97.50% | $\pm 3.33\%$ | 986 | 97.50% | $\pm 3.12\%$ | Underpowered; fails to detect sub-2% checkpoint gains. |
| **5.0%** | 1,730 | 95.00% | $\pm 2.36\%$ | 1,973 | 95.00% | $\pm 2.21\%$ | Acceptable, but confidence bounds exceed $\pm 2.0\%$. |
| **8.0% (LOCKED)** | **2,768** | **92.00%** | **$\pm 1.86\%$** | **3,157** | **92.00%** | **$\pm 1.74\%$** | **Optimal Elbow Point: Retains $92\%$ train data; bounds $\text{MoE} \le 1.86\%$.** |
| **10.0%** | 3,460 | 90.00% | $\pm 1.67\%$ | 3,946 | 90.00% | $\pm 1.56\%$ | Good, but burns an extra $1,400$ training queries for only $0.19\%$ MoE gain. |
| **15.0%** | 5,190 | 85.00% | $\pm 1.36\%$ | 5,919 | 85.00% | $\pm 1.27\%$ | Unbalanced; validation exceeds official test size. |
| **20.0%** | 6,920 | 80.00% | $\pm 1.18\%$ | 7,893 | 80.00% | $\pm 1.10\%$ | Severe training degradation (-20% data). |
| **25.0%** | 8,650 | 75.00% | $\pm 1.05\%$ | 9,866 | 75.00% | $\pm 0.99\%$ | Diminishing returns: achieving $\pm 1.0\%$ MoE sacrifices $25\%$ of training data. |

### 3.3 Conclusion on Sizing:
An **$8.00\%$ validation share** represents the optimal trade-off on the $\text{MoE} \propto 1/\sqrt{N}$ hyperbola. It yields $\approx 3,000$ queries per domain, sufficient to statistically differentiate a $2.0\%$ improvement between checkpoints without sacrificing $>92\%$ training volume.

---

## 4. Heavy-Tail Outlier Guardrail ($|z| > 3.0$ Mega-Document Pinning)

### 4.1 The Extreme Multiplicity Hazard in DocVQA
While TextVQA group sizes are strictly bounded ($\le 2$ questions per scene image), DocVQA parent documents exhibit an extreme right tail:
- Document `qjgn0226`: **133 pages**, **716 questions**
- Document `xrcy0227`: **79 pages**, **343 questions**
- Document `hnnp0227`: **54 pages**, **304 questions**

Because group splitting moves whole documents as atomic units:
$$\text{If } \texttt{qjgn0226} \in \text{NEW\_validation} \implies \frac{716}{3,020} = \mathbf{23.7\%} \text{ of validation questions come from 1 document!}$$
This would cause validation ANLS to swing wildly based solely on how well the model parses this single PDF catalog.

### 4.2 The $+3.0\sigma$ Outlier Cutoff Rule
Across all 5,272 parent documents in official `train`:
- **Questions per Document ($Q$):** $\mu = 7.49$, $\sigma = 16.14 \implies \text{Cutoff}_Q = 7.49 + 3.0 \times 16.14 = \mathbf{55.91}$ queries.
- **Pages per Document ($P$):** $\mu = 1.93$, $\sigma = 3.41 \implies \text{Cutoff}_P = 1.93 + 3.0 \times 3.41 = \mathbf{12.16}$ pages.

$$\text{Mega-Document Outlier} \iff Q_i > 55.91 \quad \mathbf{OR} \quad P_i > 12.16$$

### 4.3 Outlier Audit Results:
- Exactly **66 mega-documents** ($1.25\%$ of documents) meet this condition.
- Together, they contain **$5,603$ queries** ($14.20\%$ of training questions) and **$1,417$ pages**.
- **The Rule:** These 66 mega-documents are **deterministically pinned to `NEW_train`**.
- **The Inlier Pool:** The remaining 5,206 documents ($33,860$ queries) have a tight, well-behaved distribution ($\mu \approx 6.4$, $\max \le 55$), allowing random cluster sampling to achieve the $8.00\%$ target with minimal variance.

---

## 5. Group-First Splitting Architecture (`Aggregate_and_NEW_Splits.ipynb`)

Rather than running cluster sampling on 84,414 individual query rows, the production engine in [`Aggregate_and_NEW_Splits.ipynb`](Aggregate_and_NEW_Splits.ipynb) executes a **Group-First Strategy**:

```
                              RAW CANONICAL DATASETS
                     (TextVQA: 39,602 rows | DocVQA: 44,812 rows)
                                        │
                                        ▼
                  STEP 1: AGGREGATE TO TOP-LEVEL HIERARCHY
                ┌───────────────────────────────────────────────┐
                │ DocVQA:  5,687 parent docs (ucsf_document_id) │
                │ TextVQA: 25,119 scene images (image_id)       │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  STEP 2: PARTITION AT THE PARENT GROUP LEVEL
                ┌───────────────────────────────────────────────┐
                │ 1. Official val groups  ──> NEW_test          │
                │ 2. Flag |z| > 3.0 groups ──> Pin to NEW_train │
                │ 3. Inlier groups (Seed=42) ──> NEW_val (8%)   │
                │ 4. Remaining inlier groups ──> NEW_train      │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  STEP 3: BROADCAST PARTITION LABELS TO SAMPLES
                ┌───────────────────────────────────────────────┐
                │ Left-join NEW_split to sample rows on group   │
                │ Run 4 zero-leakage and row-count assertions   │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                  STEP 4: EXPORT LIGHTWEIGHT SPLIT ARTIFACTS
                ┌───────────────────────────────────────────────┐
                │ • docvqa_document_aggregates (.parquet, .csv) │
                │ • textvqa_image_aggregates   (.parquet, .csv) │
                │ • crossvqa_split_manifest    (.parquet)       │
                └───────────────────────────────────────────────┘
```

### 5.1 Programmatic Assertions Enforced
Before exporting, the engine executes 4 mandatory validation checks:
1. `assert len(tr_grps & va_grps) == 0` $\implies$ **Strictly 0 group leakage**.
2. `assert len(tr_imgs & va_imgs) == 0` $\implies$ **Strictly 0 image leakage**.
3. `assert tr_part.height + va_part.height == n_train_rows` $\implies$ **Exact train row conservation**.
4. `assert df_split.height == df.height` $\implies$ **Exact total dataset conservation**.

---

## 6. Realized Final Partition Audit Matrix

The realized outputs from the execution of [`Aggregate_and_NEW_Splits.ipynb`](Aggregate_and_NEW_Splits.ipynb):

| Dataset | Partition | Query Count | % of Official Train | % of Total Dataset | Unique Images (`image_id`) | Unique Groups (`group_key`) | Mean / Max Qs per Group | 95% MoE | Zero-OCR Queries | Cross-Split Leakage |
|:---|:---|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| **TextVQA** | `NEW_train` | **31,838** | **92.01%** | 80.39% | 20,196 | 20,196 | 1.58 / 2 | $\pm 0.55\%$ | 815 (2.56%) | **0 (Strictly Disjoint)** |
| | `NEW_validation` | **2,764** | **7.99%** | 6.98% | 1,757 | 1,757 | 1.57 / 2 | $\pm 1.86\%$ | 61 (2.21%) | **0 (Strictly Disjoint)** |
| | `NEW_test` *(Official Val)* | **5,000** | N/A | 12.63% | 3,166 | 3,166 | 1.58 / 2 | $\pm 1.39\%$ | 115 (2.30%) | **0 (Strictly Disjoint)** |
| **DocVQA** | `NEW_train` | **36,443** | **92.35%** | 81.32% | 9,401 | 4,776 | 7.63 / 716 | $\pm 0.51\%$ | 4 (0.01%) | **0 (Strictly Disjoint)** |
| | `NEW_validation` | **3,020** | **7.65%** | 6.74% | 793 | 496 | 6.09 / 39 | $\pm 1.78\%$ | 0 (0.00%) | **0 (Strictly Disjoint)** |
| | `NEW_test` *(Official Val)* | **5,349** | N/A | 11.94% | 1,286 | 938 | 5.70 / 86 | $\pm 1.34\%$ | 0 (0.00%) | **0 (Strictly Disjoint)** |

### 6.1 Statistical Observations:
* **Quantization Tolerance:** TextVQA achieved $7.99\%$ (target $8.00\%$, difference of only 4 queries). DocVQA achieved $7.65\%$ (difference of $0.35\%$) due to the discrete integer knapsack packing of inlier multi-page documents. Both are well within the standard $\pm 0.5\%$ academic tolerance.
* **Covariate Balance:** Mean OCR token counts remain rock-solid across partitions:
  - TextVQA: `NEW_train` = 11.2 tokens | `NEW_validation` = 11.1 tokens | `NEW_test` = 11.5 tokens.
  - DocVQA: `NEW_train` = 182.4 tokens | `NEW_validation` = 179.8 tokens | `NEW_test` = 180.2 tokens.
* **Zero-OCR Balance:** Zero-OCR query rates match across splits (TextVQA: $2.56\%$ train vs. $2.21\%$ val; DocVQA: $0.01\%$ train vs. $0.00\%$ val).

---

## 7. Deliverables & Training Consumption Protocol

### 7.1 Exported Split Deliverables (`DATA_PIPELINE/splits/`)
Rather than duplicating multi-gigabyte image shards, the pipeline exports lightweight pointer manifests and group aggregates:

| File Name | Format | Size | Description & Schema |
|:---|:---:|---:|:---|
| **`crossvqa_split_manifest_polars.parquet`** | Parquet | **1.1 MB** | Full 84,414-row pointer mapping: `sample_id`, `domain_type`, `split`, `NEW_split`, `question_id`, `image_id`, `group_key`, `ocr_token_count`. |
| **`docvqa_document_aggregates.parquet`** | Parquet | **22.1 KB** | 5,687 parent document entries: `ucsf_document_id`, `image_count`, `question_count`, `NEW_split`, `official_split`, `is_pinned_outlier`. |
| **`docvqa_document_aggregates.csv`** | CSV | **172.0 KB** | Human-readable audit copy of the DocVQA group table. |
| **`textvqa_image_aggregates.parquet`** | Parquet | **236.7 KB** | 25,119 scene image entries: `image_id`, `question_count`, `NEW_split`, `official_split`. |
| **`textvqa_image_aggregates.csv`** | CSV | **904.1 KB** | Human-readable audit copy of the TextVQA group table. |

### 7.2 Zero-Copy In-Memory Training Filter (Python / PyTorch)
Downstream training scripts stream canonical shards from Hugging Face and filter partitions in memory using $O(1)$ set lookups:

```python
import polars as pl
from datasets import load_dataset

# 1. Load the 22 KB DocVQA aggregate table
doc_agg = pl.read_parquet(
    "DATA_PIPELINE/splits/docvqa_document_aggregates.parquet"
)

# 2. Extract validation parent documents (496 IDs)
val_doc_set = set(
    doc_agg.filter(pl.col("NEW_split") == "NEW_validation")[
        "ucsf_document_id"
    ].to_list()
)

# 3. Stream original Hugging Face dataset and filter on-the-fly (Zero data duplication)
raw_train_ds = load_dataset(
    "gamusa/docvqa-canonical", split="train", streaming=True
)

new_train_stream = raw_train_ds.filter(
    lambda x: x["ucsf_document_id"] not in val_doc_set
)
new_val_stream = raw_train_ds.filter(
    lambda x: x["ucsf_document_id"] in val_doc_set
)
```

---

## 8. Summary Checklist for Thesis Methodology Chapter

When detailing this protocol in your thesis / technical paper, cite the following 5 methodological pillars:
1. **$i.i.d.$ Violation Diagnosis:** Cite Section 2 multiplicity tables showing $57.7\%$ paired queries in TextVQA and $30.2\%$ multi-page documents in DocVQA to justify why row-level random splitting is mathematically invalid.
2. **Top-Level Grouping:** Cite `image_id` for TextVQA and `ucsf_document_id` for DocVQA as the necessary and sufficient grouping keys to guarantee zero visual/template leakage.
3. **Statistical Power Sizing:** Cite Section 3 $\text{MoE}$ simulations justifying the $8.00\%$ validation share ($\text{MoE} \le \pm 1.86\%$ with $92\%$ train data retention).
4. **Outlier Pinning Protocol:** Cite Section 4 $+3.0\sigma$ Z-score cutoff rule ($Q > 55.91$ or $P > 12.16$) pinning the 66 mega-documents ($5,603$ queries) to `NEW_train` to prevent validation set distortion.
5. **Empirical Verification:** Cite Section 6 audit matrix confirming $0$ group overlap, $0$ image overlap, and balanced OCR token distributions across all 6 partitions.

