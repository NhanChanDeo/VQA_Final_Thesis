# CrossVQA 2-Week Execution Plan: Training Phase Onward (`ALT_THREE_SPLITS`)

> **Document Type:** Revised Master Execution Plan (Phase C $\rightarrow$ Phase F)  
> **Effective Date:** 2026-11-27 (`+07:00`)  
> **Target Window:** 14 Days (2026-11-27 $\rightarrow$ 2026-12-11)  
> **Locked Dataset:** `gamusa/docvqa-canonical@three-splits` & `gamusa/textvqa-canonical@three-splits` (84,414 labelled queries; 10/10 leakage audit passed)  
> **Scope Decision:** **Option A — Cut `M1–M3` from mandatory 2-week GPU runs** (`TRAIN/src/{adapters,losses,augmentations}.py` retained as *Implemented Architectural Prototype / Future Work*); focus 100% of compute on a rigorous, statistically tested **Cross-Domain Transfer & Catastrophic Forgetting Study** (`Z0`, `B1–B4`, Few-Shot `1/5/10/25%`, held-out `test` evaluation, 95% Bootstrap CIs, and 7-class Error Taxonomy).

---

## 1. Locked Baseline & Pre-Flight State (2026-11-27)

### Completed Foundations (Do Not Re-Run)
1. **Canonical Datasets & Splits (`DATA_PIPELINE/` — 100% DONE):**
   - Pooled `ALT_THREE_SPLITS` partition (`DATA_PIPELINE/ALT_THREE_SPLITS/splits/crossvqa_split_manifest_pooled.parquet`): 0 shared `ucsf_document_id` / `image_id` across `NEW_train`, `NEW_validation`, `NEW_test`; 0 cross-split `pHash` near-duplicates ($d_H \le 3$).
   - Nested Few-Shot Manifest (`DATA_PIPELINE/ALT_THREE_SPLITS/splits/textvqa_few_shot_manifest.parquet`): `1%` (`328` Q / `208` img), `5%` (`1,637` Q / `1,040` img), `10%` (`3,277` Q / `2,081` img), `25%` (`8,191` Q / `5,202` img), `100%` (`32,762` Q / `20,807` img).
   - Materialized cleanly on Hugging Face branch `three-splits`.
2. **Phase A Domain Gap & `Z0` Anchor (`00_Z0_ZeroShot_and_DomainGap.ipynb` — 100% DONE):**
   - **`smolvlm_500m` (Primary Workhorse):** `docvqa_anls = 0.005939`, `textvqa_vqa_accuracy = 0.196270`, `harmonic_mean = 0.011529`, `RKHS MMD = 0.918568`, `Proxy A-dist = 1.992429` ($N=794$ deduped/domain).
   - **`qwen2_vl_2b` (Saturation Reference):** `docvqa_anls = 0.945829`, `textvqa_vqa_accuracy = 0.815333`, `harmonic_mean = 0.875746`, `RKHS MMD = 0.809853`, `Proxy A-dist = 2.000000` ($N=500$/domain).
3. **Pipeline Fixes Applied (2026-11-27):**
   - Fixed critical EOS-masking bug in [`TRAIN/src/collator.py`](../TRAIN/src/collator.py) (`labels[inputs["attention_mask"] == 0] = -100`), preventing `<|im_end|>` loss suppression on `SmolVLM-500M-Base` (where `pad_token_id == eos_token_id == 2`).
   - Fixed Google Drive `runs_logs` mirror in notebooks `01..05` and [`build_training_notebooks.py`](../TRAIN/ALT_THREE_SPLITS/scripts/build_training_notebooks.py) to exclude heavy `checkpoints/`.

---

## 2. Revised Hypothesis-to-Pipeline Matrix (Phase C Onward)

| Hypothesis | Revised Claim (Option A Scope) | Pipeline Element (`smolvlm_500m`, `seed=42`, `ebs=64`, Track A) | Est. Colab L4 Time | Status / Tier |
| :---: | :--- | :--- | :---: | :---: |
| **H1 (RQ1)** | 2D document and 3D scene-text features form separable latent clusters ($\hat{d}_{\mathcal{A}} \gg 0$). | `Z0` (`run_z0_domain_gap.py`) on `smolvlm_500m` & `qwen2_vl_2b` | **0h** | **DONE** |
| **H2 (RQ2)** | Source-only DocVQA LoRA (`B2`) transfers reading comprehension to TextVQA (`B2 > Z0`), yet lags target specialization (`B2 < B1`) due to 3D scene-text domain shift. | Fresh **`B2`** (DocVQA, 2 ep) + Fresh **`B1`** (TextVQA, 2 ep) | **~13.0h + ~10.5h** | **P0 — MUST RUN** |
| **H3 (RQ2, RQ3)** | Sequential fine-tuning `Doc -> Text` (`B4`) causes measurable Catastrophic Forgetting on DocVQA ($\text{FR} = \text{ANLS}_{\text{B2}} - \text{ANLS}_{\text{B4}} > 0$). | **`B4`** (Sequential `Doc -> Text`, 2 ep, init from `B2/best`) | **~10.5h** | **P0 — MUST RUN** |
| **H4 (RQ2)** | Relative cross-domain transfer gain ($\text{TG} = \text{Acc}_{\text{B4}} - \text{Acc}_{\text{B1}}$) is highest in low-resource regimes (`1%–10%` TextVQA). | **Few-Shot Matrix** (`1%, 5%, 10%, 25%` $\times$ `B1, B4`; reuses `B2/best`) | **~13.4h total** | **P0 — MUST RUN** |
| **H5 (RQ3, Revised)** | Joint interleaved LoRA (`B3`) mitigates sequential forgetting ($\text{FR}_{\text{B3}} < \text{FR}_{\text{B4}}$) while revealing a negative-transfer trade-off vs. single-domain specialists (`B1`/`B2`). | **`B3`** (Joint 50/50 interleaved DocVQA+TextVQA, `--epochs 1` $\approx 70.8\text{k}$ samples / $1,106$ opt steps) | **~11.5h** | **P0 — MUST RUN** |
| **Phase D Eval** | Unbiased held-out `test` split metrics, 95% Bootstrap CIs ($B=1,000$), paired permutation tests ($p < 0.05$), and 7-class failure taxonomy (`E1–E7`). | [`eval_test_checkpoints.py`](../TRAIN/ALT_THREE_SPLITS/scripts/eval_test_checkpoints.py) + [`evaluate_statistical_suite.py`](../EVALUATION/ALT_THREE_SPLITS/evaluate_statistical_suite.py) + [`sample_error_taxonomy.py`](../EVALUATION/ALT_THREE_SPLITS/sample_error_taxonomy.py) | **~3.0h GPU + Offline CPU** | **P0 — MUST RUN** |
| **Deferred / Cut** | `M1–M3` (`DualLoRA`, `RBF-MMD`, `Curriculum`), `B5`, 3-seed sweeps (`123, 999`), training `smolvlm_base`/`qwen2_vl_2b`, Track B OCR, InfoNCE, `VALIDATION_AS_TEST` | Retained in codebase as *Implemented Architectural Prototype / Future Work* | **0h** *(saves ~280+ GPU-h)* | **CUT** |

**Total Mandatory Compute:** **~62 L4 GPU-hours** (~4–5 days at 12–14h/day, or ~2.5–3 days running `B1` and `B2` in parallel Colab tabs).

---

## 3. Exact 14-Day Execution Runbook

### Pre-Flight (Day 1 Morning — Local + HF Hub)
1. Push the `collator.py` EOS-masking fix, `ebs64_v2` namespace isolation, and Phase C/D scripts to GitHub (`git push origin main`).
2. **Namespace Isolation (`checkpoint_namespace: "ebs64_v2"`):** Already bumped in [`baselines_B1_B5.yaml`](../TRAIN/ALT_THREE_SPLITS/configs/baselines_B1_B5.yaml) so fresh `B1–B4` runs do **not** collide with or silently resume from the pre-EOS-fix `ebs64` checkpoints on Hugging Face (`gamusa/crossvqa-checkpoints`) or Google Drive. *(Optionally delete the stale `alt_three_splits/ebs64/smolvlm_500m/{B1_target_only,B2_source_only}` folders on HF Hub and Drive to reclaim space.)*

### Phase C1: Primary Baselines (Days 1–3, ~34h L4)
1. **Fresh `B1` (`01_B1_TextVQA_Only_LoRA.ipynb`, ~10.5h):**
   ```bash
   python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
     --exp B1 --model smolvlm_500m --seed 42 \
     --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
   ```
2. **Fresh `B2` (`01_B2_DocVQA_Only_LoRA.ipynb`, ~13.0h — can run in parallel with `B1`):**
   ```bash
   python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
     --exp B2 --model smolvlm_500m --seed 42 \
     --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
   ```
3. **Sequential `B4` (`03_B4_B5_Sequential_Transfer.ipynb`, ~10.5h — requires `B2/best`):**
   ```bash
   python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
     --exp B4 --model smolvlm_500m --seed 42 \
     --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
   ```

### Phase C2: Joint Baseline & Few-Shot H4 Matrix (Days 4–5, ~25h L4)
4. **Joint `B3` (`02_B3_Joint_MultiDomain_LoRA.ipynb`, `--epochs 1`, ~11.5h):**
   ```bash
   python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
     --exp B3 --model smolvlm_500m --seed 42 --epochs 1 \
     --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
   ```
5. **Few-Shot Scaling (`05_FewShot_Scaling_Matrix.ipynb`, ~13.4h total across `1%, 5%, 10%, 25%`):**
   ```bash
   for PCT in 1 5 10 25; do
     python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
       --exp B1 --model smolvlm_500m --seed 42 --few_shot_pct $PCT \
       --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
     python TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py \
       --exp B4 --model smolvlm_500m --seed 42 --few_shot_pct $PCT \
       --local_data_dir /content/data --save_every_steps 64 --resume_from_hub
   done
   ```
   *(Remember to snapshot `data/test-*.parquet` in Colab before Day 6, or pass `--local_data_dir /content/data` after downloading test shards.)*

### Phase C3: Held-Out `NEW_test` Evaluation (Day 6, ~3h L4)
6. Download `data/test-*.parquet` to `/content/data` and evaluate `Z0` + `B1..B4` (`best` checkpoints) on `test`:
   ```bash
   python TRAIN/ALT_THREE_SPLITS/scripts/eval_test_checkpoints.py \
     --model smolvlm_500m --seed 42 --exps Z0 B1 B2 B3 B4 \
     --local_data_dir /content/data
   ```
   *(Optionally add `--few_shot_pcts 1 5 10 25` for `B1`/`B4` if GPU budget permits, or use `validation` CIs for few-shot curve and `test` CIs for primary `100%` matrix.)*
   Sync all `TRAIN/ALT_THREE_SPLITS/runs/` artifacts to GitHub via Cell 5.

### Phase D: Offline Statistical Suite & 7-Class Error Taxonomy (Days 7–9, Local CPU)
7. Run [`EVALUATION/ALT_THREE_SPLITS/evaluate_statistical_suite.py`](../EVALUATION/ALT_THREE_SPLITS/evaluate_statistical_suite.py):
   - Generates `summary_metrics.json`, Markdown & LaTeX tables (with 95% Bootstrap CIs $B=1,000$ and paired permutation $p$-values), `TG`/`FR` deltas, and `few_shot_learning_curve.png`.
8. Run [`EVALUATION/ALT_THREE_SPLITS/sample_error_taxonomy.py`](../EVALUATION/ALT_THREE_SPLITS/sample_error_taxonomy.py):
   - Exports 200 stratified failure cases (`100` DocVQA + `100` TextVQA) into `EVALUATION/ALT_THREE_SPLITS/error_taxonomy_200.csv` for manual `E1–E7` labeling and qualitative case-study figures.

### Phase F (+ Optional Phase E): Thesis Writing & Defense Polish (Days 10–14)
- **Days 10–12:** Update `OFFLIMIT/DATN-template.docx` Sections `2.3.2` / `3.2.3` for empirical transfer framing, insert verified Chapter 1–3 tables/figures, and document `M1–M3` cleanly in **Implemented Prototype & Future Research**.
- **Day 13 (Optional):** Add a lightweight Gradio app (`DEMO/app.py`) loading `B2` / `B4` / `B3` LoRA checkpoints if the manuscript is 100% drafted.
- **Day 14:** Final citation, formatting, and figure cross-check.
