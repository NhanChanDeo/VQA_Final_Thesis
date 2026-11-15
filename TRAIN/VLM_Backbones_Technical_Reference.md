# CrossVQA — VLM Backbones Technical Reference & Optimization Guide

> **Location:** `TRAIN/VLM_Backbones_Technical_Reference.md`  
> **Literature Index:** [`../PLAN/Papers_List.md`](../PLAN/Papers_List.md) (Section 7 & STT 5, 6, 8)  
> **Target Splits:** [`../DATA_PIPELINE/ALT_THREE_SPLITS/Verdict on ds splitting decision (Pooled).md`](../DATA_PIPELINE/ALT_THREE_SPLITS/Verdict%20on%20ds%20splitting%20decision%20(Pooled).md) (`gamusa/docvqa-canonical@three-splits` & `gamusa/textvqa-canonical@three-splits`)

This reference consolidates verified Hugging Face `config.json` architectures, vision tokenization formulas, PEFT/LoRA target modules, quantization/memory optimizations, and OCR conditioning templates for all candidate backbones in Phase C (`TRAIN/`).

---

## 1. Candidate Backbone Architecture Comparison Matrix

All parameters below (except gated `google/paligemma-3b-pt-448`, noted from official papers) were verified directly against Hugging Face `config.json` manifests.

| Backbone Repo | HF Architecture Class | Vision Encoder | Connector / Spatial Mechanism | Language / Decoder Backbone | Primary Role in CrossVQA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen/Qwen2-VL-2B`** | `Qwen2VLForConditionalGeneration` (`qwen2_vl`) | 32L ViT (`embed_dim=1280`, `heads=16`, `patch=14`) | 2×2 Spatial Merger (`spatial_merge_size=2`, projects to `1536`) + 2D/3D M-RoPE | 28L Qwen2 (`hidden=1536`, `ffn=8960`, `12` Q-heads / `2` KV-heads GQA, `vocab=151,936`, `ctx=32,768`) | **Primary Workhorse:** Native dynamic resolution handles 4.8× Doc↔Text resolution gap |
| **`HuggingFaceTB/SmolVLM-500M-Base`** | `Idefics3ForConditionalGeneration` (`idefics3`) | 12L ViT (`hidden=768`, `heads=12`, `patch=16`, `image_size=512`, `ffn=3072`) | Pixel Shuffle connector + tile splitting | 32L Llama (`hidden=960`, `15` Q-heads / `5` KV-heads GQA, `ffn=2560`, `vocab=49,280`, `ctx=8,192`) | **Fast Pilot / Multi-Seed:** Rapid hyperparameter sweeps & ablations on 16GB T4 |
| **`HuggingFaceTB/SmolVLM-Instruct`** *(2B class)* | `Idefics3ForConditionalGeneration` (`idefics3`) | 27L ViT (`hidden=1152`, `heads=16`, `patch=14`, `image_size=384`, `ffn=4304`) | Pixel Shuffle connector + tile splitting | 24L Llama/SmolLM2 (`hidden=2048`, `heads=32`, `kv_heads=32`, `ffn=8192`, `vocab=49,155`, `ctx=16,384`) | **Compact 2B Alternative:** Note `-Instruct` requires Z0 contamination check |
| **`google/paligemma-3b-pt-448`** *(gated)* | `PaliGemmaForConditionalGeneration` | SigLIP-So400m (`hidden=1152`, 27L, `patch=14`, fixed 448²/896²) | Linear Multimodal Projector | Gemma-2B (`vocab` includes `<loc0000>..<loc1023>` spatial tokens) | **High-Res Transfer Reference:** Pre-trained (`-pt`) checkpoint; requires HF token & license |
| **`google/pix2struct-base`** | `Pix2StructForConditionalGeneration` (`pix2struct`) | 12L ViT (`hidden=768`, `heads=12`, `patch=16`, variable aspect-ratio packing) | Direct encoder-decoder cross-attention | 12L Pix2Struct Text Decoder (`hidden=768`, `heads=12`, `vocab=50,244`) | **Classic OCR-Free Reference:** Zero LLM pre-training contamination |
| **`naver-clova-ix/donut-base`** | `VisionEncoderDecoderModel` (`vision-encoder-decoder`) | 4-stage `donut-swin` (`embed=128`, `hidden=1024`, `heads=[4,8,16,32]`, `patch=4`, `image=[2560,1920]`) | Encoder-Decoder Cross-Attention | 4L mBART Decoder (`d_model=1024`, `vocab=57,525`, `ctx=1,536`) | **Classic Document OCR-Free:** Document-native baseline without LLM priors |

---

## 2. Detailed Technical Profiles

### 2.1 `Qwen/Qwen2-VL-2B` (Primary Workhorse)
- **Papers:** [`Wang_2024_Qwen2_VL.pdf`](../PLAN/Papers/Model%20Info/Wang_2024_Qwen2_VL.pdf) (`arXiv:2409.12191`), [`Dehghani_2023_NaViT.pdf`](../PLAN/Papers/Model%20Info/Dehghani_2023_NaViT.pdf) (`arXiv:2307.06304`)
- **Vision Tokenization Math:**
  - Patches are extracted with `patch_size = 14`, then grouped by `spatial_merge_size = 2` ($2 \times 2$ patches merged into 1 LLM input token).
  - Effective visual stride is $14 \times 2 = 28\text{ px}$ per token dimension:
    $$N_{\text{visual\_tokens}} \approx \left\lceil \frac{H}{28} \right\rceil \times \left\lceil \frac{W}{28} \right\rceil$$
  - Example: An unconstrained $1695 \times 2025$ DocVQA scan generates $\approx 61 \times 73 \approx 4,453$ visual tokens. Bounding `max_pixels` is therefore mandatory on 16–24 GB GPUs.
- **Positional Encoding:** Multimodal Rotary Position Embedding (M-RoPE) decomposes rotary positions into temporal, height, and width components, preserving 2D document layout and spatial alignment across arbitrary aspect ratios.
- **Contamination Hygiene:** Always load `Qwen/Qwen2-VL-2B` (Base), **not** `-Instruct`, and run `Z0` zero-shot evaluation on `NEW_test` before fine-tuning.

### 2.2 `HuggingFaceTB/SmolVLM-500M-Base` & `SmolVLM-2B` (`Idefics3`)
- **Papers:** [`Marafioti_2025_SmolVLM.pdf`](../PLAN/Papers/Model%20Info/Marafioti_2025_SmolVLM.pdf) (`arXiv:2504.05299`), [`Laurencon_2024_Idefics3.pdf`](../PLAN/Papers/Model%20Info/Laurencon_2024_Idefics3.pdf) (`arXiv:2408.12637`)
- **Architecture:** Implements the `Idefics3` image-splitting pipeline combined with **Pixel Shuffle** spatial compression, packing multiple patch embeddings into a single visual token to drastically cut context length and KV-cache VRAM.
- **Role in `TRAIN/`:** `SmolVLM-500M-Base` uses a `768`-dim 12-layer ViT and a `960`-dim 32-layer Llama decoder (`vocab_size=49,280`), making it ideal for fast 3-seed pilot tests (`42, 123, 999`) and debugging custom dual-adapter / MMD loss code before scaling to `Qwen2-VL-2B`.

### 2.3 `google/paligemma-3b-pt-*` & `PaliGemma 2`
- **Papers:** [`Beyer_2024_PaliGemma.pdf`](../PLAN/Papers/Beyer_2024_PaliGemma.pdf) (`arXiv:2407.07726`), [`Steiner_2024_PaliGemma2.pdf`](../PLAN/Papers/Model%20Info/Steiner_2024_PaliGemma2.pdf) (`arXiv:2412.03555`), [`Zhai_2023_SigLIP.pdf`](../PLAN/Papers/Model%20Info/Zhai_2023_SigLIP.pdf) (`arXiv:2303.15343`)
- **Architecture:** Combines a SigLIP-So400m vision encoder with Gemma / Gemma 2 at fixed square input resolutions (`224`, `448`, `896` px). Pre-trained checkpoints (`-pt-448`, `-pt-896`) are explicitly designed for downstream fine-tuning transfer and include 1,024 spatial location tokens (`<loc0000>`–`<loc1023>`) aligned with `[ymin, xmin, ymax, xmax]`.
- **Practical Caveat:** Hugging Face repositories for PaliGemma are gated (HTTP `401` without an authenticated token + license approval), and fixed square inputs introduce aspect-ratio padding on tall documents.

### 2.4 Classic OCR-Free Baselines (`Pix2Struct-base` & `Donut-base`)
- **Papers:** [`Lee_2023_Pix2Struct.pdf`](../PLAN/Papers/Lee_2023_Pix2Struct.pdf) (`arXiv:2210.03347`), [`Kim_2022_Donut.pdf`](../PLAN/Papers/Kim_2022_Donut.pdf) (`arXiv:2111.15664`)
- **Architecture:**
  - `google/pix2struct-base`: Preserves native aspect ratios by scaling images to fit a maximum patch budget (`patch_size=16`) with 2D absolute positional embeddings and a 12-layer text decoder.
  - `naver-clova-ix/donut-base`: Uses a 4-stage Swin Transformer (`[2560, 1920]` canvas) paired with a 4-layer mBART decoder (`1,536` max tokens).
- **Role in `TRAIN/`:** Pure visual-to-text baselines with zero causal-LLM world-knowledge leakage.

---

## 3. Memory-Saving, Quantization & PEFT Reference

- **Papers:** [`Hu_2022_LoRA.pdf`](../PLAN/Papers/Model%20Info/Hu_2022_LoRA.pdf) (`arXiv:2106.09685`), [`Dettmers_2023_QLoRA.pdf`](../PLAN/Papers/Model%20Info/Dettmers_2023_QLoRA.pdf) (`arXiv:2305.14314`)

### 3.1 Pixel / Token Budget (`Qwen2VLProcessor`)
To keep `Qwen2-VL-2B` stable inside 16 GB VRAM (Colab T4) without truncating small document text:
```python
from transformers import AutoProcessor

MIN_PIXELS = 256 * 28 * 28   # ~200k px (~256 visual tokens floor)
MAX_PIXELS = 1024 * 1024     # ~1.05M px (~1,340 visual tokens ceiling)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B",
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)
```

### 3.2 Selective 4-Bit NF4 Quantization (Keep Vision Encoder in 16-Bit)
Quantizing the early convolutional/patch layers of the ViT can blur tiny 6–8pt document fonts. Use **selective NF4 QLoRA** to quantize the LLM in 4-bit while keeping `visual` and `merger` in `fp16`/`bf16`:
```python
import torch
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["visual", "merger"],  # Preserve full-res optical precision
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B",
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",  # Use "flash_attention_2" on Ampere/Ada/Hopper (L4/A100/3090)
    device_map="auto",
)
```

### 3.3 LoRA Target Modules by Backbone Family

| Backbone Family | Recommended `target_modules` | Notes |
| :--- | :--- | :--- |
| **`Qwen2-VL` (`qwen2_vl`)** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Target LLM attention + MLP; optionally add projector linear layers for `M1` dual adapters. |
| **`SmolVLM` (`idefics3`)** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Standard Llama-style linear projections inside `text_model`. |
| **`PaliGemma` (`paligemma`)** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Matches Gemma/Gemma-2 linear layers. |
| **`Pix2Struct` / `Donut`** | `["query", "key", "value", "output"]` (or `["q_proj", "v_proj", "k_proj", "out_proj"]` for mBART) | Small enough (~200–280M) to support full fine-tuning on 16 GB VRAM with gradient checkpointing. |

### 3.4 Recommended 16 GB Colab T4 Training Stack
1. `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` (keeps `RMSNorm`/`LayerNorm` in `fp32` for stability).
2. `optim="paged_adamw_8bit"` to offload AdamW moment spikes to host RAM.
3. `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch size `16`), `group_by_length=True` to avoid padding small TextVQA crops to DocVQA lengths.

---

## 4. Input Conditioning Templates (OCR-Free vs. Given-OCR Ablation)

To cleanly separate **optical reading transfer** from **multimodal layout/scene reasoning** on the same backbone:

- **Track A — OCR-Free (Primary Spine):**
  ```text
  <image>
  Question: {question}
  Answer:
  ```
- **Track B — With Given OCR (Spatial-Prompted Ablation):**
  Uses the pre-harmonized `ocr_tokens` and normalized `[ymin, xmin, ymax, xmax]` boxes in `[0, 1000]` from `gamusa/*-canonical`:
  ```text
  <image>
  OCR Tokens ([ymin, xmin, ymax, xmax] in [0, 1000]):
  - "{token_1}" [{y0}, {x0}, {y1}, {x1}]
  - "{token_2}" [{y0}, {x0}, {y1}, {x1}]
  Question: {question}
  Answer:
  ```
- **Track C — Blind Text-Only Control (No Image):**
  Drops `<image>` and passes only sorted OCR tokens + `question` to quantify pure linguistic shortcuts.
