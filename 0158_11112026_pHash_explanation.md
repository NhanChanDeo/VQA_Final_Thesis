# Perceptual Hashing (pHash): Mathematical Foundations & Dataset De-duplication Protocol

> **Document Type:** Technical & Methodological Reference  
> **Topic:** Image De-duplication, Discrete Cosine Transform (DCT), and Hamming Distance Metric  
> **Target Application:** Multimodal VQA Data Pipeline (`CrossVQA / ALT_THREE_SPLITS`)  
> **Date Generated:** 2026-11-11 01:58  

---

## 1. Executive Summary: Cryptographic vs. Perceptual Hashing

In traditional computer science, cryptographic hash functions (e.g., MD5, SHA-256) are intentionally engineered to exhibit the **"avalanche effect"**: altering a single pixel or changing compression by $1\%$ completely randomizes $\approx 50\%$ of the output bits. This makes cryptographic hashes useless for detecting visual similarity.

In contrast, **Perceptual Hashing (pHash)** transforms visual media into a compact fingerprint based on its **frequency spectrum**. Two images that appear visually identical to a human observer will generate nearly identical hashes, regardless of minor differences in:
- Compression artifacts (JPEG vs. PNG)
- Scaling and aspect ratio adjustments
- Slight contrast, brightness, or gamma shifts
- Scanner sensor noise or minor tilt ($< 2^\circ$)

---

## 2. Step-by-Step Mathematical Derivation of pHash

The standard implementation of pHash uses the **2D Discrete Cosine Transform (DCT-II)**, the same mathematical backbone underlying JPEG compression.

```
Raw Image (H x W x 3)
         │
         ▼
[Step 1: Grayscale & Downscale to 32 x 32]
         │
         ▼
[Step 2: 2D Discrete Cosine Transform (DCT-II)]
         │
         ▼
[Step 3: Crop Top-Left 8 x 8 Low-Frequency Matrix] (Excluding DC at [0,0])
         │
         ▼
[Step 4: Median Thresholding & 64-bit Binarization]
         │
         ▼
64-bit Perceptual Hash (e.g., 0x8f3c2a1e9b4d08f1)
```

---

### Step 1: Grayscale Conversion & Fixed Downsampling
The input image is converted from RGB to single-channel luminance ($L$) and resized via bilinear interpolation to a fixed spatial grid of $32 \times 32$ pixels:

$$L(x, y) = 0.299 R(x, y) + 0.587 G(x, y) + 0.114 B(x, y)$$

Downscaling to $32 \times 32$ eliminates high-frequency texture noise (such as individual paper grain or fine sensor artifacts) while retaining macro-structures, layout blocks, and prominent edges.

---

### Step 2: 2D Discrete Cosine Transform (DCT-II)
The $32 \times 32$ matrix $f(x, y)$ is transformed into frequency space $F(u, v)$ using the 2D DCT-II:

$$F(u, v) = \alpha(u) \alpha(v) \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x, y) \cos\left[ \frac{(2x + 1)u\pi}{2N} \right] \cos\left[ \frac{(2y + 1)v\pi}{2N} \right]$$

where $N = 32$, and the normalization coefficients $\alpha(k)$ are:

$$\alpha(k) = \begin{cases} 
\sqrt{\frac{1}{N}}, & \text{if } k = 0 \\ 
\sqrt{\frac{2}{N}}, & \text{if } k > 0 
\end{cases}$$

#### Why the DCT?
- **Energy Compaction:** Natural images concentrate $>95\%$ of their perceptual information in the lowest frequencies.
- **Top-Left Concentration:** The resulting $32 \times 32$ frequency matrix concentrates the lowest horizontal and vertical frequencies in the top-left corner ($u, v \approx 0$), while high-frequency noise is pushed to the bottom-right.

---

### Step 3: Low-Frequency Matrix Extraction ($8 \times 8$)
Human perception is dominated by low- and mid-frequency spatial envelopes. The algorithm extracts only the top-left $8 \times 8$ sub-matrix:

$$F_{\text{low}}(u, v) = F(u, v) \quad \text{for } u, v \in [0, 7]$$

This gives exactly **64 frequency coefficients**.

---

### Step 4: Median Binarization
1. **Exclude the DC Component:** The coefficient at $F(0, 0)$ represents the average overall image brightness (DC bias). It is omitted from the median calculation so the hash remains invariant to global illumination or exposure changes.
2. **Compute the Median:** Compute the median value $M$ of the remaining $63$ low-frequency AC coefficients:
   $$M = \text{median}\Big( \big\{ F(u, v) \mid (u, v) \in [0, 7] \times [0, 7] \setminus \{(0, 0)\} \big\} \Big)$$
3. **Thresholding (Bit Generation):** Construct each bit $b_i$ of the 64-bit string:
   $$b_{8u + v} = \begin{cases} 
   1, & \text{if } F(u, v) > M \\ 
   0, & \text{if } F(u, v) \le M 
   \end{cases}$$

Using the median (rather than the mean) guarantees an equal $50/50$ distribution of $0$s and $1$s across the hash, maximizing entropy and distinguishing power.

---

## 3. The Similarity Metric: Hamming Distance ($d_H$)

To compare two images $A$ and $B$, compare their 64-bit hashes $h_A$ and $h_B$ using the **Hamming Distance** ($d_H$), which counts the number of differing bits:

$$d_H(h_A, h_B) = \sum_{k=0}^{63} \big( h_A[k] \oplus h_B[k] \big) = \text{popcount}(h_A \oplus h_B)$$

In hardware and Python (`numpy`), this is a single bitwise XOR followed by a population count instruction (`POPCNT`), executing in $< 1$ nanosecond.

### Threshold Interpretation Table ($0 \dots 64$ bits):

| Hamming Distance ($d_H$) | Bit Agreement (%) | Interpretation | Recommended Pipeline Action |
| :---: | :---: | :--- | :--- |
| **$d_H = 0$** | **$100.0\%$** | **Exact / Transcoded Duplicate:** Re-scanned document or identical image saved with different compression. | **Automated Merge** |
| **$1 \le d_H \le 3$** | **$95.3\% - 98.4\%$** | **Near-Duplicate:** Camera burst shot (TextVQA), slight scanner skew, crop, or identical memo form. | **Flag for Review / Merge** |
| **$4 \le d_H \le 6$** | **$90.6\% - 93.8\%$** | **Template Twin:** Similar broad document structure, but distinct content. | **Safe to split independently** |
| **$d_H \ge 7$** | **$< 89\%$** | **Distinct Images:** Normal independent visual instances. | **Ignore** |

---

## 4. Why pHash is the Sweet Spot for Dataset Partitioning

### 1. Invariance to Harmless Transforms, Sensitivity to Content
- Resizing an image from $2048 \times 1536$ down to $512 \times 384$ does not alter its low-frequency DCT components $\rightarrow d_H = 0$.
- Saving a PNG document as a $75\%$ quality JPEG produces $d_H \le 1$.
- Altering the document text or scene composition shifts the DCT energy distribution $\rightarrow d_H \ge 8$.

### 2. Immunity to "Giant Component Collapse"
Unlike deep embedding models (e.g. CLIP or DINOv2), which group images by high-level semantic meaning (e.g., embedding all invoices or all street signs with $>0.92$ cosine similarity), pHash operates on **structural instances**. It does not falsely link two different documents simply because they are both invoices.

### 3. Computational Efficiency
- **Memory:** Each hash is an 8-byte `uint64`. Storing hashes for all $37,700$ images takes **$\approx 300\text{ KB}$** of RAM.
- **Speed:** Screening 120 million cross-partition pairs with vectorized `np.bitwise_xor` executes in **$< 3\text{ seconds}$** on standard CPU memory.

---

## 5. Critical Edge Case: Blank White Page Guardrail

In **DocVQA**, multipage PDF scans frequently contain blank white separator pages. Because a uniform white image has zero variance across all AC frequencies, **all blank pages generate the exact same perceptual hash ($d_H = 0$)**.

If left unguarded, pHash would link hundreds of unrelated corporate documents into one giant false cluster.

### The Variance Guardrail Solution:
Before computing pHash, compute the standard deviation of pixel intensities:

$$\sigma_{\text{gray}} = \sqrt{\frac{1}{N}\sum_{x, y} \big( L(x, y) - \mu \big)^2}$$

If $\sigma_{\text{gray}} < 8.0$, the page is mathematically uniform (blank white or solid color) and is **excluded from visual cluster linking**.

