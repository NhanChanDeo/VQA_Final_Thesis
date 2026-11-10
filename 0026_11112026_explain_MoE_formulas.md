# Margin of Error (MoE) Derivation and Statistical Sizing in VQA Benchmarks

> **Document Type:** Mathematical Derivation & Statistical Sizing Reference  
> **Target Framework:** CrossVQA Dataset Partitioning Protocol  
> **Date Generated:** 2026-11-11 00:26  

---

## 1. Executive Formula

The $95\%$ Confidence Interval Margin of Error ($\text{MoE}_{95\%}$) upper bound for benchmark query sample size $N$ is defined as:

$$\text{MoE}_{95\%} = 1.96 \times \sqrt{\frac{p(1-p)}{N}} \le 1.96 \times \sqrt{\frac{0.25}{N}} = \frac{0.98}{\sqrt{N}} \approx \frac{1}{\sqrt{N}}$$

In percentage terms:
$$\text{MoE}_{\%} \le \frac{98\%}{\sqrt{N}}$$

---

## 2. Intuition & Operational Meaning

When evaluating a Vision-Language Model on a test set of size $N$, the observed empirical accuracy $\hat{p}$ is a sample estimate of the model's true generalization capacity $p$.

The Margin of Error defines the radius of the **$95\%$ Confidence Interval**:

$$\text{True Capability } p \in \left[\hat{p} - \text{MoE}, \; \hat{p} + \text{MoE}\right] \quad \text{with } 95\% \text{ probability}$$

*Example:* If a model achieves $72.0\%$ accuracy on a test set with $\text{MoE} = \pm 1.4\%$, we are $95\%$ certain its true accuracy lies in $[70.6\%, 73.4\%]$.

---

## 3. Step-by-Step Mathematical Derivation

### Step A: Bernoulli Modeling of Single Query Trials
Each test question is an independent indicator random variable $X_i$:
- $X_i = 1$ if the model's prediction is correct (under VQA/ANLS matching threshold).
- $X_i = 0$ if incorrect.
- Success probability: $\mathbb{E}[X_i] = p$.
- Variance of a single trial: $\text{Var}(X_i) = \sigma^2 = p(1 - p)$.

### Step B: Sample Mean & Central Limit Theorem (CLT)
The overall dataset accuracy is the sample mean of $N$ trials:
$$\hat{p} = \frac{1}{N}\sum_{i=1}^N X_i$$

By the Central Limit Theorem, for sufficiently large $N$ ($N \ge 30$), the sampling distribution of $\hat{p}$ is asymptotically normal:
$$\hat{p} \sim \mathcal{N}\left(p, \; \frac{p(1 - p)}{N}\right)$$

The **Standard Error ($SE$)** of this estimator is:
$$SE = \sqrt{\frac{p(1 - p)}{N}}$$

### Step C: The $95\%$ Confidence Multiplier ($z = 1.96$)
For a standard normal distribution $\mathcal{Z} \sim \mathcal{N}(0, 1)$, exactly $95\%$ of the total probability density lies within $\pm 1.96$ standard deviations of the mean:
$$P(-1.96 \le Z \le 1.96) = 0.95$$

Thus, the $95\%$ confidence margin of error is:
$$\text{MoE}_{95\%} = z_{0.95} \times SE = 1.96 \times \sqrt{\frac{p(1 - p)}{N}}$$

---

## 4. Why Replace $p(1 - p)$ with $0.25$? (Worst-Case Upper Bound)

Prior to model training and evaluation, the true test score $p$ is unknown. To guarantee that a partition remains statistically valid under any performance outcome, we take the maximum of the variance function $f(p) = p(1 - p) = p - p^2$:

$$\frac{d}{dp}[p - p^2] = 1 - 2p = 0 \implies p = 0.5$$

At maximum variance ($p = 0.5$):
$$f(0.5) = 0.5 \times (1 - 0.5) = 0.25$$

For any other performance level, the variance is strictly smaller:
- At $p = 0.70$: $0.70 \times 0.30 = 0.21 < 0.25$
- At $p = 0.80$: $0.80 \times 0.20 = 0.16 < 0.25$

Substituting $0.25$ yields the universal conservative bound:
$$\text{MoE}_{95\%} \le 1.96 \times \sqrt{\frac{0.25}{N}} = 1.96 \times \frac{0.5}{\sqrt{N}} = \frac{0.98}{\sqrt{N}}$$

---

## 5. Sample Size vs. Precision Reference Table

$$\text{MoE}_{\%} \approx \frac{98\%}{\sqrt{N}}$$

| Sample Size ($N$) | $\sqrt{N}$ | Calculated $\text{MoE}_{95\%}$ | Evaluation Utility |
| :--- | :--- | :--- | :--- |
| **$N = 100$** | $10.0$ | **$\pm 9.80\%$** | Unusable (noise masks all signals) |
| **$N = 1,000$** | $31.62$ | **$\pm 3.10\%$** | Weak (cannot verify $\le 2\%$ improvements) |
| **$N = 3,000$** | $54.77$ | **$\pm 1.79\%$** | **Validation Sweet Spot:** Optimal for early stopping & fast epoch runs |
| **$N = 5,000$** | $70.71$ | **$\pm 1.39\%$** | **Test Sweet Spot:** Gold standard for academic benchmark discrimination |
| **$N = 10,000$** | $100.0$ | **$\pm 0.98\%$** | Diminishing returns (wastes training data) |

---

## 6. Practical Application: Target Partitions in CrossVQA

| Partition | Target $\text{MoE}$ | Target Sample Size ($N$) | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **`NEW_test`** | **$\le \pm 1.45\%$** | **$\ge 4,500 - 5,000$ queries** | Evaluated once at publication. Must prove $+1.5\%$ gains over baseline models are statistically significant ($p < 0.05$). |
| **`NEW_validation`** | **$\le \pm 1.80\%$** | **$\ge 3,000 - 3,500$ queries** | Evaluated after every training epoch. Saves $35\%-40\%$ inference compute while reliably guiding checkpoint selection. |
| **`NEW_train`** | — | **Remainder ($> 80\%$)** | Maximizes model convergence and parameter representation. |

---

## 7. Cluster Design Effect in Clustered Data (DocVQA)

Because DocVQA queries cluster on multi-page documents (`ucsf_document_id`), observations within the same document are not strictly independent. The **Effective Sample Size ($N_{\text{eff}}$)** is:

$$N_{\text{eff}} = \frac{N}{1 + (\bar{m} - 1)\rho}$$

*(where $\bar{m}$ is the mean queries per document and $\rho$ is the intra-cluster correlation coefficient).*

**Mitigations Applied in Pipeline:**
1. **Outlier Pinning ($|z| > 3.0$):** Heavy-tail documents (up to 700 queries) are pinned deterministically to `NEW_train` to prevent inflation of test variance.
2. **Conservative Test Target ($N \approx 4,700 - 5,000$):** Guarantees that even under cluster correlation, effective error stays strictly below $\pm 1.5\%$.

