# Alignment Loss for M2 (MMD vs. InfoNCE)

In the **CrossVQA** framework (transferring knowledge from **DocVQA** to **TextVQA**), configuration **M2** augments the domain-specific adapters (M1) with an explicit **Feature Alignment Objective** $\mathcal{L}_{\text{align}}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{align}}$$

The goal of $\mathcal{L}_{\text{align}}$ is to project Document and Scene Text multimodal representations into a domain-invariant latent space, preventing the model from exploiting domain-specific artifacts while retaining textual reasoning capabilities.

---

## 1. Multi-Kernel RBF Maximum Mean Discrepancy (MK-MMD)

### 1.1 Mathematical Foundation
Maximum Mean Discrepancy (MMD) is a non-parametric two-sample test that measures the distance between two probability distributions $P_S$ (Source: DocVQA) and $P_T$ (Target: TextVQA) by mapping them into a Reproducing Kernel Hilbert Space (RKHS) $\mathcal{H}_k$:

$$\text{MMD}^2(\mathcal{H}_k, P_S, P_T) = \left\| \mathbb{E}_{x^s \sim P_S}[\phi(x^s)] - \mathbb{E}_{x^t \sim P_T}[\phi(x^t)] \right\|_{\mathcal{H}_k}^2$$

By the kernel trick, inner products in $\mathcal{H}_k$ are computed via kernel function $k(x, x') = \langle \phi(x), \phi(x') \rangle$. For empirical batches $X_S = \{z_i^s\}_{i=1}^{n_s}$ and $X_T = \{z_j^t\}_{j=1}^{n_t}$ extracted from pooled multimodal tokens, the unbiased estimator is:

$$\text{MMD}^2(X_S, X_T) = \frac{1}{n_s(n_s-1)}\sum_{i=1}^{n_s}\sum_{i' \neq i}^{n_s} k(z_i^s, z_{i'}^s) + \frac{1}{n_t(n_t-1)}\sum_{j=1}^{n_t}\sum_{j' \neq j}^{n_t} k(z_j^t, z_{j'}^t) - \frac{2}{n_s n_t}\sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^s, z_j^t)$$

### 1.2 Why Multi-Kernel RBF?
A single Gaussian Radial Basis Function (RBF) kernel:

$$k(x, x') = \exp\left( -\frac{\|x - x'\|_2^2}{2\sigma^2} \right)$$

is sensitive to the bandwidth choice $\sigma$:
* If $\sigma$ is too small, $k(x, x') \to 0$ for almost all pairs, saturating the gradient.
* If $\sigma$ is too large, $k(x, x') \to 1$, failing to distinguish subtle distribution discrepancies.

**Multi-Kernel MMD (MK-MMD)** constructs a convex combination of $M$ kernels with different bandwidth scales:

$$k_{\text{multi}}(x, x') = \sum_{m=1}^M \beta_m k_{\sigma_m}(x, x'), \quad \sum_{m=1}^M \beta_m = 1, \quad \beta_m \ge 0$$

In practice, a base bandwidth $\sigma_0$ is dynamically estimated via the **median heuristic** (the median of all pairwise Euclidean distances in the mini-batch), and a family of bandwidths is chosen around it:

$$\sigma_m = 2^{m - \lfloor M/2 \rfloor} \cdot \sigma_0, \quad m \in \{0, 1, \dots, M-1\}$$

This ensures the alignment simultaneously matches lower-order moments (global cluster centers) and higher-order moments (local density variations).

### 1.3 Characteristics in CrossVQA
* **Completely Unpaired:** Does not require DocVQA and TextVQA questions or images to be semantically matched; it operates purely on mini-batch sample distributions.
* **Batch Size Constraint:** Unbiased MMD requires pairwise matrix computations of size $\mathcal{O}((n_s + n_t)^2)$. On a single GPU with small mini-batches (e.g., $B \le 8$), empirical MMD has higher sample variance.

---

## 2. FIFO-Queue InfoNCE (Contrastive Alignment)

### 2.1 Mathematical Foundation
InfoNCE optimizes a lower bound on the mutual information between representations:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{j \in \mathcal{Q}^-} \exp(q \cdot k_j^- / \tau)}$$

where:
* $q \in \mathbb{R}^d$ is the query representation (e.g., from Target domain $X_T$).
* $k^+ \in \mathbb{R}^d$ is the positive key (semantically aligned representation from Source domain $X_S$).
* $\{k_j^-\}$ are negative keys (unrelated samples).
* $\tau$ is a learnable or fixed temperature parameter (typically $0.05 \le \tau \le 0.1$).

### 2.2 Why a FIFO Memory Queue?
Contrastive loss performance scales with the number of negative samples $K$. However, in vision-language models (e.g., Qwen2-VL), training with large batch sizes (e.g., $B \ge 512$) on a single workstation GPU (24GB VRAM) causes Out-Of-Memory (OOM).

Following the MoCo architecture, a **FIFO Memory Queue** maintains a dictionary of size $K$ (e.g., $K = 1024 \text{ to } 4096$) independent of the mini-batch size $B$:
1. **Forward Pass:** The current batch of representations $Z_{\text{batch}}$ is compared against the positive pair and all negative representations stored in the FIFO queue $\mathcal{Q}$.
2. **Enqueue & Dequeue:** After the backward step, the current representations are enqueued into $\mathcal{Q}$ with detached gradients (`.detach()`), and the oldest entries are discarded.
3. **Representation Consistency:** To avoid representation drift across iterations, keys are typically encoded by a slowly moving **momentum encoder** ($\theta_k \leftarrow m\theta_k + (1-m)\theta_q$, with $m \approx 0.999$) or normalized project heads.

### 2.3 Defining Pairs in CrossVQA
Unlike self-supervised image contrastive learning where positives are augmented views of the same image:
1. **Pseudo-Domain Pairing (for M2/M3):** A document image transformed via 3D homography/perspective warp into synthetic scene text forms a positive pair $(z_{\text{doc}}, z_{\text{synthetic}})$.
2. **Semantic Question Clustering / Matching:** If questions from DocVQA and TextVQA share semantic text similarity (e.g., cosine similarity of Question BGE/Sentence-BERT embedding $\ge 0.85$), they act as positive pairs $k^+$; all other entries in the queue serve as negatives $k^-$.
3. **Domain Prototype Centroid Alignment:** Positives are formed between domain cluster prototypes $\mathbf{c}_{\text{doc}}$ and $\mathbf{c}_{\text{scene}}$.

---

## 3. Comprehensive Comparison: MK-MMD vs. FIFO-Queue InfoNCE

| Property | Multi-Kernel RBF MMD | FIFO-Queue InfoNCE |
| :--- | :--- | :--- |
| **Primary Mechanism** | Distributional moment matching in RKHS | Hyperspherical instance discrimination (Mutual Information maximization) |
| **Sample Pairing Requirement** | **None** (Unpaired distribution matching) | **Required** (Explicit positive $(q, k^+)$ and negative $(q, k^-)$ pairing strategy) |
| **Geometric Behavior** | Pulls the support and shapes of $P_S$ and $P_T$ together | Pulls positive pairs together while spreading representations uniformly over the hypersphere |
| **Sensitivity to Batch Size $B$** | **High**: Variance increases when $B < 16$ | **Low**: FIFO queue decouples negative dictionary size $K$ from batch size $B$ |
| **VRAM Overhead** | Minimal ($\mathcal{O}(B^2)$ pairwise kernel matrix in float32) | Small fixed tensor allocation for queue $\mathcal{Q} \in \mathbb{R}^{K \times d}$ |
| **Risk of Representation Collapse** | Moderate (if $\lambda_{\text{align}}$ is too high, representations become degenerate) | Low (InfoNCE negative repulsion guarantees uniform dispersion) |
| **Integration Complexity** | **Low** (plug-and-play loss module on pooled features) | **Medium-High** (requires queue pointer management, momentum updates, and positive pairing logic) |

---

## 4. PyTorch Reference Implementations

### 4.1 Multi-Kernel RBF MMD Loss
```python
import torch
import torch.nn as nn

class MultiKernelRBF_MMD(nn.Module):
    def __init__(self, kernels_scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.scales = kernels_scales

    def forward(self, source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_feats: [N_s, D] pooled tokens from DocVQA
            target_feats: [N_t, D] pooled tokens from TextVQA
        Returns:
            Scalar MMD loss
        """
        n_s = source_feats.size(0)
        n_t = target_feats.size(0)
        
        # Concatenate features to compute joint distance matrix
        all_feats = torch.cat([source_feats, target_feats], dim=0)  # [N_s + N_t, D]
        dist_sq = torch.cdist(all_feats, all_feats, p=2).pow(2)     # [N, N]
        
        # Median heuristic for base bandwidth sigma_0
        with torch.no_grad():
            median_dist = torch.median(dist_sq)
            base_sigma = median_dist if median_dist > 0 else torch.tensor(1.0, device=source_feats.device)
            
        # Multi-kernel accumulation
        kernel_val = 0.0
        for scale in self.scales:
            gamma = 1.0 / (2.0 * base_sigma * scale + 1e-8)
            kernel_val = kernel_val + torch.exp(-gamma * dist_sq)
            
        # Partition kernel blocks
        K_ss = kernel_val[:n_s, :n_s]
        K_tt = kernel_val[n_s:, n_s:]
        K_st = kernel_val[:n_s, n_s:]
        
        # Unbiased estimators (exclude diagonal self-similarity)
        loss_ss = (K_ss.sum() - torch.diagonal(K_ss).sum()) / (n_s * (n_s - 1) + 1e-8)
        loss_tt = (K_tt.sum() - torch.diagonal(K_tt).sum()) / (n_t * (n_t - 1) + 1e-8)
        loss_st = 2.0 * K_st.mean()
        
        mmd_loss = loss_ss + loss_tt - loss_st
        return torch.clamp(mmd_loss, min=0.0)
```

### 4.2 FIFO-Queue InfoNCE Loss
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FIFOQueueInfoNCE(nn.Module):
    def __init__(self, feat_dim: int = 768, queue_size: int = 2048, temperature: float = 0.07):
        super().__init__()
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.temperature = temperature

        # Register FIFO memory queue as persistent buffer (no gradients)
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, feat_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr.item())
        
        # Ensure batch fits or wrap around
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
            ptr = batch_size - remaining
            
        self.queue_ptr[0] = ptr

    def forward(self, query: torch.Tensor, positive_key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, D] L2-normalized representations (e.g., Target / Scene)
            positive_key: [B, D] L2-normalized representations (e.g., Source / Doc)
        Returns:
            Scalar InfoNCE loss
        """
        q = F.normalize(query, dim=-1)
        k_pos = F.normalize(positive_key, dim=-1)
        
        # Positive logits: [B, 1]
        l_pos = torch.sum(q * k_pos, dim=-1, keepdim=True) / self.temperature
        
        # Negative logits against FIFO queue: [B, K]
        l_neg = torch.matmul(q, self.queue.T) / self.temperature
        
        # Concatenate logits: pos at index 0
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1 + K]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Enqueue current keys into FIFO dictionary
        self._dequeue_and_enqueue(k_pos.detach())
        
        return loss
```

---

## 5. Practical Recommendation for CrossVQA Thesis

1. **Default Recommendation for Baseline M2:**
   * Use **Multi-Kernel RBF MMD**.
   * *Rationale:* In standard VQA datasets, DocVQA and TextVQA samples are collected independently and do not naturally form aligned pairs $(x_{\text{doc}}, x_{\text{scene}})$. Forcing artificial positive pairs can introduce noise or negative transfer. MK-MMD operates on unpaired sample batches without requiring synthetic paired data or momentum encoders.
2. **When to Adopt FIFO-Queue InfoNCE (Configuration M3):**
   * If you introduce **Curriculum Pseudo-Domains** (e.g., DocVQA documents transformed via perspective homography and background texturing to resemble scene documents), the original document and the warped document form exact natural positive pairs $(z_{\text{orig}}, z_{\text{warped}})$. In that setting, FIFO-Queue InfoNCE offers superior stability and prevents representation collapse.
3. **Balancing Weight Schedule:**
   * Regardless of the alignment loss chosen, avoid setting static high weights early in training. Apply a standard ramp-up schedule (as in DANN/CDAN):
     $$\lambda(p) = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1$$
     where $p \in [0, 1]$ represents training progress, allowing the backbone task loss to stabilize the adapter representations before imposing strong cross-domain invariance.
