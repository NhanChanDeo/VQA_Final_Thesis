"""
Proposed CrossVQA M2 Alignment Losses & Phase A9 Domain Discrepancy Metrics:
- FeatureQueue: FIFO ring buffer for small micro-batches on 16–24 GB GPUs
- MultiKernelRBFMMDLoss: Default M2 alignment objective
- CrossDomainInfoNCELoss: Contrastive ablation objective
- compute_rkhs_mmd & compute_proxy_a_distance: Phase A9 evaluation metrics
"""

from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class FeatureQueue:
    """Fixed-capacity FIFO ring buffer storing detached L2-normalized embeddings."""

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = int(max_size)
        self._buffer: torch.Tensor | None = None

    def enqueue(self, embs: torch.Tensor) -> None:
        if embs.numel() == 0:
            return
        clean = F.normalize(embs.detach().float().cpu(), dim=-1)
        if self._buffer is None:
            self._buffer = clean[-self.max_size :]
        else:
            combined = torch.cat([self._buffer, clean], dim=0)
            self._buffer = combined[-self.max_size :]

    def get(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self._buffer is None or self._buffer.numel() == 0:
            return None
        return self._buffer.to(device=device, dtype=dtype)

    def __len__(self) -> int:
        return 0 if self._buffer is None else int(self._buffer.shape[0])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "buffer": self._buffer.clone() if self._buffer is not None else None,
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            self._buffer = None
            return
        buf = state.get("buffer")
        if buf is None:
            self._buffer = None
        else:
            self._buffer = buf.detach().float().cpu()[-self.max_size :]


def _rbf_kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Iterable[float],
) -> torch.Tensor:
    sq_dists = torch.cdist(x, y, p=2).pow(2)
    kernel = torch.zeros_like(sq_dists)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
        kernel = kernel + torch.exp(-gamma * sq_dists)
    return kernel


def _mmd_from_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> torch.Tensor:
    k_xx = _rbf_kernel_matrix(x, x, sigmas).mean()
    k_yy = _rbf_kernel_matrix(y, y, sigmas).mean()
    k_xy = _rbf_kernel_matrix(x, y, sigmas).mean()
    return torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0)


class MultiKernelRBFMMDLoss(nn.Module):
    """
    Multi-bandwidth RBF Maximum Mean Discrepancy loss with dual-domain FIFO queues.
    Gradients flow cleanly through current micro-batch embeddings while the queue
    stabilizes distribution estimates on small batches (B=2..4).
    """

    def __init__(
        self,
        sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
        queue_size: int = 256,
        use_queue: bool = True,
    ) -> None:
        super().__init__()
        self.sigmas = tuple(sigmas)
        self.use_queue = bool(use_queue)
        self.doc_queue = FeatureQueue(max_size=queue_size)
        self.scene_queue = FeatureQueue(max_size=queue_size)

    def forward(
        self,
        pooled_vis: torch.Tensor,
        domain_ids: torch.Tensor,
        update_queue: bool = True,
    ) -> torch.Tensor:
        norm_vis = F.normalize(pooled_vis.float(), dim=-1)
        doc_batch = norm_vis[domain_ids == 0]
        scene_batch = norm_vis[domain_ids == 1]

        doc_parts = [doc_batch] if doc_batch.shape[0] > 0 else []
        scene_parts = [scene_batch] if scene_batch.shape[0] > 0 else []

        if self.use_queue:
            q_doc = self.doc_queue.get(norm_vis.device, norm_vis.dtype)
            q_scene = self.scene_queue.get(norm_vis.device, norm_vis.dtype)
            if q_doc is not None:
                doc_parts.append(q_doc)
            if q_scene is not None:
                scene_parts.append(q_scene)

        if not doc_parts or not scene_parts:
            loss = pooled_vis.sum() * 0.0
        else:
            x = torch.cat(doc_parts, dim=0)
            y = torch.cat(scene_parts, dim=0)
            loss = _mmd_from_tensors(x, y, self.sigmas)

        if update_queue and self.use_queue:
            self.doc_queue.enqueue(doc_batch)
            self.scene_queue.enqueue(scene_batch)

        return loss

    def get_queue_state(self) -> Dict[str, Any]:
        return {
            "doc_queue": self.doc_queue.state_dict(),
            "scene_queue": self.scene_queue.state_dict(),
        }

    def load_queue_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.doc_queue.load_state_dict(state.get("doc_queue"))
        self.scene_queue.load_state_dict(state.get("scene_queue"))


class CrossDomainInfoNCELoss(nn.Module):
    """
    Temperature-scaled cross-domain InfoNCE ablation objective.
    Pulls scene/doc embeddings toward the opposite domain's normalized prototype
    while contrasting against queue features from the same domain to prevent collapse.
    """

    def __init__(self, temperature: float = 0.07, queue_size: int = 256) -> None:
        super().__init__()
        self.temperature = float(max(temperature, 1e-4))
        self.doc_queue = FeatureQueue(max_size=queue_size)
        self.scene_queue = FeatureQueue(max_size=queue_size)

    def forward(
        self,
        pooled_vis: torch.Tensor,
        domain_ids: torch.Tensor,
        update_queue: bool = True,
    ) -> torch.Tensor:
        norm_vis = F.normalize(pooled_vis.float(), dim=-1)
        doc_batch = norm_vis[domain_ids == 0]
        scene_batch = norm_vis[domain_ids == 1]

        losses = []
        q_doc = self.doc_queue.get(norm_vis.device, norm_vis.dtype)
        q_scene = self.scene_queue.get(norm_vis.device, norm_vis.dtype)

        doc_pool = torch.cat([doc_batch, q_doc], dim=0) if (q_doc is not None and doc_batch.shape[0] > 0) else (
            doc_batch if doc_batch.shape[0] > 0 else q_doc
        )
        scene_pool = torch.cat([scene_batch, q_scene], dim=0) if (q_scene is not None and scene_batch.shape[0] > 0) else (
            scene_batch if scene_batch.shape[0] > 0 else q_scene
        )

        if scene_batch.shape[0] > 0 and doc_pool is not None and doc_pool.shape[0] > 0:
            doc_proto = F.normalize(doc_pool.mean(dim=0, keepdim=True), dim=-1)
            pos_logits = (scene_batch @ doc_proto.T) / self.temperature  # [Ns, 1]
            neg_pool = scene_pool if scene_pool is not None else scene_batch
            neg_logits = (scene_batch @ neg_pool.T) / self.temperature  # [Ns, Npool]
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            targets = torch.zeros(scene_batch.shape[0], dtype=torch.long, device=norm_vis.device)
            losses.append(F.cross_entropy(logits, targets))

        if doc_batch.shape[0] > 0 and scene_pool is not None and scene_pool.shape[0] > 0:
            scene_proto = F.normalize(scene_pool.mean(dim=0, keepdim=True), dim=-1)
            pos_logits = (doc_batch @ scene_proto.T) / self.temperature
            neg_pool = doc_pool if doc_pool is not None else doc_batch
            neg_logits = (doc_batch @ neg_pool.T) / self.temperature
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            targets = torch.zeros(doc_batch.shape[0], dtype=torch.long, device=norm_vis.device)
            losses.append(F.cross_entropy(logits, targets))

        if update_queue:
            self.doc_queue.enqueue(doc_batch)
            self.scene_queue.enqueue(scene_batch)

        if not losses:
            return pooled_vis.sum() * 0.0
        return torch.stack(losses).mean()

    def get_queue_state(self) -> Dict[str, Any]:
        return {
            "doc_queue": self.doc_queue.state_dict(),
            "scene_queue": self.scene_queue.state_dict(),
        }

    def load_queue_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.doc_queue.load_state_dict(state.get("doc_queue"))
        self.scene_queue.load_state_dict(state.get("scene_queue"))


def compute_rkhs_mmd(
    doc_embs: np.ndarray,
    scene_embs: np.ndarray,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    device: Optional[str] = None,
) -> float:
    """Computes normalized multi-kernel RBF MMD between NumPy embedding matrices (Phase A9)."""
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        x = F.normalize(torch.as_tensor(doc_embs, dtype=torch.float32, device=target_device), dim=-1)
        y = F.normalize(torch.as_tensor(scene_embs, dtype=torch.float32, device=target_device), dim=-1)
        return float(_mmd_from_tensors(x, y, sigmas=sigmas).item())


def compute_proxy_a_distance(
    doc_embs: np.ndarray,
    scene_embs: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes Proxy A-distance d_A = 2 * (1 - 2 * epsilon) via 5-fold stratified
    linear domain classification with strict per-fold StandardScaler fitting.
    """
    x = np.vstack([np.asarray(doc_embs, dtype=np.float32), np.asarray(scene_embs, dtype=np.float32)])
    y = np.concatenate(
        [
            np.zeros(len(doc_embs), dtype=np.int64),
            np.ones(len(scene_embs), dtype=np.int64),
        ]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_errors = []
    for train_idx, test_idx in skf.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(x_train, y[train_idx])
        preds = clf.predict(x_test)
        err = float(np.mean(preds != y[test_idx]))
        fold_errors.append(err)

    epsilon = float(np.mean(fold_errors))
    proxy_a_dist = max(0.0, 2.0 * (1.0 - 2.0 * epsilon))
    return {
        "error_rate": epsilon,
        "accuracy": 1.0 - epsilon,
        "proxy_a_distance": proxy_a_dist,
    }
