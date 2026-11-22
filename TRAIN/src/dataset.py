"""
Unified streaming and local NVMe dataset loader for CrossVQA:
- Supports 'ALT_THREE_SPLITS' (via local Parquet snapshot or revision='three-splits')
- Supports 'VALIDATION_AS_TEST' (via revision='main' + manifest pointer filtering)
- Provides InterleavedCrossVQADataset (B3) and PairedDomainDataset (M1-M3)
"""

import io
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import polars as pl
from PIL import Image
from torch.utils.data import IterableDataset

DATASET_REPOS = {
    "docvqa": "gamusa/docvqa-canonical",
    "textvqa": "gamusa/textvqa-canonical",
}


def _find_local_split_files(local_data_dir: str, dataset_name: str, split: str):
    root = Path(local_data_dir) / dataset_name
    for candidate_dir in (root / "data", root):
        if candidate_dir.exists():
            matches = sorted(str(p) for p in candidate_dir.glob(f"{split}-*.parquet"))
            if matches:
                return matches
    return None


def _apply_few_shot_filter(
    ds,
    dataset_name: str,
    split: str,
    few_shot_pct: Optional[int],
    few_shot_manifest_path: Optional[str],
):
    if few_shot_pct is None or int(few_shot_pct) >= 100 or dataset_name != "textvqa" or split != "train":
        return ds
    assert few_shot_manifest_path is not None, "few_shot_manifest_path required when few_shot_pct < 100"
    col_name = f"in_{int(few_shot_pct)}pct"
    manifest_df = pl.read_parquet(few_shot_manifest_path)
    valid_ids = set(manifest_df.filter(pl.col(col_name))["sample_id"].to_list())
    return ds.filter(lambda x: x["sample_id"] in valid_ids)


def load_crossvqa_dataset(
    dataset_name: str,
    split: str,
    split_regime: str = "alt_three_splits",
    manifest_path: Optional[str] = None,
    streaming: bool = True,
    local_data_dir: Optional[str] = None,
    few_shot_pct: Optional[int] = None,
    few_shot_manifest_path: Optional[str] = None,
):
    """
    Loads samples for training/eval from Colab NVMe Parquet snapshot or HF Hub.
    """
    from datasets import load_dataset

    assert dataset_name in DATASET_REPOS, f"Invalid dataset: {dataset_name}"
    repo_id = DATASET_REPOS[dataset_name]

    if split_regime.lower() == "alt_three_splits":
        local_files = _find_local_split_files(local_data_dir, dataset_name, split) if local_data_dir else None
        if local_files:
            ds = load_dataset(
                "parquet",
                data_files={split: local_files},
                split=split,
                streaming=streaming,
            )
        else:
            ds = load_dataset(
                repo_id,
                revision="three-splits",
                split=split,
                streaming=streaming,
            )
        return _apply_few_shot_filter(ds, dataset_name, split, few_shot_pct, few_shot_manifest_path)

    if split_regime.lower() == "validation_as_test":
        if split == "test":
            return load_dataset(repo_id, revision="main", split="validation", streaming=streaming)

        assert manifest_path is not None, "manifest_path required for validation_as_test train/val split"
        manifest_df = pl.read_parquet(manifest_path)
        target_split_name = "NEW_train" if split == "train" else "NEW_validation"
        valid_ids = set(
            manifest_df.filter(
                (pl.col("dataset") == dataset_name) & (pl.col("NEW_split") == target_split_name)
            )["sample_id"].to_list()
        )
        base_ds = load_dataset(repo_id, revision="main", split="train", streaming=streaming)
        ds = base_ds.filter(lambda x: x["sample_id"] in valid_ids)
        return _apply_few_shot_filter(ds, dataset_name, split, few_shot_pct, few_shot_manifest_path)

    raise ValueError(f"Unknown split regime: {split_regime}")


class InterleavedCrossVQADataset(IterableDataset):
    """Interleaves DocVQA and TextVQA streams for B3 joint training."""

    def __init__(
        self,
        doc_dataset,
        scene_dataset,
        doc_prob: float = 0.5,
        seed: int = 42,
        shuffle_buffer: int = 256,
    ) -> None:
        super().__init__()
        self.doc_dataset = doc_dataset
        self.scene_dataset = scene_dataset
        self.doc_prob = float(doc_prob)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import random

        rng = random.Random(self.seed)
        doc_ds = (
            self.doc_dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            if hasattr(self.doc_dataset, "shuffle")
            else self.doc_dataset
        )
        scene_ds = (
            self.scene_dataset.shuffle(seed=self.seed + 1, buffer_size=self.shuffle_buffer)
            if hasattr(self.scene_dataset, "shuffle")
            else self.scene_dataset
        )
        doc_it = iter(doc_ds)
        scene_it = iter(scene_ds)
        doc_done = False
        scene_done = False

        while not (doc_done and scene_done):
            pick_doc = (not doc_done) and (scene_done or rng.random() < self.doc_prob)
            if pick_doc:
                try:
                    yield next(doc_it)
                except StopIteration:
                    doc_done = True
            else:
                try:
                    yield next(scene_it)
                except StopIteration:
                    scene_done = True


class PairedDomainDataset(IterableDataset):
    """
    Yields alternating (DocVQA, TextVQA) sample pairs so every micro-batch >= 2
    contains both domains for M1 routing and M2 MMD/InfoNCE feature alignment.
    """

    def __init__(
        self,
        doc_dataset,
        scene_dataset,
        max_pairs: Optional[int] = None,
        seed: int = 42,
        shuffle_buffer: int = 256,
    ) -> None:
        super().__init__()
        self.doc_dataset = doc_dataset
        self.scene_dataset = scene_dataset
        self.max_pairs = max_pairs
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        doc_ds = (
            self.doc_dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            if hasattr(self.doc_dataset, "shuffle")
            else self.doc_dataset
        )
        scene_ds = (
            self.scene_dataset.shuffle(seed=self.seed + 1, buffer_size=self.shuffle_buffer)
            if hasattr(self.scene_dataset, "shuffle")
            else self.scene_dataset
        )
        doc_it = iter(doc_ds)
        scene_it = iter(scene_ds)

        pairs_emitted = 0
        # Primary target epoch stops once TextVQA (target) completes unless max_pairs is set
        while True:
            if self.max_pairs is not None and pairs_emitted >= self.max_pairs:
                break
            try:
                scene_sample = next(scene_it)
            except StopIteration:
                break

            try:
                doc_sample = next(doc_it)
            except StopIteration:
                doc_it = iter(doc_ds)
                try:
                    doc_sample = next(doc_it)
                except StopIteration:
                    break

            yield doc_sample
            yield scene_sample
            pairs_emitted += 1


def decode_image_payload(image_payload) -> Image.Image:
    """Helper to convert HF image bytes or PIL object into an RGB PIL Image."""
    if isinstance(image_payload, Image.Image):
        return image_payload.convert("RGB")
    if isinstance(image_payload, dict) and "bytes" in image_payload:
        return Image.open(io.BytesIO(image_payload["bytes"])).convert("RGB")
    if isinstance(image_payload, bytes):
        return Image.open(io.BytesIO(image_payload)).convert("RGB")
    raise ValueError(f"Unsupported image format: {type(image_payload)}")
