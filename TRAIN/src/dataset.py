"""
Unified streaming and local NVMe dataset loader for CrossVQA:
- Supports 'ALT_THREE_SPLITS' (via local Parquet snapshot or revision='three-splits')
- Supports 'VALIDATION_AS_TEST' (via revision='main' + manifest pointer filtering)
- Provides InterleavedCrossVQADataset (B3) and PairedDomainDataset (M1-M3)
"""

import io
import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import polars as pl
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

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
    if _is_map_style_dataset(ds) and hasattr(ds, "select"):
        sample_ids = ds["sample_id"]
        keep_indices = [i for i, sid in enumerate(sample_ids) if sid in valid_ids]
        return ds.select(keep_indices)
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
            num_proc = None if streaming else min(4, os.cpu_count() or 1)
            dl_kwargs = {"num_proc": num_proc} if (num_proc and num_proc > 1 and len(local_files) > 1) else {}
            ds = load_dataset(
                "parquet",
                data_files={split: local_files},
                split=split,
                streaming=streaming,
                **dl_kwargs,
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


def _is_map_style_dataset(ds: Any) -> bool:
    return (
        hasattr(ds, "__len__")
        and hasattr(ds, "__getitem__")
        and not isinstance(ds, IterableDataset)
    )


def _iter_shuffled_domain(ds: Any, seed: int, shuffle_buffer: int) -> Iterator[Dict[str, Any]]:
    import random

    if _is_map_style_dataset(ds):
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        for idx in indices:
            yield ds[idx]
        return
    stream_ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer) if hasattr(ds, "shuffle") else ds
    yield from iter(stream_ds)


def _image_area_from_payload(payload: Any) -> int:
    if isinstance(payload, Image.Image):
        return int(payload.width) * int(payload.height)
    if isinstance(payload, dict):
        if payload.get("bytes") is not None:
            with Image.open(io.BytesIO(payload["bytes"])) as im:
                return int(im.width) * int(im.height)
        if payload.get("path"):
            with Image.open(payload["path"]) as im:
                return int(im.width) * int(im.height)
    if isinstance(payload, bytes):
        with Image.open(io.BytesIO(payload)) as im:
            return int(im.width) * int(im.height)
    return 0


def extract_dataset_image_areas(dataset: Any) -> Optional[List[int]]:
    """
    Lazily extracts pixel area (width * height) per sample for a map-style dataset
    using metadata columns or image header inspection without decoding pixel arrays.
    """
    if not _is_map_style_dataset(dataset):
        return None
    n = len(dataset)
    if n == 0:
        return []
    try:
        col_names = list(getattr(dataset, "column_names", []) or [])
        for w_col, h_col in (("image_width", "image_height"), ("width", "height")):
            if w_col in col_names and h_col in col_names:
                widths = dataset[w_col]
                heights = dataset[h_col]
                return [int(w) * int(h) for w, h in zip(widths, heights)]

        if hasattr(dataset, "cast_column") and "image" in getattr(dataset, "features", {}):
            from datasets import Image as HFImage

            raw_ds = dataset.select_columns(["image"]) if hasattr(dataset, "select_columns") else dataset
            raw_ds = raw_ds.cast_column("image", HFImage(decode=False))
            return [_image_area_from_payload(raw_ds[i]["image"]) for i in range(len(raw_ds))]

        if isinstance(dataset, (list, tuple)):
            return [_image_area_from_payload(item.get("image")) for item in dataset]

        return [_image_area_from_payload(dataset[i].get("image")) for i in range(n)]
    except Exception:
        return None


def mega_batch_bucket_indices(
    indices: List[int],
    areas: Optional[List[int]],
    batch_size: int,
    seed: int,
    mega_batch_mult: int = 32,
) -> List[int]:
    """
    Groups shuffled indices into resolution-coherent micro-batches inside mega-batches,
    then shuffles micro-batch order (keeping any trailing partial batch at the end so
    mid-epoch resume offsets remain exact multiples of `batch_size`).
    """
    import random

    bs = int(batch_size)
    if bs <= 1 or not areas or len(indices) <= bs:
        return indices

    n = len(indices)
    full_end = (n // bs) * bs
    tail = indices[full_end:]
    head = indices[:full_end]
    mega_size = max(bs * max(1, int(mega_batch_mult)), bs)

    micro_batches: List[List[int]] = []
    for start in range(0, len(head), mega_size):
        chunk = sorted(head[start : start + mega_size], key=lambda idx: areas[idx], reverse=True)
        for mb_start in range(0, len(chunk), bs):
            micro_batches.append(chunk[mb_start : mb_start + bs])

    random.Random(int(seed)).shuffle(micro_batches)
    out = [idx for mb in micro_batches for idx in mb]
    if tail:
        out.extend(tail)
    return out


def _iter_shuffled_domain_lazy(
    ds: Any,
    seed: int,
    shuffle_buffer: int,
    bucket_batch_size: int = 0,
    areas: Optional[List[int]] = None,
):
    """Yields zero-decode callables (lambda: ds[idx]) for map-style datasets so skipping is instant."""
    import random

    if _is_map_style_dataset(ds):
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        if bucket_batch_size > 1 and areas is not None:
            indices = mega_batch_bucket_indices(indices, areas, bucket_batch_size, seed + 77)
        for idx in indices:
            yield (lambda i=idx: ds[i])
        return
    stream_ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer) if hasattr(ds, "shuffle") else ds
    for item in iter(stream_ds):
        yield (lambda x=item: x)


class InterleavedCrossVQADataset(IterableDataset):
    """Interleaves DocVQA and TextVQA streams (or map-style Arrow datasets) for B3 joint training."""

    def __init__(
        self,
        doc_dataset,
        scene_dataset,
        doc_prob: float = 0.5,
        seed: int = 42,
        shuffle_buffer: int = 256,
        bucket_batch_size: int = 0,
    ) -> None:
        super().__init__()
        self.doc_dataset = doc_dataset
        self.scene_dataset = scene_dataset
        self.doc_prob = float(doc_prob)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.bucket_batch_size = int(bucket_batch_size)
        self._epoch = 0
        self._skip_samples = 0
        self._doc_areas: Optional[List[int]] = None
        self._scene_areas: Optional[List[int]] = None

    def set_epoch(
        self,
        epoch: int,
        skip_samples: int = 0,
        bucket_batch_size: Optional[int] = None,
    ) -> None:
        self._epoch = int(epoch)
        self._skip_samples = max(0, int(skip_samples))
        if bucket_batch_size is not None:
            self.bucket_batch_size = int(bucket_batch_size)

    def _ensure_areas(self) -> None:
        if self.bucket_batch_size <= 1:
            return
        if self._doc_areas is None:
            self._doc_areas = extract_dataset_image_areas(self.doc_dataset)
        if self._scene_areas is None:
            self._scene_areas = extract_dataset_image_areas(self.scene_dataset)

    def __len__(self) -> int:
        if _is_map_style_dataset(self.doc_dataset) and _is_map_style_dataset(self.scene_dataset):
            return len(self.doc_dataset) + len(self.scene_dataset)
        raise TypeError("Length unavailable for streaming InterleavedCrossVQADataset.")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import random

        self._ensure_areas()
        sub_bs = max(1, self.bucket_batch_size // 2) if self.bucket_batch_size > 1 else 0
        ep_seed = self.seed + self._epoch * 1009
        rng = random.Random(ep_seed)
        doc_it = _iter_shuffled_domain_lazy(
            self.doc_dataset, ep_seed, self.shuffle_buffer, bucket_batch_size=sub_bs, areas=self._doc_areas
        )
        scene_it = _iter_shuffled_domain_lazy(
            self.scene_dataset, ep_seed + 1, self.shuffle_buffer, bucket_batch_size=sub_bs, areas=self._scene_areas
        )
        doc_done = False
        scene_done = False
        remaining_skip = self._skip_samples
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        emit_idx = 0

        while not (doc_done and scene_done):
            pick_doc = (not doc_done) and (scene_done or rng.random() < self.doc_prob)
            if pick_doc:
                try:
                    getter = next(doc_it)
                except StopIteration:
                    doc_done = True
                    continue
            else:
                try:
                    getter = next(scene_it)
                except StopIteration:
                    scene_done = True
                    continue
            if remaining_skip > 0:
                remaining_skip -= 1
                continue
            if num_workers <= 1 or (emit_idx % num_workers) == worker_id:
                yield getter()
            emit_idx += 1


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
        bucket_batch_size: int = 0,
    ) -> None:
        super().__init__()
        self.doc_dataset = doc_dataset
        self.scene_dataset = scene_dataset
        self.max_pairs = max_pairs
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.bucket_batch_size = int(bucket_batch_size)
        self._epoch = 0
        self._skip_samples = 0
        self._doc_areas: Optional[List[int]] = None
        self._scene_areas: Optional[List[int]] = None

    def set_epoch(
        self,
        epoch: int,
        skip_samples: int = 0,
        bucket_batch_size: Optional[int] = None,
    ) -> None:
        self._epoch = int(epoch)
        self._skip_samples = max(0, int(skip_samples))
        if bucket_batch_size is not None:
            self.bucket_batch_size = int(bucket_batch_size)

    def _ensure_areas(self) -> None:
        if self.bucket_batch_size <= 1:
            return
        if self._doc_areas is None:
            self._doc_areas = extract_dataset_image_areas(self.doc_dataset)
        if self._scene_areas is None:
            self._scene_areas = extract_dataset_image_areas(self.scene_dataset)

    def __len__(self) -> int:
        if _is_map_style_dataset(self.scene_dataset):
            n_pairs = len(self.scene_dataset)
            if self.max_pairs is not None:
                n_pairs = min(n_pairs, int(self.max_pairs))
            return 2 * n_pairs
        if self.max_pairs is not None:
            return 2 * int(self.max_pairs)
        raise TypeError("Length unavailable for streaming PairedDomainDataset without max_pairs.")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._ensure_areas()
        sub_bs = max(1, self.bucket_batch_size // 2) if self.bucket_batch_size > 1 else 0
        ep_seed = self.seed + self._epoch * 1009
        doc_cycle = 0
        doc_it = _iter_shuffled_domain_lazy(
            self.doc_dataset, ep_seed, self.shuffle_buffer, bucket_batch_size=sub_bs, areas=self._doc_areas
        )
        scene_it = _iter_shuffled_domain_lazy(
            self.scene_dataset, ep_seed + 1, self.shuffle_buffer, bucket_batch_size=sub_bs, areas=self._scene_areas
        )

        pairs_emitted = 0
        unskipped_pair_idx = 0
        remaining_skip = self._skip_samples
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Primary target epoch stops once TextVQA (target) completes unless max_pairs is set
        while True:
            if self.max_pairs is not None and pairs_emitted >= self.max_pairs:
                break
            try:
                scene_getter = next(scene_it)
            except StopIteration:
                break

            try:
                doc_getter = next(doc_it)
            except StopIteration:
                doc_cycle += 1
                doc_it = _iter_shuffled_domain_lazy(
                    self.doc_dataset,
                    ep_seed + doc_cycle * 37,
                    self.shuffle_buffer,
                    bucket_batch_size=sub_bs,
                    areas=self._doc_areas,
                )
                try:
                    doc_getter = next(doc_it)
                except StopIteration:
                    break

            skip_doc = remaining_skip > 0
            if skip_doc:
                remaining_skip -= 1
            skip_scene = remaining_skip > 0
            if skip_scene:
                remaining_skip -= 1

            if not (skip_doc and skip_scene):
                mine = num_workers <= 1 or (unskipped_pair_idx % num_workers) == worker_id
                if mine:
                    if not skip_doc:
                        yield doc_getter()
                    if not skip_scene:
                        yield scene_getter()
                unskipped_pair_idx += 1
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
