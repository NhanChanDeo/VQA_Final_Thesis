"""
Unified streaming and local dataset loader for CrossVQA:
- Supports 'ALT_THREE_SPLITS' (via revision='three-splits')
- Supports 'VALIDATION_AS_TEST' (via revision='main' + manifest pointer filtering)
"""

import io
from PIL import Image
from datasets import load_dataset
import polars as pl

DATASET_REPOS = {
    "docvqa": "gamusa/docvqa-canonical",
    "textvqa": "gamusa/textvqa-canonical",
}

def load_crossvqa_dataset(
    dataset_name: str,
    split: str,
    split_regime: str = "alt_three_splits",
    manifest_path: str = None,
    streaming: bool = True,
):
    """
    Loads samples for training/eval from Hugging Face or filtered manifest.
    
    Args:
        dataset_name: 'docvqa' or 'textvqa'
        split: 'train', 'validation', or 'test'
        split_regime: 'alt_three_splits' or 'validation_as_test'
        manifest_path: Optional path to split manifest parquet (if filtering main)
        streaming: Whether to stream samples without full dataset download
    """
    assert dataset_name in DATASET_REPOS, f"Invalid dataset: {dataset_name}"
    repo_id = DATASET_REPOS[dataset_name]

    if split_regime.lower() == "alt_three_splits":
        # Native three-splits branch on Hugging Face Hub
        ds = load_dataset(
            repo_id,
            revision="three-splits",
            split=split,
            streaming=streaming,
        )
        return ds

    elif split_regime.lower() == "validation_as_test":
        # Original 'main' branch + manifest pointer lookup
        if split == "test":
            # In validation_as_test, official validation acts as the held-out test set
            ds = load_dataset(repo_id, revision="main", split="validation", streaming=streaming)
            return ds

        assert manifest_path is not None, "manifest_path required for validation_as_test train/val split"
        manifest_df = pl.read_parquet(manifest_path)
        target_split_name = "NEW_train" if split == "train" else "NEW_validation"
        valid_ids = set(
            manifest_df.filter(
                (pl.col("dataset") == dataset_name) & (pl.col("NEW_split") == target_split_name)
            )["sample_id"].to_list()
        )

        base_ds = load_dataset(repo_id, revision="main", split="train", streaming=streaming)
        if streaming:
            return base_ds.filter(lambda x: x["sample_id"] in valid_ids)
        else:
            return base_ds.filter(lambda x: x["sample_id"] in valid_ids)

    else:
        raise ValueError(f"Unknown split regime: {split_regime}")


def decode_image_payload(image_payload) -> Image.Image:
    """Helper to convert HF image bytes or PIL object into an RGB PIL Image."""
    if isinstance(image_payload, Image.Image):
        return image_payload.convert("RGB")
    elif isinstance(image_payload, dict) and "bytes" in image_payload:
        return Image.open(io.BytesIO(image_payload["bytes"])).convert("RGB")
    elif isinstance(image_payload, bytes):
        return Image.open(io.BytesIO(image_payload)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {type(image_payload)}")
