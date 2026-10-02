"""
Normalization Script for TextVQA (gamusa/textvqa-v0.5.1)
Transforms raw TextVQA annotations, images, and Rosetta OCR into the Canonical VQA Schema.
"""

import os
import io
import json
import zipfile
from typing import Dict, Any, List, Optional
from PIL import Image, ImageOps
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from canonical_schema import (
    CanonicalVQAItem,
    extract_canonical_answer,
    normalize_rosetta_box,
)


def load_rosetta_ocr(ocr_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads Rosetta OCR JSON and indexes by image_id.
    """
    print(f"Loading OCR from: {ocr_path}...")
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    ocr_lookup = {}
    for item in data:
        img_id = item["image_id"]
        tokens = item.get("ocr_tokens", [])
        ocr_info = item.get("ocr_info", [])

        boxes = []
        for word_info in ocr_info:
            bbox = word_info.get("bounding_box", {})
            boxes.append(normalize_rosetta_box(bbox))

        ocr_lookup[img_id] = {
            "tokens": tokens,
            "boxes": boxes
        }
    print(f"Indexed OCR for {len(ocr_lookup)} images.")
    return ocr_lookup


def process_textvqa_split(
    split_name: str,
    raw_split_key: str,
    ann_file: str,
    ocr_file: str,
    zip_path: str,
    output_parquet: str,
    embed_images: bool = True
) -> int:
    """
    Processes a single TextVQA split ('train', 'validation', or 'test') and writes to Parquet.
    """
    print(f"\n==========================================")
    print(f"Processing TextVQA Split: {split_name} (from {raw_split_key})")
    print(f"Annotation file: {ann_file}")
    print(f"==========================================")

    # 1. Load Annotations
    with open(ann_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)["data"]
    print(f"Total raw questions: {len(raw_items)}")

    # 2. Load OCR Lookup
    ocr_lookup = load_rosetta_ocr(ocr_file)

    # 3. Open Zip file for reading images
    print(f"Opening image archive: {zip_path}...")
    zip_ref = zipfile.ZipFile(zip_path, "r")
    # Identify internal folder prefix ('train_images/' or 'test_images/')
    sample_prefix = "train_images/" if split_name != "test" else "test_images/"

    processed_rows = []

    for item in tqdm(raw_items, desc=f"Converting {split_name}"):
        q_id = str(item["question_id"])
        img_id = item["image_id"]
        question = item["question"].strip()
        q_tokens = item.get("question_tokens", None)
        answers = item.get("answers", None)

        # Retrieve OCR
        ocr_data = ocr_lookup.get(img_id, {"tokens": [], "boxes": []})

        # Visual handling
        img_filename = f"{sample_prefix}{img_id}.jpg"
        img_bytes = None
        orig_w = item.get("image_width", 0)
        orig_h = item.get("image_height", 0)

        try:
            raw_img_bytes = zip_ref.read(img_filename)
            # Correct EXIF rotation
            img = Image.open(io.BytesIO(raw_img_bytes))
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            if embed_images:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()
        except KeyError:
            print(f"Warning: Image {img_filename} not found in {zip_path}")
            continue

        canonical_ans = extract_canonical_answer(answers) if answers else None

        metadata = {
            "image_classes": item.get("image_classes", []),
            "flickr_original_url": item.get("flickr_original_url", ""),
            "flickr_300k_url": item.get("flickr_300k_url", ""),
        }

        row = {
            "sample_id": f"textvqa_{q_id}",
            "domain_type": "scene_text",
            "split": split_name,
            "question_id": q_id,
            "question": question,
            "question_tokens": q_tokens if q_tokens else [],
            "image_id": img_id,
            "image_width": orig_w,
            "image_height": orig_h,
            "answers": answers if answers else [],
            "canonical_answer": canonical_ans,
            "ocr_tokens": ocr_data["tokens"],
            "ocr_boxes": ocr_data["boxes"],
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }
        if embed_images and img_bytes:
            row["image"] = {"bytes": img_bytes, "path": f"{img_id}.jpg"}

        processed_rows.append(row)

    zip_ref.close()

    # 4. Serialize to Parquet
    print(f"Writing {len(processed_rows)} rows to {output_parquet}...")
    df = pd.DataFrame(processed_rows)
    df.to_parquet(output_parquet, index=False, engine="pyarrow", compression="zstd")
    print(f"✓ Saved Parquet: {output_parquet} ({os.path.getsize(output_parquet) / (1024**2):.2f} MB)")

    return len(processed_rows)


if __name__ == "__main__":
    # Example execution paths (configurable)
    BASE_DIR = "./data/textvqa"
    OUT_DIR = "./data/normalized/textvqa"
    os.makedirs(OUT_DIR, exist_ok=True)

    splits_config = [
        {
            "split": "train",
            "raw_key": "train",
            "ann": f"{BASE_DIR}/annotations/TextVQA_0.5.1_train.json",
            "ocr": f"{BASE_DIR}/ocr/TextVQA_Rosetta_OCR_v0.2_train.json",
            "zip": f"{BASE_DIR}/images/train_val_images.zip",
            "out": f"{OUT_DIR}/train.parquet"
        },
        {
            "split": "validation",
            "raw_key": "val",
            "ann": f"{BASE_DIR}/annotations/TextVQA_0.5.1_val.json",
            "ocr": f"{BASE_DIR}/ocr/TextVQA_Rosetta_OCR_v0.2_val.json",
            "zip": f"{BASE_DIR}/images/train_val_images.zip",
            "out": f"{OUT_DIR}/validation.parquet"
        },
        {
            "split": "test",
            "raw_key": "test",
            "ann": f"{BASE_DIR}/annotations/TextVQA_0.5.1_test.json",
            "ocr": f"{BASE_DIR}/ocr/TextVQA_Rosetta_OCR_v0.2_test.json",
            "zip": f"{BASE_DIR}/images/test_images.zip",
            "out": f"{OUT_DIR}/test.parquet"
        }
    ]

    for cfg in splits_config:
        if os.path.exists(cfg["ann"]):
            process_textvqa_split(
                split_name=cfg["split"],
                raw_split_key=cfg["raw_key"],
                ann_file=cfg["ann"],
                ocr_file=cfg["ocr"],
                zip_path=cfg["zip"],
                output_parquet=cfg["out"],
                embed_images=True
            )
        else:
            print(f"Skipping {cfg['split']} (annotation file not found at {cfg['ann']})")
