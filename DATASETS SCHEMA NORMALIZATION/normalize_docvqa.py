"""
Normalization Script for DocVQA Task 1 (gamusa/docvqa-task1-spdocvqa)
Transforms raw DocVQA annotations, document images, and Microsoft OCR into the Canonical VQA Schema.
"""

import os
import io
import json
import tarfile
from typing import Dict, Any, List, Optional
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from canonical_schema import (
    CanonicalVQAItem,
    extract_canonical_answer,
    normalize_ms_ocr_polygon,
)


def extract_ocr_from_ms_json(ocr_json_dict: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    """
    Parses Microsoft Azure OCR JSON result and returns normalized tokens and boxes.
    """
    tokens = []
    boxes = []

    rec_results = ocr_json_dict.get("recognitionResults", [])
    for page in rec_results:
        for line in page.get("lines", []):
            for word_obj in line.get("words", []):
                word_text = word_obj.get("text", "").strip()
                poly = word_obj.get("boundingBox", [])
                if word_text and poly:
                    tokens.append(word_text)
                    boxes.append(normalize_ms_ocr_polygon(poly, img_w, img_h))

    return {"tokens": tokens, "boxes": boxes}


def process_docvqa_split(
    split_name: str,
    raw_split_key: str,
    ann_file: str,
    images_tar_path: str,
    ocr_tar_path: Optional[str],
    output_parquet: str,
    embed_images: bool = True
) -> int:
    """
    Processes a single DocVQA split ('train', 'validation', or 'test') and writes to Parquet.
    """
    print(f"\n==========================================")
    print(f"Processing DocVQA Split: {split_name} (from {raw_split_key})")
    print(f"Annotation file: {ann_file}")
    print(f"==========================================")

    # 1. Load Annotations
    with open(ann_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)["data"]
    print(f"Total raw questions: {len(raw_items)}")

    # 2. Open Images Tar Archive
    print(f"Opening images tar: {images_tar_path}...")
    img_tar = tarfile.open(images_tar_path, "r:*")

    # 3. Open OCR Tar Archive (if provided)
    ocr_tar = None
    if ocr_tar_path and os.path.exists(ocr_tar_path):
        print(f"Opening OCR tar: {ocr_tar_path}...")
        ocr_tar = tarfile.open(ocr_tar_path, "r:*")

    processed_rows = []

    for item in tqdm(raw_items, desc=f"Converting {split_name}"):
        q_id = str(item["questionId"])
        question = item["question"].strip()
        answers = item.get("answers", None)
        img_rel_path = item["image"]  # e.g., 'documents/xnbl0037_1.png'
        doc_id = item.get("docId", None)

        # Image extraction
        # Normalize internal path in tar
        clean_img_path = img_rel_path.lstrip("./")
        img_bytes = None
        orig_w, orig_h = 0, 0

        try:
            member = img_tar.getmember(clean_img_path)
            f_img = img_tar.extractfile(member)
            raw_bytes = f_img.read()

            img = Image.open(io.BytesIO(raw_bytes))
            orig_w, orig_h = img.size

            if embed_images:
                buf = io.BytesIO()
                # Save as PNG or high quality JPEG
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
        except KeyError:
            # Fallback if path prefix is different
            alt_path = os.path.basename(clean_img_path)
            try:
                member = img_tar.getmember(alt_path)
                f_img = img_tar.extractfile(member)
                raw_bytes = f_img.read()
                img = Image.open(io.BytesIO(raw_bytes))
                orig_w, orig_h = img.size
                if embed_images:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
            except KeyError:
                print(f"Warning: Image {clean_img_path} not found in tar.")
                continue

        # OCR extraction
        ocr_tokens = []
        ocr_boxes = []
        if ocr_tar:
            base_name = os.path.splitext(os.path.basename(clean_img_path))[0]
            ocr_candidate_paths = [
                f"ocr_results/{base_name}.json",
                f"{base_name}.json"
            ]
            for candidate in ocr_candidate_paths:
                try:
                    ocr_member = ocr_tar.getmember(candidate)
                    f_ocr = ocr_tar.extractfile(ocr_member)
                    ocr_json = json.loads(f_ocr.read().decode("utf-8"))
                    ocr_res = extract_ocr_from_ms_json(ocr_json, orig_w, orig_h)
                    ocr_tokens = ocr_res["tokens"]
                    ocr_boxes = ocr_res["boxes"]
                    break
                except KeyError:
                    continue

        canonical_ans = extract_canonical_answer(answers) if answers else None

        metadata = {
            "question_types": item.get("question_types", []),
            "docId": doc_id,
            "ucsf_document_id": item.get("ucsf_document_id", ""),
            "ucsf_document_page_no": item.get("ucsf_document_page_no", ""),
            "original_image_path": img_rel_path
        }

        row = {
            "sample_id": f"docvqa_{q_id}",
            "domain_type": "document_text",
            "split": split_name,
            "question_id": q_id,
            "question": question,
            "question_tokens": question.lower().split(),
            "image_id": os.path.basename(clean_img_path),
            "image_width": orig_w,
            "image_height": orig_h,
            "answers": answers if answers else [],
            "canonical_answer": canonical_ans,
            "ocr_tokens": ocr_tokens,
            "ocr_boxes": ocr_boxes,
            "metadata": json.dumps(metadata, ensure_ascii=False),
        }
        if embed_images and img_bytes:
            row["image"] = {"bytes": img_bytes, "path": os.path.basename(clean_img_path)}

        processed_rows.append(row)

    img_tar.close()
    if ocr_tar:
        ocr_tar.close()

    # 4. Serialize to Parquet
    print(f"Writing {len(processed_rows)} rows to {output_parquet}...")
    df = pd.DataFrame(processed_rows)
    df.to_parquet(output_parquet, index=False, engine="pyarrow", compression="zstd")
    print(f"✓ Saved Parquet: {output_parquet} ({os.path.getsize(output_parquet) / (1024**2):.2f} MB)")

    return len(processed_rows)


if __name__ == "__main__":
    BASE_DIR = "./data/docvqa"
    OUT_DIR = "./data/normalized/docvqa"
    os.makedirs(OUT_DIR, exist_ok=True)

    splits_config = [
        {
            "split": "train",
            "raw_key": "train",
            "ann": f"{BASE_DIR}/annotations/train_v1.0_withQT.json",
            "img_tar": f"{BASE_DIR}/images/spdocvqa_images.tar.gz",
            "ocr_tar": f"{BASE_DIR}/ocr/spdocvqa_ocr.tar.gz",
            "out": f"{OUT_DIR}/train.parquet"
        },
        {
            "split": "validation",
            "raw_key": "val",
            "ann": f"{BASE_DIR}/annotations/val_v1.0_withQT.json",
            "img_tar": f"{BASE_DIR}/images/spdocvqa_images.tar.gz",
            "ocr_tar": f"{BASE_DIR}/ocr/spdocvqa_ocr.tar.gz",
            "out": f"{OUT_DIR}/validation.parquet"
        },
        {
            "split": "test",
            "raw_key": "test",
            "ann": f"{BASE_DIR}/annotations/test_v1.0.json",
            "img_tar": f"{BASE_DIR}/images/spdocvqa_images.tar.gz",
            "ocr_tar": f"{BASE_DIR}/ocr/spdocvqa_ocr.tar.gz",
            "out": f"{OUT_DIR}/test.parquet"
        }
    ]

    for cfg in splits_config:
        if os.path.exists(cfg["ann"]):
            process_docvqa_split(
                split_name=cfg["split"],
                raw_split_key=cfg["raw_key"],
                ann_file=cfg["ann"],
                images_tar_path=cfg["img_tar"],
                ocr_tar_path=cfg["ocr_tar"],
                output_parquet=cfg["out"],
                embed_images=True
            )
        else:
            print(f"Skipping {cfg['split']} (annotation file not found at {cfg['ann']})")
