"""
Canonical Schema Definition for CrossVQA
Harmonizing TextVQA (Scene Text) and DocVQA (Document Text)
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter
import datasets


@dataclass
class CanonicalVQAItem:
    """
    Unified Data Contract for CrossVQA Multi-Modal Examples.
    Compatible with both OCR-free VLMs and OCR-grounded spatial models.
    """
    # 1. Unique Identifiers
    sample_id: str                      # e.g., "textvqa_34602" or "docvqa_337"
    domain_type: str                    # "scene_text" or "document_text"
    split: str                          # "train", "validation", or "test"

    # 2. Text Question Input
    question_id: str                    # Original ID cast to string
    question: str                       # Cleaned text question
    question_tokens: Optional[List[str]] = None

    # 3. Visual Input
    image_id: str                       # Base image filename or hash
    image_width: int                    # Native image width
    image_height: int                   # Native image height

    # 4. Supervised Labels (None for unlabelled test split)
    answers: Optional[List[str]] = None
    canonical_answer: Optional[str] = None  # Majority vote answer

    # 5. Spatial OCR Grounding (Normalized to [0, 1000])
    ocr_tokens: Optional[List[str]] = None
    ocr_boxes: Optional[List[List[int]]] = None  # Format: [ymin, xmin, ymax, xmax]

    # 6. Domain-Specific Metadata
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_canonical_answer(answers: Optional[List[str]]) -> Optional[str]:
    """
    Computes majority voting canonical answer.
    Returns None if answers list is empty or None (e.g. in test set).
    """
    if not answers or len(answers) == 0:
        return None
    # Filter empty or whitespace-only answers
    cleaned = [ans.strip() for ans in answers if ans and ans.strip()]
    if not cleaned:
        return None
    # Majority vote: highest frequency
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def normalize_rosetta_box(box_dict: Dict[str, float]) -> List[int]:
    """
    Normalizes Rosetta OCR bounding box to [ymin, xmin, ymax, xmax] in [0, 1000].
    Rosetta boxes already have top_left_x, top_left_y, width, height in [0.0, 1.0].
    """
    tl_x = box_dict.get("top_left_x", 0.0)
    tl_y = box_dict.get("top_left_y", 0.0)
    w = box_dict.get("width", 0.0)
    h = box_dict.get("height", 0.0)

    xmin = int(round(max(0.0, min(1.0, tl_x)) * 1000))
    ymin = int(round(max(0.0, min(1.0, tl_y)) * 1000))
    xmax = int(round(max(0.0, min(1.0, tl_x + w)) * 1000))
    ymax = int(round(max(0.0, min(1.0, tl_y + h)) * 1000))

    return [ymin, xmin, ymax, xmax]


def normalize_ms_ocr_polygon(polygon: List[float], img_w: int, img_h: int) -> List[int]:
    """
    Normalizes Microsoft Azure OCR polygon [x1, y1, x2, y2, x3, y3, x4, y4]
    to an axis-aligned bounding box [ymin, xmin, ymax, xmax] in [0, 1000].
    """
    if not polygon or len(polygon) < 8 or img_w <= 0 or img_h <= 0:
        return [0, 0, 0, 0]

    xs = polygon[0::2]
    ys = polygon[1::2]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    xmin = int(round(max(0.0, min(1.0, min_x / img_w)) * 1000))
    xmax = int(round(max(0.0, min(1.0, max_x / img_w)) * 1000))
    ymin = int(round(max(0.0, min(1.0, min_y / img_h)) * 1000))
    ymax = int(round(max(0.0, min(1.0, max_y / img_h)) * 1000))

    return [ymin, xmin, ymax, xmax]


def get_canonical_features(include_image_bytes: bool = True) -> datasets.Features:
    """
    Returns Hugging Face datasets.Features schema for Parquet serialization.
    """
    features_dict = {
        "sample_id": datasets.Value("string"),
        "domain_type": datasets.Value("string"),
        "split": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "question_tokens": datasets.Sequence(datasets.Value("string")),
        "image_id": datasets.Value("string"),
        "image_width": datasets.Value("int32"),
        "image_height": datasets.Value("int32"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "canonical_answer": datasets.Value("string"),
        "ocr_tokens": datasets.Sequence(datasets.Value("string")),
        "ocr_boxes": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
        "metadata": datasets.Value("string"),  # Serialized JSON string for complex nested structures
    }
    if include_image_bytes:
        features_dict["image"] = datasets.Image()

    return datasets.Features(features_dict)
