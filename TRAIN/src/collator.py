"""
Multimodal Data Collator for Qwen2-VL-2B and SmolVLM-Base:
Handles dynamic image tokenization, prompt formatting, OCR-conditioning, and label masking.
"""

from typing import List, Dict, Any
import torch
from .dataset import decode_image_payload

class CrossVQADataCollator:
    def __init__(
        self,
        processor,
        model_type: str = "qwen2_vl",
        include_ocr: bool = False,
        max_ocr_tokens: int = 50,
        is_train: bool = True,
    ):
        self.processor = processor
        self.model_type = model_type.lower()
        self.include_ocr = include_ocr
        self.max_ocr_tokens = max_ocr_tokens
        self.is_train = is_train

    def _format_prompt(self, question: str, ocr_tokens: List[str] = None, ocr_boxes: List[List[int]] = None) -> str:
        prompt_text = ""
        if self.include_ocr and ocr_tokens:
            prompt_text += "Detected Text Context:\n"
            for t, b in zip(ocr_tokens[:self.max_ocr_tokens], ocr_boxes[:self.max_ocr_tokens]):
                prompt_text += f"- '{t}' at {b}\n"
        prompt_text += f"Question: {question}\nAnswer:"
        return prompt_text

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        texts = []

        for item in batch:
            img = decode_image_payload(item["image"])
            images.append(img)

            ocr_tokens = item.get("ocr_tokens", [])
            ocr_boxes = item.get("ocr_boxes", [])
            prompt = self._format_prompt(item["question"], ocr_tokens, ocr_boxes)

            if self.is_train:
                # Append target answer for causal language modeling loss
                target_answer = item.get("canonical_answer", "")
                if not target_answer and item.get("answers"):
                    target_answer = item["answers"][0]
                full_text = f"{prompt} {target_answer}"
            else:
                full_text = prompt

            texts.append(full_text)

        # Batch encode with processor
        if "qwen2" in self.model_type:
            # Qwen2-VL handles images and text via chat/prompt template or visual input
            # Standard Qwen2VLProcessor accepts images and text lists
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        if self.is_train and "input_ids" in inputs:
            labels = inputs["input_ids"].clone()
            # Mask out padding tokens from loss
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

        return inputs
