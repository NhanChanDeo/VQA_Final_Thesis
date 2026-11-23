"""
Multimodal Data Collator for Qwen2-VL-2B and SmolVLM-Base / SmolVLM-500M-Base:
Handles model-aware image placeholders, prompt formatting, OCR conditioning,
strict answer-only label masking, EOS appending, domain IDs, and M3 transforms.
"""

from typing import Any, Callable, Dict, List, Optional
import torch
from .dataset import decode_image_payload


def _infer_domain_id(item: Dict[str, Any]) -> int:
    """Returns 0 for document_text (DocVQA) and 1 for scene_text (TextVQA)."""
    domain_type = str(item.get("domain_type", "")).lower()
    source_dataset = str(item.get("source_dataset", item.get("dataset", ""))).lower()
    if domain_type == "document_text" or source_dataset == "docvqa":
        return 0
    return 1


def _compute_soft_weight(item: Dict[str, Any], domain_id: int) -> float:
    """Computes consensus-based sample weight in [0.33, 1.0] for TextVQA and 1.0 for DocVQA."""
    if domain_id == 0:
        return 1.0
    answers = item.get("answers") or []
    if not answers:
        return 1.0
    canon = item.get("canonical_answer") or answers[0]
    canon_norm = str(canon).strip().lower()
    match_count = sum(1 for a in answers if str(a).strip().lower() == canon_norm)
    return float(min(1.0, max(0.33, match_count / 3.0)))


class CrossVQADataCollator:
    def __init__(
        self,
        processor,
        model_type: str = "qwen2_vl",
        include_ocr: bool = False,
        max_ocr_tokens: int = 50,
        is_train: bool = True,
        use_soft_weights: bool = False,
        image_transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.processor = processor
        self.model_type = model_type.lower()
        self.include_ocr = include_ocr
        self.max_ocr_tokens = max_ocr_tokens
        self.is_train = is_train
        self.use_soft_weights = use_soft_weights
        self.image_transform = image_transform

    def _format_prompt(
        self,
        question: str,
        ocr_tokens: Optional[List[str]] = None,
        ocr_boxes: Optional[List[List[Any]]] = None,
    ) -> str:
        prompt_text = ""
        if self.include_ocr and ocr_tokens:
            tokens_slice = list(ocr_tokens)[: self.max_ocr_tokens]
            boxes_slice = (
                list(ocr_boxes)[: self.max_ocr_tokens]
                if ocr_boxes is not None
                else [[] for _ in tokens_slice]
            )
            if len(boxes_slice) < len(tokens_slice):
                boxes_slice = boxes_slice + [[] for _ in range(len(tokens_slice) - len(boxes_slice))]
            prompt_text += "Detected Text Context:\n"
            for tok, box in zip(tokens_slice, boxes_slice):
                prompt_text += f"- '{tok}' at {box}\n"
        prompt_text += f"Question: {question}\nAnswer:"
        return prompt_text

    def _apply_image_placeholder(self, prompt_text: str) -> str:
        """Prepends or wraps the prompt with model-specific visual placeholder tokens."""
        if "qwen2" in self.model_type:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(self.processor, "chat_template", None) or getattr(
                tokenizer, "chat_template", None
            )
            if chat_template and hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                try:
                    templated = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if isinstance(templated, str) and "<|image_pad|>" in templated:
                        return templated
                except Exception:
                    pass
            return f"<|vision_start|><|image_pad|><|vision_end|>\n{prompt_text}"

        image_token = getattr(self.processor, "image_token", None) or "<image>"
        return f"{image_token}\n{prompt_text}"

    def _set_padding_side(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right" if self.is_train else "left"
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self._set_padding_side()

        images = []
        formatted_prompts = []
        full_texts = []
        domain_ids = []
        sample_weights = []

        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token = (getattr(tokenizer, "eos_token", None) or "") if tokenizer is not None else ""

        for item in batch:
            domain_id = _infer_domain_id(item)
            domain_ids.append(domain_id)

            img = decode_image_payload(item["image"])
            if self.image_transform is not None and domain_id == 0:
                img = self.image_transform(img)
            images.append(img)

            ocr_tokens = item.get("ocr_tokens") or []
            ocr_boxes = item.get("ocr_boxes") or []
            base_prompt = self._format_prompt(item["question"], ocr_tokens, ocr_boxes)
            formatted_prompt = self._apply_image_placeholder(base_prompt)
            formatted_prompts.append(formatted_prompt)

            if self.is_train:
                target_answer = item.get("canonical_answer", "")
                if not target_answer and item.get("answers"):
                    target_answer = item["answers"][0]
                target_str = str(target_answer).strip()
                if eos_token and not target_str.endswith(eos_token):
                    full_text = f"{formatted_prompt} {target_str}{eos_token}"
                else:
                    full_text = f"{formatted_prompt} {target_str}"
                full_texts.append(full_text)

            if self.use_soft_weights:
                sample_weights.append(_compute_soft_weight(item, domain_id))

        if self.is_train:
            prompt_inputs = self.processor(
                text=formatted_prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = self.processor(
                text=full_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

            if "input_ids" in inputs:
                labels = inputs["input_ids"].clone()
                pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
                if pad_id is not None:
                    labels[labels == pad_id] = -100

                if "attention_mask" in prompt_inputs:
                    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
                else:
                    prompt_lens = torch.full(
                        (labels.shape[0],),
                        prompt_inputs["input_ids"].shape[1],
                        dtype=torch.long,
                    )

                for idx, p_len in enumerate(prompt_lens.tolist()):
                    mask_end = min(int(p_len), labels.shape[1])
                    labels[idx, :mask_end] = -100

                inputs["labels"] = labels
        else:
            inputs = self.processor(
                text=formatted_prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )

        inputs["domain_ids"] = torch.tensor(domain_ids, dtype=torch.long)
        if self.use_soft_weights:
            inputs["sample_weights"] = torch.tensor(sample_weights, dtype=torch.float32)

        return inputs
