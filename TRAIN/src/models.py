"""
Unified model initialization and PEFT/LoRA configuration for CrossVQA backbones:
- Qwen/Qwen2-VL-2B (Base)
- HuggingFaceTB/SmolVLM-Base (2.2B Base)
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    AutoModelForVision2Seq,
)

SUPPORTED_MODELS = {
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B",
    "smolvlm_base": "HuggingFaceTB/SmolVLM-Base",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Base",
}


def build_vlm_model_and_processor(
    model_name_or_key: str = "qwen2_vl_2b",
    load_in_4bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    min_pixels: int = 256 * 28 * 28,   # ~200k pixels
    max_pixels: int = 1024 * 1024,     # ~1.05M pixels (~1280 visual tokens)
    gradient_checkpointing: bool = True,
):
    """
    Builds the model and processor with memory safeguards and optional QLoRA/LoRA.
    """
    model_id = SUPPORTED_MODELS.get(model_name_or_key, model_name_or_key)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 1. Quantization Configuration
    quantization_config = None
    if load_in_4bit:
        # Exempt vision encoder from quantization to preserve character strokes
        skip_modules = ["visual", "merger"] if "Qwen2" in model_id else ["vision_model", "connector"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules,
        )

    # 2. Processor Initialization
    processor_kwargs = {}
    if "Qwen2" in model_id:
        processor_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    # 3. Model Loading
    attn_impl = "flash_attention_2" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else "sdpa"
    
    print(f"Loading backbone '{model_id}' (4-bit: {load_in_4bit}, attn: {attn_impl})...")
    if "Qwen2" in model_id:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            attn_implementation=attn_impl,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            attn_implementation=attn_impl,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    # 4. Activation Checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 5. PEFT / LoRA Configuration
    if use_lora:
        # Standard target modules across causal language backbones
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor
