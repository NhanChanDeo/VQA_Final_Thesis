"""
Unified model initialization and PEFT/LoRA configuration for CrossVQA backbones:
1. Qwen/Qwen2-VL-2B (Primary Workhorse)
2. HuggingFaceTB/SmolVLM-Base (Secondary 1: 2.2B Cross-Architecture)
3. HuggingFaceTB/SmolVLM-500M-Base (Secondary 2: Fast Pilot / Scale Ablation)
"""

import importlib.util
from typing import List, Optional
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLMModel


SUPPORTED_MODELS = {
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B",
    "smolvlm_base": "HuggingFaceTB/SmolVLM-Base",
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Base",
}

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _resolve_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _resolve_attn_implementation(attn_implementation: str = "auto") -> str:
    if attn_implementation != "auto":
        return attn_implementation
    has_flash = importlib.util.find_spec("flash_attn") is not None
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and has_flash:
        return "flash_attention_2"
    return "sdpa"


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
    attn_implementation: str = "auto",
    adapter_checkpoint: Optional[str] = None,
    architecture_mode: str = "standard_lora",
    target_modules: Optional[List[str]] = None,
    router_tau: float = 1.0,
):
    """
    Builds the backbone model and processor with memory safeguards, optional
    selective 4-bit QLoRA, sequential adapter resume, or M1 Dual-LoRA wrapping.
    """
    model_id = SUPPORTED_MODELS.get(model_name_or_key, model_name_or_key)
    compute_dtype = _resolve_compute_dtype()
    attn_impl = _resolve_attn_implementation(attn_implementation)
    lora_targets = target_modules or list(DEFAULT_TARGET_MODULES)

    # 1. Selective 4-bit NF4 Quantization (exempt visual encoder & connector/merger)
    quantization_config = None
    effective_4bit = bool(load_in_4bit and torch.cuda.is_available())
    if load_in_4bit and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; disabling 4-bit BitsAndBytes quantization for CPU execution.")

    if effective_4bit:
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

    # 3. Backbone Loading
    print(f"Loading backbone '{model_id}' (4-bit: {effective_4bit}, dtype: {compute_dtype}, attn: {attn_impl})...")
    load_kwargs = {
        "torch_dtype": compute_dtype,
        "attn_implementation": attn_impl,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if "Qwen2" in model_id:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoVLMModel.from_pretrained(model_id, **load_kwargs)

    # 4. Activation Checkpointing & k-bit preparation
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if effective_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 5. Adapter Architecture Selection
    mode = architecture_mode.lower()
    if mode == "none" or (not use_lora and adapter_checkpoint is None and mode == "standard_lora"):
        if mode == "none":
            for param in model.parameters():
                param.requires_grad = False
        return model, processor

    if mode == "dual_lora":
        from .adapters import CrossVQADualAdapterModel

        dual_model = CrossVQADualAdapterModel(
            backbone=model,
            target_modules=lora_targets,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            tau=router_tau,
        )
        if adapter_checkpoint is not None:
            dual_model.load_pretrained(adapter_checkpoint)
        return dual_model, processor

    if mode == "standard_lora":
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_checkpoint is not None:
            print(f"Loading existing LoRA adapter from '{adapter_checkpoint}' (is_trainable={use_lora})...")
            model = PeftModel.from_pretrained(model, adapter_checkpoint, is_trainable=use_lora)
        elif use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_targets,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        return model, processor

    raise ValueError(f"Unknown architecture_mode: {architecture_mode}")
