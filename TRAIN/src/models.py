"""
Unified model initialization and PEFT/LoRA configuration for CrossVQA backbones:
1. HuggingFaceTB/SmolVLM-500M-Base (Primary Full-Matrix Workhorse)
2. HuggingFaceTB/SmolVLM-Base (Secondary 1: Intra-Family 2.2B Scale Control)
3. Qwen/Qwen2-VL-2B (Secondary 2: Cross-Architecture & Saturation Reference)
"""

import importlib.util
import inspect
from typing import List, Optional, Sequence
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLMModel


SUPPORTED_MODELS = {
    "smolvlm_500m": "HuggingFaceTB/SmolVLM-500M-Base",
    "smolvlm_base": "HuggingFaceTB/SmolVLM-Base",
    "qwen2_vl_2b": "Qwen/Qwen2-VL-2B",
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

DEFAULT_EXCLUDE_VISION_PREFIXES = ["vision_model", "visual"]


def _resolve_compute_dtype() -> torch.dtype:
    """Requires Ampere+ (SM >= 8) for native bfloat16; uses float16 on T4 (SM 7.5)."""
    if torch.cuda.is_available():
        sm_major = torch.cuda.get_device_capability(0)[0]
        if sm_major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_attn_implementation(attn_implementation: str = "auto") -> str:
    if attn_implementation != "auto":
        return attn_implementation
    has_flash = importlib.util.find_spec("flash_attn") is not None
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and has_flash:
        return "flash_attention_2"
    return "sdpa"


def _build_lora_config(
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    exclude_modules: Sequence[str] = DEFAULT_EXCLUDE_VISION_PREFIXES,
):
    """Builds a PEFT LoraConfig that strictly excludes vision encoder projections."""
    from peft import LoraConfig

    sig_params = inspect.signature(LoraConfig.__init__).parameters
    kwargs = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    excl = list(exclude_modules) if exclude_modules else []
    if excl:
        if "exclude_modules" in sig_params:
            kwargs["exclude_modules"] = excl
        else:
            # Regex fallback for older PEFT versions lacking exclude_modules
            proj_alt = "|".join(target_modules)
            excl_alt = "|".join(excl)
            kwargs["target_modules"] = rf"^(?!.*(?:{excl_alt})).*\.(?:{proj_alt})$"
    return LoraConfig(**kwargs)


def _disable_incompatible_peft_torchao() -> None:
    """Prevents PEFT from crashing when an older system torchao (e.g. Colab 0.10.0) is installed."""
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available()
    except ImportError:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = lambda: False
        try:
            import peft.tuners.lora.torchao as peft_lora_torchao

            peft_lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass
        print("Warning: Incompatible torchao version detected for PEFT; disabled PEFT torchao dispatch.")
    except Exception:
        pass


def build_vlm_model_and_processor(
    model_name_or_key: str = "smolvlm_500m",
    load_in_4bit: bool = True,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    min_pixels: int = 256 * 28 * 28,   # ~200k pixels (Qwen2-VL)
    max_pixels: int = 1024 * 1024,     # ~1.05M pixels (Qwen2-VL)
    smolvlm_longest_edge: Optional[int] = 2048,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "auto",
    adapter_checkpoint: Optional[str] = None,
    architecture_mode: str = "standard_lora",
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[Sequence[str]] = None,
    router_tau: float = 1.0,
    align_adapter_rank: int = 0,
):
    """
    Builds the backbone model and processor with memory safeguards, optional
    selective 4-bit QLoRA (pure 16-bit by default for smolvlm_500m),
    vision-excluded LoRA parity, sequential adapter resume, or M1–M3 Dual-LoRA wrapping.
    """
    model_id = SUPPORTED_MODELS.get(model_name_or_key, model_name_or_key)
    compute_dtype = _resolve_compute_dtype()
    attn_impl = _resolve_attn_implementation(attn_implementation)
    lora_targets = target_modules or list(DEFAULT_TARGET_MODULES)
    lora_excludes = list(exclude_modules) if exclude_modules is not None else list(DEFAULT_EXCLUDE_VISION_PREFIXES)
    is_qwen2 = "Qwen2" in model_id
    is_small_smol = "500M" in model_id or model_name_or_key == "smolvlm_500m"

    # 1. Selective 4-bit NF4 Quantization (disabled by default for SmolVLM-500M-Base)
    quantization_config = None
    effective_4bit = bool(load_in_4bit and torch.cuda.is_available() and not is_small_smol)
    if load_in_4bit and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; disabling 4-bit BitsAndBytes quantization for CPU execution.")
    elif load_in_4bit and is_small_smol:
        print(f"Using pure {compute_dtype} LoRA for lightweight '{model_id}' (4-bit NF4 disabled).")

    if effective_4bit:
        skip_modules = (
            ["lm_head", "visual", "merger"]
            if is_qwen2
            else ["lm_head", "vision_model", "connector"]
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules,
        )

    # 2. Processor Initialization
    processor_kwargs = {}
    if is_qwen2:
        processor_kwargs = {"min_pixels": min_pixels, "max_pixels": max_pixels}
    elif smolvlm_longest_edge is not None:
        processor_kwargs = {"size": {"longest_edge": int(smolvlm_longest_edge)}}
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

    if is_qwen2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoVLMModel.from_pretrained(model_id, **load_kwargs)

    # 4. Activation Checkpointing & k-bit preparation
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        cfg = getattr(model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None and hasattr(text_cfg, "use_cache"):
                text_cfg.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
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
        _disable_incompatible_peft_torchao()
        from .adapters import CrossVQADualAdapterModel

        dual_model = CrossVQADualAdapterModel(
            backbone=model,
            target_modules=lora_targets,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            tau=router_tau,
            skip_vision_prefixes=tuple(lora_excludes),
            align_adapter_rank=align_adapter_rank,
        )
        if adapter_checkpoint is not None:
            dual_model.load_pretrained(adapter_checkpoint)
        return dual_model, processor

    if mode == "standard_lora":
        _disable_incompatible_peft_torchao()
        from peft import PeftModel, get_peft_model

        if adapter_checkpoint is not None:
            print(f"Loading existing LoRA adapter from '{adapter_checkpoint}' (is_trainable={use_lora})...")
            model = PeftModel.from_pretrained(model, adapter_checkpoint, is_trainable=use_lora)
        elif use_lora:
            lora_config = _build_lora_config(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_targets,
                exclude_modules=lora_excludes,
            )
            model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        return model, processor

    raise ValueError(f"Unknown architecture_mode: {architecture_mode}")
