"""
Proposed CrossVQA M1 Architecture:
Dual Domain-Specific LoRA Adapters (A_doc, A_scene) with Visual Soft Gating Router.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


class DualLoRAContext:
    """Shared per-forward routing context holding [B, 2] gate weights (doc, scene)."""

    def __init__(self) -> None:
        self.gate_weights: Optional[torch.Tensor] = None

    def set(self, gate_weights: Optional[torch.Tensor]) -> None:
        self.gate_weights = gate_weights

    def clear(self) -> None:
        self.gate_weights = None


def _expand_gate_column(col: torch.Tensor, target_batch: int) -> torch.Tensor:
    """Aligns [B, 1] gate weight with 2D [B, D] or flattened [B*K, D] activations."""
    if col.shape[0] == target_batch:
        return col
    if target_batch % col.shape[0] == 0:
        repeat_factor = target_batch // col.shape[0]
        return col.repeat_interleave(repeat_factor, dim=0)
    raise ValueError(
        f"Gate batch dimension ({col.shape[0]}) is incompatible with activation batch ({target_batch})."
    )


class DualLoRALinear(nn.Module):
    """
    Wraps a frozen linear projection with parallel Document and Scene LoRA branches
    modulated per-sample by soft gating weights from DualLoRAContext.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        context: DualLoRAContext,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.context = context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(lora_r)

        in_features = int(getattr(base_layer, "in_features"))
        out_features = int(getattr(base_layer, "out_features"))

        ref_weight = getattr(base_layer, "weight", None)
        device = ref_weight.device if ref_weight is not None else torch.device("cpu")
        dtype = (
            ref_weight.dtype
            if (ref_weight is not None and ref_weight.is_floating_point())
            else torch.float32
        )

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self.lora_A_doc = nn.Linear(in_features, lora_r, bias=False, device=device, dtype=dtype)
        self.lora_B_doc = nn.Linear(lora_r, out_features, bias=False, device=device, dtype=dtype)
        self.lora_A_scene = nn.Linear(in_features, lora_r, bias=False, device=device, dtype=dtype)
        self.lora_B_scene = nn.Linear(lora_r, out_features, bias=False, device=device, dtype=dtype)

        self.reset_lora_parameters()

        # Freeze base projection parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self) -> None:
        for layer_a, layer_b in (
            (self.lora_A_doc, self.lora_B_doc),
            (self.lora_A_scene, self.lora_B_scene),
        ):
            nn.init.kaiming_uniform_(layer_a.weight, a=math.sqrt(5))
            nn.init.zeros_(layer_b.weight)

    def set_trainable_branch(self, branch: Literal["doc", "scene", "both", "none"]) -> None:
        train_doc = branch in ("doc", "both")
        train_scene = branch in ("scene", "both")
        for p in (*self.lora_A_doc.parameters(), *self.lora_B_doc.parameters()):
            p.requires_grad = train_doc
        for p in (*self.lora_A_scene.parameters(), *self.lora_B_scene.parameters()):
            p.requires_grad = train_scene

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        gates = self.context.gate_weights
        if gates is None:
            w_doc = x.new_full((x.shape[0], 1), 0.5)
            w_scene = x.new_full((x.shape[0], 1), 0.5)
        else:
            w_doc = _expand_gate_column(gates[:, 0:1], x.shape[0]).to(device=x.device, dtype=base_out.dtype)
            w_scene = _expand_gate_column(gates[:, 1:2], x.shape[0]).to(device=x.device, dtype=base_out.dtype)

        if x.dim() == 3:
            w_doc = w_doc.unsqueeze(1)
            w_scene = w_scene.unsqueeze(1)
        elif x.dim() > 3:
            extra_dims = (1,) * (x.dim() - 2)
            w_doc = w_doc.view(w_doc.shape[0], *extra_dims, 1)
            w_scene = w_scene.view(w_scene.shape[0], *extra_dims, 1)

        x_in = self.lora_dropout(x).to(dtype=self.lora_A_doc.weight.dtype)
        delta_doc = self.lora_B_doc(self.lora_A_doc(x_in)).to(dtype=base_out.dtype) * self.scaling
        delta_scene = self.lora_B_scene(self.lora_A_scene(x_in)).to(dtype=base_out.dtype) * self.scaling

        return base_out + w_doc * delta_doc + w_scene * delta_scene


class VisualSoftDomainRouter(nn.Module):
    """Lightweight 2-layer MLP predicting soft domain probabilities (doc, scene)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, tau: float = 1.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(max(tau, 1e-4))
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, pooled_vis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(pooled_vis)
        weights = F.softmax(logits / self.tau, dim=-1)
        return weights, logits


def _infer_router_in_dim(backbone: nn.Module) -> Optional[int]:
    config = getattr(backbone, "config", None)
    if config is None:
        return None
    for attr in ("hidden_size", "d_model"):
        if hasattr(config, attr) and isinstance(getattr(config, attr), int):
            return int(getattr(config, attr))
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return int(text_cfg.hidden_size)
    return None


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _resolve_projector_module(
    backbone: nn.Module,
    projector_module_name: Optional[str] = None,
) -> Tuple[str, nn.Module]:
    named_modules = dict(backbone.named_modules())
    candidates: List[str] = []
    if projector_module_name:
        candidates.append(projector_module_name)
    candidates.extend(
        [
            "visual.merger",
            "model.visual.merger",
            "model.connector",
            "connector",
            "merger",
            "multi_modal_projector",
        ]
    )
    for cand in candidates:
        if cand in named_modules:
            return cand, named_modules[cand]
    # Fallback suffix search
    for name, mod in named_modules.items():
        if name.endswith(("visual.merger", ".connector", "multi_modal_projector")):
            return name, mod
    raise ValueError(
        "Could not locate visual projector/merger module on backbone; pass projector_module_name explicitly."
    )


def _extract_idefics3_tile_counts(
    batch_size: int,
    total_tiles: int,
    pixel_attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
) -> Optional[List[int]]:
    """Computes exact non-pad tile count per sample for Idefics3/SmolVLM variable-tile batches."""
    if pixel_attention_mask is not None and pixel_attention_mask.dim() == 4 and pixel_attention_mask.shape[0] == batch_size:
        counts = [max(1, int(c)) for c in pixel_attention_mask.any(dim=(-2, -1)).sum(dim=1).tolist()]
        if sum(counts) == total_tiles:
            return counts
    if pixel_values is not None and pixel_values.dim() == 5 and pixel_values.shape[0] == batch_size:
        counts = [max(1, int(c)) for c in (pixel_values.abs().sum(dim=(-3, -2, -1)) > 0).sum(dim=1).tolist()]
        if sum(counts) == total_tiles:
            return counts
    return None


def _pool_visual_tokens(
    vis_out: Any,
    batch_size: int,
    image_grid_thw: Optional[torch.Tensor] = None,
    pixel_attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(vis_out, (tuple, list)):
        vis_out = vis_out[0]
    if vis_out.dim() == 3:
        if vis_out.shape[0] == batch_size:
            return vis_out.mean(dim=1)
        tile_counts = _extract_idefics3_tile_counts(
            batch_size=batch_size,
            total_tiles=int(vis_out.shape[0]),
            pixel_attention_mask=pixel_attention_mask,
            pixel_values=pixel_values,
        )
        if tile_counts is not None:
            chunks = torch.split(vis_out, tile_counts, dim=0)
            return torch.stack([c.mean(dim=(0, 1)) for c in chunks], dim=0)
        if vis_out.shape[0] % batch_size == 0:
            return vis_out.view(batch_size, -1, vis_out.shape[-1]).mean(dim=1)
    if vis_out.dim() == 2:
        if vis_out.shape[0] == batch_size:
            return vis_out
        if image_grid_thw is not None and image_grid_thw.shape[0] == batch_size:
            # Qwen2-VL merges 2x2 patches in spatial_merge_size=2
            counts = ((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) // 4).tolist()
            if sum(counts) == vis_out.shape[0] and all(c > 0 for c in counts):
                chunks = torch.split(vis_out, counts, dim=0)
                return torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        if vis_out.shape[0] >= batch_size:
            chunks = torch.tensor_split(vis_out, batch_size, dim=0)
            return torch.stack([c.mean(dim=0) for c in chunks], dim=0)
    raise ValueError(f"Unsupported visual projector output shape {tuple(vis_out.shape)} for batch_size={batch_size}.")


class ProjectorAlignmentAdapter(nn.Module):
    """Zero-initialized residual bottleneck on projector tokens so M2/M3 alignment losses have non-zero gradients."""

    def __init__(self, hidden_dim: int, rank: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.down = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(self.rank, self.hidden_dim, bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, vis_tokens: torch.Tensor) -> torch.Tensor:
        orig_dtype = vis_tokens.dtype
        p_dtype = self.up.weight.dtype
        delta = self.up(self.act(self.down(self.norm(vis_tokens.to(dtype=p_dtype)))))
        return vis_tokens + delta.to(dtype=orig_dtype)


class CrossVQADualAdapterModel(nn.Module):
    """
    Wraps a frozen VLM backbone with DualLoRALinear layers, a VisualSoftDomainRouter,
    and an optional ProjectorAlignmentAdapter (M2/M3) hooked onto the visual projector/merger output.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_modules: Optional[Sequence[str]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        tau: float = 1.0,
        router_hidden_dim: int = 256,
        router_in_dim: Optional[int] = None,
        projector_module_name: Optional[str] = None,
        skip_vision_prefixes: Sequence[str] = ("visual", "vision_model"),
        align_adapter_rank: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.target_modules = list(
            target_modules
            or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.tau = float(tau)
        self.router_hidden_dim = int(router_hidden_dim)

        # Freeze entire base backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.context = DualLoRAContext()
        self.dual_layers: List[DualLoRALinear] = []

        # Inject DualLoRALinear into matching LLM projections
        targets_to_replace: List[Tuple[str, nn.Module]] = []
        for full_name, mod in self.backbone.named_modules():
            if not full_name:
                continue
            leaf = full_name.split(".")[-1]
            if leaf not in self.target_modules:
                continue
            parts = full_name.split(".")
            if any(p in skip_vision_prefixes for p in parts):
                continue
            if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
                targets_to_replace.append((full_name, mod))

        if not targets_to_replace:
            raise ValueError(f"No linear target modules matched {self.target_modules} on backbone.")

        for full_name, mod in targets_to_replace:
            parent, attr = _get_parent_and_attr(self.backbone, full_name)
            wrapped = DualLoRALinear(
                base_layer=mod,
                context=self.context,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )
            setattr(parent, attr, wrapped)
            self.dual_layers.append(wrapped)

        in_dim = router_in_dim or _infer_router_in_dim(self.backbone)
        if in_dim is None:
            raise ValueError("Unable to infer router_in_dim from backbone config; specify router_in_dim.")

        ref_param = next(self.backbone.parameters(), None)
        router_device = ref_param.device if ref_param is not None else torch.device("cpu")
        router_dtype = (
            ref_param.dtype
            if (ref_param is not None and ref_param.is_floating_point())
            else torch.float32
        )
        self.router = VisualSoftDomainRouter(
            in_dim=in_dim,
            hidden_dim=self.router_hidden_dim,
            tau=self.tau,
        ).to(device=router_device, dtype=router_dtype)

        self.align_adapter_rank = int(align_adapter_rank)
        self.align_adapter: Optional[ProjectorAlignmentAdapter] = (
            ProjectorAlignmentAdapter(hidden_dim=in_dim, rank=self.align_adapter_rank).to(
                device=router_device, dtype=router_dtype
            )
            if self.align_adapter_rank > 0
            else None
        )

        self.projector_name, projector_mod = _resolve_projector_module(self.backbone, projector_module_name)
        self._current_batch_size: Optional[int] = None
        self._current_image_grid_thw: Optional[torch.Tensor] = None
        self._current_pixel_attention_mask: Optional[torch.Tensor] = None
        self._current_pixel_values: Optional[torch.Tensor] = None
        self._forced_domain_ids: Optional[torch.Tensor] = None
        self.last_routing_outputs: Dict[str, torch.Tensor] = {}

        self._hook_handle = projector_mod.register_forward_hook(self._projector_hook)

    def _projector_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if self._current_batch_size is None:
            return None
        vis_tensor = output[0] if isinstance(output, (tuple, list)) else output
        adapted = self.align_adapter(vis_tensor) if self.align_adapter is not None else vis_tensor

        pooled_vis = _pool_visual_tokens(
            adapted,
            batch_size=self._current_batch_size,
            image_grid_thw=self._current_image_grid_thw,
            pixel_attention_mask=self._current_pixel_attention_mask,
            pixel_values=self._current_pixel_values,
        )
        router_param = next(self.router.parameters())
        gate_weights, gate_logits = self.router(
            pooled_vis.to(device=router_param.device, dtype=router_param.dtype)
        )
        if self._forced_domain_ids is not None:
            forced = F.one_hot(self._forced_domain_ids.to(gate_weights.device), num_classes=2).to(
                dtype=gate_weights.dtype
            )
            self.context.set(forced)
        else:
            self.context.set(gate_weights)

        self.last_routing_outputs = {
            "pooled_vis": pooled_vis,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
        }
        if self.align_adapter is not None:
            if isinstance(output, tuple):
                return (adapted, *output[1:])
            if isinstance(output, list):
                return [adapted, *output[1:]]
            return adapted
        return None

    def set_trainable_stage(self, stage: Literal["doc_only", "scene_only_frozen_doc", "joint"]) -> None:
        if stage == "doc_only":
            branch = "doc"
        elif stage == "scene_only_frozen_doc":
            branch = "scene"
        elif stage == "joint":
            branch = "both"
        else:
            raise ValueError(f"Unknown training stage: {stage}")

        for layer in self.dual_layers:
            layer.set_trainable_branch(branch)
        for p in self.router.parameters():
            p.requires_grad = True
        if self.align_adapter is not None:
            train_align = stage in ("scene_only_frozen_doc", "joint")
            for p in self.align_adapter.parameters():
                p.requires_grad = train_align

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        forced_domain_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_ids is not None:
            self._current_batch_size = int(input_ids.shape[0])
        elif "pixel_values" in kwargs and isinstance(kwargs["pixel_values"], torch.Tensor):
            self._current_batch_size = int(kwargs["pixel_values"].shape[0])
        self._current_image_grid_thw = kwargs.get("image_grid_thw", None)
        self._current_pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
        self._current_pixel_values = kwargs.get("pixel_values", None)
        self._forced_domain_ids = forced_domain_ids
        self.last_routing_outputs = {}
        try:
            return self.backbone(input_ids=input_ids, **kwargs)
        finally:
            self._current_batch_size = None
            self._current_image_grid_thw = None
            self._current_pixel_attention_mask = None
            self._current_pixel_values = None
            self._forced_domain_ids = None

    def generate(self, input_ids: Optional[torch.Tensor] = None, **kwargs):
        if input_ids is not None:
            self._current_batch_size = int(input_ids.shape[0])
        self._current_image_grid_thw = kwargs.get("image_grid_thw", None)
        self._current_pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
        self._current_pixel_values = kwargs.get("pixel_values", None)
        self._forced_domain_ids = None
        try:
            return self.backbone.generate(input_ids=input_ids, **kwargs)
        finally:
            self._current_batch_size = None
            self._current_image_grid_thw = None
            self._current_pixel_attention_mask = None
            self._current_pixel_values = None
            self.context.clear()

    def _adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        trainable_prefixes = ("dual_layers.", "router.", "align_adapter.")
        state = {}
        for k, v in self.state_dict().items():
            if k.startswith(trainable_prefixes) or ".lora_A_" in k or ".lora_B_" in k:
                if "base_layer." not in k:
                    state[k] = v.detach().cpu().contiguous()
        return state

    def save_pretrained(self, save_directory: str) -> None:
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_safetensors(self._adapter_state_dict(), str(out_dir / "dual_lora_weights.safetensors"))
        cfg = {
            "architecture": "CrossVQADualAdapterModel",
            "target_modules": self.target_modules,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "tau": self.tau,
            "router_in_dim": self.router.in_dim,
            "router_hidden_dim": self.router_hidden_dim,
            "align_adapter_rank": self.align_adapter_rank,
            "projector_name": self.projector_name,
        }
        (out_dir / "crossvqa_adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def load_pretrained(self, load_directory: str) -> None:
        weights_path = Path(load_directory) / "dual_lora_weights.safetensors"
        loaded = load_safetensors(str(weights_path))
        # Allow loading Stage-1 (align_adapter_rank=0) checkpoints into M2/M3 (align_adapter_rank>0)
        expected_core = {k for k in self._adapter_state_dict().keys() if not k.startswith("align_adapter.")}
        missing = sorted(expected_core - set(loaded.keys()))
        if missing:
            raise KeyError(f"Missing keys in Dual-LoRA checkpoint '{weights_path}': {missing[:5]}")
        self.load_state_dict(loaded, strict=False)

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = (100.0 * trainable / total) if total > 0 else 0.0
        print(f"Dual-LoRA trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}%")

    @property
    def config(self):
        return getattr(self.backbone, "config", None)
