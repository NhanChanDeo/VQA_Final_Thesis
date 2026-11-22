"""
Offline CPU unit and smoke tests for the CrossVQA Phase C training pipeline.
Validates collator masking, M1 Dual-LoRA + router, M2 MMD/InfoNCE + Proxy A-distance,
M3 curriculum homography transform, metrics, hardware profiling, CheckpointManager,
few-shot manifest integrity, and generated Colab notebooks without GPU or network.
"""

import json
import re
import sys
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TRAIN.src import (
    CheckpointManager,
    CrossDomainInfoNCELoss,
    CrossVQADataCollator,
    CrossVQADualAdapterModel,
    CrossVQATrainer,
    CurriculumPseudoDomainTransform,
    DualLoRAContext,
    DualLoRALinear,
    FeatureQueue,
    MultiKernelRBFMMDLoss,
    PairedDomainDataset,
    SUPPORTED_MODELS,
    compute_anls,
    compute_forgetting_rate,
    compute_proxy_a_distance,
    compute_rkhs_mmd,
    compute_transfer_gain,
    compute_vqa_accuracy,
    resolve_hardware_profile,
    save_run_card,
)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self._vocab = {"<pad>": 0, "<eos>": 2}

    def _encode_one(self, text: str):
        tokens = text.replace("<eos>", " <eos> ").split()
        ids = []
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab) + 10
            ids.append(self._vocab[t])
        return ids

    def batch_decode(self, seqs, skip_special_tokens=True):
        inv = {v: k for k, v in self._vocab.items()}
        out = []
        for row in seqs:
            words = []
            for tid in row.tolist():
                tok = inv.get(tid, "ans")
                if skip_special_tokens and tok in ("<pad>", "<eos>"):
                    continue
                words.append(tok)
            out.append(" ".join(words) if words else "ans")
        return out


class _DummyProcessor:
    def __init__(self, with_chat_template: bool = False) -> None:
        self.tokenizer = _DummyTokenizer()
        self.chat_template = "dummy_template" if with_chat_template else None
        self.image_token = "<image>"
        self.last_texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        txt = messages[0]["content"][1]["text"]
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{txt}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        self.last_texts.append(list(text))
        seqs = [self.tokenizer._encode_one(t) for t in text]
        max_len = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [self.tokenizer.pad_token_id] * (max_len - len(s))
            if self.tokenizer.padding_side == "right":
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
            else:
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "pixel_values": torch.randn(len(images), 3, 16, 16),
        }


class _TinyVLM(nn.Module):
    """Minimal synthetic VLM matching projector + LLM projection structure."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 256) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_dim})()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.connector = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4),
            nn.Flatten(2),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None):
        bsz = input_ids.shape[0]
        if pixel_values is None:
            pixel_values = torch.zeros(bsz, 3, 16, 16, device=input_ids.device)
        # Connector outputs [B, hidden_dim, num_patches] -> transpose to [B, num_patches, hidden_dim]
        vis = self.connector[0](pixel_values)
        vis = vis.flatten(2).transpose(1, 2)  # triggers hook? Let's call a dedicated Linear/Module
        x = self.emb(input_ids) + vis.mean(dim=1, keepdim=True)
        h = self.q_proj(x) + self.v_proj(x)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()

    def generate(self, input_ids, attention_mask=None, pixel_values=None, max_new_tokens=4, do_sample=False):
        _ = self.forward(input_ids=input_ids, pixel_values=pixel_values)
        gen = torch.full((input_ids.shape[0], 2), 11, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen], dim=1)


class _TinyVLMWithLinearConnector(nn.Module):
    """Uses an nn.Linear connector module that outputs [B, patches, D] directly."""

    def __init__(self, hidden_dim: int = 16, vocab_size: int = 256) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_dim})()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.connector = nn.Linear(12, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None):
        bsz = input_ids.shape[0]
        if pixel_values is None:
            pixel_values = torch.zeros(bsz, 3, 16, 16, device=input_ids.device)
        vis_in = pixel_values.mean(dim=-1).view(bsz, 4, 12)
        vis_emb = self.connector(vis_in)  # [B, 4, D]
        x = self.emb(input_ids) + vis_emb.mean(dim=1, keepdim=True)
        h = self.q_proj(x) + self.v_proj(x)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()

    def generate(self, input_ids, attention_mask=None, pixel_values=None, max_new_tokens=4, do_sample=False):
        _ = self.forward(input_ids=input_ids, pixel_values=pixel_values)
        gen = torch.full((input_ids.shape[0], 2), 11, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, gen], dim=1)


def test_supported_models_locked_order():
    assert list(SUPPORTED_MODELS.items()) == [
        ("qwen2_vl_2b", "Qwen/Qwen2-VL-2B"),
        ("smolvlm_base", "HuggingFaceTB/SmolVLM-Base"),
        ("smolvlm_500m", "HuggingFaceTB/SmolVLM-500M-Base"),
    ]


def test_collator_masking_placeholders_and_eval_left_padding():
    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    batch = [
        {
            "image": img,
            "question": "What is total?",
            "canonical_answer": "42",
            "domain_type": "document_text",
            "ocr_tokens": ["total", "42"],
            "ocr_boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
        },
        {
            "image": img,
            "question": "What brand is shown on the green sign?",
            "canonical_answer": "acme coffee",
            "domain_type": "scene_text",
            "answers": ["acme coffee", "acme coffee", "coffee"],
        },
    ]

    proc_q = _DummyProcessor(with_chat_template=True)
    col_train = CrossVQADataCollator(
        proc_q, model_type="qwen2_vl", include_ocr=True, is_train=True, use_soft_weights=True
    )
    out = col_train(batch)
    assert proc_q.tokenizer.padding_side == "right"
    assert "<|vision_start|><|image_pad|><|vision_end|>" in proc_q.last_texts[0][0]
    assert "<eos>" in proc_q.last_texts[1][0]
    assert out["domain_ids"].tolist() == [0, 1]
    assert pytest.approx(out["sample_weights"].tolist(), abs=1e-4) == [1.0, 2.0 / 3.0]

    # Prompt tokens must be -100 while answer + eos tokens remain >= 0
    prompt_only = proc_q(proc_q.last_texts[0], [img, img], padding=True, return_tensors="pt")
    p_len_0 = int(prompt_only["attention_mask"][0].sum().item())
    assert (out["labels"][0, :p_len_0] == -100).all()
    assert (out["labels"][0, p_len_0:] != -100).any()

    # Eval mode must enforce left padding and SmolVLM <image> tag
    proc_s = _DummyProcessor(with_chat_template=False)
    col_eval = CrossVQADataCollator(proc_s, model_type="smolvlm", include_ocr=False, is_train=False)
    _ = col_eval(batch)
    assert proc_s.tokenizer.padding_side == "left"
    assert proc_s.last_texts[0][0].startswith("<image>\nQuestion:")


def test_dual_lora_linear_and_wrapper_stage_freezing_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    ctx = DualLoRAContext()
    dual = DualLoRALinear(base, ctx, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    # Set non-zero weights so branches differ
    with torch.no_grad():
        dual.lora_B_doc.weight.fill_(0.2)
        dual.lora_B_scene.weight.fill_(-0.3)

    x = torch.randn(2, 5, 8)
    ctx.set(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    y = dual(x)

    # Analytical verification
    expected_0 = base(x[0:1]) + dual.scaling * dual.lora_B_doc(dual.lora_A_doc(x[0:1]))
    expected_1 = base(x[1:2]) + dual.scaling * dual.lora_B_scene(dual.lora_A_scene(x[1:2]))
    assert torch.allclose(y[0:1], expected_0, atol=1e-5)
    assert torch.allclose(y[1:2], expected_1, atol=1e-5)

    # Full wrapper test: stage freezing + save/load roundtrip
    backbone = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    backbone2 = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    backbone2.load_state_dict(backbone.state_dict())
    wrapped = CrossVQADualAdapterModel(
        backbone=backbone,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        projector_module_name="connector",
    )
    wrapped.set_trainable_stage("scene_only_frozen_doc")
    for layer in wrapped.dual_layers:
        assert not layer.lora_A_doc.weight.requires_grad
        assert not layer.lora_B_doc.weight.requires_grad
        assert layer.lora_A_scene.weight.requires_grad
        assert layer.lora_B_scene.weight.requires_grad

    ids = torch.randint(0, 100, (2, 6))
    pix = torch.randn(2, 3, 16, 16)
    labels = ids.clone()
    labels[:, :3] = -100
    out = wrapped(input_ids=ids, pixel_values=pix, labels=labels)
    out.loss.backward()
    for layer in wrapped.dual_layers:
        assert layer.lora_A_doc.weight.grad is None
        assert layer.lora_B_doc.weight.grad is None
    assert wrapped.last_routing_outputs["pooled_vis"].shape == (2, 16)

    save_dir = tmp_path / "dual_ckpt"
    wrapped.save_pretrained(str(save_dir))
    assert (save_dir / "dual_lora_weights.safetensors").exists()

    wrapped2 = CrossVQADualAdapterModel(
        backbone=backbone2,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        projector_module_name="connector",
    )
    wrapped2.load_pretrained(str(save_dir))
    with torch.no_grad():
        wrapped.eval()
        wrapped2.eval()
        o1 = wrapped(input_ids=ids, pixel_values=pix).logits
        o2 = wrapped2(input_ids=ids, pixel_values=pix).logits
    assert torch.allclose(o1, o2, atol=1e-6)


def test_losses_queue_and_proxy_a_distance():
    torch.manual_seed(7)
    q = FeatureQueue(max_size=5)
    q.enqueue(torch.ones(3, 4))
    q.enqueue(torch.ones(4, 4) * 2.0)
    assert len(q) == 5

    mmd_loss = MultiKernelRBFMMDLoss(queue_size=16)
    vis = torch.randn(4, 16, requires_grad=True)
    dom = torch.tensor([0, 0, 1, 1])
    loss_mmd = mmd_loss(vis, dom)
    assert torch.isfinite(loss_mmd) and loss_mmd.item() >= 0.0
    loss_mmd.backward()
    assert vis.grad is not None and vis.grad.norm().item() > 0.0

    # Single-domain batch fallback (no NaN)
    vis_single = torch.randn(2, 16, requires_grad=True)
    loss_single = MultiKernelRBFMMDLoss(use_queue=False)(vis_single, torch.tensor([0, 0]))
    assert loss_single.item() == 0.0

    infonce = CrossDomainInfoNCELoss(queue_size=16)
    vis2 = torch.randn(4, 16, requires_grad=True)
    loss_nce = infonce(vis2, dom)
    assert torch.isfinite(loss_nce)
    loss_nce.backward()
    assert vis2.grad is not None

    # Proxy A-distance: linearly separable vs identical
    rng = np.random.default_rng(42)
    doc_sep = rng.normal(loc=3.0, scale=0.3, size=(60, 8))
    scene_sep = rng.normal(loc=-3.0, scale=0.3, size=(60, 8))
    res_sep = compute_proxy_a_distance(doc_sep, scene_sep, n_splits=5, seed=42)
    assert res_sep["proxy_a_distance"] > 1.8
    assert compute_rkhs_mmd(doc_sep, scene_sep) > 0.1

    same_a = rng.normal(loc=0.0, scale=1.0, size=(80, 8))
    same_b = rng.normal(loc=0.0, scale=1.0, size=(80, 8))
    res_same = compute_proxy_a_distance(same_a, same_b, n_splits=5, seed=42)
    assert res_same["proxy_a_distance"] < 0.5


def test_curriculum_transform_determinism_and_rgb():
    img = Image.new("RGB", (96, 64), color=(180, 190, 200))
    for sev in (0.0, 0.5, 1.0):
        t1 = CurriculumPseudoDomainTransform(severity=sev, seed=123)
        t2 = CurriculumPseudoDomainTransform(severity=sev, seed=123)
        out1 = t1(img)
        out2 = t2(img)
        assert out1.mode == "RGB" and out1.size == img.size
        assert np.array_equal(np.asarray(out1), np.asarray(out2))
    assert CurriculumPseudoDomainTransform.linear_schedule(0, 11, 0.2, 1.0) == pytest.approx(0.2)
    assert CurriculumPseudoDomainTransform.linear_schedule(10, 11, 0.2, 1.0) == pytest.approx(1.0)


def test_metrics_hardware_checkpoint_and_trainer_e2e(tmp_path):
    assert compute_anls("Acme Inc.", ["acme inc."]) == pytest.approx(1.0)
    assert compute_vqa_accuracy("red", ["red", "red", "blue", "green"]) == pytest.approx(2.0 / 3.0)
    assert compute_transfer_gain(0.62, 0.58) == pytest.approx(0.04)
    assert compute_forgetting_rate(0.75, 0.72) == pytest.approx(0.03)

    hw_cpu = resolve_hardware_profile("auto")
    assert hw_cpu["name"] == "cpu"
    hw_l4 = resolve_hardware_profile("l4_24gb")
    assert hw_l4["dtype"] == "bf16" and hw_l4["per_device_train_batch_size"] == 4
    hw_t4 = resolve_hardware_profile("t4_16gb")
    assert hw_t4["dtype"] == "fp16" and hw_t4["per_device_train_batch_size"] == 2

    # End-to-end trainer + CheckpointManager with Drive backup mirror (no network)
    backbone = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    wrapped = CrossVQADualAdapterModel(
        backbone=backbone,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        projector_module_name="connector",
    )
    wrapped.set_trainable_stage("scene_only_frozen_doc")
    proc = _DummyProcessor(with_chat_template=False)

    img = Image.new("RGB", (16, 16), color=(120, 120, 120))
    doc_samples = [
        {"image": img, "question": "doc q1?", "canonical_answer": "yes", "domain_type": "document_text"},
        {"image": img, "question": "doc q2?", "canonical_answer": "no", "domain_type": "document_text"},
    ]
    tv_samples = [
        {"image": img, "question": "scene q1?", "canonical_answer": "blue", "domain_type": "scene_text"},
        {"image": img, "question": "scene q2?", "canonical_answer": "red", "domain_type": "scene_text"},
    ]
    paired = PairedDomainDataset(doc_samples, tv_samples, max_pairs=2, seed=42)

    drive_dir = tmp_path / "gdrive_backup"
    ckpt_mgr = CheckpointManager(
        local_ckpt_dir=str(tmp_path / "local_ckpts"),
        hf_repo_id=None,
        hf_token=None,
        gdrive_backup_dir=str(drive_dir),
    )
    trainer = CrossVQATrainer(
        model=wrapped,
        processor=proc,
        run_dir=str(tmp_path / "run"),
        model_type="smolvlm",
        device="cpu",
        lr=1e-3,
        gradient_accumulation_steps=1,
        optim_name="adamw",
        align_loss_fn=MultiKernelRBFMMDLoss(queue_size=8),
        lambda_align=0.1,
        lambda_gate=0.1,
        checkpoint_manager=ckpt_mgr,
    )
    summary = trainer.fit(
        paired,
        epochs=1,
        batch_size=2,
        max_steps_per_epoch=2,
        eval_fn=lambda: {"selection_score": 0.5},
    )
    assert summary["opt_steps"] == 2
    assert (tmp_path / "run" / "train_log.jsonl").exists()
    assert (drive_dir / "best" / "dual_lora_weights.safetensors").exists()

    # Verify Drive fallback restore when local directory is cleared
    local_best = tmp_path / "local_ckpts" / "best"
    for f in local_best.iterdir():
        f.unlink()
    local_best.rmdir()
    restored = ckpt_mgr.restore_checkpoint("best")
    assert restored is not None and (restored / "dual_lora_weights.safetensors").exists()

    card_path = save_run_card(str(tmp_path / "run"), {"experiment": {"id": "smoke"}}, {"val": 0.5})
    assert card_path.exists() and (tmp_path / "run" / "val_metrics.json").exists()


def test_few_shot_manifest_integrity():
    manifest_path = REPO_ROOT / "DATA_PIPELINE/ALT_THREE_SPLITS/splits/textvqa_few_shot_manifest.parquet"
    assert manifest_path.exists(), "Run generate_few_shot_splits.py first"
    df = pl.read_parquet(manifest_path)
    assert df.height == 32762
    assert df["image_id"].n_unique() == 20807

    # Nested inclusion and 0 split image_ids across boundaries
    for small, large in [("in_1pct", "in_5pct"), ("in_5pct", "in_10pct"), ("in_10pct", "in_25pct"), ("in_25pct", "in_100pct")]:
        assert df.filter(pl.col(small) & (~pl.col(large))).height == 0

    for col in ("in_1pct", "in_5pct", "in_10pct", "in_25pct"):
        in_imgs = set(df.filter(pl.col(col))["image_id"].to_list())
        out_imgs = set(df.filter(~pl.col(col))["image_id"].to_list())
        assert len(in_imgs & out_imgs) == 0


def test_generated_colab_notebooks_valid_and_secret_free():
    nb_dir = REPO_ROOT / "TRAIN/ALT_THREE_SPLITS/notebooks"
    expected = [
        "00_Z0_ZeroShot_and_DomainGap.ipynb",
        "01_B1_B2_SingleDomain_LoRA.ipynb",
        "02_B3_Joint_MultiDomain_LoRA.ipynb",
        "03_B4_B5_Sequential_Transfer.ipynb",
        "04_M1_M3_CrossVQA_Proposed.ipynb",
        "05_FewShot_Scaling_Matrix.ipynb",
    ]
    for name in expected:
        p = nb_dir / name
        assert p.exists(), f"Missing notebook: {p}"
        raw = p.read_text(encoding="utf-8")
        assert "ghp_" not in raw and "github_pat_" not in raw
        assert not re.search(r"hf_[A-Za-z0-9]{15,}", raw)
        assert "getpass" in raw
        nb = json.loads(raw)
        assert nb["nbformat"] == 4 and len(nb["cells"]) >= 5
