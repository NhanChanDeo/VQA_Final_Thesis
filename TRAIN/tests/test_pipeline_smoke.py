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
    ProjectorAlignmentAdapter,
    SUPPORTED_MODELS,
    compute_anls,
    compute_forgetting_rate,
    compute_proxy_a_distance,
    compute_rkhs_mmd,
    compute_transfer_gain,
    compute_vqa_accuracy,
    resolve_hardware_profile,
    run_evaluation,
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
        ("smolvlm_500m", "HuggingFaceTB/SmolVLM-500M-Base"),
        ("smolvlm_base", "HuggingFaceTB/SmolVLM-Base"),
        ("qwen2_vl_2b", "Qwen/Qwen2-VL-2B"),
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

    # Qwen2-VL fallback when chat_template exists but omits <|image_pad|>
    class _BrokenTemplateProc(_DummyProcessor):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<|im_start|>user\nQuestion only without image pad<|im_end|>\n"

    proc_broken = _BrokenTemplateProc(with_chat_template=True)
    col_broken = CrossVQADataCollator(proc_broken, model_type="qwen2_vl", include_ocr=False, is_train=False)
    _ = col_broken(batch)
    assert "<|vision_start|><|image_pad|><|vision_end|>" in proc_broken.last_texts[0][0]


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
    hw_a100 = resolve_hardware_profile("a100_40gb")
    assert hw_a100["dtype"] == "bf16" and hw_a100["per_device_train_batch_size"] == 8 and hw_a100["eval_batch_size"] == 16
    hw_g4 = resolve_hardware_profile("g4_96gb")
    assert hw_g4["dtype"] == "bf16" and hw_g4["load_in_4bit"] is False and hw_g4["per_device_train_batch_size"] == 16
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


def test_z0_tier1_resolution_bucketing_order_preservation_and_early_exit():
    model = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    proc = _DummyProcessor(with_chat_template=False)

    # Deliberately unordered resolutions: 64x64, 16x16, 48x48, 32x32
    eval_samples = [
        {"sample_id": "s0", "domain_type": "document_text", "image": Image.new("RGB", (64, 64)), "question": "q0?", "answers": ["ans"]},
        {"sample_id": "s1", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q1?", "answers": ["ans"]},
        {"sample_id": "s2", "domain_type": "document_text", "image": Image.new("RGB", (48, 48)), "question": "q2?", "answers": ["ans"]},
        {"sample_id": "s3", "domain_type": "scene_text", "image": Image.new("RGB", (32, 32)), "question": "q3?", "answers": ["ans"]},
    ]

    observed_batches = []

    def _on_batch(meta_list, _model_inputs):
        observed_batches.append([m["sample_id"] for m in meta_list])

    metrics, details = run_evaluation(
        model=model,
        processor=proc,
        eval_dataset=eval_samples,
        device="cpu",
        max_new_tokens=2,
        eval_batch_size=2,
        return_details=True,
        bucket_by_resolution=True,
        on_batch_evaluated=_on_batch,
    )
    assert "anls" in metrics and "vqa_accuracy" in metrics
    # Bucketing groups small images (s1: 16x16, s3: 32x32) and large images (s2: 48x48, s0: 64x64)
    assert observed_batches == [["s1", "s3"], ["s2", "s0"]]
    # Returned details must be restored to original order
    assert [d["sample_id"] for d in details] == ["s0", "s1", "s2", "s3"]

    # Verify extract_domain_embeddings exits after projector hook without executing LLM q_proj
    sys.path.insert(0, str(REPO_ROOT / "TRAIN/ALT_THREE_SPLITS/scripts"))
    from run_z0_domain_gap import extract_domain_embeddings

    llm_calls = {"count": 0}
    h = model.q_proj.register_forward_hook(lambda *_args: llm_calls.__setitem__("count", llm_calls["count"] + 1))
    try:
        embs = extract_domain_embeddings(
            model=model,
            processor=proc,
            model_type="smolvlm",
            dataset_name="docvqa",
            split="validation",
            max_samples=3,
            batch_size=2,
            device="cpu",
            eval_dataset=eval_samples,
        )
    finally:
        h.remove()

    assert llm_calls["count"] == 0
    assert embs.shape == (3, 16)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Verify CPU compute_rkhs_mmd still produces finite non-negative output
    assert compute_rkhs_mmd(embs, embs, device="cpu") == pytest.approx(0.0, abs=1e-6)

    # Verify 00_Z0 notebook contains RUN_MODE and updated primary SmolVLM-500M defaults
    z0_obj = json.loads((REPO_ROOT / "TRAIN/ALT_THREE_SPLITS/notebooks/00_Z0_ZeroShot_and_DomainGap.ipynb").read_text(encoding="utf-8"))
    z0_src = "\n".join("".join(c.get("source", [])) for c in z0_obj["cells"])
    assert 'RUN_MODE = "val_and_domain_gap"' in z0_src
    assert 'MODEL_KEY = "smolvlm_500m"' in z0_src
    assert "DOMAIN_GAP_SAMPLES = 794" in z0_src
    assert "MAX_EVAL_SAMPLES = None" in z0_src
    assert "--domain_gap_only" in z0_src
    assert "domain_gap_metrics.json" in z0_src


def test_idefics3_variable_tile_pooling():
    from TRAIN.src.adapters import _pool_visual_tokens

    # Batch of 2 samples with unequal tiles: sample 0 has 3 valid tiles, sample 1 has 1 valid tile
    t0 = torch.full((3, 4, 16), 2.0)
    t1 = torch.full((1, 4, 16), 5.0)
    vis_out = torch.cat([t0, t1], dim=0)  # [4, 4, 16]

    # Case A: 4D pixel_attention_mask [B=2, max_tiles=3, H=8, W=8]
    mask = torch.zeros((2, 3, 8, 8), dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :1] = True
    pooled_mask = _pool_visual_tokens(vis_out, batch_size=2, pixel_attention_mask=mask)
    assert pooled_mask.shape == (2, 16)
    assert torch.allclose(pooled_mask[0], torch.full((16,), 2.0), atol=1e-5)
    assert torch.allclose(pooled_mask[1], torch.full((16,), 5.0), atol=1e-5)

    # Case B: 5D pixel_values fallback [B=2, max_tiles=3, C=3, H=8, W=8]
    pv = torch.zeros((2, 3, 3, 8, 8), dtype=torch.float32)
    pv[0, :3] = 1.0
    pv[1, :1] = 1.0
    pooled_pv = _pool_visual_tokens(vis_out, batch_size=2, pixel_values=pv)
    assert pooled_pv.shape == (2, 16)
    assert torch.allclose(pooled_pv[0], torch.full((16,), 2.0), atol=1e-5)
    assert torch.allclose(pooled_pv[1], torch.full((16,), 5.0), atol=1e-5)


def test_smolvlm_nested_image_collation():
    class _ImageCapturingProcessor(_DummyProcessor):
        def __init__(self) -> None:
            super().__init__(with_chat_template=False)
            self.captured_images = []

        def __call__(self, text, images, padding=True, return_tensors="pt"):
            self.captured_images.append(images)
            return super().__call__(text=text, images=images, padding=padding, return_tensors=return_tensors)

    img0 = Image.new("RGB", (24, 24), color=(10, 20, 30))
    img1 = Image.new("RGB", (24, 24), color=(40, 50, 60))
    samples = [
        {"domain_type": "document_text", "image": img0, "question": "Q0?", "answers": ["A0"]},
        {"domain_type": "scene_text", "image": img1, "question": "Q1?", "answers": ["A1"]},
    ]

    smol_proc = _ImageCapturingProcessor()
    smol_collator = CrossVQADataCollator(smol_proc, model_type="smolvlm", is_train=False)
    smol_collator(samples)
    assert len(smol_proc.captured_images) == 1
    assert isinstance(smol_proc.captured_images[0], list)
    assert len(smol_proc.captured_images[0]) == 2
    assert isinstance(smol_proc.captured_images[0][0], list) and len(smol_proc.captured_images[0][0]) == 1

    qwen_proc = _ImageCapturingProcessor()
    qwen_collator = CrossVQADataCollator(qwen_proc, model_type="qwen2_vl", is_train=False)
    qwen_collator(samples)
    assert isinstance(qwen_proc.captured_images[0], list)
    assert not isinstance(qwen_proc.captured_images[0][0], list)


def test_lora_vision_exclusion_parity(monkeypatch):
    import types
    from TRAIN.src.adapters import DualLoRALinear
    from TRAIN.src.models import _build_lora_config

    class _ModernLoraConfig:
        def __init__(self, r, lora_alpha, target_modules, lora_dropout, bias, task_type, exclude_modules=None):
            self.r = r
            self.target_modules = target_modules
            self.exclude_modules = exclude_modules

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = _ModernLoraConfig
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    cfg_modern = _build_lora_config(
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    assert list(cfg_modern.exclude_modules) == ["vision_model", "visual"]

    class _LegacyLoraConfig:
        def __init__(self, r, lora_alpha, target_modules, lora_dropout, bias, task_type):
            self.r = r
            self.target_modules = target_modules

    fake_peft_legacy = types.ModuleType("peft")
    fake_peft_legacy.LoraConfig = _LegacyLoraConfig
    monkeypatch.setitem(sys.modules, "peft", fake_peft_legacy)
    cfg_legacy = _build_lora_config(
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    assert isinstance(cfg_legacy.target_modules, str)
    assert re.search(cfg_legacy.target_modules, "text_model.layers.0.self_attn.q_proj")
    assert not re.search(cfg_legacy.target_modules, "vision_model.encoder.q_proj")
    assert not re.search(cfg_legacy.target_modules, "visual.blocks.0.v_proj")

    class _DummyVisionAndTextBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 8})()
            self.connector = nn.Linear(8, 8)
            self.vision_model = nn.Module()
            self.vision_model.q_proj = nn.Linear(8, 8)
            self.visual = nn.Module()
            self.visual.v_proj = nn.Linear(8, 8)
            self.text_model = nn.Module()
            self.text_model.q_proj = nn.Linear(8, 8)
            self.text_model.v_proj = nn.Linear(8, 8)

    wrapped = CrossVQADualAdapterModel(
        backbone=_DummyVisionAndTextBackbone(),
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        projector_module_name="connector",
    )
    assert isinstance(wrapped.backbone.text_model.q_proj, DualLoRALinear)
    assert not isinstance(wrapped.backbone.vision_model.q_proj, DualLoRALinear)
    assert not isinstance(wrapped.backbone.visual.v_proj, DualLoRALinear)


def test_m2_alignment_loss_has_nonzero_gradients():
    base = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    wrapper = CrossVQADualAdapterModel(
        backbone=base,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        align_adapter_rank=8,
    )
    assert wrapper.align_adapter is not None
    # Verify exact zero initialization (identity start)
    assert torch.count_nonzero(wrapper.align_adapter.up.weight).item() == 0
    assert torch.count_nonzero(wrapper.align_adapter.up.bias).item() == 0

    proc = _DummyProcessor(with_chat_template=False)
    collator = CrossVQADataCollator(proc, model_type="smolvlm", is_train=True)
    batch = collator([
        {"domain_type": "document_text", "image": Image.new("RGB", (32, 32), (255, 0, 0)), "question": "doc?", "answers": ["d"]},
        {"domain_type": "scene_text", "image": Image.new("RGB", (32, 32), (0, 255, 0)), "question": "scn?", "answers": ["s"]},
    ])
    mmd_fn = MultiKernelRBFMMDLoss(queue_size=16)

    wrapper.train()
    wrapper.zero_grad()
    outputs = wrapper(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )
    pooled_vis = wrapper.last_routing_outputs["pooled_vis"]
    d_ids = batch["domain_ids"]
    loss_align = mmd_fn(pooled_vis, d_ids)
    total_loss = outputs.loss + loss_align
    total_loss.backward()

    align_grads = [
        p.grad.detach().abs().sum().item()
        for p in wrapper.align_adapter.parameters()
        if p.grad is not None
    ]
    assert len(align_grads) > 0 and max(align_grads) > 0.0


def test_map_style_cosine_lr_schedule_and_mid_epoch_ckpt(tmp_path):
    class _MapDataset(torch.utils.data.Dataset):
        def __init__(self, n: int = 16) -> None:
            self.items = [
                {
                    "domain_type": "document_text" if i % 2 == 0 else "scene_text",
                    "image": Image.new("RGB", (16, 16)),
                    "question": f"q{i}?",
                    "answers": ["ans"],
                }
                for i in range(n)
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            return self.items[idx]

    model = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    proc = _DummyProcessor(with_chat_template=False)
    run_dir = tmp_path / "run_map"
    ckpt_mgr = CheckpointManager(local_ckpt_dir=str(run_dir / "checkpoints"), hf_token=None)
    saved_calls = []
    orig_save = ckpt_mgr.save

    def _spy_save(mdl, tag="latest", trainer_state=None):
        saved_calls.append((tag, dict(trainer_state or {})))
        return orig_save(mdl, tag=tag, trainer_state=trainer_state)

    ckpt_mgr.save = _spy_save
    trainer = CrossVQATrainer(
        model=model,
        processor=proc,
        run_dir=str(run_dir),
        model_type="smolvlm",
        device="cpu",
        lr=1e-3,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        checkpoint_manager=ckpt_mgr,
    )
    summary = trainer.fit(
        train_dataset=_MapDataset(16),
        epochs=1,
        batch_size=4,
        save_every_steps=2,
        seed=42,
    )
    assert summary["opt_steps"] == 4
    logs = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lrs = [entry["lr"] for entry in logs]
    # Exact len(dataloader)=4 yields strictly decreasing cosine LRs with non-zero midpoint & terminal decay
    assert len(lrs) == 4
    assert lrs[0] > lrs[1] > lrs[2] > lrs[3]
    assert ("latest", {"epoch": 1, "opt_step": 2}) in saved_calls
    assert (run_dir / "checkpoints" / "latest" / "trainer_state.json").exists()
    assert (run_dir / "checkpoints" / "best" / "trainer_state.json").exists()


def test_z0_image_id_dedup_and_notebook_defaults():
    sys.path.insert(0, str(REPO_ROOT / "TRAIN/ALT_THREE_SPLITS/scripts"))
    from run_z0_domain_gap import _select_unique_image_indices, extract_domain_embeddings

    samples = [
        {"sample_id": "s0", "image_id": "img_a", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q0?", "answers": ["a"]},
        {"sample_id": "s1", "image_id": "img_a", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q1?", "answers": ["b"]},
        {"sample_id": "s2", "image_id": "img_b", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q2?", "answers": ["c"]},
        {"sample_id": "s3", "image_id": "img_b", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q3?", "answers": ["d"]},
        {"sample_id": "s4", "image_id": "img_c", "domain_type": "scene_text", "image": Image.new("RGB", (16, 16)), "question": "q4?", "answers": ["e"]},
    ]
    indices, keys = _select_unique_image_indices(samples, max_samples=10, seed=42)
    assert len(indices) == 3
    assert set(keys) == {"img_a", "img_b", "img_c"}

    model = _TinyVLMWithLinearConnector(hidden_dim=16, vocab_size=128)
    proc = _DummyProcessor(with_chat_template=False)
    embs, ret_keys = extract_domain_embeddings(
        model=model,
        processor=proc,
        model_type="smolvlm",
        dataset_name="textvqa",
        split="validation",
        max_samples=10,
        batch_size=2,
        device="cpu",
        eval_dataset=samples,
        dedup_by_image_id=True,
        seed=42,
        return_keys=True,
    )
    assert embs.shape == (3, 16)
    assert len(set(ret_keys)) == 3

    # Verify all 6 canonical regenerated notebooks default to smolvlm_500m
    nb_dir = REPO_ROOT / "TRAIN/ALT_THREE_SPLITS/notebooks"
    canonical_nbs = [
        "00_Z0_ZeroShot_and_DomainGap.ipynb",
        "01_B1_B2_SingleDomain_LoRA.ipynb",
        "02_B3_Joint_MultiDomain_LoRA.ipynb",
        "03_B4_B5_Sequential_Transfer.ipynb",
        "04_M1_M3_CrossVQA_Proposed.ipynb",
        "05_FewShot_Scaling_Matrix.ipynb",
    ]
    for name in canonical_nbs:
        p = nb_dir / name
        src = "\n".join("".join(c.get("source", [])) for c in json.loads(p.read_text(encoding="utf-8"))["cells"])
        assert 'MODEL_KEY = "smolvlm_500m"' in src, f"Expected MODEL_KEY='smolvlm_500m' in {p.name}"
        if p.name != "00_Z0_ZeroShot_and_DomainGap.ipynb":
            assert "--save_every_steps" in src, f"Expected --save_every_steps in {p.name}"
