"""
Compiles all 6 Google Colab GPU training notebooks under TRAIN/ALT_THREE_SPLITS/notebooks/:
- 00_Z0_ZeroShot_and_DomainGap.ipynb
- 01_B1_B2_SingleDomain_LoRA.ipynb
- 02_B3_Joint_MultiDomain_LoRA.ipynb
- 03_B4_B5_Sequential_Transfer.ipynb
- 04_M1_M3_CrossVQA_Proposed.ipynb
- 05_FewShot_Scaling_Matrix.ipynb

Each notebook features:
1. Auto-clone / git pull + GPU profile check (L4 preferred, T4 fallback)
2. Masked interactive getpass() input box for HF_TOKEN (private gamusa/crossvqa-checkpoints)
   and optional GH_TOKEN, plus Google Drive backup mount
3. Fast hf_transfer snapshot download of three-splits Parquet shards to /content/data/
4. CLI execution cell + lightweight Git log sync (never committing model weights)

Usage:
    python TRAIN/ALT_THREE_SPLITS/scripts/build_training_notebooks.py
"""

import json
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path("TRAIN/ALT_THREE_SPLITS/notebooks")
REPO_URL = "https://github.com/NhanChanDeo/VQA_Final_Thesis.git"


def _md_cell(text: str) -> Dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").splitlines()],
    }


def _code_cell(code: str) -> Dict:
    lines = code.rstrip("\n").splitlines()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
    }


CELL_BOOTSTRAP = f"""# ==============================================================================
# 1. COLAB SETUP, REPO SYNC & GPU CHECK (L4 > A100 > G4 > T4 FALLBACK)
# ==============================================================================
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

def run_streamed(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

if not Path("TRAIN/src").exists():
    if Path("VQA_Final_Thesis/TRAIN/src").exists():
        os.chdir("VQA_Final_Thesis")
    else:
        subprocess.run(["git", "clone", "{REPO_URL}"], check=True)
        os.chdir("VQA_Final_Thesis")

subprocess.run(["git", "pull", "origin", "main"], check=True)
print("Working directory:", Path.cwd())

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers>=4.49.0", "peft", "bitsandbytes", "accelerate",
        "datasets", "polars", "pyyaml", "safetensors", "huggingface_hub",
        "hf_transfer", "umap-learn", "scikit-learn", "qwen-vl-utils",
    ],
    check=True,
)

import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"GPU: {{props.name}} | VRAM: {{vram_gb:.1f}} GB | bf16: {{torch.cuda.is_bf16_supported()}}")
else:
    print("WARNING: No CUDA GPU detected. Switch Colab runtime to L4, A100, G4, or T4.")
"""

CELL_AUTH_AND_DRIVE = """# ==============================================================================
# 2. MASKED API KEY INPUT BOX & GOOGLE DRIVE CHECKPOINT BACKUP
# ==============================================================================
import os
from getpass import getpass
from pathlib import Path
from huggingface_hub import HfApi

# 2A. Interactive masked HF WRITE token (never stored in notebook source/outputs)
HF_TOKEN = os.environ.get("HF_TOKEN") or None
try:
    from google.colab import userdata
    HF_TOKEN = HF_TOKEN or userdata.get("HF_TOKEN")
except Exception:
    pass

if not HF_TOKEN:
    print("Enter your Hugging Face WRITE Token (input is masked and never saved):")
    HF_TOKEN = getpass("HF Write Token: ").strip()

os.environ["HF_TOKEN"] = HF_TOKEN
assert HF_TOKEN, "HF_TOKEN is required to save/restore private checkpoints."

api = HfApi(token=HF_TOKEN)
user_info = api.whoami()
print(f"Authenticated Hugging Face account: {user_info['name']}")

# 2B. Mount Google Drive for dual-redundant checkpoint backup
GDRIVE_BACKUP_DIR = None
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    drive_ckpt_root = Path("/content/drive/MyDrive/CrossVQA_Checkpoints/ALT_THREE_SPLITS")
    drive_ckpt_root.mkdir(parents=True, exist_ok=True)
    GDRIVE_BACKUP_DIR = str(drive_ckpt_root)
    print(f"Google Drive backup active: {GDRIVE_BACKUP_DIR}")
except Exception as e:
    print(f"Google Drive mount skipped ({e}); continuing with private HF Hub checkpoints only.")
"""

CELL_DATA_SNAPSHOT = """# ==============================================================================
# 3. FAST NVMe SNAPSHOT OF `three-splits` SHARDS TO /content/data/
# ==============================================================================
from pathlib import Path
from huggingface_hub import snapshot_download

LOCAL_DATA_DIR = Path("/content/data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

for ds_name, repo_id in [
    ("docvqa", "gamusa/docvqa-canonical"),
    ("textvqa", "gamusa/textvqa-canonical"),
]:
    target = LOCAL_DATA_DIR / ds_name
    print(f"Syncing {repo_id}@three-splits -> {target} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="three-splits",
        allow_patterns=["data/*.parquet"],
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
    )
print("Snapshot ready on ephemeral NVMe scratch.")
"""

CELL_SYNC_LOGS = """# ==============================================================================
# 5. INSPECT RESULTS & SYNC LIGHTWEIGHT LOGS TO GITHUB (WEIGHTS EXCLUDED)
# ==============================================================================
import json
import subprocess
from getpass import getpass
from pathlib import Path

# Preview generated metrics/run cards
for p in sorted(Path("TRAIN/ALT_THREE_SPLITS/runs").rglob("val_metrics.json")):
    print("===", p, "===")
    print(p.read_text(encoding="utf-8")[:600])

# Optional masked GitHub push for lightweight JSON/YAML/PNG run artifacts
PUSH_LOGS_TO_GITHUB = False  # Set True when ready to commit logs from Colab
if PUSH_LOGS_TO_GITHUB:
    subprocess.run(["git", "add", "TRAIN/ALT_THREE_SPLITS/runs/"], check=True)
    subprocess.run(["git", "commit", "-m", "chore(runs): sync Colab training metrics and logs"], check=False)
    gh_token = os.environ.get("GH_TOKEN") or getpass("GitHub PAT (masked, optional): ").strip()
    if gh_token:
        remote_url = f"https://{gh_token}@github.com/NhanChanDeo/VQA_Final_Thesis.git"
        subprocess.run(["git", "push", remote_url, "HEAD:main"], check=True)
        print("Synced lightweight logs to origin/main.")
"""

NOTEBOOK_SPECS: List[Dict[str, str]] = [
    {
        "filename": "00_Z0_ZeroShot_and_DomainGap.ipynb",
        "title": "# 00 — Z0 Zero-Shot Benchmark & Phase A8/A9 Domain Gap Diagnostics\n\n"
                 "Evaluates the frozen backbone (`Qwen/Qwen2-VL-2B` primary, `SmolVLM-Base`, `SmolVLM-500M-Base`) "
                 "on `NEW_validation` + `NEW_test`, generates 2D UMAP projections (A8), and computes RKHS MMD & 5-fold Proxy $\\mathcal{A}$-distance (A9).",
        "run_code": """# ==============================================================================
# 4. RUN Z0 ZERO-SHOT + PHASE A8 UMAP + PHASE A9 MMD / PROXY A-DISTANCE
# ==============================================================================
import subprocess
import sys

MODEL_KEY = "qwen2_vl_2b"        # Options: "qwen2_vl_2b", "smolvlm_base", "smolvlm_500m"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
DOMAIN_GAP_SAMPLES = 2000        # Number of validation samples per domain for UMAP/MMD/Proxy-A
MAX_EVAL_SAMPLES = None          # Set e.g. 100 for a quick pilot smoke test

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_z0_domain_gap.py",
    "--model", MODEL_KEY,
    "--hardware_profile", HARDWARE_PROFILE,
    "--domain_gap_samples", str(DOMAIN_GAP_SAMPLES),
    "--local_data_dir", str(LOCAL_DATA_DIR),
]
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

print("Running:", " ".join(cmd))
run_streamed(cmd)
""",
    },
    {
        "filename": "01_B1_B2_SingleDomain_LoRA.ipynb",
        "title": "# 01 — Single-Domain LoRA Baselines (`B1` Target-Only & `B2` Source-Only)\n\n"
                 "Trains `B1` (TextVQA-only) and `B2` (DocVQA-only), syncing checkpoints to private HF Hub (`gamusa/crossvqa-checkpoints`) and personal Google Drive.",
        "run_code": """# ==============================================================================
# 4. RUN B1 (TEXTVQA-ONLY) OR B2 (DOCVQA-ONLY) BASELINE
# ==============================================================================
import subprocess
import sys

EXP = "B1"                       # "B1" (TextVQA) or "B2" (DocVQA)
MODEL_KEY = "qwen2_vl_2b"        # "qwen2_vl_2b" | "smolvlm_base" | "smolvlm_500m"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42                        # Seeds: 42, 123, 999
INCLUDE_OCR = False              # False = Track A (OCR-free primary); True = Track B ablation
MAX_TRAIN_STEPS = None           # Set e.g. 20 for smoke test
MAX_EVAL_SAMPLES = None

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py",
    "--exp", EXP,
    "--model", MODEL_KEY,
    "--seed", str(SEED),
    "--hardware_profile", HARDWARE_PROFILE,
    "--local_data_dir", str(LOCAL_DATA_DIR),
    "--hf_ckpt_repo", "gamusa/crossvqa-checkpoints",
    "--resume_from_hub",
]
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if INCLUDE_OCR:
    cmd.append("--include_ocr")
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

run_streamed(cmd)
""",
    },
    {
        "filename": "02_B3_Joint_MultiDomain_LoRA.ipynb",
        "title": "# 02 — Naive Joint Training Baseline (`B3`: DocVQA + TextVQA)\n\n"
                 "Interleaves DocVQA and TextVQA streams 50/50 with a single shared LoRA adapter to measure negative transfer.",
        "run_code": """# ==============================================================================
# 4. RUN B3 (JOINT DOCVQA + TEXTVQA) BASELINE
# ==============================================================================
import subprocess
import sys

MODEL_KEY = "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
MAX_TRAIN_STEPS = None
MAX_EVAL_SAMPLES = None

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py",
    "--exp", "B3",
    "--model", MODEL_KEY,
    "--seed", str(SEED),
    "--hardware_profile", HARDWARE_PROFILE,
    "--local_data_dir", str(LOCAL_DATA_DIR),
    "--hf_ckpt_repo", "gamusa/crossvqa-checkpoints",
    "--resume_from_hub",
]
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

run_streamed(cmd)
""",
    },
    {
        "filename": "03_B4_B5_Sequential_Transfer.ipynb",
        "title": "# 03 — Sequential Transfer Baselines (`B4`: Doc→Text & `B5`: Text→Doc)\n\n"
                 "Automatically restores Stage-1 checkpoints (`B2` for `B4`, `B1` for `B5`) from private HF Hub or Google Drive and runs Stage-2 fine-tuning to measure catastrophic forgetting.",
        "run_code": """# ==============================================================================
# 4. RUN B4 (DOC -> TEXT) OR B5 (TEXT -> DOC) SEQUENTIAL TRANSFER
# ==============================================================================
import subprocess
import sys

EXP = "B4"                       # "B4" (Doc->Text) or "B5" (Text->Doc)
MODEL_KEY = "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
MAX_TRAIN_STEPS = None
MAX_EVAL_SAMPLES = None

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_baseline.py",
    "--exp", EXP,
    "--model", MODEL_KEY,
    "--seed", str(SEED),
    "--hardware_profile", HARDWARE_PROFILE,
    "--local_data_dir", str(LOCAL_DATA_DIR),
    "--hf_ckpt_repo", "gamusa/crossvqa-checkpoints",
    "--resume_from_hub",
]
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

run_streamed(cmd)
""",
    },
    {
        "filename": "04_M1_M3_CrossVQA_Proposed.ipynb",
        "title": "# 04 — Proposed CrossVQA Architectures (`M1` Dual-LoRA, `M2` + Alignment, `M3` + Curriculum)\n\n"
                 "Executes the proposed CrossVQA multi-stage pipeline with Dual Domain-Specific LoRA adapters, Visual Soft Gating Router, Multi-Kernel RBF MMD (or InfoNCE ablation), and M3 Homography Curriculum.",
        "run_code": """# ==============================================================================
# 4. RUN PROPOSED CROSSVQA (M1, M2, OR M3)
# ==============================================================================
import subprocess
import sys

EXP = "M2"                       # "M1" | "M2" | "M3"
MODEL_KEY = "qwen2_vl_2b"        # Pilot first with "smolvlm_500m", then run "qwen2_vl_2b" & "smolvlm_base"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
ALIGN_TYPE = "mmd"               # "mmd" (default) or "infonce" (ablation)
INCLUDE_OCR = False              # True for Track B Given-OCR ablation on M2
SKIP_STAGE1_IF_CACHED = True     # Reuses M_stage1_doc checkpoint across M1/M2/M3
MAX_TRAIN_STEPS = None
MAX_EVAL_SAMPLES = None

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_crossvqa.py",
    "--exp", EXP,
    "--model", MODEL_KEY,
    "--seed", str(SEED),
    "--align_type", ALIGN_TYPE,
    "--hardware_profile", HARDWARE_PROFILE,
    "--local_data_dir", str(LOCAL_DATA_DIR),
    "--hf_ckpt_repo", "gamusa/crossvqa-checkpoints",
    "--resume_from_hub",
]
if SKIP_STAGE1_IF_CACHED:
    cmd.append("--skip_stage1")
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if INCLUDE_OCR:
    cmd.append("--include_ocr")
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

run_streamed(cmd)
""",
    },
    {
        "filename": "05_FewShot_Scaling_Matrix.ipynb",
        "title": "# 05 — Few-Shot Target Scaling Matrix (`1%`, `5%`, `10%`, `25%` TextVQA — Hypothesis H4)\n\n"
                 "Evaluates `B1`, `B4`, and `M2` across nested `image_id`-grouped stratified TextVQA subsets.",
        "run_code": """# ==============================================================================
# 4. RUN FEW-SHOT SCALING MATRIX (1%, 5%, 10%, 25%)
# ==============================================================================
import subprocess
import sys

MODEL_KEY = "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
FRACTIONS = [1, 5, 10, 25]
METHODS = ["B1", "B4", "M2"]
MAX_TRAIN_STEPS = None
MAX_EVAL_SAMPLES = None

for pct in FRACTIONS:
    for method in METHODS:
        script = "run_crossvqa.py" if method.startswith("M") else "run_baseline.py"
        cmd = [
            sys.executable, f"TRAIN/ALT_THREE_SPLITS/scripts/{script}",
            "--exp", method,
            "--model", MODEL_KEY,
            "--seed", str(SEED),
            "--few_shot_pct", str(pct),
            "--hardware_profile", HARDWARE_PROFILE,
            "--local_data_dir", str(LOCAL_DATA_DIR),
            "--hf_ckpt_repo", "gamusa/crossvqa-checkpoints",
            "--resume_from_hub",
        ]
        if method == "M2":
            cmd.append("--skip_stage1")
        if GDRIVE_BACKUP_DIR:
            cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
        if MAX_TRAIN_STEPS is not None:
            cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
        if MAX_EVAL_SAMPLES is not None:
            cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])
        print(f"\\n>>> Running {method} at {pct}% target TextVQA...")
        run_streamed(cmd)
""",
    },
]


def build_notebook(spec: Dict[str, str]) -> Dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "cells": [
            _md_cell(spec["title"]),
            _code_cell(CELL_BOOTSTRAP),
            _code_cell(CELL_AUTH_AND_DRIVE),
            _code_cell(CELL_DATA_SNAPSHOT),
            _code_cell(spec["run_code"]),
            _code_cell(CELL_SYNC_LOGS),
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in NOTEBOOK_SPECS:
        nb_path = OUT_DIR / spec["filename"]
        nb_obj = build_notebook(spec)
        nb_path.write_text(json.dumps(nb_obj, indent=2), encoding="utf-8")
        print(f"Generated {nb_path}")


if __name__ == "__main__":
    main()
