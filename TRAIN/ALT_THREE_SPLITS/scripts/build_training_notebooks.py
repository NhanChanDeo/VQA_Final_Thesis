"""
Compiles all 6 Google Colab GPU training notebooks under TRAIN/ALT_THREE_SPLITS/notebooks/:
- 00_Z0_ZeroShot_and_DomainGap.ipynb
- 01_B1_B2_SingleDomain_LoRA.ipynb
- 02_B3_Joint_MultiDomain_LoRA.ipynb
- 03_B4_B5_Sequential_Transfer.ipynb
- 04_M1_M3_CrossVQA_Proposed.ipynb
- 05_FewShot_Scaling_Matrix.ipynb

Each notebook features:
1. Fail-fast GPU/disk checks, Xet high-performance transfers, uv/cached installs, TF32, & git identity setup
2. Non-blocking upfront HF_TOKEN / GH_TOKEN secret resolution + Google Drive backup mount
3. Split-aware, idempotent NVMe snapshot of three-splits Parquet shards to /content/data/
4. Exception-safe CLI execution with stage wall-clock timing & peak CUDA memory reporting
5. Unattended Google Drive + GitHub lightweight run-log sync (never committing model weights)
6. Compute-economy summary + safe 300s delayed auto-disconnect (with interactive Cancel / Disconnect Now controls)

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


_BOOTSTRAP_TEMPLATE = """# ==============================================================================
# 1. COLAB SETUP, FAIL-FAST GUARDS, REPO SYNC & FAST DEPS
# ==============================================================================
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

REQUIRE_GPU = True          # Set False only for intentional CPU smoke tests
MIN_FREE_DISK_GB = 25.0     # Safety floor for dataset shards + model weights
STAGE_TIMINGS = {}
_t0 = time.perf_counter()

import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    sm_major = torch.cuda.get_device_capability(0)[0]
    if sm_major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {props.name} | VRAM: {vram_gb:.1f} GB | bf16: {torch.cuda.is_bf16_supported()} | TF32: {sm_major >= 8}")
else:
    msg = "No CUDA GPU detected. Switch Colab runtime to L4, A100, G4, or T4 (or set REQUIRE_GPU = False)."
    if REQUIRE_GPU:
        raise RuntimeError(msg)
    print("WARNING:", msg)

_disk_probe = Path("/content") if Path("/content").exists() else Path.cwd()
_free_gb = shutil.disk_usage(_disk_probe).free / (1024 ** 3)
print(f"Free disk at {_disk_probe}: {_free_gb:.1f} GB")
if _free_gb < MIN_FREE_DISK_GB:
    raise RuntimeError(f"Insufficient free disk ({_free_gb:.1f} GB < {MIN_FREE_DISK_GB:.1f} GB required).")

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
        subprocess.run(["git", "clone", "__REPO_URL__"], check=True)
        os.chdir("VQA_Final_Thesis")

subprocess.run(["git", "pull", "origin", "main"], check=True)
print("Working directory:", Path.cwd())

for _k, _v in [("user.name", "Colab Runner"), ("user.email", "colab-runner@users.noreply.github.com")]:
    _chk = subprocess.run(["git", "config", "--get", _k], capture_output=True, text=True)
    if _chk.returncode != 0 or not _chk.stdout.strip():
        subprocess.run(["git", "config", _k, _v], check=True)

_sentinel_dir = Path("/content") if Path("/content").exists() else Path("/tmp")
_deps_sentinel = _sentinel_dir / ".crossvqa_deps___DEPS_TAG__"
_pkgs = [
    "transformers>=4.49.0", "peft", "bitsandbytes", "accelerate",
    "datasets", "polars", "pyyaml", "safetensors", "huggingface_hub>=0.32.0",
    "hf_xet", "scikit-learn", "qwen-vl-utils",__EXTRA_PKGS__
]
if not _deps_sentinel.exists():
    _uv = shutil.which("uv")
    if _uv:
        subprocess.run([_uv, "pip", "install", "--system", "--quiet", *_pkgs], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_pkgs], check=True)
    _deps_sentinel.write_text("ok", encoding="utf-8")
else:
    print("Dependencies already verified in this runtime; skipping install.")

STAGE_TIMINGS["prep_sec"] = round(time.perf_counter() - _t0, 2)
print(f"Bootstrap completed in {STAGE_TIMINGS['prep_sec']:.1f}s")
"""


def _make_bootstrap_cell(needs_umap: bool = False) -> str:
    extra_pkgs = ' "umap-learn",' if needs_umap else ""
    deps_tag = "umap" if needs_umap else "base"
    return (
        _BOOTSTRAP_TEMPLATE
        .replace("__REPO_URL__", REPO_URL)
        .replace("__EXTRA_PKGS__", extra_pkgs)
        .replace("__DEPS_TAG__", deps_tag)
    )


CELL_AUTH_AND_DRIVE = """# ==============================================================================
# 2. NON-BLOCKING SECRET LOADING & GOOGLE DRIVE CHECKPOINT BACKUP
# ==============================================================================
import os
import time
from getpass import getpass
from pathlib import Path
from huggingface_hub import HfApi

_t0 = time.perf_counter()
PROMPT_FOR_GH_TOKEN = False  # Set True only if GH_TOKEN isn't in Colab Secrets and you want a prompt

# 2A. Interactive masked HF WRITE token (never stored in notebook source/outputs)
HF_TOKEN = os.environ.get("HF_TOKEN") or None
GH_TOKEN = os.environ.get("GH_TOKEN") or None
try:
    from google.colab import userdata
    HF_TOKEN = HF_TOKEN or userdata.get("HF_TOKEN")
    try:
        GH_TOKEN = GH_TOKEN or userdata.get("GH_TOKEN")
    except Exception:
        pass
except Exception:
    pass

if not HF_TOKEN:
    print("Enter your Hugging Face WRITE Token (input is masked and never saved):")
    HF_TOKEN = getpass("HF Write Token: ").strip()

if not GH_TOKEN and PROMPT_FOR_GH_TOKEN:
    GH_TOKEN = getpass("GitHub PAT (masked, optional — press Enter to skip): ").strip() or None

os.environ["HF_TOKEN"] = HF_TOKEN
if GH_TOKEN:
    os.environ["GH_TOKEN"] = GH_TOKEN
assert HF_TOKEN, "HF_TOKEN is required to save/restore private checkpoints."

api = HfApi(token=HF_TOKEN)
user_info = api.whoami()
print(f"Authenticated Hugging Face account: {user_info['name']} | GH_TOKEN loaded: {bool(GH_TOKEN)}")

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

STAGE_TIMINGS["auth_sec"] = round(time.perf_counter() - _t0, 2)
"""

_SNAPSHOT_TEMPLATE = """# ==============================================================================
# 3. SPLIT-AWARE, IDEMPOTENT NVMe SNAPSHOT OF `three-splits` SHARDS
# ==============================================================================
import time
from pathlib import Path
from huggingface_hub import snapshot_download

_t0 = time.perf_counter()
LOCAL_DATA_DIR = Path("/content/data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RESYNC = False

_ALLOW_PATTERNS = __ALLOW_PATTERNS__
_REQUIRED_PREFIXES = __REQUIRED_PREFIXES__

for ds_name, repo_id in [
    ("docvqa", "gamusa/docvqa-canonical"),
    ("textvqa", "gamusa/textvqa-canonical"),
]:
    target = LOCAL_DATA_DIR / ds_name
    data_dir = target / "data"
    cached_ok = (
        not FORCE_RESYNC
        and data_dir.exists()
        and all(any(data_dir.glob(f"{pfx}*.parquet")) for pfx in _REQUIRED_PREFIXES)
    )
    if cached_ok:
        print(f"Cached on NVMe ({ds_name}): skipping download.")
        continue
    print(f"Syncing {repo_id}@three-splits ({_ALLOW_PATTERNS}) -> {target} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="three-splits",
        allow_patterns=_ALLOW_PATTERNS,
        local_dir=str(target),
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )

STAGE_TIMINGS["snapshot_sec"] = round(time.perf_counter() - _t0, 2)
print(f"Snapshot ready on ephemeral NVMe scratch ({STAGE_TIMINGS['snapshot_sec']:.1f}s).")
"""


def _make_snapshot_cell(include_train: bool = True, include_test: bool = False) -> str:
    prefixes = []
    if include_train:
        prefixes.append("train-")
    prefixes.append("validation-")
    if include_test:
        prefixes.append("test-")
    patterns = [f"data/{p}*.parquet" for p in prefixes]
    return (
        _SNAPSHOT_TEMPLATE
        .replace("__ALLOW_PATTERNS__", json.dumps(patterns))
        .replace("__REQUIRED_PREFIXES__", json.dumps(prefixes))
    )


CELL_SYNC_LOGS = """# ==============================================================================
# 5. INSPECT RESULTS & UNATTENDED SYNC (DRIVE + OPTIONAL GITHUB)
# ==============================================================================
import os
import shutil
import subprocess
import time
from pathlib import Path

_t0 = time.perf_counter()

# Preview generated metrics/run cards & display UMAP plot if present
for p in sorted(Path("TRAIN/ALT_THREE_SPLITS/runs").rglob("*.json")):
    if p.name in ("val_metrics.json", "domain_gap_metrics.json"):
        print("===", p, "===")
        print(p.read_text(encoding="utf-8")[:1000])

try:
    from IPython.display import Image as IPyImage, display
    for img_p in sorted(Path("TRAIN/ALT_THREE_SPLITS/runs").rglob("umap_domain_gap.png")):
        print("===", img_p, "===")
        display(IPyImage(filename=str(img_p)))
except Exception:
    pass

# Resolve Google Drive backup dir safely (even if Cell 5 is run independently)
_gdrive_dir = globals().get("GDRIVE_BACKUP_DIR")
if not _gdrive_dir and Path("/content/drive/MyDrive").exists():
    _default_drive = Path("/content/drive/MyDrive/CrossVQA_Checkpoints/ALT_THREE_SPLITS")
    _default_drive.mkdir(parents=True, exist_ok=True)
    _gdrive_dir = str(_default_drive)

# Mirror lightweight logs/plots to Google Drive if mounted
if _gdrive_dir:
    drive_runs = Path(_gdrive_dir) / "runs_logs"
    drive_runs.mkdir(parents=True, exist_ok=True)
    shutil.copytree("TRAIN/ALT_THREE_SPLITS/runs", drive_runs, dirs_exist_ok=True)
    print(f"Mirrored lightweight run logs to {drive_runs}")

# Optional unattended GitHub push for lightweight JSON/YAML/PNG run artifacts
PUSH_LOGS_TO_GITHUB = True  # Set True when ready to commit logs from Colab
if PUSH_LOGS_TO_GITHUB:
    status = subprocess.run(
        ["git", "status", "--porcelain", "TRAIN/ALT_THREE_SPLITS/runs/"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if status:
        subprocess.run(["git", "add", "TRAIN/ALT_THREE_SPLITS/runs/"], check=True)
        subprocess.run(["git", "commit", "-m", "chore(runs): sync Colab training metrics and logs"], check=False)
        gh_token = os.environ.get("GH_TOKEN")
        if gh_token:
            remote_url = f"https://{gh_token}@github.com/NhanChanDeo/VQA_Final_Thesis.git"
            subprocess.run(["git", "push", remote_url, "HEAD:main"], check=True)
            print("Synced lightweight logs to origin/main.")
        else:
            print("GH_TOKEN not set in environment/Colab Secrets; skipped git push (committed locally).")
    else:
        print("No new run artifacts to commit.")

STAGE_TIMINGS = globals().get("STAGE_TIMINGS", {})
STAGE_TIMINGS["sync_sec"] = round(time.perf_counter() - _t0, 2)
"""

CELL_AUTO_DISCONNECT = """# ==============================================================================
# 6. COMPUTE ECONOMY SUMMARY & SAFE DELAYED AUTO-DISCONNECT (300s)
# ==============================================================================
import time

AUTO_DISCONNECT = True            # Automatically release Colab GPU after delay
DISCONNECT_DELAY_SEC = 300        # 5-minute grace window before runtime.unassign()
REQUIRE_EXPLICIT_CONFIRM = True  # True -> manual y/N prompt (blocks unattended runs!)

_prep = STAGE_TIMINGS.get("prep_sec", 0.0) + STAGE_TIMINGS.get("auth_sec", 0.0) + STAGE_TIMINGS.get("snapshot_sec", 0.0)
_train = STAGE_TIMINGS.get("train_sec", 0.0)
_sync = STAGE_TIMINGS.get("sync_sec", 0.0)
_total = max(_prep + _train + _sync, 1e-6)
print(
    f"Compute Summary -> Prep/Snapshot: {_prep:.1f}s | Train/Eval: {_train:.1f}s ({100.0 * _train / _total:.1f}%) "
    f"| Sync: {_sync:.1f}s | Total: {_total:.1f}s | Run Succeeded: {globals().get('RUN_SUCCEEDED', True)}"
)

def _flush_drive_and_unassign():
    try:
        from google.colab import drive
        if Path("/content/drive/MyDrive").exists():
            drive.flush_and_unmount()
            print("Flushed Google Drive buffers.")
    except Exception:
        pass
    try:
        from google.colab import runtime
        print("Disconnecting and unassigning Colab runtime now...")
        runtime.unassign()
    except ImportError:
        print("Not running in Google Colab runtime; skipping runtime.unassign().")

if REQUIRE_EXPLICIT_CONFIRM:
    ans = input("Disconnect Colab runtime now? [y/N]: ").strip().lower()
    if ans in ("y", "yes"):
        _flush_drive_and_unassign()
    else:
        print("Keeping runtime alive.")
elif AUTO_DISCONNECT:
    _state = {"cancelled": False, "now": False}
    try:
        import ipywidgets as widgets
        from IPython.display import display

        _status = widgets.HTML(value=f"<b>Auto-disconnecting in {DISCONNECT_DELAY_SEC}s...</b>")
        _btn_cancel = widgets.Button(description="Cancel Auto-Disconnect", button_style="warning")
        _btn_now = widgets.Button(description="Disconnect Now", button_style="danger")

        def _on_cancel(_):
            _state["cancelled"] = True
            _status.value = "<b>Auto-disconnect cancelled; keeping runtime alive.</b>"

        def _on_now(_):
            _state["now"] = True
            _status.value = "<b>Disconnecting immediately...</b>"

        _btn_cancel.on_click(_on_cancel)
        _btn_now.on_click(_on_now)
        display(widgets.HBox([_btn_cancel, _btn_now, _status]))

        for _rem in range(DISCONNECT_DELAY_SEC, 0, -1):
            if _state["cancelled"] or _state["now"]:
                break
            _status.value = f"<b>Auto-disconnecting in {_rem}s...</b> (click Cancel to keep VM)"
            time.sleep(1)
    except Exception:
        print(f"Auto-disconnecting in {DISCONNECT_DELAY_SEC}s (interrupt cell to cancel)...")
        time.sleep(DISCONNECT_DELAY_SEC)

    if not _state["cancelled"]:
        _flush_drive_and_unassign()
else:
    print("AUTO_DISCONNECT=False; runtime left active.")
"""

NOTEBOOK_SPECS: List[Dict] = [
    {
        "filename": "00_Z0_ZeroShot_and_DomainGap.ipynb",
        "needs_umap": True,
        "include_train_shards": False,
        "include_test_shards": False,
        "title": "# 00 — Z0 Zero-Shot Benchmark & Phase A8/A9 Domain Gap Diagnostics\n\n"
                 "Evaluates the frozen backbone (`HuggingFaceTB/SmolVLM-500M-Base` primary, `SmolVLM-Base` 2B scale control, `Qwen/Qwen2-VL-2B` cross-arch anchor) "
                 "on `NEW_validation` (default, or `NEW_validation` + `NEW_test`), generates deduplicated 2D UMAP projections (A8), "
                 "and computes balanced RKHS MMD & 5-fold Proxy $\\mathcal{A}$-distance (A9). Supports fast subset or domain-gap-only runs.",
        "run_code": """# ==============================================================================
# 4. RUN Z0 ZERO-SHOT + PHASE A8 UMAP + PHASE A9 MMD / PROXY A-DISTANCE
# ==============================================================================
import sys
import time
import traceback
import torch
from huggingface_hub import snapshot_download

MODEL_KEY = "smolvlm_500m"            # Primary: "smolvlm_500m" | Scale 2B: "smolvlm_base" | Cross-Arch: "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"             # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"

# Execution Mode:
#   - "val_and_domain_gap" (Default): Zero-shot VQA on NEW_validation + fused Phase A8/A9 Domain Gap
#   - "domain_gap_only"   (Fastest):  Skips VQA text generation & LLM layers; runs UMAP + MMD + Proxy-A on NEW_validation
#   - "full_val_and_test" (Full):     Evaluates both NEW_validation and NEW_test + Domain Gap
RUN_MODE = "val_and_domain_gap"

DOMAIN_GAP_SAMPLES = 794              # Balanced unique image_ids/domain on NEW_validation (794 = max DocVQA unique pages)
MAX_EVAL_SAMPLES = None               # None = evaluate all NEW_validation queries (3,141 DocVQA / 2,860 TextVQA)

if RUN_MODE == "domain_gap_only":
    EVAL_SPLITS = []
elif RUN_MODE == "full_val_and_test":
    EVAL_SPLITS = ["validation", "test"]
else:
    EVAL_SPLITS = ["validation"]

# Ensure test shards are synced if "test" split was requested
if "test" in EVAL_SPLITS:
    for _ds_name, _repo_id in [("docvqa", "gamusa/docvqa-canonical"), ("textvqa", "gamusa/textvqa-canonical")]:
        _tdir = LOCAL_DATA_DIR / _ds_name
        if not any((_tdir / "data").glob("test-*.parquet")):
            print(f"Syncing test shards for {_ds_name}...")
            snapshot_download(
                repo_id=_repo_id, repo_type="dataset", revision="three-splits",
                allow_patterns=["data/test-*.parquet"], local_dir=str(_tdir),
                max_workers=8, token=os.environ.get("HF_TOKEN"),
            )

cmd = [
    sys.executable, "TRAIN/ALT_THREE_SPLITS/scripts/run_z0_domain_gap.py",
    "--model", MODEL_KEY,
    "--hardware_profile", HARDWARE_PROFILE,
    "--domain_gap_samples", str(DOMAIN_GAP_SAMPLES),
    "--local_data_dir", str(LOCAL_DATA_DIR),
]
if RUN_MODE == "domain_gap_only":
    cmd.append("--domain_gap_only")
else:
    cmd.extend(["--eval_splits", *EVAL_SPLITS])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

print("Running:", " ".join(cmd))
_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
    run_streamed(cmd)
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during Z0 execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
    {
        "filename": "01_B1_B2_SingleDomain_LoRA.ipynb",
        "needs_umap": False,
        "include_train_shards": True,
        "include_test_shards": False,
        "title": "# 01 — Single-Domain LoRA Baselines (`B1` Target-Only & `B2` Source-Only)\n\n"
                 "Trains `B1` (TextVQA-only) and `B2` (DocVQA-only), syncing checkpoints to private HF Hub (`gamusa/crossvqa-checkpoints`) and personal Google Drive.",
        "run_code": """# ==============================================================================
# 4. RUN B1 (TEXTVQA-ONLY) OR B2 (DOCVQA-ONLY) BASELINE
# ==============================================================================
import sys
import time
import traceback
import torch

EXP = "B1"                       # "B1" (TextVQA) or "B2" (DocVQA)
MODEL_KEY = "smolvlm_500m"       # Primary: "smolvlm_500m" | Scale 2B: "smolvlm_base" | Cross-Arch: "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42                        # Seeds: 42, 123, 999
INCLUDE_OCR = False              # False = Track A (OCR-free primary); True = Track B ablation
SAVE_EVERY_STEPS = 64           # Mid-epoch checkpoint frequency (resilience against Colab disconnects)
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
    "--save_every_steps", str(SAVE_EVERY_STEPS),
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

_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
    run_streamed(cmd)
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during {EXP} execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
    {
        "filename": "02_B3_Joint_MultiDomain_LoRA.ipynb",
        "needs_umap": False,
        "include_train_shards": True,
        "include_test_shards": False,
        "title": "# 02 — Naive Joint Training Baseline (`B3`: DocVQA + TextVQA)\n\n"
                 "Interleaves DocVQA and TextVQA streams 50/50 with a single shared LoRA adapter to measure negative transfer.",
        "run_code": """# ==============================================================================
# 4. RUN B3 (JOINT DOCVQA + TEXTVQA) BASELINE
# ==============================================================================
import sys
import time
import traceback
import torch

MODEL_KEY = "smolvlm_500m"       # Primary: "smolvlm_500m" | Scale 2B: "smolvlm_base" | Cross-Arch: "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
SAVE_EVERY_STEPS = 64
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
    "--save_every_steps", str(SAVE_EVERY_STEPS),
    "--resume_from_hub",
]
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
    run_streamed(cmd)
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during B3 execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
    {
        "filename": "03_B4_B5_Sequential_Transfer.ipynb",
        "needs_umap": False,
        "include_train_shards": True,
        "include_test_shards": False,
        "title": "# 03 — Sequential Transfer Baselines (`B4`: Doc→Text & `B5`: Text→Doc)\n\n"
                 "Automatically restores Stage-1 checkpoints (`B2` for `B4`, `B1` for `B5`) from private HF Hub or Google Drive and runs Stage-2 fine-tuning to measure catastrophic forgetting.",
        "run_code": """# ==============================================================================
# 4. RUN B4 (DOC -> TEXT) OR B5 (TEXT -> DOC) SEQUENTIAL TRANSFER
# ==============================================================================
import sys
import time
import traceback
import torch

EXP = "B4"                       # "B4" (Doc->Text) or "B5" (Text->Doc)
MODEL_KEY = "smolvlm_500m"       # Primary: "smolvlm_500m" | Scale 2B: "smolvlm_base" | Cross-Arch: "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
SAVE_EVERY_STEPS = 64
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
    "--save_every_steps", str(SAVE_EVERY_STEPS),
    "--resume_from_hub",
]
if GDRIVE_BACKUP_DIR:
    cmd.extend(["--gdrive_backup_dir", GDRIVE_BACKUP_DIR])
if MAX_TRAIN_STEPS is not None:
    cmd.extend(["--max_train_steps", str(MAX_TRAIN_STEPS)])
if MAX_EVAL_SAMPLES is not None:
    cmd.extend(["--max_eval_samples", str(MAX_EVAL_SAMPLES)])

_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
    run_streamed(cmd)
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during {EXP} execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
    {
        "filename": "04_M1_M3_CrossVQA_Proposed.ipynb",
        "needs_umap": False,
        "include_train_shards": True,
        "include_test_shards": False,
        "title": "# 04 — Proposed CrossVQA Architectures (`M1` Dual-LoRA, `M2` + Alignment, `M3` + Curriculum)\n\n"
                 "Executes the proposed CrossVQA multi-stage pipeline with Dual Domain-Specific LoRA adapters, Visual Soft Gating Router, Multi-Kernel RBF MMD (or InfoNCE ablation), and M3 Homography Curriculum.",
        "run_code": """# ==============================================================================
# 4. RUN PROPOSED CROSSVQA (M1, M2, OR M3)
# ==============================================================================
import sys
import time
import traceback
import torch

EXP = "M2"                       # "M1" | "M2" | "M3"
MODEL_KEY = "smolvlm_500m"       # Primary: "smolvlm_500m" | Scale 2B: "smolvlm_base" | Cross-Arch: "qwen2_vl_2b"
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
ALIGN_TYPE = "mmd"               # "mmd" (default) or "infonce" (ablation)
INCLUDE_OCR = False              # True for Track B Given-OCR ablation on M2
SKIP_STAGE1_IF_CACHED = True     # Reuses M_stage1_doc checkpoint across M1/M2/M3
SAVE_EVERY_STEPS = 64
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
    "--save_every_steps", str(SAVE_EVERY_STEPS),
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

_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
    run_streamed(cmd)
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during {EXP} execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
    {
        "filename": "05_FewShot_Scaling_Matrix.ipynb",
        "needs_umap": False,
        "include_train_shards": True,
        "include_test_shards": False,
        "title": "# 05 — Few-Shot Target Scaling Matrix (`1%`, `5%`, `10%`, `25%` TextVQA — Hypothesis H4)\n\n"
                 "Evaluates `B1`, `B4`, and `M2` across nested `image_id`-grouped stratified TextVQA subsets.",
        "run_code": """# ==============================================================================
# 4. RUN FEW-SHOT SCALING MATRIX (1%, 5%, 10%, 25%)
# ==============================================================================
import sys
import time
import traceback
import torch

MODEL_KEY = "smolvlm_500m"       # Primary full-matrix backbone for Hypothesis H4
HARDWARE_PROFILE = "auto"        # Options: "auto", "l4_24gb", "a100_40gb", "g4_96gb", "t4_16gb"
SEED = 42
FRACTIONS = [1, 5, 10, 25]
METHODS = ["B1", "B4", "M2"]
SAVE_EVERY_STEPS = 64
MAX_TRAIN_STEPS = None
MAX_EVAL_SAMPLES = None

_t0 = time.perf_counter()
RUN_SUCCEEDED = False
try:
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
                "--save_every_steps", str(SAVE_EVERY_STEPS),
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
    RUN_SUCCEEDED = True
except Exception as exc:
    print(f"ERROR during few-shot matrix execution: {exc}")
    traceback.print_exc()
finally:
    STAGE_TIMINGS["train_sec"] = round(time.perf_counter() - _t0, 2)
    if torch.cuda.is_available():
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Stage elapsed: {STAGE_TIMINGS['train_sec']:.1f}s | Success: {RUN_SUCCEEDED}")
""",
    },
]


def build_notebook(spec: Dict) -> Dict:
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
            _code_cell(_make_bootstrap_cell(needs_umap=bool(spec.get("needs_umap", False)))),
            _code_cell(CELL_AUTH_AND_DRIVE),
            _code_cell(
                _make_snapshot_cell(
                    include_train=bool(spec.get("include_train_shards", True)),
                    include_test=bool(spec.get("include_test_shards", False)),
                )
            ),
            _code_cell(spec["run_code"]),
            _code_cell(CELL_SYNC_LOGS),
            _code_cell(CELL_AUTO_DISCONNECT),
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
