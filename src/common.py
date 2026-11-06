"""Shared code for the bit-flip study. Every other script imports from here."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic GPU matmuls

import hashlib, json, subprocess
import numpy as np
from numpy.random import SeedSequence, default_rng
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ---------------- Seeds ----------------
MASTER = 20261104   # chosen once; also written in the README
ROLES = {"train": 0, "step4": 1, "mat1": 2, "mat2": 3}

def rng(role):
    """Independent random stream for one purpose, derived from the master seed."""
    return default_rng(SeedSequence(MASTER, spawn_key=(ROLES[role],)))

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

# ---------------- Precision ----------------
# FP16 on GPU. If no GPU, fall back to FP32 (9,600 faults); say so in the report.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    DTYPE, INT_DTYPE, NBITS = torch.float16, torch.int16, 16
else:
    DTYPE, INT_DTYPE, NBITS = torch.float32, torch.int32, 32

def bit_field(bit):
    """Which field a bit belongs to (sign / exponent / mantissa)."""
    if NBITS == 16:
        return "sign" if bit == 15 else ("exponent" if bit >= 10 else "mantissa")
    return "sign" if bit == 31 else ("exponent" if bit >= 23 else "mantissa")

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "mlp_fp32.pt")
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR = os.path.join(ROOT, "logs")
CONFIG_DIR = os.path.join(ROOT, "configs")
FIG_DIR = os.path.join(ROOT, "figures")
for d in (os.path.dirname(MODEL_PATH), LOG_DIR, CONFIG_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------- Model and data ----------------
def make_model():
    # model[1] = first weight matrix (128x784), model[3] = second (10x128)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def load_test_batch():
    """All 10,000 test images as ONE fixed batch, so evaluation never varies."""
    ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TFM)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return X, y

def load_fault_model():
    """Load the trained FP32 weights and convert to the study precision."""
    model = make_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model.to(DEVICE).to(DTYPE).eval()

# ---------------- Fault injection ----------------
def fingerprint(W):
    return hashlib.sha256(W.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def flip_bit(W, idx, bit):
    """Flip one bit IN PLACE on the real parameter tensor. Returns the original bit pattern."""
    flat = W.data.view(INT_DTYPE).view(-1)          # shares memory with W
    old = int(flat[idx])
    mask = (1 << bit) if bit < NBITS - 1 else -(1 << (NBITS - 1))  # sign bit as signed int
    flat[idx] = old ^ mask
    return old

def restore(W, idx, old):
    """Write the original pattern back (do not flip again)."""
    W.data.view(INT_DTYPE).view(-1)[idx] = old

# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, X, y):
    out = model(X)
    finite = bool(torch.isfinite(out).all())
    preds = out.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return finite, preds, acc

def classify(finite, preds, golden_preds):
    """Order matters: non-finite first, because argmax of NaN is arbitrary."""
    if not finite:
        return "non-finite"
    if bool((preds != golden_preds).any()):
        return "SDC"
    return "masked"

# ---------------- Run metadata ----------------
def run_metadata():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return "unknown"
    return {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": git("status", "--porcelain", "--", "src") != "",
        "master_seed": MASTER,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "dtype": str(DTYPE),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
