"""Week 2: train one architecture on one dataset, save the weights once, record everything.

Usage:  python src/train_cnn.py vgg fmnist
        python src/train_cnn.py vgg cifar10
        python src/train_cnn.py resnet20 fmnist
        python src/train_cnn.py resnet20 cifar10

GPU training is not bit-for-bit repeatable, so each (architecture, dataset) is trained ONCE.
The saved file and its SHA-256 are the reference; every later campaign loads that file.
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np
import torch, torch.nn as nn, torchvision

from data import ROOT, DATASETS, EVAL_BATCH, train_loader, test_tensors, transforms_for
from models import make_model, EXPECTED_PARAMS
from seeds import MASTER, ROLE_TRAIN, ROLE_SAMPLE, ARCH_IDS, DATA_IDS, training_seed

arch, data = sys.argv[1], sys.argv[2]
assert arch in ("vgg", "resnet20") and data in DATASETS, "usage: train_cnn.py {vgg|resnet20} {fmnist|cifar10}"
name = f"{arch}_{data}"
DEVICE = "cuda"
assert torch.cuda.is_available(), "Use a GPU runtime (Runtime -> Change runtime type -> T4 GPU)"

# ---------------- Training settings ----------------
EPOCHS = 30 if data == "cifar10" else 15
BATCH = 128
LR, MOMENTUM, WEIGHT_DECAY = 0.1, 0.9, 5e-4     # SGD with cosine learning-rate decay
AUGMENT = data == "cifar10"                     # random crop + horizontal flip for CIFAR-10 only

seed = training_seed(arch, data)
torch.manual_seed(seed)                         # initial weights
in_ch = DATASETS[data][1]

model = make_model(arch, in_ch).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_conv = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
print(f"{name}: {n_params:,} parameters ({100 * n_conv / n_params:.1f}% in convolutions), training seed {seed}")
assert n_params == EXPECTED_PARAMS[(arch, in_ch)], f"expected {EXPECTED_PARAMS[(arch, in_ch)]:,} parameters"

loader = train_loader(data, AUGMENT, BATCH, seed)  # batch order from the same training seed
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
loss_fn = nn.CrossEntropyLoss()

t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        sched.step()
        loss_sum += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    print(f"epoch {epoch + 1:2d}/{EPOCHS}  loss {loss_sum / total:.4f}  train acc {correct / total:.4f}")
train_minutes = (time.time() - t0) / 60

# ---------------- Save once, record the SHA-256 ----------------
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
path = os.path.join(ROOT, "models", f"{name}_fp32.pt")
state = {k: v.detach().cpu() for k, v in model.state_dict().items()}  # includes BN running stats
torch.save(state, path)
file_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(f"Saved {path}\nSHA-256 {file_sha}")


# ---------------- Accuracy in FP32 and FP16, from the saved file ----------------
@torch.no_grad()
def accuracy(m, X, y, dtype):
    m = m.to(DEVICE, dtype).eval()
    correct = 0
    for i in range(0, len(X), EVAL_BATCH):
        out = m(X[i:i + EVAL_BATCH].to(DEVICE, dtype))
        correct += (out.argmax(1).cpu() == y[i:i + EVAL_BATCH]).sum().item()
    return correct / len(X)


X, y = test_tensors(data)
loaded = make_model(arch, in_ch)
loaded.load_state_dict(torch.load(path, map_location="cpu"))
acc32 = accuracy(loaded, X, y, torch.float32)
acc16 = accuracy(loaded, X, y, torch.float16)       # same weights, converted to FP16
print(f"FP32 test accuracy {acc32:.4f}   FP16 {acc16:.4f}   difference {acc16 - acc32:+.4f}")


# ---------------- Run metadata ----------------
def git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


_, _, mean, std = transforms_for(data, AUGMENT)
meta = {
    "model": arch, "dataset": data,
    "dataset_detail": f"torchvision {DATASETS[data][0].__name__}, {in_ch} channel(s), "
                      f"{DATASETS[data][2]}x{DATASETS[data][2]}, 10,000 test images",
    "normalization": {"mean": mean, "std": std, "computed_from": "training set, float64, rounded to 4 decimals"},
    "loader": {"batch_size": BATCH, "shuffle": True, "num_workers": 2, "drop_last": False,
               "augmentation": "RandomCrop(pad 4) + RandomHorizontalFlip" if AUGMENT else "none",
               "eval_batch_size": EVAL_BATCH},
    "training": {"epochs": EPOCHS, "optimizer": "SGD", "lr": LR, "momentum": MOMENTUM,
                 "weight_decay": WEIGHT_DECAY, "schedule": "cosine, per step", "minutes": round(train_minutes, 1)},
    "seeds": {"master": MASTER,
              "training_seed": seed, "training_spawn_key": [ROLE_TRAIN, ARCH_IDS[arch], DATA_IDS[data]],
              "sampling_spawn_key": [ROLE_SAMPLE, ARCH_IDS[arch], DATA_IDS[data]]},
    "parameters": n_params, "conv_weight_share": round(n_conv / n_params, 4),
    "weights_file": f"models/{name}_fp32.pt", "weights_sha256": file_sha,
    "acc_fp32": acc32, "acc_fp16": acc16,
    "device": torch.cuda.get_device_name(0),
    "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                 "torchvision": torchvision.__version__, "numpy": np.__version__,
                 "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
    "commit": git("rev-parse", "HEAD"),
    "uncommitted_changes": git("status", "--porcelain", "--", "src") != "",
}
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
meta_path = os.path.join(ROOT, "logs", f"train_{name}.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print("Saved", meta_path)
