"""Load each saved network the way a campaign will, and check it is the recorded one.

Usage: python src/check_cnn.py            (checks every model that has a train_*.json)
For each model: verifies the file's SHA-256 against the metadata, then recomputes the
FP32 and FP16 test accuracy and compares them with the recorded values.
"""
import glob, hashlib, json, os
import torch

from data import ROOT, DATASETS, EVAL_BATCH, test_tensors
from models import make_model


def load_trained(arch, data, dtype=torch.float16, device="cuda"):
    """The ONLY way later scripts should load a week-2 network: refuses a changed file."""
    meta = json.load(open(os.path.join(ROOT, "logs", f"train_{arch}_{data}.json")))
    path = os.path.join(ROOT, meta["weights_file"])
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != meta["weights_sha256"]:
        raise RuntimeError(f"{path}: SHA-256 {sha} does not match recorded {meta['weights_sha256']}")
    model = make_model(arch, DATASETS[data][1])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device, dtype).eval(), meta


@torch.no_grad()
def accuracy(model, X, y, dtype):
    correct = 0
    for i in range(0, len(X), EVAL_BATCH):
        out = model(X[i:i + EVAL_BATCH].to("cuda", dtype))
        correct += (out.argmax(1).cpu() == y[i:i + EVAL_BATCH]).sum().item()
    return correct / len(X)


if __name__ == "__main__":
    for meta_path in sorted(glob.glob(os.path.join(ROOT, "logs", "train_*_*.json"))):
        meta = json.load(open(meta_path))
        if "weights_sha256" not in meta:       # skip the week-1 MLP metadata
            continue
        arch, data = meta["model"], meta["dataset"]
        X, y = test_tensors(data)
        row = [f"{arch:9s} {data:8s}  SHA ok"]
        for dtype, key in [(torch.float32, "acc_fp32"), (torch.float16, "acc_fp16")]:
            model, _ = load_trained(arch, data, dtype)
            acc = accuracy(model, X, y, dtype)
            row.append(f"{key} {acc:.4f} ({'matches' if acc == meta[key] else 'DIFFERS from ' + str(meta[key])})")
        print("  ".join(row))
