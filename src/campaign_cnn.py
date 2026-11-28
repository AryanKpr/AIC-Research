"""Bit-flip campaign on a trained week 2 network: every bit of N sampled elements per layer.

Usage:
    python src/campaign_cnn.py vgg cifar10 run1            # 100 elements per layer (default)
    python src/campaign_cnn.py vgg cifar10 run1 --n 150    # more elements per layer
    python src/campaign_cnn.py vgg cifar10 run1 --bn       # also fault batch-norm tensors
    python src/campaign_cnn.py vgg cifar10 test --n 2      # quick pipeline test

If Colab disconnects, run the SAME command again: finished faults are skipped and the run
continues where it stopped, so the final file is the same as an uninterrupted run.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse, csv, json, subprocess, sys, time, zlib
import numpy as np
import torch, torch.nn as nn, torchvision
from numpy.random import SeedSequence, default_rng

from check_cnn import load_trained
from data import ROOT, EVAL_BATCH, test_tensors
from faults import flip_bit, restore, fingerprint, bit_field, nbits
from seeds import MASTER, ROLE_SAMPLE, ARCH_IDS, DATA_IDS

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

ap = argparse.ArgumentParser()
ap.add_argument("arch"); ap.add_argument("data"); ap.add_argument("tag")
ap.add_argument("--n", type=int, default=100, help="elements sampled per layer")
ap.add_argument("--bn", action="store_true", help="also fault batch-norm gamma, beta, running mean, running var")
args = ap.parse_args()
assert torch.cuda.is_available(), "Use a GPU runtime"
name = f"{args.arch}_{args.data}"

model, meta = load_trained(args.arch, args.data, torch.float16)   # refuses a file with the wrong SHA-256


# ---------------- Fault targets, in network order, with readable labels ----------------
def fault_targets(model, include_bn):
    targets, n_conv, n_bn = {}, 0, 0
    for mod_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            n_conv += 1
            targets[f"conv{n_conv}"] = (f"{mod_name}.weight", m.weight)
        elif isinstance(m, nn.Linear):
            targets["fc"] = (f"{mod_name}.weight", m.weight)
        elif isinstance(m, nn.BatchNorm2d) and include_bn:
            n_bn += 1
            for short, attr in [("gamma", "weight"), ("beta", "bias"), ("mean", "running_mean"), ("var", "running_var")]:
                targets[f"bn{n_bn}.{short}"] = (f"{mod_name}.{attr}", getattr(m, attr))
    return targets


targets = fault_targets(model, args.bn)


def layer_rng(tensor_name):
    """Own sampling stream per layer: (role 5, architecture, dataset, layer). The layer key
    comes from the tensor's name, so adding or removing other targets never changes it."""
    key = (ROLE_SAMPLE, ARCH_IDS[args.arch], DATA_IDS[args.data], zlib.crc32(tensor_name.encode()))
    return default_rng(SeedSequence(MASTER, spawn_key=key))


elements = {}
for label, (tname, t) in targets.items():
    k = min(args.n, t.numel())
    elements[label] = sorted(int(i) for i in layer_rng(tname).choice(t.numel(), k, replace=False))
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
with open(os.path.join(ROOT, "configs", f"elements_{name}_n{args.n}{'_bn' if args.bn else ''}.json"), "w") as f:
    json.dump({label: {"tensor": targets[label][0], "numel": targets[label][1].numel(), "elements": e}
               for label, e in elements.items()}, f, indent=1)

# ---------------- Test set and fault-free (golden) run ----------------
X, y = test_tensors(args.data)
X, y = X.to("cuda", torch.float16), y.to("cuda")


@torch.no_grad()
def evaluate():
    out = torch.cat([model(X[i:i + EVAL_BATCH]) for i in range(0, len(X), EVAL_BATCH)])
    return bool(torch.isfinite(out).all()), out.argmax(1)


_, golden = evaluate()
acc0 = (golden == y).float().mean().item()
print(f"{name}: golden FP16 accuracy {acc0:.4f} (recorded at training: {meta['acc_fp16']:.4f})")
if abs(acc0 - meta["acc_fp16"]) > 1e-9:
    print("  NOTE: differs from the training record, probably a different GPU type. "
          "Results are still valid; both runs of this campaign must use the same GPU type.")

# ---------------- Sanity check: a flip must change the real network ----------------
label0 = [k for k in targets if k.startswith("conv")][-1]   # last conv layer
t0 = targets[label0][1]
fp = fingerprint(t0)
with torch.no_grad():
    before = model(X[:EVAL_BATCH]).clone()
    old = flip_bit(t0, 0, nbits(t0) - 2)         # top exponent bit
    after = model(X[:EVAL_BATCH]).clone()
    restore(t0, 0, old)
assert not torch.equal(before, after), "Flip did not change outputs: wrong tensor or a copy"
assert fingerprint(t0) == fp, "Restore failed in sanity check"
print(f"Sanity check passed on {label0}: flip changes outputs, restore is exact")

# ---------------- Campaign (resumable) ----------------
FIELDS = ["layer", "tensor", "element", "bit", "field", "original_value", "new_value", "class", "accuracy", "changed"]
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
csv_path = os.path.join(ROOT, "logs", f"campaign_{name}_{args.tag}.csv")
done = set()
if os.path.exists(csv_path):
    with open(csv_path) as f:
        done = {(r["layer"], int(r["element"]), int(r["bit"])) for r in csv.DictReader(f)}
    print(f"Resuming: {len(done)} faults already in {csv_path}")

total = sum(len(e) for e in elements.values()) * 16
count, t_start = len(done), time.time()
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
    for label, (tname, t) in targets.items():
        nb = nbits(t)
        fp = fingerprint(t)
        flat = t.data.view(-1)
        for idx in elements[label]:
            orig = flat[idx].item()
            for bit in range(nb):
                if (label, idx, bit) in done:
                    continue
                old = flip_bit(t, idx, bit)
                try:
                    new = flat[idx].item()
                    finite, preds = evaluate()
                finally:
                    restore(t, idx, old)
                if fingerprint(t) != fp:
                    sys.exit(f"STOP: restore failed at {label}[{idx}] bit {bit}")
                changed = int((preds != golden).sum())
                cls = "non-finite" if not finite else ("SDC" if changed else "masked")
                writer.writerow({
                    "layer": label, "tensor": tname, "element": idx, "bit": bit, "field": bit_field(bit, nb),
                    "original_value": repr(orig), "new_value": repr(new), "class": cls,
                    # argmax of NaN is arbitrary, so accuracy and changed are left blank for non-finite
                    "accuracy": f"{(preds == y).float().mean().item():.4f}" if finite else "",
                    "changed": changed if finite else "",
                })
                count += 1
            f.flush()
            new_done = count - len(done)
            if new_done and new_done % (10 * nb) == 0:
                rate = (time.time() - t_start) / new_done
                print(f"{label:9s} {count}/{total} faults   ~{rate * (total - count) / 60:.0f} min left")

_, preds = evaluate()
assert torch.equal(preds, golden), "Model changed during the campaign"


def git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


with open(os.path.join(ROOT, "logs", f"campaign_{name}_{args.tag}_meta.json"), "w") as f:
    json.dump({"model": args.arch, "dataset": args.data, "tag": args.tag, "faults": total,
               "elements_per_layer": args.n, "batch_norm_targets": args.bn,
               "layers": {k: {"tensor": v[0], "numel": v[1].numel(), "sampled": len(elements[k])}
                          for k, v in targets.items()},
               "golden_acc_fp16": acc0, "weights_sha256": meta["weights_sha256"],
               "master_seed": MASTER, "sampling_role": ROLE_SAMPLE,
               "device": torch.cuda.get_device_name(0),
               "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "numpy": np.__version__, "cuda": torch.version.cuda},
               "commit": git("rev-parse", "HEAD"),
               "uncommitted_changes": git("status", "--porcelain", "--", "src") != ""}, f, indent=2)
print(f"Done: {total} faults in {csv_path}")
