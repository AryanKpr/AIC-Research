"""Step 4: pick one weight with |w| < 1 from the first matrix and flip each of its bits."""
import csv
from common import *

model = load_fault_model()
X, y = load_test_batch()
X, y = X.to(DEVICE, DTYPE), y.to(DEVICE)
_, golden, acc0 = evaluate(model, X, y)
W = model[1].weight
flat = W.data.view(-1)

r = rng("step4")
while True:
    idx = int(r.integers(W.numel()))
    if abs(flat[idx].item()) < 1:
        break
orig = flat[idx].item()
print(f"Chosen element {idx}, value {orig!r}, golden accuracy {acc0:.4f}\n")

rows = []
fp = fingerprint(W)
for bit in reversed(range(NBITS)):
    old = flip_bit(W, idx, bit)
    try:
        new = flat[idx].item()
        finite, preds, acc = evaluate(model, X, y)
    finally:
        restore(W, idx, old)
    assert fingerprint(W) == fp, "Restore failed"
    cls = classify(finite, preds, golden)
    rows.append({"bit": bit, "field": bit_field(bit), "new_value": repr(new), "class": cls,
                 "accuracy": f"{acc:.4f}" if finite else ""})
    print(f"bit {bit:2d} {bit_field(bit):9s} new value {new!r:>22}  {cls:10s} {rows[-1]['accuracy']}")

path = os.path.join(LOG_DIR, "step4.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
save_json({"element": idx, "original_value": orig, **run_metadata()}, os.path.join(LOG_DIR, "step4_meta.json"))
print("\nSaved", path)
