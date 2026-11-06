"""Step 5: 300 elements (150 per weight matrix) x all bits. Usage: python campaign.py run1"""
import csv, sys
from common import *

tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
N_PER_MATRIX = 150

model = load_fault_model()
X, y = load_test_batch()
X, y = X.to(DEVICE, DTYPE), y.to(DEVICE)
_, golden, acc0 = evaluate(model, X, y)
print(f"Golden accuracy: {acc0:.4f}")

matrices = {"W1": model[1].weight, "W2": model[3].weight}
elements = {
    "W1": sorted(int(i) for i in rng("mat1").choice(matrices["W1"].numel(), N_PER_MATRIX, replace=False)),
    "W2": sorted(int(i) for i in rng("mat2").choice(matrices["W2"].numel(), N_PER_MATRIX, replace=False)),
}
save_json(elements, os.path.join(CONFIG_DIR, "elements.json"))

fields = ["tensor", "element", "bit", "field", "original_value", "new_value", "class", "accuracy"]
csv_path = os.path.join(LOG_DIR, f"campaign_{tag}.csv")
total = sum(len(v) for v in elements.values()) * NBITS
done = 0
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for name, W in matrices.items():
        flat = W.data.view(-1)
        for idx in elements[name]:
            orig = flat[idx].item()
            for bit in range(NBITS):
                fp = fingerprint(W)
                old = flip_bit(W, idx, bit)
                try:
                    new = flat[idx].item()
                    finite, preds, acc = evaluate(model, X, y)
                finally:
                    restore(W, idx, old)
                if fingerprint(W) != fp:
                    sys.exit(f"STOP: restore failed at {name}[{idx}] bit {bit}")
                writer.writerow({"tensor": name, "element": idx, "bit": bit, "field": bit_field(bit),
                                 "original_value": repr(orig), "new_value": repr(new),
                                 "class": classify(finite, preds, golden),
                                 "accuracy": f"{acc:.4f}" if finite else ""})  # blank: NaN argmax is arbitrary
                done += 1
            if done % (20 * NBITS) == 0:
                print(f"{done}/{total} faults")

# Final check: the model is exactly as it started
_, preds, acc1 = evaluate(model, X, y)
assert torch.equal(preds, golden) and acc1 == acc0, "Model changed during campaign"

save_json({"tag": tag, "faults": total, "golden_accuracy": acc0, **run_metadata()},
          os.path.join(LOG_DIR, f"campaign_{tag}_meta.json"))
print(f"Done: {total} faults written to {csv_path}")
