"""Step 3: test flip_bit by hand, then sanity-check it on the real network."""
import math, torch
from common import *

# ---- Hand tests on single numbers ----
if NBITS == 16:
    cases = [(0.5, 14, 32768.0), (1.0, 14, math.inf), (0.5, 0, 0.50048828125), (0.5, 15, -0.5)]
    top_exp = 14
else:
    cases = [(0.5, 30, 0.5 * 2.0**128), (1.0, 30, math.inf), (0.5, 0, 0.5 * (1 + 2.0**-23)), (0.5, 31, -0.5)]
    top_exp = 30

for value, bit, expected in cases:
    t = torch.tensor([value], dtype=DTYPE)
    old = flip_bit(t, 0, bit)
    got = t.item()
    restore(t, 0, old)
    ok = got == expected and t.item() == value
    print(f"{value} bit {bit:2d} -> {got!r:>24}  expected {expected!r}  {'OK' if ok else 'FAIL'}")
    assert ok

# ---- Sanity check on the network ----
model = load_fault_model()
X, y = load_test_batch()
X, y = X.to(DEVICE, DTYPE), y.to(DEVICE)
_, golden, acc0 = evaluate(model, X, y)
W = model[1].weight

out_before = model(X).detach().clone()
fp_before = fingerprint(W)
old = flip_bit(W, 0, top_exp)
print("weight[0] after flip:", W.view(-1)[0].item())
out_after = model(X).detach().clone()
restore(W, 0, old)

changed = not torch.equal(out_before, out_after)
print("Outputs changed after flip:", changed)
assert changed, "Outputs did not change: you probably flipped a copy or the wrong tensor"
assert fingerprint(W) == fp_before, "Restore failed: fingerprint differs"
_, preds, acc1 = evaluate(model, X, y)
assert torch.equal(preds, golden) and acc1 == acc0
print("Restore verified: fingerprint and predictions match. All tests passed.")
