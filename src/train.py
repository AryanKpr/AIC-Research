"""Steps 1-2: train the 784-128-10 MLP, save weights, compare FP32 vs study-precision accuracy."""
import torch, torch.nn as nn
from torchvision import datasets
from common import *

EPOCHS, BATCH, LR = 5, 128, 1e-3

seed = int(rng("train").integers(2**31))
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed)

train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TFM)
loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=gen)

model = make_model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
n_bias = sum(p.numel() for n, p in model.named_parameters() if "bias" in n)
print(f"Parameters: {n_params} (biases: {n_bias})")
assert n_params == 101_770 and n_bias == 138, "Wrong architecture"

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print(f"epoch {epoch + 1}/{EPOCHS}  last batch loss {loss.item():.4f}")

torch.save(model.cpu().state_dict(), MODEL_PATH)
print("Saved", MODEL_PATH)

X, y = load_test_batch()
model32 = make_model(); model32.load_state_dict(torch.load(MODEL_PATH)); model32 = model32.to(DEVICE).eval()
_, _, acc32 = evaluate(model32, X.to(DEVICE), y.to(DEVICE))
model_low = load_fault_model()
_, _, acc_low = evaluate(model_low, X.to(DEVICE, DTYPE), y.to(DEVICE))
print(f"FP32 test accuracy: {acc32:.4f}")
print(f"{DTYPE} test accuracy: {acc_low:.4f}  (difference {acc_low - acc32:+.4f})")

save_json({"train_seed": seed, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
           "acc_fp32": acc32, "acc_study_precision": acc_low, **run_metadata()},
          os.path.join(LOG_DIR, "train_meta.json"))
