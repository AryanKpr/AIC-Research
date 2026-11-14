## Week 2: small VGG on Fashion-MNIST and CIFAR-10

Our team uses the small VGG-style network: three stages of two 3×3 convolutions
(32, 64, 128 filters), each followed by batch normalization and ReLU, with 2×2 max-pooling
after the first two stages, then global average pooling and a linear layer 128 → 10.
Over 99% of its weights are in convolutions.

### Seeds
Same master seed (20261104). Week 2 adds two roles, keyed by (role, architecture, dataset):
role 4 is the training seed (initial weights and batch order), role 5 the sampling seed for
the fault element draw. Roles 0–3 belong to the week 1 MLP and are unchanged.

### Reproducibility
GPU training is not bit-for-bit repeatable, so each network was trained once. The saved
weights and their SHA-256 are the reference; later scripts load them through
`load_trained()`, which refuses a file whose hash has changed. Weights are kept outside git.

### Trained networks
| | Fashion-MNIST | CIFAR-10 |
|---|---|---|
| Input | 1 × 28 × 28 | 3 × 32 × 32 |
| Parameters | 288,170 | 288,746 |
| Training seed | 1938730673 | 1358819710 |
| Epochs | 15 | 30 |
| Augmentation | none | random crop (pad 4) + horizontal flip |
| Normalization mean | 0.2860 | 0.4914, 0.4822, 0.4465 |
| Normalization std | 0.3530 | 0.2470, 0.2435, 0.2616 |
| FP32 test accuracy | 93.60% | 89.30% |
| FP16 test accuracy | 93.61% | 89.30% |
| Weights SHA-256 | `TODO` | `TODO` |

Training: SGD (lr 0.1, momentum 0.9, weight decay 5e-4), batch 128, cosine learning-rate
decay. Full metadata: `logs/train_vgg_fmnist.json`, `logs/train_vgg_cifar10.json`.
Trained at commit `TODO`.

### Reproduce
```bash
python src/train_cnn.py vgg fmnist
python src/train_cnn.py vgg cifar10
python src/check_cnn.py      # verifies the SHA-256 and both accuracies
```
Retraining gives slightly different weights; to reproduce our results exactly, use the
saved weight files whose SHA-256 values are listed above.