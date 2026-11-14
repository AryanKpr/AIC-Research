"""Week 2 datasets. Normalization values are computed from the TRAINING set in float64,
rounded to 4 decimals, and recorded in the run metadata."""
import os
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DATASETS = {
    # name: (torchvision class, input channels, image size)
    "fmnist": (datasets.FashionMNIST, 1, 28),
    "cifar10": (datasets.CIFAR10, 3, 32),
}

EVAL_BATCH = 1000   # fixed test batch size, so evaluation order never varies


def normalization(name):
    """Per-channel mean and std of the training images, scaled to 0-1."""
    cls = DATASETS[name][0]
    raw = cls(DATA_DIR, train=True, download=True).data
    x = np.asarray(raw, dtype=np.float64) / 255.0
    axes = (0, 1, 2) if x.ndim == 4 else None        # CIFAR: (N, H, W, C); FMNIST: (N, H, W)
    mean = np.atleast_1d(x.mean(axis=axes))
    std = np.atleast_1d(x.std(axis=axes))
    return [round(float(m), 4) for m in mean], [round(float(s), 4) for s in std]


def transforms_for(name, augment):
    mean, std = normalization(name)
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    size = DATASETS[name][2]
    train_aug = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    return transforms.Compose(train_aug + norm), transforms.Compose(norm), mean, std


def train_loader(name, augment, batch_size, seed):
    train_tf, _, _, _ = transforms_for(name, augment)
    ds = DATASETS[name][0](DATA_DIR, train=True, download=True, transform=train_tf)
    gen = torch.Generator().manual_seed(seed)          # batch order comes from the training seed
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=2, generator=gen, drop_last=False)


def test_tensors(name):
    """The full 10,000-image test set, preprocessed once, in a fixed order."""
    _, test_tf, _, _ = transforms_for(name, augment=False)
    ds = DATASETS[name][0](DATA_DIR, train=False, download=True, transform=test_tf)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    return X, y
