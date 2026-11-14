"""Seeds for week 2. Same master seed as the MNIST pilot (common.py).

Roles 0-3 are used by the pilot and stay untouched. Week 2 adds:
  role 4 = training seed (initial weights and batch order), one per model + dataset
  role 5 = sampling seed for the fault element draw (reserved for the campaign)
Each stream is keyed by (role, architecture, dataset), so the four trainings and the
four future draws are all independent of each other.
"""
from numpy.random import SeedSequence, default_rng

MASTER = 20261104
ARCH_IDS = {"vgg": 0, "resnet20": 1}
DATA_IDS = {"fmnist": 0, "cifar10": 1}
ROLE_TRAIN = 4
ROLE_SAMPLE = 5


def stream(role, arch, data):
    return default_rng(SeedSequence(MASTER, spawn_key=(role, ARCH_IDS[arch], DATA_IDS[data])))


def training_seed(arch, data):
    """The integer used for torch.manual_seed and the batch-order generator."""
    return int(stream(ROLE_TRAIN, arch, data).integers(2**31))


def sampling_rng(arch, data):
    """Random stream for drawing fault elements (used by the campaign, not by training)."""
    return stream(ROLE_SAMPLE, arch, data)
