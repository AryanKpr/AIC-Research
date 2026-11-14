"""Week 2 architectures. Both take the number of input channels, so the same code
serves Fashion-MNIST (1 x 28 x 28) and CIFAR-10 (3 x 32 x 32)."""
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Small VGG-style network ----------------
def conv_bn_relu(cin, cout):
    # bias=False because batch normalization directly after the convolution has its own shift
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class SmallVGG(nn.Module):
    """Stage 1: 2 x conv 32, max-pool. Stage 2: 2 x conv 64, max-pool.
    Stage 3: 2 x conv 128. Output: global average pool, linear 128 -> 10.
    No skip connections, no large linear layers."""

    def __init__(self, in_ch, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(in_ch, 32), *conv_bn_relu(32, 32), nn.MaxPool2d(2),
            *conv_bn_relu(32, 64), *conv_bn_relu(64, 64), nn.MaxPool2d(2),
            *conv_bn_relu(64, 128), *conv_bn_relu(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


# ---------------- ResNet-20 (He et al. 2016, CIFAR version) ----------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        # Shortcut "option A" from the paper: when the size changes, subsample by 2 and
        # pad the extra channels with zeros. It has no weights, so every weight is in a
        # convolution, batch norm or the final linear layer.
        self.pad = cout - cin
        self.stride = stride

    def shortcut(self, x):
        if self.stride == 1 and self.pad == 0:
            return x
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """conv 3x3 16, then three stages of 3 basic blocks (16, 32, 64 channels; the
    second and third stages halve the image size), global average pool, linear 64 -> 10."""

    def __init__(self, in_ch, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(3):
                layers.append(BasicBlock(cin, cout, stride if i == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.fc(self.pool(self.layers(x)).flatten(1))


def make_model(arch, in_ch):
    return {"vgg": SmallVGG, "resnet20": ResNet20}[arch](in_ch)


# Expected trainable parameter counts, worked out by hand from the layer sizes.
EXPECTED_PARAMS = {
    ("vgg", 1): 288_170, ("vgg", 3): 288_746,
    ("resnet20", 1): 269_434, ("resnet20", 3): 269_722,
}
