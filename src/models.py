"""Residual CNNs for 32x32 CIFAR-10, all under 5M parameters.

`PreActResNet` is a standard pre-activation ResNet with average pooling.
`AirResNet` follows the airbench96 design (Jordan, 2024): a frozen 2x2 patch-whitening stem,
then three stages of conv -> maxpool -> BN -> GELU followed by residual pairs of 3x3 convs.
`depth` sets how many residual pairs each stage gets.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BatchNorm(nn.BatchNorm2d):
    # Scale is fixed to 1 (only the bias trains), as in airbench.
    def __init__(self, num_features, eps=1e-12, weight=False):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight


class Conv(nn.Conv2d):
    def __init__(self, cin, cout, k=3, padding="same", bias=False):
        super().__init__(cin, cout, kernel_size=k, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ResPair(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1, self.norm1 = Conv(ch, ch), BatchNorm(ch)
        self.conv2, self.norm2 = Conv(ch, ch), BatchNorm(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class Stage(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        self.conv = Conv(cin, cout)
        self.pool = nn.MaxPool2d(2)
        self.norm = BatchNorm(cout)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(*[ResPair(cout) for _ in range(depth)])

    def forward(self, x):
        x = self.act(self.norm(self.pool(self.conv(x))))
        return self.blocks(x)


class AirResNet(nn.Module):
    """`ssl` adds a rotation self-supervision head on the shared features:
    "sla" a joint (class x rotation) head with num_classes * 4 outputs, label y * 4 + k for an image
    rotated by k * 90 degrees (Lee et al., 2020); "rot" a 4-way rotation head (Hendrycks et al., 2019).
    `head` stays the plain num_classes classifier, so forward() is unchanged."""

    def __init__(self, widths=(128, 256, 384), depth=(1, 1, 1), scale=1 / 9, num_classes=10, ssl="none"):
        super().__init__()
        ws = 2 * 3 * 2 * 2
        self.whiten = Conv(3, ws, k=2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        stages, cin = [], ws
        for w, d in zip(widths, depth):
            stages.append(Stage(cin, w, d))
            cin = w
        self.act = nn.GELU()
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(cin, num_classes, bias=False)
        self.scale = scale
        self.ssl = ssl
        if ssl == "sla":
            self.joint_head = nn.Linear(cin, num_classes * 4, bias=False)
        elif ssl == "rot":
            self.rot_head = nn.Linear(cin, 4, bias=False)

    def features(self, x):
        return self.pool(self.stages(self.act(self.whiten(x)))).flatten(1)

    def forward(self, x):
        return self.head(self.features(x)) * self.scale


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def init_whitening(layer, images, eps=5e-4):
    """Set the stem to a ZCA-like whitening of 2x2 patches from (normalized) training images."""
    h, w = layer.weight.shape[2:]
    c = images.shape[1]
    patches = images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    flat = patches.view(len(patches), -1)
    cov = (flat.T @ flat) / len(flat)
    evals, evecs = torch.linalg.eigh(cov.cpu())
    evals, evecs = evals.flip(0).view(-1, 1, 1, 1), evecs.T.reshape(c * h * w, c, h, w).flip(0)
    scaled = evecs / torch.sqrt(evals + eps)
    layer.weight.data[:] = torch.cat((scaled, -scaled)).to(layer.weight.device)


class PreActBlock(nn.Module):
    """Pre-activation basic block (He et al., 2016b): BN-ReLU-conv-BN-ReLU-conv, plus an identity or
    1x1 projection skip connection."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = self.skip(out) if self.skip is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + skip


class PreActResNet(nn.Module):
    """Standard CIFAR pre-activation ResNet: 3x3 stem, stages of PreActBlocks (stride 2 between stages),
    global average pooling and a linear classifier. Same `ssl` heads as AirResNet."""

    def __init__(self, widths=(64, 128, 256), depth=(4, 4, 3), num_classes=10, ssl="none"):
        super().__init__()
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        blocks, cin = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depth)):
            for j in range(d):
                blocks.append(PreActBlock(cin, w, 2 if (i > 0 and j == 0) else 1))
                cin = w
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin)
        self.head = nn.Linear(cin, num_classes)
        self.scale = 1.0
        self.ssl = ssl
        if ssl == "sla":
            self.joint_head = nn.Linear(cin, num_classes * 4)
        elif ssl == "rot":
            self.rot_head = nn.Linear(cin, 4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def features(self, x):
        x = F.relu(self.bn(self.blocks(self.stem(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x)) * self.scale
