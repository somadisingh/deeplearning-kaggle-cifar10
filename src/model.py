"""Pre-activation ResNet for 32x32 CIFAR-10 with an SLA joint (class x rotation) head.

The network is a standard CIFAR pre-activation ResNet (He et al., 2016): a 3x3 stem, three stages of basic blocks
(stride 2 between stages), BN-ReLU, global average pooling and a linear classifier. SLA (Lee et al., 2020) adds a
second linear head on the same features with 10 x 4 outputs: label y * 4 + k for an image rotated by k * 90 degrees.
"""
import torch.nn.functional as F
from torch import nn

PARAM_LIMIT = 5_000_000


class PreActBlock(nn.Module):
    """BN-ReLU-conv3x3-BN-ReLU-conv3x3, plus an identity skip or a 1x1 projection when the shape changes."""

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
    """Widths 64/128/256 and 4/4/3 blocks: 4,696,394 parameters with the 10-class head, 4,706,674 with the SLA
    head. forward() gives the 10-class logits; joint_head is used in training and in the rotation vote."""

    def __init__(self, widths=(64, 128, 256), depth=(4, 4, 3), num_classes=10):
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
        self.joint_head = nn.Linear(cin, num_classes * 4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def features(self, x):
        x = F.relu(self.bn(self.blocks(self.stem(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
