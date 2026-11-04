"""Prediction: test-time augmentation, the SLA rotation vote and the head + vote combination.

The prediction for an image is 0.5 x softmax(head) + 0.5 x softmax(rotation vote). Both parts use the same TTA:
the image and its mirror, plus the two diagonal 1-pixel shifts and their mirrors. The TTA is adapted from
cifar10-airbench by Keller Jordan (MIT License); see THIRD_PARTY_NOTICES.md.

Usage: python src/predict.py <model.pt> <out.csv>   (writes the submission for the 2,000 test images)
"""
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from data import ROOT, load_data
from model import PreActResNet


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(path, dev):
    net = PreActResNet().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    return net.eval()


def rotate4(x):
    """The 4 rotations of each image, interleaved: row 4 * i + k is image i rotated by k * 90 degrees."""
    return torch.stack([torch.rot90(x, k, (2, 3)) for k in range(4)], 1).flatten(0, 1)


def rotation_vote(net, x):
    """SLA aggregation (Lee et al., 2020, eq. 4): for the copy rotated by k, keep the joint logits of the labels
    (class, k), then average them over the 4 rotations."""
    joint = net.joint_head(net.features(rotate4(x))).view(len(x), 4, 10, 4)
    return torch.stack([joint[:, k, :, k] for k in range(4)]).mean(0)


@torch.no_grad()
def tta_logits(net, images, fn=None, bs=2000):
    """Logits averaged over the mirror and the two diagonal 1-pixel shifts. fn(net, x) replaces the forward pass."""
    net.eval()
    fn = fn or (lambda m, x: m(x))

    def mirror(x):
        return 0.5 * fn(net, x) + 0.5 * fn(net, x.flip(-1))

    out = []
    for x in images.split(bs):
        p = F.pad(x, (1,) * 4, mode="reflect")
        shifted = 0.5 * (mirror(p[:, :, :32, :32]) + mirror(p[:, :, 2:, 2:]))
        out.append(0.5 * mirror(x) + 0.5 * shifted)
    return torch.cat(out).float()


def predict_probs(net, images):
    """Head + vote probabilities (N, 10) on the CPU."""
    head = tta_logits(net, images).cpu()
    vote = tta_logits(net, images, rotation_vote).cpu()
    return 0.5 * head.softmax(1) + 0.5 * vote.softmax(1)


def write_submission(labels, path):
    ids = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))["id"]
    assert len(labels) == len(ids)
    pd.DataFrame({"id": ids, "label": labels}).to_csv(path, index=False)


if __name__ == "__main__":
    dev = device()
    net = load_model(sys.argv[1], dev)
    _, _, x_test, _ = load_data(dev)
    write_submission(predict_probs(net, x_test).argmax(1).numpy(), sys.argv[2])
    print("wrote", sys.argv[2])
