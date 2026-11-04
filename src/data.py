"""Data loading, the holdout split and the training augmentation (all on the GPU except TrivialAugment).

batch_crop and batch_cutout are adapted from cifar10-airbench by Keller Jordan (MIT License); see
THIRD_PARTY_NOTICES.md.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = os.path.join(ROOT, "data", "cifar10_cache.npz")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def normalize(x_uint8, device):
    """(N, H, W, C) or (N, C, H, W) uint8 tensor -> normalized float (N, C, H, W) on device."""
    x = x_uint8.to(device).float().div(255)
    return (x - MEAN.to(device)) / STD.to(device)


def load_data(device):
    """Normalized train images and labels, normalized test images and test ids (from src/prepare_data.py)."""
    d = np.load(CACHE)
    x_train = normalize(torch.from_numpy(d["x_train"]).permute(0, 3, 1, 2), device)
    x_test = normalize(torch.from_numpy(d["x_test"]).permute(0, 3, 1, 2), device)
    return x_train, torch.from_numpy(d["y_train"]).to(device), x_test, d["test_ids"]


def split(y, val, seed=0):
    """Stratified holdout of `val` training images (val // 10 per class), fixed by seed. val = 0: no holdout."""
    if val == 0:
        return np.arange(len(y)), np.array([], dtype=np.int64)
    rng = np.random.default_rng(1000 + seed)
    val_idx = np.concatenate([rng.permutation(np.where(y == c)[0])[: val // 10] for c in range(10)])
    return np.setdiff1d(np.arange(len(y)), val_idx), np.sort(val_idx)


def batch_crop(padded, size):
    """Random translation: one random crop of `size` per image from the padded batch."""
    r = (padded.size(-1) - size) // 2
    n = len(padded)
    shifts = torch.randint(-r, r + 1, size=(n, 2), device=padded.device)
    tmp = torch.empty((n, 3, size, size + 2 * r), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 0] == s
        tmp[m] = padded[m, :, r + s : r + s + size, :]
    out = torch.empty((n, 3, size, size), device=padded.device, dtype=padded.dtype)
    for s in range(-r, r + 1):
        m = shifts[:, 1] == s
        out[m] = tmp[m, :, :, r + s : r + s + size]
    return out


def batch_cutout(x, size):
    """Cutout (DeVries and Taylor, 2017): set one random size x size square of each image to zero."""
    n, _, h, w = x.shape
    cy = torch.randint(0, h - size + 1, (n,), device=x.device).view(-1, 1, 1, 1)
    cx = torch.randint(0, w - size + 1, (n,), device=x.device).view(-1, 1, 1, 1)
    ys = torch.arange(h, device=x.device).view(1, 1, h, 1) - cy
    xs = torch.arange(w, device=x.device).view(1, 1, 1, w) - cx
    mask = (ys >= 0) & (ys < size) & (xs >= 0) & (xs < size)
    return x.masked_fill(mask, 0)


_TA_IMAGES = None


def _ta_init(images):
    global _TA_IMAGES
    torch.set_num_threads(1)
    _TA_IMAGES = images


def _ta_chunk(start, end, seed):
    from torchvision.transforms import v2
    torch.manual_seed(seed)
    ta = v2.TrivialAugmentWide()
    return torch.stack([ta(img) for img in _TA_IMAGES[start:end]]).numpy()


class TrivialAugment:
    """TrivialAugmentWide (Muller and Hutter, 2021) per image on CPU worker processes. The next epoch is
    calculated in the background while the GPU trains on the current epoch. Each chunk of 1,000 images has its own
    seed, so the result does not depend on the number of workers."""

    def __init__(self, images_uint8, device, seed, chunk=1000):
        from concurrent.futures import ProcessPoolExecutor
        if hasattr(os, "sched_getaffinity"):
            workers = max(1, len(os.sched_getaffinity(0)) - 1)
        else:
            workers = max(1, os.cpu_count() - 2)
        self.n, self.device, self.seed, self.chunk = len(images_uint8), device, seed, chunk
        self.pool = ProcessPoolExecutor(workers, initializer=_ta_init, initargs=(images_uint8,))
        self.epoch = 0
        self.pending = self._submit()

    def _submit(self):
        s = self.seed * 100_000 + self.epoch * 1000
        return [self.pool.submit(_ta_chunk, i, min(i + self.chunk, self.n), s + k)
                for k, i in enumerate(range(0, self.n, self.chunk))]

    def seek(self, epoch):
        """After a resume: drop the queued epoch and queue `epoch` instead."""
        for f in self.pending:
            f.cancel()
        self.epoch = epoch
        self.pending = self._submit()

    def next(self):
        x = np.concatenate([f.result() for f in self.pending])
        self.epoch += 1
        self.pending = self._submit()
        return normalize(torch.from_numpy(x), self.device)

    def close(self):
        self.pool.shutdown(cancel_futures=True)


class Loader:
    """One augmented epoch per iteration: TrivialAugment, random flip, 4-pixel reflect-pad translation, cutout,
    then shuffled batches (the last incomplete batch is dropped)."""

    def __init__(self, source, labels, batch_size, translate=4, cutout=12):
        self.source, self.labels, self.bs = source, labels, batch_size
        self.translate, self.cutout = translate, cutout
        # Not used. This draw keeps the random stream equal to the runs in docs/LOGBOOK.md.
        torch.rand(len(labels), device=labels.device)

    def __len__(self):
        return len(self.labels) // self.bs

    def __iter__(self):
        x = self.source.next()
        flip = (torch.rand(len(x), device=x.device) < 0.5).view(-1, 1, 1, 1)
        x = torch.where(flip, x.flip(-1), x)
        x = batch_crop(F.pad(x, (self.translate,) * 4, mode="reflect"), x.shape[-1])
        x = batch_cutout(x, self.cutout)
        idx = torch.randperm(len(x), device=x.device)
        for i in range(len(self)):
            j = idx[i * self.bs : (i + 1) * self.bs]
            yield x[j], self.labels[j]
