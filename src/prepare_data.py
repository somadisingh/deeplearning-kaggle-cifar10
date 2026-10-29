"""Decode the PNGs once into uint8 numpy arrays (N, 32, 32, 3) for fast loading."""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def decode(folder, filenames):
    paths = [os.path.join(ROOT, folder, f) for f in filenames]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        return np.stack(list(ex.map(load, paths, chunksize=512)))


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(ROOT, "train_labels.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    assert train["id"].is_unique and test["id"].is_unique
    x_train = decode("train_images", train["filename"])
    x_test = decode("test_images", test["filename"])
    np.savez(os.path.join(ROOT, "cifar10_cache.npz"),
             x_train=x_train, y_train=train["label"].to_numpy(np.int64),
             x_test=x_test, test_ids=test["id"].to_numpy(np.int64))
    print(x_train.shape, x_test.shape, np.bincount(train["label"]))
    print("train mean", (x_train / 255).mean((0, 1, 2)), "std", (x_train / 255).std((0, 1, 2)))
