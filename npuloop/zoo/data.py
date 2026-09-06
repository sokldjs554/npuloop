"""CIFAR-10 loading from a single .npz (no torchvision download needed).

Augmentation is implemented in NumPy on uint8 arrays (random crop with 4px padding
+ horizontal flip), which is faster than a DataLoader with worker processes on a
small CPU box.
"""
from __future__ import annotations
import numpy as np
import torch

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


class CIFAR10NPZ:
    def __init__(self, path: str):
        d = np.load(path)
        self.x_train = d["x_train"]  # (N,32,32,3) uint8
        self.y_train = d["y_train"]
        self.x_test = d["x_test"]
        self.y_test = d["y_test"]
        self.classes = [str(c) for c in d["classes"]]

    @staticmethod
    def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
        x = x_uint8.astype(np.float32) / 255.0
        x = (x - CIFAR_MEAN) / CIFAR_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))

    def train_batches(self, batch_size: int, rng: np.random.Generator, augment: bool = True):
        n = len(self.x_train)
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb = self.x_train[idx]
            if augment:
                xb = augment_batch(xb, rng)
            yield self.to_tensor(xb), torch.from_numpy(self.y_train[idx])

    def test_batches(self, batch_size: int = 500):
        for i in range(0, len(self.x_test), batch_size):
            yield self.to_tensor(self.x_test[i:i + batch_size]), torch.from_numpy(self.y_test[i:i + batch_size])

    def calib_batch(self, n: int = 512, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.x_train), size=n, replace=False)
        return self.to_tensor(self.x_train[idx])


def augment_batch(xb: np.ndarray, rng: np.random.Generator, pad: int = 4) -> np.ndarray:
    n, h, w, c = xb.shape
    padded = np.zeros((n, h + 2 * pad, w + 2 * pad, c), dtype=xb.dtype)
    padded[:, pad:pad + h, pad:pad + w] = xb
    ox = rng.integers(0, 2 * pad + 1, size=n)
    oy = rng.integers(0, 2 * pad + 1, size=n)
    flip = rng.random(n) < 0.5
    out = np.empty_like(xb)
    for i in range(n):
        crop = padded[i, oy[i]:oy[i] + h, ox[i]:ox[i] + w]
        out[i] = crop[:, ::-1] if flip[i] else crop
    return out
