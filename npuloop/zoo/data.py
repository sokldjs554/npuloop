"""CIFAR-10 loading from a single .npz (no torchvision download needed).

Splits: the 50,000 official training images are divided once, with a fixed seed, into a
45,000-image training split and a 5,000-image stratified validation split (500 per class).
Checkpoint selection and every training-time decision use the validation split; the
10,000-image test split is evaluated once, for the selected checkpoint.

Augmentation is implemented in NumPy on uint8 arrays (random crop with 4px padding
+ horizontal flip), which is faster than a DataLoader with worker processes on a
small CPU box.
"""
from __future__ import annotations
import numpy as np
import torch

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
VAL_SEED = 2024          # fixed: the same 5,000 images are held out in every run, every experiment
VAL_PER_CLASS = 500


class CIFAR10NPZ:
    """Image classification .npz: x_train/y_train/x_test/y_test (+ optional mean/std/pad/val_per_class).

    Despite the name it serves any file with this layout (tools/prepare_imagenette.py writes one at 128px);
    per-channel normalization constants and the crop padding come from the file when present.
    """

    def __init__(self, path: str, val_per_class: int | None = None, val_seed: int = VAL_SEED):
        d = np.load(path)
        x_all, y_all = d["x_train"], d["y_train"]     # (N,H,W,3) uint8, (N,)
        self.mean = d["mean"].astype(np.float32) if "mean" in d else CIFAR_MEAN
        self.std = d["std"].astype(np.float32) if "std" in d else CIFAR_STD
        self.pad = int(d["pad"]) if "pad" in d else 4
        if val_per_class is None:
            val_per_class = int(d["val_per_class"]) if "val_per_class" in d else VAL_PER_CLASS
        self.val_per_class = val_per_class
        self.img_size = int(x_all.shape[1])
        self.val_idx = stratified_holdout(y_all, val_per_class, val_seed)
        keep = np.ones(len(y_all), dtype=bool); keep[self.val_idx] = False
        self.x_train = x_all[keep]  # e.g. (45000,32,32,3) uint8 for CIFAR-10
        self.y_train = y_all[keep]
        self.x_val = x_all[self.val_idx]
        self.y_val = y_all[self.val_idx]
        # The npz may be stored class-sorted (e.g. built from per-class image folders); shuffle the test split with a
        # fixed permutation so that any prefix (`limit=N` evaluations, agreement batches) is a class-mixed sample.
        order = np.random.default_rng(1234).permutation(len(d["y_test"]))
        self.x_test = d["x_test"][order]
        self.y_test = d["y_test"][order]
        self.test_order = order
        self.classes = [str(c) for c in d["classes"]]

    def batches(self, split: str, batch_size: int = 500):
        """Un-augmented, deterministic batches of the 'val' or 'test' split."""
        x, y = (self.x_val, self.y_val) if split == "val" else (self.x_test, self.y_test)
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        for i in range(0, len(x), batch_size):
            yield self.to_tensor(x[i:i + batch_size]), torch.from_numpy(y[i:i + batch_size])

    def to_tensor(self, x_uint8: np.ndarray) -> torch.Tensor:
        x = x_uint8.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))

    def train_batches(self, batch_size: int, rng: np.random.Generator, augment: bool = True):
        n = len(self.x_train)
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb = self.x_train[idx]
            if augment:
                xb = augment_batch(xb, rng, pad=self.pad)
            yield self.to_tensor(xb), torch.from_numpy(self.y_train[idx])

    def test_batches(self, batch_size: int = 500):
        return self.batches("test", batch_size)

    def val_batches(self, batch_size: int = 500):
        return self.batches("val", batch_size)

    def calib_batch(self, n: int = 512, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.x_train), size=n, replace=False)
        return self.to_tensor(self.x_train[idx])


def stratified_holdout(y: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Indices of a class-balanced hold-out set (`per_class` images of every class), shuffled with the same seed
    so that any prefix of the validation split is class-mixed."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=per_class, replace=False) for c in np.unique(y)])
    return idx[rng.permutation(len(idx))]


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
