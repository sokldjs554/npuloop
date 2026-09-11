"""Imagenette (10 ImageNet classes, fast.ai) -> one .npz with the same layout as the CIFAR-10 file.

    python tools/prepare_imagenette.py --src data/imagenette2-160 --out data/imagenette128.npz --size 128

The official `val` folder becomes the test split (never used for selection); the trainer holds out its own
validation split from `train`. Images are resized so the shorter side is `size` and center-cropped to size x size.
Per-channel mean/std and the augmentation padding are stored in the file so the loader needs no constants.
"""
import argparse, os
import numpy as np
from PIL import Image

NAMES = {"n01440764": "tench", "n02102040": "English springer", "n02979186": "cassette player", "n03000684": "chain saw",
         "n03028079": "church", "n03394916": "French horn", "n03417042": "garbage truck", "n03425413": "gas pump",
         "n03445777": "golf ball", "n03888257": "parachute"}


def class_folders(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_split(root, size, classes):
    xs, ys = [], []
    for ci, wnid in enumerate(classes):
        d = os.path.join(root, wnid)
        for fn in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, fn)).convert("RGB")
            w, h = im.size; s = size / min(w, h)
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
            w, h = im.size; left, top = (w - size) // 2, (h - size) // 2
            xs.append(np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)); ys.append(ci)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/imagenette2-160"); p.add_argument("--out", default="data/imagenette128.npz")
    p.add_argument("--size", type=int, default=128)
    a = p.parse_args()
    classes = class_folders(os.path.join(a.src, "train"))
    xtr, ytr = load_split(os.path.join(a.src, "train"), a.size, classes)
    xte, yte = load_split(os.path.join(a.src, "val"), a.size, classes)
    mean = (xtr.astype(np.float64) / 255).mean(axis=(0, 1, 2)); std = (xtr.astype(np.float64) / 255).std(axis=(0, 1, 2))
    np.savez_compressed(a.out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, classes=np.array([NAMES.get(c, c) for c in classes]),
                        mean=mean.astype(np.float32), std=std.astype(np.float32), pad=np.int64(a.size // 8),
                        val_per_class=np.int64(100), source="imagenette2-160 (fast.ai), shorter side -> %d, center crop" % a.size)
    print(f"train {xtr.shape} test {xte.shape} mean {mean.round(4)} std {std.round(4)} -> {a.out}")


if __name__ == "__main__":
    main()
