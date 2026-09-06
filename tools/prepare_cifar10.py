"""Build data/cifar10.npz from either the original python batches (cifar-10-batches-py/) or the
fast.ai PNG mirror (cifar10/{train,test}/<class>/*.png). The npz holds uint8 NHWC arrays.

python tools/prepare_cifar10.py --src /path/to/cifar-10-batches-py --out data/cifar10.npz
python tools/prepare_cifar10.py --src /path/to/cifar10            --out data/cifar10.npz
"""
import argparse, os, pickle
import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def from_batches(src):
    def load(f):
        d = pickle.load(open(os.path.join(src, f), "rb"), encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); y = np.array(d[b"labels"], dtype=np.int64)
        return x, y
    xs, ys = zip(*[load(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = load("test_batch")
    return np.concatenate(xs), np.concatenate(ys), xte, yte


def from_png(src):
    from PIL import Image
    def load(split):
        X, Y = [], []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(src, split, c)
            for f in sorted(os.listdir(d)):
                X.append(np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.uint8)); Y.append(ci)
        return np.stack(X), np.array(Y, dtype=np.int64)
    xtr, ytr = load("train"); xte, yte = load("test")
    return xtr, ytr, xte, yte


def main():
    p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", default="data/cifar10.npz")
    a = p.parse_args()
    if os.path.exists(os.path.join(a.src, "data_batch_1")):
        xtr, ytr, xte, yte = from_batches(a.src)
    else:
        xtr, ytr, xte, yte = from_png(a.src)
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, classes=np.array(CLASSES))
    print(f"wrote {a.out}: train {xtr.shape} test {xte.shape} class counts {np.bincount(ytr).tolist()}")


if __name__ == "__main__":
    main()
