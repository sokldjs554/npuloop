"""ResNet-50 with ImageNet weights, in torch, laid out exactly like keras.applications.ResNet50.

Why not torchvision: torchvision's Bottleneck is ResNet **v1.5** -- it puts the stride on the 3x3 -- while
Keras's is v1, with the stride on the first 1x1. The two have identical parameter shapes, so loading Keras
weights into torchvision produces a network that runs without error and computes the wrong thing. This module
mirrors the Keras graph so the weights mean what they meant, and ``check_against_keras`` proves it by running
both and comparing logits.

The weights are Keras's own file (``storage.googleapis.com``); nothing here re-trains or fine-tunes. They come
with Keras's preprocessing, which is the ``caffe`` mode: RGB -> BGR, then subtract a per-channel mean, no
scaling. ``preprocess`` applies it so a dataset built for this model matches what the weights expect.

    m = resnet50_keras()                    # 25.6M parameters, 53 convolutions, 1000-way head
    x = preprocess(uint8_rgb_nhwc)          # -> float32 NCHW, Keras 'caffe' preprocessing
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

WEIGHTS_URL = ("https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
               "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
WEIGHTS_PATH = os.path.expanduser("~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)
STAGES = ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2))       # filters, blocks, stride of block 1


class Block(nn.Module):
    """Keras ``resnet.block1``: stride on the first 1x1, and on the shortcut when there is one."""

    def __init__(self, c_in: int, filters: int, stride: int, projection: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, filters, 1, stride)
        self.bn1 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.conv3 = nn.Conv2d(filters, 4 * filters, 1)
        self.bn3 = nn.BatchNorm2d(4 * filters, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=False)
        if projection:
            self.down = nn.Conv2d(c_in, 4 * filters, 1, stride)
            self.down_bn = nn.BatchNorm2d(4 * filters, eps=1.001e-5)
        else:
            self.down = self.down_bn = None

    def pairs(self):
        """(conv, bn) in Keras order: the projection shortcut first, then the 1x1/3x3/1x1."""
        head = [(self.down, self.down_bn)] if self.down is not None else []
        return head + [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]

    def forward(self, x):
        s = self.down_bn(self.down(x)) if self.down is not None else x
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + s)


class ResNet50Keras(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # Keras pads explicitly (ZeroPadding2D(3) then a 'valid' 7x7/2); torch's padding=3 is the same map.
        self.conv1 = nn.Conv2d(3, 64, 7, 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1.001e-5)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        blocks, c = [], 64
        for filters, n, stride in STAGES:
            for i in range(n):
                blocks.append(Block(c, filters, stride if i == 0 else 1, projection=(i == 0)))
                c = 4 * filters
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def pairs(self):
        """Every (conv, bn) in the order _keras_names() lists them. Not modules(): a block registers its
        projection shortcut last, while Keras names it first, and the shapes line up either way for long
        enough to load a wrong network."""
        out = [(self.conv1, self.bn1)]
        for b in self.blocks:
            out += b.pairs()
        return out

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.blocks(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def _keras_names():
    """(conv, bn) Keras layer names in this module's parameter order."""
    names = [("conv1_conv", "conv1_bn")]
    for stage, (_, n, _) in enumerate(STAGES, start=2):
        for i in range(1, n + 1):
            p = f"conv{stage}_block{i}"
            if i == 1:
                names.append((f"{p}_0_conv", f"{p}_0_bn"))
            names += [(f"{p}_{k}_conv", f"{p}_{k}_bn") for k in (1, 2, 3)]
    return names


def load_keras_weights(model: nn.Module, path: str = WEIGHTS_PATH) -> nn.Module:
    """Copy the Keras HDF5 weights in. Kernels are (H,W,in,out) there and (out,in,H,W) here."""
    import h5py
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; fetch {WEIGHTS_URL} (Keras caches it at this path)")
    f = h5py.File(path, "r")

    def get(layer, leaf):
        g = f[layer]
        hits = []
        g.visititems(lambda n, o: hits.append(o) if isinstance(o, h5py.Dataset) and n.endswith(leaf) else None)
        if len(hits) != 1:
            raise KeyError(f"{layer}/{leaf}: {len(hits)} matches")
        return np.array(hits[0])

    pairs, names = model.pairs(), _keras_names()
    if len(pairs) != len(names):
        raise RuntimeError(f"{len(pairs)} conv/bn pairs but {len(names)} Keras names")
    with torch.no_grad():
        for (conv, bn), (cn, bnn) in zip(pairs, names):
            k = get(cn, "kernel:0")
            want = tuple(conv.weight.shape)
            got = (k.shape[3], k.shape[2], k.shape[0], k.shape[1])
            if want != got:
                raise RuntimeError(f"{cn}: Keras kernel {k.shape} -> {got}, but this layer wants {want}")
            conv.weight.copy_(torch.from_numpy(k.transpose(3, 2, 0, 1).copy()))
            conv.bias.copy_(torch.from_numpy(get(cn, "bias:0")))
            bn.weight.copy_(torch.from_numpy(get(bnn, "gamma:0")))
            bn.bias.copy_(torch.from_numpy(get(bnn, "beta:0")))
            bn.running_mean.copy_(torch.from_numpy(get(bnn, "moving_mean:0")))
            bn.running_var.copy_(torch.from_numpy(get(bnn, "moving_variance:0")))
        model.fc.weight.copy_(torch.from_numpy(get("probs", "kernel:0").T.copy()))
        model.fc.bias.copy_(torch.from_numpy(get("probs", "bias:0")))
    f.close()
    return model


def resnet50_keras(pretrained: bool = True, path: str = WEIGHTS_PATH) -> nn.Module:
    m = ResNet50Keras()
    if pretrained:
        load_keras_weights(m, path)
    return m.eval()


def preprocess(images_uint8_nhwc: np.ndarray) -> torch.Tensor:
    """Keras 'caffe' preprocessing: RGB -> BGR, subtract the per-channel mean, no scaling. Returns NCHW."""
    x = np.asarray(images_uint8_nhwc, dtype=np.float32)[..., ::-1]              # RGB -> BGR
    x = x - np.array(CAFFE_MEAN_BGR, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def check_against_keras(n: int = 2, size: int = 224, seed: int = 0) -> dict:
    """Run the Keras model and this one on the same pixels and report the largest logit difference."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    px = rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    keras_model = tf.keras.applications.ResNet50(weights="imagenet")
    ref = keras_model.predict(tf.keras.applications.resnet50.preprocess_input(px.astype("float32")), verbose=0)
    with torch.no_grad():
        mine = torch.softmax(resnet50_keras()(preprocess(px)), dim=1).numpy()
    return {"max_abs_prob_diff": float(np.abs(ref - mine).max()),
            "top1_agree": int((ref.argmax(1) == mine.argmax(1)).sum()), "n": n}
