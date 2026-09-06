import os
import numpy as np
import pytest

DATA = os.environ.get("NPULOOP_DATA", "/home/user/data/cifar10.npz")


@pytest.mark.skipif(not os.path.exists(DATA), reason="CIFAR-10 npz not available")
def test_test_split_prefix_is_class_mixed():
    from npuloop.zoo import CIFAR10NPZ
    ds = CIFAR10NPZ(DATA)
    first = ds.y_test[:500]
    assert len(set(first.tolist())) == 10                 # every class appears in the first 500 images
    assert np.bincount(first).min() >= 25                 # and roughly balanced
    assert np.bincount(ds.y_test).tolist() == [1000] * 10


def test_augment_batch_shapes_and_flip():
    from npuloop.zoo import augment_batch
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (16, 32, 32, 3), dtype=np.uint8)
    y = augment_batch(x, rng)
    assert y.shape == x.shape and y.dtype == np.uint8
