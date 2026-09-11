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


@pytest.mark.skipif(not os.path.exists(DATA), reason="CIFAR-10 npz not available")
def test_validation_split_is_stratified_disjoint_and_fixed():
    from npuloop.zoo import CIFAR10NPZ
    ds = CIFAR10NPZ(DATA)
    assert len(ds.x_train) == 45000 and len(ds.x_val) == 5000 and len(ds.x_test) == 10000
    assert np.bincount(ds.y_val).tolist() == [500] * 10                    # 500 held-out images per class
    assert np.bincount(ds.y_train).tolist() == [4500] * 10
    assert len(set(ds.y_val[:200].tolist())) == 10                          # prefix is class-mixed
    again = CIFAR10NPZ(DATA)
    assert np.array_equal(again.val_idx, ds.val_idx)                       # same hold-out every time
    # disjoint: no validation image appears in the training split (compare raw bytes of a sample)
    train_keys = {ds.x_train[i].tobytes() for i in range(0, 45000, 9)}
    assert sum(ds.x_val[i].tobytes() in train_keys for i in range(500)) == 0
    val_batches = list(ds.val_batches(1000))
    assert len(val_batches) == 5 and val_batches[0][0].shape == (1000, 3, 32, 32)


class _TinyDataset:
    """A CIFAR10NPZ look-alike with random uint8 images, small enough for a trainer test to run in seconds."""

    def __init__(self, n_train=256, n_val=64, n_test=64, seed=0):
        from npuloop.zoo import CIFAR10NPZ
        rng = np.random.default_rng(seed)
        mk = lambda n: (rng.integers(0, 255, (n, 32, 32, 3), dtype=np.uint8), rng.integers(0, 10, n))
        self.x_train, self.y_train = mk(n_train)
        self.x_val, self.y_val = mk(n_val)
        self.x_test, self.y_test = mk(n_test)
        from npuloop.zoo.data import CIFAR_MEAN, CIFAR_STD
        self.mean, self.std, self.pad = CIFAR_MEAN, CIFAR_STD, 4
        self.to_tensor = CIFAR10NPZ.to_tensor.__get__(self)
        self.batches = CIFAR10NPZ.batches.__get__(self)
        self.train_batches = CIFAR10NPZ.train_batches.__get__(self)
        self.test_batches = CIFAR10NPZ.test_batches.__get__(self)
        self.val_batches = CIFAR10NPZ.val_batches.__get__(self)


def test_fit_selects_best_checkpoint_on_validation_split(tmp_path):
    import json, torch
    from npuloop.zoo import fit, build_model, load_checkpoint
    from npuloop.zoo.train import evaluate
    ds = _TinyDataset()
    # width 16 on purpose: the 1x1 stride-2 shortcut convs then have 16 and 32 input channels, a multiple of both
    # oneDNN channel blocks (8 on AVX2, 16 on AVX-512). torch 2.14.0 (oneDNN 3.12) writes past the rtus workspace
    # in 1x1 backward_weights when a channels_last, strided conv has fewer input channels than the block: an
    # endless spin on the AMD EPYC CI runners (runs 12-16) or a segfault. Report, validated patch and reproducer:
    # docs/upstream/ and tools/onednn_1x1_repro.py. ONEDNN_MAX_CPU_ISA=AVX2 reproduces the AVX2 case on any CPU.
    model = build_model(dict(arch="resnet", depth=8, width=16, act="relu"))
    log = fit(model, ds, epochs=3, lr=0.05, bs=32, seed=0, out=str(tmp_path))
    assert [e["epoch"] for e in log["epochs"]] == [1, 2, 3]
    assert all("val_acc" in e and "test_acc" not in e for e in log["epochs"])      # test never scored per epoch
    vals = [e["val_acc"] for e in log["epochs"]]
    sel = log["selected_epoch"]
    assert log["best_val_acc"] == max(vals) and vals[sel - 1] == max(vals)
    assert sel == max(i + 1 for i, v in enumerate(vals) if v == max(vals))        # ties -> later epoch
    assert log["splits"] == dict(train=256, val=64, test=64)
    # best.pt really is the selected epoch: its test accuracy is what the log reports for it
    best = load_checkpoint(str(tmp_path / "best.pt"))
    assert evaluate(best, ds) == log["test_acc"]
    last = load_checkpoint(str(tmp_path / "last.pt"))
    assert evaluate(last, ds) == log["final_test_acc"]
    assert json.load(open(tmp_path / "log.json"))["selection"].startswith("best.pt = highest val_acc")
    # and the caller keeps the final-epoch weights
    for (k, a), (_, b) in zip(model.state_dict().items(), last.state_dict().items()):
        assert torch.equal(a.contiguous(), b.contiguous()), k


def test_augment_batch_shapes_and_flip():
    from npuloop.zoo import augment_batch
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (16, 32, 32, 3), dtype=np.uint8)
    y = augment_batch(x, rng)
    assert y.shape == x.shape and y.dtype == np.uint8
