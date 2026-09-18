"""The ResNet-50 port: structure always, numerical identity when the weights and TensorFlow are present.

torchvision's Bottleneck is ResNet v1.5 -- stride on the 3x3 -- while Keras's is v1, stride on the first 1x1.
The parameter shapes are identical either way, so loading Keras weights into the wrong layout gives a network
that runs and computes nonsense. The structural test pins the stride placement; the numerical one proves the
whole port against Keras.
"""
import os
import numpy as np
import pytest
import torch

from npuloop.zoo.resnet50_keras import WEIGHTS_PATH, ResNet50Keras, _keras_names, resnet50_keras


def test_layout_is_keras_v1_not_torchvision_v1_5():
    m = ResNet50Keras()
    convs = [mod for mod in m.modules() if isinstance(mod, torch.nn.Conv2d)]
    assert len(convs) == 53, len(convs)
    # Every (conv, bn) pair the loader will fill must line up with a Keras layer name, one for one.
    assert len(m.pairs()) == len(_keras_names()) == 53          # one per convolution, projections included
    strided = [b for b in m.blocks if b.conv1.stride == (2, 2)]
    assert len(strided) == 3, "stages 3-5 start with a stride-2 block"
    for b in strided:
        assert b.conv2.stride == (1, 1), "v1 puts the stride on the 1x1; v1.5 would put it on the 3x3"
        assert b.down is not None and b.down.stride == (2, 2), "the shortcut carries the same stride"
    assert m(torch.zeros(1, 3, 160, 160)).shape == (1, 1000)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Keras ResNet-50 weights not downloaded")
def test_matches_keras_on_the_same_pixels():
    pytest.importorskip("tensorflow")
    from npuloop.zoo.resnet50_keras import check_against_keras
    r = check_against_keras(n=2)
    assert r["top1_agree"] == r["n"]
    assert r["max_abs_prob_diff"] < 1e-4, r


@pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Keras ResNet-50 weights not downloaded")
def test_loaded_weights_are_not_the_initialization():
    """A loader that silently no-ops would still pass the structural test."""
    fresh, loaded = ResNet50Keras(), resnet50_keras()
    assert not torch.allclose(fresh.conv1.weight, loaded.conv1.weight)
    assert float(loaded.bn1.running_var.mean()) != 1.0, "batch-norm statistics were not loaded"
    assert np.isfinite(loaded.fc.weight.detach().numpy()).all()
