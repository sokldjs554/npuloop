import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)


@pytest.fixture(scope="session")
def small_resnet():
    """Tiny, deterministic ResNet with randomized BN statistics (so BN folding is non-trivial)."""
    from npuloop.zoo import ResNetCIFAR
    torch.manual_seed(0)
    m = ResNetCIFAR(depth=8, width=8, act="relu")
    _randomize_bn(m)
    return m.eval()


@pytest.fixture(scope="session")
def small_silu_resnet():
    from npuloop.zoo import ResNetCIFAR
    torch.manual_seed(1)
    m = ResNetCIFAR(depth=8, width=8, act="silu")
    _randomize_bn(m)
    return m.eval()


@pytest.fixture(scope="session")
def small_mobilenet():
    from npuloop.zoo import MobileNetV2CIFAR
    torch.manual_seed(2)
    m = MobileNetV2CIFAR(width_mult=0.35, act="relu6", setting=[[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 2, 2]], last_channels=128)
    _randomize_bn(m)
    return m.eval()


def _randomize_bn(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.3, 0.3)


@pytest.fixture(scope="session")
def batch():
    torch.manual_seed(0)
    return torch.randn(8, 3, 32, 32)


@pytest.fixture(scope="session")
def calib_batches():
    torch.manual_seed(3)
    return [torch.randn(16, 3, 32, 32) for _ in range(2)]
