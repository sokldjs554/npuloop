from .models import ResNetCIFAR, MobileNetV2CIFAR, build_model, count_params, make_act, ACTS
from .data import CIFAR10NPZ, augment_batch, CIFAR_MEAN, CIFAR_STD

_LAZY = {"fit", "evaluate", "load_checkpoint"}


def __getattr__(name):      # PEP 562: import the trainer on first use, so `python -m npuloop.zoo.train`
    if name in _LAZY:       # does not import it twice (RuntimeWarning) via this package
        from . import train
        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
