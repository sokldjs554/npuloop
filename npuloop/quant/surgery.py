"""Model surgery passes that change the architecture *before* quantization, plus QAT helpers."""
from __future__ import annotations
import copy
import torch.nn as nn
from ..zoo.models import make_act
from ..graph.ir import ACT_MODULE_KINDS


def swap_activations(model: nn.Module, mapping: dict[str, str]) -> nn.Module:
    """Return a deep copy where every activation module of kind k in `mapping` is replaced by mapping[k]."""
    model = copy.deepcopy(model)
    n = 0
    for name, m in list(model.named_modules()):
        kind = ACT_MODULE_KINDS.get(type(m))
        if kind in mapping:
            *path, leaf = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, leaf, make_act(mapping[kind])); n += 1
    if hasattr(model, "config"):
        model.config = dict(model.config, act=mapping.get(model.config.get("act"), model.config.get("act")))
    model.swapped = n
    return model


def qat(qm, ds, epochs: int = 3, lr: float = 0.01, seed: int = 0, out: str | None = None, freeze_ranges: bool = True, **kw):
    """Quantization-aware fine-tuning of a prepared+calibrated fake-quant GraphModule."""
    from ..zoo.train import fit
    from .prepare import set_mode
    set_mode(qm, enabled=True, w_enabled=True, calibrating=not freeze_ranges)
    log = fit(qm, ds, epochs=epochs, lr=lr, seed=seed, out=out, channels_last=False, warmup_pct=0.1, div_factor=5, **kw)
    set_mode(qm, calibrating=False)
    return log
