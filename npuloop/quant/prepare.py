"""Insert NPU-style fake quantization into an fx graph and calibrate it.

Quantization points mirror an INT8 NPU datapath exactly:
  * the network input
  * every conv/linear/add output ...
      - if its single consumer is a ReLU-family activation: quantize *after* the activation
        (ReLU is fused into the requantization clamp; no separate tensor exists)
      - otherwise: quantize the op output itself (pre-activation / residual-branch tensor)
  * every non-ReLU activation output (the int8 LUT result)
  * global average pool output, tied to its input's scale/zero-point (TFLite AveragePool semantics)
"""
from __future__ import annotations
import operator
from typing import Iterable
import torch
import torch.nn as nn
import torch.fx as fx
from ..graph.ir import ACT_MODULE_KINDS, RELU_FAMILY
from .scheme import QScheme
from .fold import fold_bn, _set_module
from .fake import FakeQuantAct, QConv2d, QLinear


def prepare(model: nn.Module, scheme: QScheme = QScheme()) -> fx.GraphModule:
    """Fold BN, swap conv/linear for fake-quant versions, insert activation quantizers. Returns GraphModule."""
    gm = fold_bn(model)
    modules = dict(gm.named_modules())
    q_kwargs = dict(w_bits=scheme.w_bits, per_channel=scheme.w_per_channel, w_method=scheme.w_method,
                    learnable=scheme.learnable, pow2=scheme.pow2)
    # 1. weight quantizers
    for node in gm.graph.nodes:
        if node.op == "call_module":
            m = modules[node.target]
            if isinstance(m, nn.Conv2d) and not isinstance(m, QConv2d):
                _set_module(gm, node.target, QConv2d.from_conv(m, **q_kwargs))
            elif isinstance(m, nn.Linear) and not isinstance(m, QLinear):
                _set_module(gm, node.target, QLinear.from_linear(m, **q_kwargs))
    modules = dict(gm.named_modules())

    def act_kind(node: fx.Node):
        if node.op == "call_module" and type(modules[node.target]) in ACT_MODULE_KINDS:
            return ACT_MODULE_KINDS[type(modules[node.target])]
        return None

    def is_compute(node: fx.Node):
        return (node.op == "call_module" and isinstance(modules[node.target], (QConv2d, QLinear))) or \
               (node.op == "call_function" and node.target in (operator.add, torch.add))

    def new_fq(name: str) -> str:
        fq = FakeQuantAct(scheme.a_bits, scheme.a_symmetric, scheme.a_observer, scheme.ema, scheme.learnable, scheme.pow2, name=name)
        attr = f"fq_{name}"
        gm.add_submodule(attr, fq)
        return attr

    def insert_after(node: fx.Node, attr: str) -> fx.Node:
        with gm.graph.inserting_after(node):
            q = gm.graph.call_module(attr, args=(node,))
        node.replace_all_uses_with(q)
        q.args = (node,)
        return q

    # 2. activation quantizers
    for node in list(gm.graph.nodes):
        if node.op == "placeholder":
            insert_after(node, new_fq("input"))
        elif is_compute(node):
            users = list(node.users)
            if len(users) == 1 and act_kind(users[0]) in RELU_FAMILY:
                continue   # fused: the quantizer goes after the activation (handled below)
            insert_after(node, new_fq(node.name))
        elif act_kind(node) is not None:
            insert_after(node, new_fq(node.name))
        elif node.op == "call_module" and isinstance(modules[node.target], nn.AdaptiveAvgPool2d):
            src = node.args[0]
            while src.op == "call_module" and isinstance(gm.get_submodule(src.target), (nn.Identity, nn.Dropout)):
                src = src.args[0]                       # look through pass-through modules
            q = insert_after(node, new_fq(node.name))
            src_fq = gm.get_submodule(src.target) if src.op == "call_module" else None
            if isinstance(src_fq, FakeQuantAct):
                gm.get_submodule(q.target).tied_to = src_fq
            else:
                raise ValueError(f"pool input {src.name} is not a quantizer; cannot tie pool scale")
    gm.graph.lint(); gm.recompile()
    return gm


def quantizers(gm: fx.GraphModule) -> list[FakeQuantAct]:
    return [m for m in gm.modules() if isinstance(m, FakeQuantAct)]


def qlayers(gm: fx.GraphModule) -> list[QConv2d | QLinear]:
    return [m for m in gm.modules() if isinstance(m, (QConv2d, QLinear))]


def set_mode(gm: fx.GraphModule, calibrating: bool | None = None, enabled: bool | None = None, w_enabled: bool | None = None):
    for fq in quantizers(gm):
        if calibrating is not None:
            fq.calibrating = calibrating
        if enabled is not None:
            fq.enabled = enabled
    for ql in qlayers(gm):
        if w_enabled is not None:
            ql.w_enabled = w_enabled


@torch.no_grad()
def calibrate(gm: fx.GraphModule, batches: Iterable[torch.Tensor], weights: bool = True) -> fx.GraphModule:
    """PTQ calibration: init weight scales, observe activation ranges with float weights+acts, then enable."""
    gm.eval()
    if weights:
        for ql in qlayers(gm):
            ql.init_w_scale()
    for fq in quantizers(gm):
        fq.observer.reset() if hasattr(fq.observer, "reset") else None
    set_mode(gm, calibrating=True, enabled=False, w_enabled=False)
    for xb in batches:
        gm(xb)
    set_mode(gm, calibrating=False)
    for fq in quantizers(gm):
        fq.compute_qparams()
    set_mode(gm, enabled=True, w_enabled=True)
    return gm


@torch.no_grad()
def calibrate_sequential(gm: fx.GraphModule, batches: list[torch.Tensor]) -> fx.GraphModule:
    """Calibrate activation ranges *with quantization already applied upstream* (layer-by-layer).

    The plain `calibrate` observes float activations; on a real toolchain the ranges are often
    collected while earlier layers are already quantized, which shifts later ranges slightly.
    """
    gm.eval()
    for ql in qlayers(gm):
        ql.init_w_scale()
    fqs = [m for m in gm.modules() if isinstance(m, FakeQuantAct)]
    for fq in fqs:
        fq.observer.reset() if hasattr(fq.observer, "reset") else None
        fq.enabled = False; fq.calibrating = False
    set_mode(gm, w_enabled=True)
    # graph order == module registration order in our insertion pass, so iterate in that order
    order = [gm.get_submodule(n.target) for n in gm.graph.nodes if n.op == "call_module" and isinstance(gm.get_submodule(n.target), FakeQuantAct)]
    for fq in order:
        fq.calibrating = True
        for xb in batches:
            gm(xb)
        fq.calibrating = False
        fq.compute_qparams()
        fq.enabled = True
    return gm


@torch.no_grad()
def evaluate(gm: nn.Module, ds, batch_size: int = 500, channels_last: bool = False, limit: int | None = None) -> float:
    """Top-1 accuracy on the test split (optionally only the first `limit` images)."""
    gm.eval()
    correct = 0; total = 0
    for xb, yb in ds.test_batches(batch_size):
        if channels_last:
            xb = xb.to(memory_format=torch.channels_last)
        correct += (gm(xb).argmax(1) == yb).sum().item(); total += len(yb)
        if limit and total >= limit:
            break
    return correct / total


def describe(gm: fx.GraphModule) -> str:
    rows = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            m = gm.get_submodule(node.target)
            if isinstance(m, FakeQuantAct):
                s, z = m.qparams()
                rows.append(f"{node.target:36s} act  scale={s:.6f} zp={z}" + ("  (tied)" if m.tied_to is not None else ""))
            elif isinstance(m, (QConv2d, QLinear)):
                rows.append(f"{node.target:36s} w    scale[min/max]={m.w_scale.min():.6f}/{m.w_scale.max():.6f}")
    return "\n".join(rows)
