"""Structured (channel) pruning with MAC-array alignment, driven by the virtual-NPU cost model.

A *prunable group* is a set of tensors that must be sliced together:
    ResNet BasicBlock      : conv1.out -> bn1 -> act -> conv2.in           (inner channels)
    MobileNetV2 block      : pw.out -> pw_bn -> dw (groups) -> dw_bn -> proj.in   (expanded channels)
Residual-stream channels are left alone (they are shared by many layers and by the shortcut adds).

Channel importance = |gamma| of the BN that follows the producing conv (Network Slimming, Liu 2017)
times the L1 norm of the consumer's input-channel weights — cheap, data-free, and reasonable for a
short fine-tune afterwards.

Strategies for choosing how many channels to keep per group:
    "uniform"   : keep = round(ratio * C)                  (what FLOP-driven pruning does)
    "aligned"   : keep = align * round(ratio * C / align)  (multiples of the PE array width)
    "cost-greedy": remove `align` channels at a time from the group with the best
                  (cycle saving on the target NPU) / (importance lost) until the cycle budget is met
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
from ..zoo.models import ResNetCIFAR, MobileNetV2CIFAR, BasicBlock, InvertedResidual
from ..graph.ir import trace
from ..npu.cost import estimate
from ..npu.spec import get_spec


@dataclass
class Group:
    name: str
    producer: str            # conv whose output channels are pruned
    bn: str                  # its BatchNorm
    consumers: list[str]     # convs whose input channels are pruned
    dw: str | None = None    # depthwise conv (+bn) in between, if any
    dw_bn: str | None = None
    channels: int = 0


def find_groups(model: nn.Module) -> list[Group]:
    groups = []
    for name, m in model.named_modules():
        if isinstance(m, BasicBlock):
            groups.append(Group(name, f"{name}.conv1", f"{name}.bn1", [f"{name}.conv2"], channels=m.conv1.out_channels))
        elif isinstance(m, InvertedResidual) and m.expand:
            groups.append(Group(name, f"{name}.pw_conv", f"{name}.pw_bn", [f"{name}.proj_conv"], dw=f"{name}.dw_conv",
                                dw_bn=f"{name}.dw_bn", channels=m.pw_conv.out_channels))
    return groups


def _get(model, name):
    return model.get_submodule(name)


def _set(model, name, module):
    *path, leaf = name.split(".")
    parent = model
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, module)


@torch.no_grad()
def importance(model: nn.Module, g: Group) -> torch.Tensor:
    bn = _get(model, g.bn)
    gamma = bn.weight.detach().abs()
    l1 = torch.zeros_like(gamma)
    for c in g.consumers:
        w = _get(model, c).weight.detach()
        l1 += w.abs().sum(dim=(0, 2, 3))
    if g.dw:
        wd = _get(model, g.dw).weight.detach()
        l1 = l1 * wd.abs().sum(dim=(1, 2, 3))
    return gamma * l1


@torch.no_grad()
def apply_keep(model: nn.Module, g: Group, keep: torch.Tensor) -> None:
    """Physically slice the group's modules to the kept channel indices (in place)."""
    keep = keep.sort().values
    conv = _get(model, g.producer)
    new = nn.Conv2d(conv.in_channels, len(keep), conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=conv.bias is not None)
    new.weight.data = conv.weight.data[keep].clone()
    if conv.bias is not None:
        new.bias.data = conv.bias.data[keep].clone()
    new.train(conv.training)
    _set(model, g.producer, new)
    _slice_bn(model, g.bn, keep)
    if g.dw:
        dw = _get(model, g.dw)
        newdw = nn.Conv2d(len(keep), len(keep), dw.kernel_size, dw.stride, dw.padding, dw.dilation, groups=len(keep), bias=dw.bias is not None)
        newdw.weight.data = dw.weight.data[keep].clone()
        if dw.bias is not None:
            newdw.bias.data = dw.bias.data[keep].clone()
        newdw.train(dw.training)
        _set(model, g.dw, newdw)
        _slice_bn(model, g.dw_bn, keep)
    for c in g.consumers:
        conv = _get(model, c)
        new = nn.Conv2d(len(keep), conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=conv.bias is not None)
        new.weight.data = conv.weight.data[:, keep].clone()
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
        new.train(conv.training)
        _set(model, c, new)
    g.channels = len(keep)


def _slice_bn(model, name, keep):
    bn = _get(model, name)
    new = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum)
    new.weight.data = bn.weight.data[keep].clone(); new.bias.data = bn.bias.data[keep].clone()
    new.running_mean = bn.running_mean[keep].clone(); new.running_var = bn.running_var[keep].clone()
    new.train(bn.training)     # a fresh module defaults to training mode; inherit the replaced module's mode
    _set(model, name, new)


def keep_counts(groups: list[Group], ratio: float, strategy: str, align: int = 16, min_channels: int = 8) -> list[int]:
    out = []
    for g in groups:
        if strategy == "uniform":
            k = max(min_channels, int(round(ratio * g.channels)))
        elif strategy == "aligned":
            k = max(align, int(round(ratio * g.channels / align)) * align)
            k = min(k, g.channels)
        else:
            raise ValueError(strategy)
        out.append(min(k, g.channels))
    return out


@torch.no_grad()
def prune(model: nn.Module, ratio: float | None = None, strategy: str = "aligned", align: int = 16,
          keep: list[int] | None = None) -> tuple[nn.Module, list[Group]]:
    """Return (pruned deep copy, groups). Either give `ratio`+`strategy` or explicit per-group `keep` counts."""
    model = copy.deepcopy(model).eval()
    groups = find_groups(model)
    counts = keep if keep is not None else keep_counts(groups, ratio, strategy, align)
    for g, k in zip(groups, counts):
        imp = importance(model, g)
        idx = torch.argsort(imp, descending=True)[:k]
        apply_keep(model, g, idx)
    _update_config(model, groups)
    return model, groups


def _update_config(model, groups):
    if hasattr(model, "config"):
        model.config = dict(model.config, pruned_channels=[g.channels for g in groups])


def rebuild_from_config(config: dict) -> nn.Module:
    """Build a model with the pruned channel counts stored in `config['pruned_channels']`."""
    from ..zoo.models import build_model
    cfg = dict(config); pruned = cfg.pop("pruned_channels", None)
    m = build_model(cfg)
    if pruned:
        groups = find_groups(m)
        for g, k in zip(groups, pruned):
            apply_keep(m, g, torch.arange(k))
        m.config = dict(config)
    return m


def cycles_of(model: nn.Module, spec) -> float:
    return estimate(trace(copy.deepcopy(model).eval()), spec).total_cycles


@torch.no_grad()
def prune_cost_greedy(model: nn.Module, spec, target_ratio: float, align: int = 16, min_channels: int = 16,
                      max_steps: int = 10_000, verbose: bool = False) -> tuple[nn.Module, list[Group], list[dict]]:
    """Greedy: repeatedly drop `align` channels from the group with the largest
    (cycle saving on `spec`) / (importance mass removed) until cycles <= target_ratio * baseline."""
    spec = get_spec(spec)
    base = copy.deepcopy(model).eval()
    groups = find_groups(base)
    counts = [g.channels for g in groups]
    base_cycles = cycles_of(base, spec)
    target = base_cycles * target_ratio
    history = [dict(step=0, cycles=base_cycles, counts=list(counts))]
    imps = [importance(base, g).sort(descending=True).values for g in groups]
    cur_cycles = base_cycles
    for step in range(1, max_steps + 1):
        if cur_cycles <= target:
            break
        best = None
        for gi, g in enumerate(groups):
            if counts[gi] - align < min_channels:
                continue
            trial = list(counts); trial[gi] -= align
            m_trial, _ = prune(base, keep=trial)
            cyc = cycles_of(m_trial, spec)
            saving = cur_cycles - cyc
            lost = float(imps[gi][trial[gi]:counts[gi]].sum()) + 1e-12
            score = saving / lost
            if best is None or score > best[0]:
                best = (score, gi, cyc, saving)
        if best is None:
            break
        _, gi, cyc, saving = best
        counts[gi] -= align; cur_cycles = cyc
        history.append(dict(step=step, group=groups[gi].name, cycles=cyc, saving=saving, counts=list(counts)))
        if verbose:
            print(f"step {step}: -{align}ch from {groups[gi].name} -> cycles {cyc:,.0f} ({cyc/base_cycles*100:.1f}%)")
    pruned, groups = prune(base, keep=counts)
    return pruned, groups, history
