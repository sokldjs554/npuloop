"""Structured (channel) pruning driven by the fx graph and priced by the virtual-NPU cost model.

A *prunable group* is a set of tensors that must be sliced together. Groups are discovered on the traced
graph, not by recognising module classes: starting from every conv/linear *producer*, its output channels are
followed through channel-preserving nodes (activations, LUT activations, a depthwise conv) until they are
consumed. The group is prunable when every path ends in a conv/linear that reads those channels as its input
channels — directly, or through a `concat` (in which case the consumer's input-channel slice that belongs to
this branch is what gets pruned). Paths that reach an `add` (residual stream), a pool, an attention matmul,
a LayerNorm or the output are not prunable, so those channels are left alone.

This finds, without any per-architecture code:
    ResNet BasicBlock      conv1 -> act -> conv2                      (inner channels)
    MobileNetV2            pw -> act -> dw -> act -> proj             (expanded channels)  and non-residual block outputs
    Inception              p5_red -> act -> p5_conv  and  branch -> concat -> next block's convs
    Transformer MLP        fc1 -> act -> fc2                          (hidden dimension)

Channel importance = |gamma| of the BN that follows the producer (Network Slimming, Liu 2017; the producer's
own output-row L1 norm when there is no BN) times the L1 norm of the consumers' input-channel weights —
cheap, data-free, reasonable for a short fine-tune afterwards.

Strategies for choosing how many channels to keep per group:
    "uniform"    : keep = round(ratio * C)                  (what FLOP-driven pruning does)
    "aligned"    : keep = align * round(ratio * C / align)  (multiples of the PE array width)
    "cost-greedy": remove `align` channels at a time from the group with the best
                   (cycle saving on the target NPU) / (importance lost) until the cycle budget is met
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from ..graph.ir import trace, StaticGraph, Node
from ..npu.cost import estimate
from ..npu.spec import get_spec


@dataclass
class Consumer:
    module: str                       # conv/linear module whose input channels are sliced
    concat: str | None = None         # name of the concat node the channels pass through, if any
    branches: list[str] = field(default_factory=list)   # producer modules feeding that concat, in concat order
    branch: int = 0                   # this group's position among them


@dataclass
class Group:
    name: str
    producer: str                     # conv/linear module whose output channels are pruned
    bn: str | None                    # its BatchNorm (None for linear producers or convs without BN)
    consumers: list[Consumer]
    dw: str | None = None             # depthwise conv (+bn) in between, if any
    dw_bn: str | None = None
    channels: int = 0
    kind: str = "conv"                # conv | linear
    via: str = "direct"               # direct | concat

    def to_dict(self):
        return dict(name=self.name, producer=self.producer, channels=self.channels, kind=self.kind, via=self.via,
                    consumers=[c.module for c in self.consumers], dw=self.dw)


PASS_THROUGH = {"act", "lut"}         # elementwise ops that keep the channel axis


def _module_of(node: Node) -> str | None:
    return node.attrs.get("module")


def _concat_branches(g: StaticGraph, cat: Node) -> list[str] | None:
    """Producer conv modules feeding a channel concat, in order (None if some input has no conv producer)."""
    out = []
    for src in cat.inputs:
        n = g[src]
        while n.op in PASS_THROUGH:
            n = g[n.inputs[0]]
        if n.op != "conv" or n.attrs.get("depthwise"):
            return None
        out.append(_module_of(n))
    return out


def find_groups(model: nn.Module, input_shape=(3, 32, 32)) -> list[Group]:
    """Discover prunable channel groups from the traced graph (see module docstring)."""
    g = trace(copy.deepcopy(model).eval(), input_shape)
    groups: list[Group] = []
    for p in g.nodes:
        if p.op not in ("conv", "linear") or (p.op == "conv" and p.attrs.get("depthwise")):
            continue
        if p.op == "conv" and p.attrs["groups"] != 1:
            continue
        consumers: list[Consumer] = []; dw: Node | None = None; ok = True; via = "direct"
        stack = [(p, None)]           # (node, concat context)
        seen = set()
        while stack and ok:
            node, ctx = stack.pop()
            for u in g.users(node.name):
                if u.name in seen:
                    continue
                seen.add(u.name)
                if u.op in PASS_THROUGH:
                    stack.append((u, ctx))
                elif u.op == "conv" and u.attrs.get("depthwise") and ctx is None and dw is None:
                    dw = u; stack.append((u, ctx))
                elif u.op in ("conv", "linear") and not (u.op == "conv" and u.attrs["groups"] != 1):
                    consumers.append(Consumer(_module_of(u), *ctx) if ctx else Consumer(_module_of(u)))
                elif u.op == "concat" and ctx is None and int(u.attrs.get("dim", 1)) == 1:
                    branches = _concat_branches(g, u)
                    if branches is None or _module_of(p) not in branches:
                        ok = False; break
                    via = "concat"
                    stack.append((u, (u.name, branches, branches.index(_module_of(p)))))
                else:
                    ok = False; break
        if not ok or not consumers:
            continue
        bn = p.attrs.get("bn_module")
        channels = int(p.attrs["out_channels"] if p.op == "conv" else p.attrs["out_features"])
        groups.append(Group(name=_module_of(p), producer=_module_of(p), bn=bn, consumers=consumers,
                            dw=_module_of(dw) if dw is not None else None,
                            dw_bn=dw.attrs.get("bn_module") if dw is not None else None,
                            channels=channels, kind=p.op, via=via))
    return groups


def _get(model, name):
    return model.get_submodule(name)


def _set(model, name, module):
    *path, leaf = name.split(".")
    parent = model
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, module)


def _offset(model: nn.Module, c: Consumer) -> int:
    """Input-channel offset of this group's branch inside the consumer (0 unless the channels pass a concat)."""
    if c.concat is None:
        return 0
    return sum(_out_width(_get(model, b)) for b in c.branches[:c.branch])


def _out_width(m: nn.Module) -> int:
    return m.out_channels if isinstance(m, nn.Conv2d) else m.out_features


@torch.no_grad()
def importance(model: nn.Module, g: Group) -> torch.Tensor:
    prod = _get(model, g.producer)
    if g.bn:
        gamma = _get(model, g.bn).weight.detach().abs()
    else:
        w = prod.weight.detach()
        gamma = w.abs().reshape(w.shape[0], -1).sum(1) + (prod.bias.detach().abs() if prod.bias is not None else 0)
    l1 = torch.zeros_like(gamma)
    for c in g.consumers:
        w = _get(model, c.module).weight.detach()
        off = _offset(model, c)
        sl = w[:, off:off + g.channels]
        l1 += sl.abs().sum(dim=(0, 2, 3)) if sl.ndim == 4 else sl.abs().sum(dim=0)
    if g.dw:
        wd = _get(model, g.dw).weight.detach()
        l1 = l1 * wd.abs().sum(dim=(1, 2, 3)) * _get(model, g.dw_bn).weight.detach().abs()
    return gamma * l1


def _slice_out(m: nn.Module, keep: torch.Tensor) -> nn.Module:
    if isinstance(m, nn.Conv2d):
        new = nn.Conv2d(m.in_channels, len(keep), m.kernel_size, m.stride, m.padding, m.dilation, m.groups, bias=m.bias is not None)
    else:
        new = nn.Linear(m.in_features, len(keep), bias=m.bias is not None)
    new.weight.data = m.weight.data[keep].clone()
    if m.bias is not None:
        new.bias.data = m.bias.data[keep].clone()
    new.train(m.training)
    return new


def _slice_in(m: nn.Module, keep_abs: torch.Tensor) -> nn.Module:
    """Keep only the input channels `keep_abs` (absolute indices into the consumer's input axis)."""
    if isinstance(m, nn.Conv2d):
        new = nn.Conv2d(len(keep_abs), m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, bias=m.bias is not None)
    else:
        new = nn.Linear(len(keep_abs), m.out_features, bias=m.bias is not None)
    new.weight.data = m.weight.data[:, keep_abs].clone()
    if m.bias is not None:
        new.bias.data = m.bias.data.clone()
    new.train(m.training)
    return new


@torch.no_grad()
def apply_keep(model: nn.Module, g: Group, keep: torch.Tensor) -> None:
    """Physically slice the group's modules to the kept channel indices (in place)."""
    keep = keep.sort().values
    # consumers first: their offsets depend on the producer widths *before* this group shrinks
    for c in g.consumers:
        m = _get(model, c.module)
        off = _offset(model, c)
        n_in = m.in_channels if isinstance(m, nn.Conv2d) else m.in_features
        keep_abs = torch.cat([torch.arange(0, off), keep + off, torch.arange(off + g.channels, n_in)])
        _set(model, c.module, _slice_in(m, keep_abs))
    _set(model, g.producer, _slice_out(_get(model, g.producer), keep))
    if g.bn:
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
          keep: list[int] | None = None, input_shape=(3, 32, 32)) -> tuple[nn.Module, list[Group]]:
    """Return (pruned deep copy, groups). Either give `ratio`+`strategy` or explicit per-group `keep` counts."""
    model = copy.deepcopy(model).eval()
    groups = find_groups(model, input_shape)
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


def rebuild_from_config(config: dict, input_shape=(3, 32, 32)) -> nn.Module:
    """Build a model with the pruned channel counts stored in `config['pruned_channels']` (group order = find_groups)."""
    from ..zoo.models import build_model
    cfg = dict(config); pruned = cfg.pop("pruned_channels", None)
    m = build_model(cfg)
    if pruned:
        groups = find_groups(m, input_shape)
        if len(groups) != len(pruned):
            raise ValueError(f"config lists {len(pruned)} pruned groups but the graph has {len(groups)}")
        for g, k in zip(groups, pruned):
            apply_keep(m, g, torch.arange(k))
        m.config = dict(config)
    return m.eval()


def cycles_of(model: nn.Module, spec, input_shape=(3, 32, 32)) -> float:
    return estimate(trace(copy.deepcopy(model).eval(), input_shape), spec).total_cycles


@torch.no_grad()
def prune_cost_greedy(model: nn.Module, spec, target_ratio: float, align: int = 16, min_channels: int = 16,
                      max_steps: int = 10_000, verbose: bool = False, input_shape=(3, 32, 32)) -> tuple[nn.Module, list[Group], list[dict]]:
    """Greedy: repeatedly drop `align` channels from the group with the largest
    (cycle saving on `spec`) / (importance mass removed) until cycles <= target_ratio * baseline."""
    spec = get_spec(spec)
    base = copy.deepcopy(model).eval()
    groups = find_groups(base, input_shape)
    counts = [g.channels for g in groups]
    base_cycles = cycles_of(base, spec, input_shape)
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
            m_trial, _ = prune(base, keep=trial, input_shape=input_shape)
            cyc = cycles_of(m_trial, spec, input_shape)
            saving = cur_cycles - cyc
            if saving <= 0:
                continue                     # removing these channels would not save a single cycle on this NPU
            lost = float(imps[gi][trial[gi]:counts[gi]].sum()) + 1e-12
            score = saving / lost
            if best is None or score > best[0]:
                best = (score, gi, cyc, saving)
        if best is None:
            break                            # no candidate saves cycles any more (staircase floor reached)
        _, gi, cyc, saving = best
        counts[gi] -= align; cur_cycles = cyc
        history.append(dict(step=step, group=groups[gi].name, cycles=cyc, saving=saving, counts=list(counts)))
        if verbose:
            print(f"step {step}: -{align}ch from {groups[gi].name} -> cycles {cyc:,.0f} ({cyc/base_cycles*100:.1f}%)")
    pruned, groups = prune(base, keep=counts, input_shape=input_shape)
    history[-1]["target_reached"] = bool(cur_cycles <= target)
    history[-1]["achieved_ratio"] = cur_cycles / base_cycles
    return pruned, groups, history
