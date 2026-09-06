"""BatchNorm folding as an fx graph transformation (conv -> bn becomes conv with bias)."""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.fx as fx
from ..graph.ir import fold_bn_params


def fold_bn(model: nn.Module) -> fx.GraphModule:
    """Return a traced copy of `model` (eval mode) with every Conv2d->BatchNorm2d pair folded."""
    model = copy.deepcopy(model).eval()
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(modules[node.target], nn.BatchNorm2d):
            continue
        prev = node.args[0]
        if not (isinstance(prev, fx.Node) and prev.op == "call_module" and isinstance(modules[prev.target], nn.Conv2d)):
            raise ValueError(f"BatchNorm {node.target} does not directly follow a conv")
        if len(prev.users) != 1:
            raise ValueError(f"conv {prev.target} has several users; cannot fold BN {node.target}")
        conv, bn = modules[prev.target], modules[node.target]
        w, b = fold_bn_params(conv.weight, conv.bias, bn)
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                             conv.dilation, conv.groups, bias=True)
        new_conv.weight.data.copy_(w); new_conv.bias.data.copy_(b)
        _set_module(gm, prev.target, new_conv)
        node.replace_all_uses_with(prev)
        gm.graph.erase_node(node)
        _set_module(gm, node.target, nn.Identity())
    gm.graph.lint(); gm.recompile()
    gm.delete_all_unused_submodules()
    return gm


def _set_module(gm: fx.GraphModule, target: str, module: nn.Module):
    *path, name = target.split(".")
    parent = gm
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, name, module)
