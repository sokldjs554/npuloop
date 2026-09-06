"""Static graph IR extracted from an nn.Module with torch.fx.

Every downstream tool (cost model, lint, quantizer export, integer engine) consumes this
representation instead of poking at PyTorch modules. BatchNorm is folded into the preceding
conv here, because the NPU never sees a BN layer.

Supported ops: input, conv (incl. depthwise / grouped), linear, add, act, pool (global avg),
flatten, output. Anything else raises UnsupportedOpError with the fx node for diagnosis.
"""
from __future__ import annotations
import operator
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

RELU_FAMILY = {"relu", "relu6"}         # monotone, zero-preserving -> fusable into requant clamp
LUT_ACTS = {"silu", "gelu", "hswish", "lrelu", "sigmoid", "tanh", "hsigmoid"}  # int8 LUT on NPU

ACT_MODULE_KINDS = {
    nn.ReLU: "relu", nn.ReLU6: "relu6", nn.SiLU: "silu", nn.GELU: "gelu", nn.Hardswish: "hswish",
    nn.LeakyReLU: "lrelu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh", nn.Hardsigmoid: "hsigmoid",
}


class UnsupportedOpError(Exception):
    pass


@dataclass
class Node:
    name: str
    op: str                       # input|conv|linear|add|act|pool|flatten|output
    inputs: list[str]
    out_shape: tuple[int, ...]    # excludes batch dim
    attrs: dict[str, Any] = field(default_factory=dict)
    # conv/linear: weight (np.float32, BN-folded), bias (np.float32, BN-folded)
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None

    @property
    def macs(self) -> int:
        if self.op == "conv":
            cout, cin_g, kh, kw = self.weight.shape
            _, h, w = self.out_shape
            return int(cout * cin_g * kh * kw * h * w)
        if self.op == "linear":
            return int(self.weight.shape[0] * self.weight.shape[1])
        return 0

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.out_shape))


@dataclass
class StaticGraph:
    nodes: list[Node]
    input_shape: tuple[int, ...]

    def __getitem__(self, name: str) -> Node:
        return self.by_name[name]

    @property
    def by_name(self) -> dict[str, Node]:
        return {n.name: n for n in self.nodes}

    def users(self, name: str) -> list[Node]:
        return [n for n in self.nodes if name in n.inputs]

    def compute_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.op in ("conv", "linear")]

    @property
    def total_macs(self) -> int:
        return sum(n.macs for n in self.nodes)

    @property
    def total_params(self) -> int:
        return sum(int(n.weight.size + (n.bias.size if n.bias is not None else 0)) for n in self.compute_nodes())

    def summary(self) -> str:
        rows = [f"{'name':28s} {'op':8s} {'out_shape':16s} {'MACs':>12s}  attrs"]
        for n in self.nodes:
            a = {k: v for k, v in n.attrs.items() if k not in ("module",)}
            rows.append(f"{n.name:28s} {n.op:8s} {str(n.out_shape):16s} {n.macs:12,d}  {a}")
        rows.append(f"total MACs={self.total_macs:,} params={self.total_params:,}")
        return "\n".join(rows)


def fold_bn_params(conv_w: torch.Tensor, conv_b: torch.Tensor | None, bn: nn.BatchNorm2d):
    """Return (w, b) with BN (eval statistics) folded into the conv."""
    w = conv_w.detach().clone()
    b = conv_b.detach().clone() if conv_b is not None else torch.zeros(w.shape[0], dtype=w.dtype)
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones(w.shape[0])
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros(w.shape[0])
    inv_std = torch.rsqrt(bn.running_var.detach() + bn.eps)
    scale = gamma * inv_std
    w = w * scale.reshape(-1, 1, 1, 1)
    b = (b - bn.running_mean.detach()) * scale + beta
    return w, b


def trace(model: nn.Module, input_shape=(3, 32, 32)) -> StaticGraph:
    """Trace an eval-mode model into a StaticGraph (BN folded, shapes propagated)."""
    model = model.eval()
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(torch.zeros(1, *input_shape))
    modules = dict(gm.named_modules())
    nodes: list[Node] = []
    fxname_to_node: dict[str, str] = {}   # fx node name -> IR node name (BN maps to its conv)

    def shape_of(n: fx.Node) -> tuple[int, ...]:
        return tuple(n.meta["tensor_meta"].shape)[1:]

    def src(arg) -> str:
        if isinstance(arg, fx.Node):
            return fxname_to_node[arg.name]
        raise UnsupportedOpError(f"non-tensor input {arg!r}")

    for n in gm.graph.nodes:
        if n.op == "placeholder":
            nodes.append(Node(n.name, "input", [], shape_of(n)))
            fxname_to_node[n.name] = n.name
        elif n.op == "output":
            arg = n.args[0]
            nodes.append(Node("output", "output", [src(arg)], shape_of(arg)))
            fxname_to_node[n.name] = "output"
        elif n.op == "call_module":
            m = modules[n.target]
            if isinstance(m, nn.Conv2d):
                w = m.weight.detach().clone(); b = m.bias.detach().clone() if m.bias is not None else torch.zeros(m.out_channels)
                node = Node(n.name, "conv", [src(n.args[0])], shape_of(n),
                            attrs=dict(stride=tuple(m.stride), padding=tuple(m.padding), groups=m.groups,
                                       kernel=tuple(m.kernel_size), in_channels=m.in_channels,
                                       out_channels=m.out_channels, depthwise=(m.groups == m.in_channels == m.out_channels and m.groups > 1),
                                       bn_folded=False, module=n.target),
                            weight=w.numpy().astype(np.float32), bias=b.numpy().astype(np.float32))
                if m.dilation != (1, 1):
                    raise UnsupportedOpError(f"dilated conv {n.target}")
                nodes.append(node); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.BatchNorm2d):
                prev_name = src(n.args[0]); prev = nodes[-1] if nodes and nodes[-1].name == prev_name else None
                if prev is None or prev.op != "conv" or prev.attrs.get("bn_folded"):
                    raise UnsupportedOpError(f"BatchNorm {n.target} not directly after a conv")
                if len(n.args[0].users) != 1:
                    raise UnsupportedOpError(f"conv feeding BN {n.target} has multiple users; cannot fold")
                w, b = fold_bn_params(torch.from_numpy(prev.weight), torch.from_numpy(prev.bias), m)
                prev.weight = w.numpy().astype(np.float32); prev.bias = b.numpy().astype(np.float32)
                prev.attrs["bn_folded"] = True; prev.attrs["bn_module"] = n.target
                fxname_to_node[n.name] = prev.name
            elif type(m) in ACT_MODULE_KINDS:
                kind = ACT_MODULE_KINDS[type(m)]
                attrs = dict(kind=kind, module=n.target)
                if isinstance(m, nn.LeakyReLU):
                    attrs["negative_slope"] = m.negative_slope
                nodes.append(Node(n.name, "act", [src(n.args[0])], shape_of(n), attrs)); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                if tuple(np.atleast_1d(m.output_size)) not in ((1,), (1, 1)):
                    raise UnsupportedOpError("only global average pooling is supported")
                nodes.append(Node(n.name, "pool", [src(n.args[0])], shape_of(n), dict(kind="global_avg", module=n.target))); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.Flatten):
                nodes.append(Node(n.name, "flatten", [src(n.args[0])], shape_of(n), dict(module=n.target))); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.Linear):
                w = m.weight.detach().clone().numpy().astype(np.float32)
                b = (m.bias.detach().clone().numpy() if m.bias is not None else np.zeros(m.out_features)).astype(np.float32)
                nodes.append(Node(n.name, "linear", [src(n.args[0])], shape_of(n),
                                  dict(in_features=m.in_features, out_features=m.out_features, module=n.target),
                                  weight=w, bias=b)); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.Identity):
                fxname_to_node[n.name] = src(n.args[0])
            elif isinstance(m, nn.Dropout):
                fxname_to_node[n.name] = src(n.args[0])
            else:
                raise UnsupportedOpError(f"module {n.target}: {type(m).__name__}")
        elif n.op == "call_function":
            if n.target in (operator.add, torch.add):
                a, b = n.args[0], n.args[1]
                nodes.append(Node(n.name, "add", [src(a), src(b)], shape_of(n))); fxname_to_node[n.name] = n.name
            elif n.target in (torch.flatten,):
                nodes.append(Node(n.name, "flatten", [src(n.args[0])], shape_of(n))); fxname_to_node[n.name] = n.name
            else:
                raise UnsupportedOpError(f"function {n.target}")
        elif n.op == "call_method":
            if n.target in ("flatten", "view", "reshape") and len(shape_of(n)) == 1:
                nodes.append(Node(n.name, "flatten", [src(n.args[0])], shape_of(n))); fxname_to_node[n.name] = n.name
            else:
                raise UnsupportedOpError(f"method {n.target}")
        elif n.op == "get_attr":
            raise UnsupportedOpError(f"get_attr {n.target}")
    # sanity: every conv should have had its BN folded (models in the zoo always have conv->bn)
    return StaticGraph(nodes, tuple(input_shape))


def run_reference(graph: StaticGraph, x: np.ndarray) -> dict[str, np.ndarray]:
    """Float32 NumPy execution of the StaticGraph (used to validate BN folding / tracing)."""
    import torch.nn.functional as F
    vals: dict[str, np.ndarray] = {}
    for n in graph.nodes:
        if n.op == "input":
            vals[n.name] = x.astype(np.float32)
        elif n.op == "conv":
            inp = torch.from_numpy(vals[n.inputs[0]])
            out = F.conv2d(inp, torch.from_numpy(n.weight), torch.from_numpy(n.bias), stride=n.attrs["stride"],
                           padding=n.attrs["padding"], groups=n.attrs["groups"])
            vals[n.name] = out.numpy()
        elif n.op == "linear":
            vals[n.name] = vals[n.inputs[0]] @ n.weight.T + n.bias
        elif n.op == "add":
            vals[n.name] = vals[n.inputs[0]] + vals[n.inputs[1]]
        elif n.op == "act":
            vals[n.name] = apply_act(vals[n.inputs[0]], n.attrs["kind"], n.attrs)
        elif n.op == "pool":
            vals[n.name] = vals[n.inputs[0]].mean(axis=(2, 3), keepdims=True)
        elif n.op == "flatten":
            vals[n.name] = vals[n.inputs[0]].reshape(vals[n.inputs[0]].shape[0], -1)
        elif n.op == "output":
            vals[n.name] = vals[n.inputs[0]]
    return vals


def apply_act(x: np.ndarray, kind: str, attrs: dict | None = None) -> np.ndarray:
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    attrs = attrs or {}
    if kind == "relu":
        y = torch.relu(t)
    elif kind == "relu6":
        y = torch.clamp(t, 0, 6)
    elif kind == "silu":
        y = torch.nn.functional.silu(t)
    elif kind == "gelu":
        y = torch.nn.functional.gelu(t)
    elif kind == "hswish":
        y = torch.nn.functional.hardswish(t)
    elif kind == "lrelu":
        y = torch.nn.functional.leaky_relu(t, attrs.get("negative_slope", 0.01))
    elif kind == "sigmoid":
        y = torch.sigmoid(t)
    elif kind == "tanh":
        y = torch.tanh(t)
    elif kind == "hsigmoid":
        y = torch.nn.functional.hardsigmoid(t)
    else:
        raise ValueError(kind)
    return y.numpy()
