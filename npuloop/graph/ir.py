"""Static graph IR extracted from an nn.Module with torch.fx.

Every downstream tool (cost model, lint, quantizer export, integer engine) consumes this
representation instead of poking at PyTorch modules. BatchNorm is folded into the preceding
conv here, because the NPU never sees a BN layer.

Supported ops: input, const, conv (incl. depthwise / grouped), linear (2D or token-wise 3D),
add, mul (elementwise or by a constant), concat, matmul (activation x activation), softmax,
layernorm, act, pool (global avg / token mean), transpose, reshape, flatten, output.
Anything else raises UnsupportedOpError with the fx node for diagnosis.
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
    op: str                       # input|const|conv|linear|add|mul|concat|matmul|softmax|layernorm|act|pool|transpose|reshape|flatten|output
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
            return int(self.weight.shape[0] * self.weight.shape[1] * self.attrs.get("tokens", 1))
        if self.op == "matmul":     # activation x activation: no weight reuse across the batch
            a = self.attrs
            return int(a["batch"] * a["m"] * a["k"] * a["n"])
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
        """Nodes with quantizable weights (the ones per-channel scales and pruning apply to)."""
        return [n for n in self.nodes if n.op in ("conv", "linear")]

    def has_op(self, *ops: str) -> bool:
        return any(n.op in ops for n in self.nodes)

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


def _pos_dim(dim: int, ndim: int) -> int:
    """Normalize a (possibly negative) dim against the full tensor rank, batch included."""
    return dim if dim >= 0 else ndim + dim


def _transpose_node(name: str, inp: str, out_shape, d0: int, d1: int) -> Node:
    ndim = len(out_shape) + 1
    perm = list(range(ndim))
    a, b = _pos_dim(d0, ndim), _pos_dim(d1, ndim)
    perm[a], perm[b] = perm[b], perm[a]
    return Node(name, "transpose", [inp], out_shape, dict(perm=perm))


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
                prev_name = src(n.args[0]); prev = next((x for x in nodes if x.name == prev_name), None)
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
                osh = shape_of(n)
                nodes.append(Node(n.name, "linear", [src(n.args[0])], osh,
                                  dict(in_features=m.in_features, out_features=m.out_features, module=n.target,
                                       tokens=int(np.prod(osh[:-1])) if len(osh) > 1 else 1),
                                  weight=w, bias=b)); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.LayerNorm):
                if len(m.normalized_shape) != 1:
                    raise UnsupportedOpError(f"LayerNorm {n.target}: only last-dim normalization is supported")
                w = (m.weight.detach().clone().numpy() if m.weight is not None else np.ones(m.normalized_shape[0])).astype(np.float32)
                b = (m.bias.detach().clone().numpy() if m.bias is not None else np.zeros(m.normalized_shape[0])).astype(np.float32)
                nodes.append(Node(n.name, "layernorm", [src(n.args[0])], shape_of(n),
                                  dict(eps=float(m.eps), channels=int(m.normalized_shape[0]), module=n.target),
                                  weight=w, bias=b)); fxname_to_node[n.name] = n.name
            elif isinstance(m, nn.Softmax):
                nodes.append(Node(n.name, "softmax", [src(n.args[0])], shape_of(n),
                                  dict(dim=int(m.dim), module=n.target))); fxname_to_node[n.name] = n.name
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
            elif n.target in (torch.cat, torch.concat):
                seq = n.args[0]
                if not isinstance(seq, (list, tuple)):
                    raise UnsupportedOpError("cat needs a static list of tensors")
                dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else 0)
                nodes.append(Node(n.name, "concat", [src(a) for a in seq], shape_of(n),
                                  dict(dim=_pos_dim(int(dim), len(shape_of(n)) + 1)))); fxname_to_node[n.name] = n.name
            elif n.target in (torch.matmul, operator.matmul):
                a, b = n.args[0], n.args[1]
                sa, sb = shape_of(a), shape_of(b)          # both exclude the batch dim
                nodes.append(Node(n.name, "matmul", [src(a), src(b)], shape_of(n),
                                  dict(m=int(sa[-2]), k=int(sa[-1]), n=int(sb[-1]),
                                       batch=int(np.prod(sa[:-2])) if len(sa) > 2 else 1)))
                fxname_to_node[n.name] = n.name
            elif n.target in (operator.mul, torch.mul):
                a, b = n.args[0], n.args[1]
                if isinstance(a, fx.Node) and isinstance(b, fx.Node):
                    nodes.append(Node(n.name, "mul", [src(a), src(b)], shape_of(n), dict(kind="elementwise")))
                else:
                    t, c = (a, b) if isinstance(a, fx.Node) else (b, a)
                    if not isinstance(c, (int, float)):
                        raise UnsupportedOpError(f"mul by {type(c).__name__}")
                    nodes.append(Node(n.name, "mul", [src(t)], shape_of(n), dict(kind="scalar", scalar=float(c))))
                fxname_to_node[n.name] = n.name
            elif n.target in (torch.nn.functional.softmax,):
                dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else -1)
                nodes.append(Node(n.name, "softmax", [src(n.args[0])], shape_of(n),
                                  dict(dim=_pos_dim(int(dim), len(shape_of(n)) + 1)))); fxname_to_node[n.name] = n.name
            elif n.target in (torch.nn.functional.gelu,):
                nodes.append(Node(n.name, "act", [src(n.args[0])], shape_of(n), dict(kind="gelu"))); fxname_to_node[n.name] = n.name
            elif n.target in (torch.transpose,):
                nodes.append(_transpose_node(n.name, src(n.args[0]), shape_of(n), int(n.args[1]), int(n.args[2])))
                fxname_to_node[n.name] = n.name
            else:
                raise UnsupportedOpError(f"function {n.target}")
        elif n.op == "call_method":
            if n.target in ("flatten", "view", "reshape") and len(shape_of(n)) == 1:
                nodes.append(Node(n.name, "flatten", [src(n.args[0])], shape_of(n))); fxname_to_node[n.name] = n.name
            elif n.target in ("view", "reshape", "flatten"):
                nodes.append(Node(n.name, "reshape", [src(n.args[0])], shape_of(n))); fxname_to_node[n.name] = n.name
            elif n.target == "transpose":
                nodes.append(_transpose_node(n.name, src(n.args[0]), shape_of(n), int(n.args[1]), int(n.args[2])))
                fxname_to_node[n.name] = n.name
            elif n.target == "permute":
                perm = [int(a) for a in (n.args[1] if isinstance(n.args[1], (list, tuple)) else n.args[1:])]
                nodes.append(Node(n.name, "transpose", [src(n.args[0])], shape_of(n), dict(perm=perm)))
                fxname_to_node[n.name] = n.name
            elif n.target == "softmax":
                dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else -1)
                nodes.append(Node(n.name, "softmax", [src(n.args[0])], shape_of(n),
                                  dict(dim=_pos_dim(int(dim), len(shape_of(n)) + 1)))); fxname_to_node[n.name] = n.name
            elif n.target == "mean":
                dim = n.kwargs.get("dim", n.args[1] if len(n.args) > 1 else None)
                if dim is None:
                    raise UnsupportedOpError("mean over all dims")
                dims = tuple(int(d) for d in (dim if isinstance(dim, (list, tuple)) else [dim]))
                if dims != (1,):
                    raise UnsupportedOpError(f"mean over dims {dims}: only token mean (dim=1) is supported")
                nodes.append(Node(n.name, "pool", [src(n.args[0])], shape_of(n),
                                  dict(kind="token_mean", keepdim=bool(n.kwargs.get("keepdim", False)))))
                fxname_to_node[n.name] = n.name
            else:
                raise UnsupportedOpError(f"method {n.target}")
        elif n.op == "get_attr":
            t = getattr(gm, n.target, None)
            if t is None:
                for holder, attr in [(gm.get_submodule(n.target.rsplit(".", 1)[0]), n.target.rsplit(".", 1)[1])] if "." in n.target else []:
                    t = getattr(holder, attr, None)
            if not isinstance(t, torch.Tensor):
                raise UnsupportedOpError(f"get_attr {n.target}")
            arr = t.detach().clone().numpy().astype(np.float32)
            nodes.append(Node(n.name, "const", [], tuple(arr.shape[1:]) if arr.ndim > 1 else tuple(arr.shape),
                              dict(target=n.target), weight=arr))
            fxname_to_node[n.name] = n.name
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
        elif n.op == "const":
            vals[n.name] = n.weight
        elif n.op == "linear":
            vals[n.name] = vals[n.inputs[0]] @ n.weight.T + n.bias
        elif n.op == "add":
            vals[n.name] = vals[n.inputs[0]] + vals[n.inputs[1]]
        elif n.op == "mul":
            vals[n.name] = (vals[n.inputs[0]] * n.attrs["scalar"] if n.attrs["kind"] == "scalar"
                            else vals[n.inputs[0]] * vals[n.inputs[1]])
        elif n.op == "concat":
            vals[n.name] = np.concatenate([vals[i] for i in n.inputs], axis=n.attrs["dim"])
        elif n.op == "matmul":
            vals[n.name] = np.matmul(vals[n.inputs[0]], vals[n.inputs[1]])
        elif n.op == "softmax":
            z = vals[n.inputs[0]]; d = n.attrs["dim"]
            e = np.exp(z - z.max(axis=d, keepdims=True))
            vals[n.name] = e / e.sum(axis=d, keepdims=True)
        elif n.op == "layernorm":
            z = vals[n.inputs[0]]
            mu = z.mean(axis=-1, keepdims=True); var = z.var(axis=-1, keepdims=True)
            vals[n.name] = (z - mu) / np.sqrt(var + n.attrs["eps"]) * n.weight + n.bias
        elif n.op == "transpose":
            vals[n.name] = np.transpose(vals[n.inputs[0]], n.attrs["perm"])
        elif n.op == "reshape":
            vals[n.name] = vals[n.inputs[0]].reshape((-1, *n.out_shape))
        elif n.op == "act":
            vals[n.name] = apply_act(vals[n.inputs[0]], n.attrs["kind"], n.attrs)
        elif n.op == "pool":
            vals[n.name] = (vals[n.inputs[0]].mean(axis=1, keepdims=n.attrs.get("keepdim", False))
                            if n.attrs.get("kind") == "token_mean" else vals[n.inputs[0]].mean(axis=(2, 3), keepdims=True))
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
