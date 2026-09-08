"""IntGraph: an integer-only executable description of a calibrated fake-quant model.

Every tensor is int8/uint8 (stored as int64 for convenience); every op is defined in terms of
integer arithmetic only. Building it from the fx GraphModule produced by `quant.prepare` is the
"export" step a real NPU compiler performs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from ..graph.ir import ACT_MODULE_KINDS, RELU_FAMILY, apply_act
from ..quant.fake import FakeQuantAct, QConv2d, QLinear
from .requant import quantize_multiplier, RequantConfig


@dataclass
class QParams:
    scale: float
    zero_point: int
    qmin: int
    qmax: int

    def to_dict(self):
        return dict(scale=self.scale, zero_point=self.zero_point, qmin=self.qmin, qmax=self.qmax)


@dataclass
class IntNode:
    name: str
    op: str                          # input|const|conv|linear|add|matmul|concat|softmax|layernorm|lut|pool|transpose|reshape|flatten|output
    inputs: list[str]
    out_q: QParams | None = None
    attrs: dict = field(default_factory=dict)
    w_int: np.ndarray | None = None          # int8 weights (as int64)
    w_scale: np.ndarray | None = None        # float64 per-output-channel scale
    bias_int: np.ndarray | None = None       # int32 bias (as int64) in units of s_in*s_w[c]
    bias_float: np.ndarray | None = None     # original float bias (for reference / ablations)
    mult: np.ndarray | None = None           # per-channel q31 multipliers
    shift: np.ndarray | None = None          # per-channel shifts
    lut: np.ndarray | None = None            # 256-entry table for activation nodes
    add_params: dict | None = None           # TFLite add parameters
    in_mult: np.ndarray | None = None        # concat: per-input q31 multiplier into the output scale
    in_shift: np.ndarray | None = None       # concat: per-input shift
    codes: np.ndarray | None = None          # const: the quantized tensor

    @property
    def out_shape(self):
        return self.attrs.get("out_shape")


@dataclass
class IntGraph:
    nodes: list[IntNode]
    input_q: QParams
    requant: RequantConfig = RequantConfig()

    def __getitem__(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def compute_nodes(self):
        return [n for n in self.nodes if n.op in ("conv", "linear")]

    def summary(self) -> str:
        rows = []
        for n in self.nodes:
            q = f"s={n.out_q.scale:.5g} zp={n.out_q.zero_point}" if n.out_q else ""
            extra = ""
            if n.op in ("conv", "linear"):
                extra = f"w{tuple(n.w_int.shape)} shift[{n.shift.min()},{n.shift.max()}] fused={n.attrs.get('fused_act')}"
            rows.append(f"{n.name:28s} {n.op:8s} <- {','.join(n.inputs):40s} {q:24s} {extra}")
        return "\n".join(rows)


def _qp(fq: FakeQuantAct) -> QParams:
    s, z = fq.qparams()
    return QParams(float(s), int(z), fq.qmin, fq.qmax)


def export_int_graph(gm: fx.GraphModule, requant: RequantConfig = RequantConfig(), input_shape=(3, 32, 32)) -> IntGraph:
    """Turn a calibrated fake-quant GraphModule into an IntGraph (the NPU compiler's job)."""
    from torch.fx.passes.shape_prop import ShapeProp
    gm.eval()
    ShapeProp(gm).propagate(torch.zeros(1, *input_shape))
    modules = dict(gm.named_modules())
    nodes: list[IntNode] = []
    by_name: dict[str, IntNode] = {}
    alias: dict[str, str] = {}          # fx node name -> IntNode name producing that tensor
    input_q: QParams | None = None

    def shape_of(n):
        return tuple(n.meta["tensor_meta"].shape)[1:]

    def add_node(node: IntNode):
        nodes.append(node); by_name[node.name] = node
        return node

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            add_node(IntNode(node.name, "input", [], attrs=dict(out_shape=shape_of(node))))
            alias[node.name] = node.name
        elif node.op == "output":
            src = node.args[0]
            add_node(IntNode("output", "output", [alias[src.name]], out_q=by_name[alias[src.name]].out_q,
                             attrs=dict(out_shape=shape_of(src))))
        elif node.op == "call_module":
            m = modules[node.target]
            src = node.args[0]
            if isinstance(m, FakeQuantAct):
                producer = by_name[alias[src.name]]
                producer.out_q = _qp(m)
                producer.attrs["fq_target"] = node.target
                if producer.op == "input":
                    input_q = producer.out_q
                alias[node.name] = producer.name
            elif isinstance(m, (QConv2d, QLinear)):
                w_int = m.int_weight().numpy().astype(np.int64)
                w_scale = (m.w_scale.detach().double().numpy() if m.per_channel
                           else np.full(w_int.shape[0], float(m.w_scale.detach()), dtype=np.float64))
                n = IntNode(node.name, "conv" if isinstance(m, QConv2d) else "linear", [alias[src.name]],
                            attrs=dict(out_shape=shape_of(node)), w_int=w_int, w_scale=w_scale,
                            bias_float=m.bias.detach().double().numpy())
                if isinstance(m, QConv2d):
                    n.attrs.update(stride=m.stride, padding=m.padding, groups=m.groups,
                                   depthwise=(m.groups == m.in_channels == m.out_channels and m.groups > 1))
                add_node(n); alias[node.name] = n.name
            elif type(m) in ACT_MODULE_KINDS:
                kind = ACT_MODULE_KINDS[type(m)]
                producer = by_name[alias[src.name]]
                fusable = (kind in RELU_FAMILY and producer.op in ("conv", "linear", "add")
                           and producer.out_q is None and "fused_act" not in producer.attrs)
                if fusable:
                    producer.attrs["fused_act"] = kind
                    alias[node.name] = producer.name
                else:
                    n = add_node(IntNode(node.name, "lut", [alias[src.name]],
                                         attrs=dict(kind=kind, out_shape=shape_of(node),
                                                    negative_slope=getattr(m, "negative_slope", None))))
                    alias[node.name] = n.name
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                n = add_node(IntNode(node.name, "pool", [alias[src.name]], attrs=dict(out_shape=shape_of(node))))
                alias[node.name] = n.name
            elif isinstance(m, nn.LayerNorm):
                n = add_node(IntNode(node.name, "layernorm", [alias[src.name]],
                                     attrs=dict(out_shape=shape_of(node), eps=float(m.eps),
                                                channels=int(m.normalized_shape[0]),
                                                gamma=m.weight.detach().double().numpy() if m.weight is not None else np.ones(m.normalized_shape[0]),
                                                beta=m.bias.detach().double().numpy() if m.bias is not None else np.zeros(m.normalized_shape[0]))))
                alias[node.name] = n.name
            elif isinstance(m, nn.Softmax):
                osh = shape_of(node); dim = int(m.dim) if int(m.dim) >= 0 else len(osh) + 1 + int(m.dim)
                n = add_node(IntNode(node.name, "softmax", [alias[src.name]], attrs=dict(out_shape=osh, dim=dim)))
                alias[node.name] = n.name
            elif isinstance(m, nn.Flatten):
                n = add_node(IntNode(node.name, "flatten", [alias[src.name]], attrs=dict(out_shape=shape_of(node))))
                n.out_q = by_name[alias[src.name]].out_q
                alias[node.name] = n.name
            elif isinstance(m, (nn.Identity, nn.Dropout)):
                alias[node.name] = alias[src.name]
            else:
                raise ValueError(f"export: unsupported module {type(m).__name__} at {node.target}")
        elif node.op == "get_attr":
            t = gm.get_parameter(node.target) if node.target in dict(gm.named_parameters()) else getattr(gm, node.target)
            n = add_node(IntNode(node.name, "const", [], attrs=dict(out_shape=tuple(t.shape)[1:],
                                                                   value=t.detach().double().numpy())))
            alias[node.name] = n.name
        elif node.op == "call_function" and node.target in (operator.add, torch.add):
            a, b = node.args
            n = add_node(IntNode(node.name, "add", [alias[a.name], alias[b.name]], attrs=dict(out_shape=shape_of(node))))
            alias[node.name] = n.name
        elif node.op == "call_function" and node.target in (torch.matmul, operator.matmul):
            a, b = node.args
            n = add_node(IntNode(node.name, "matmul", [alias[a.name], alias[b.name]], attrs=dict(out_shape=shape_of(node))))
            alias[node.name] = n.name
        elif node.op == "call_function" and node.target in (torch.cat, torch.concat):
            seq = node.args[0]
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
            dim = int(dim) if int(dim) >= 0 else len(shape_of(node)) + 1 + int(dim)
            n = add_node(IntNode(node.name, "concat", [alias[a.name] for a in seq],
                                 attrs=dict(out_shape=shape_of(node), dim=dim)))
            alias[node.name] = n.name
        elif node.op == "call_function" and node.target in (operator.mul, torch.mul):
            tensors = [a for a in node.args if isinstance(a, fx.Node)]
            if len(tensors) != 1:
                raise ValueError("export: activation x activation multiply is not supported")
            scalar = float([a for a in node.args if not isinstance(a, fx.Node)][0])
            producer = by_name[alias[tensors[0].name]]
            if producer.op not in ("conv", "linear", "matmul") or producer.out_q is not None:
                raise ValueError(f"export: scalar multiply at {node.name} must follow an un-requantized compute op")
            producer.attrs["fold_scale"] = producer.attrs.get("fold_scale", 1.0) * scalar
            alias[node.name] = producer.name
        elif (node.op == "call_method" and node.target in ("transpose", "permute")) or \
             (node.op == "call_function" and node.target is torch.transpose):
            src = node.args[0]
            osh = shape_of(node); ndim = len(osh) + 1
            if node.target == "permute":
                perm = [int(a) for a in (node.args[1] if isinstance(node.args[1], (list, tuple)) else node.args[1:])]
            else:
                perm = list(range(ndim))
                d0, d1 = int(node.args[1]), int(node.args[2])
                d0 = d0 if d0 >= 0 else ndim + d0; d1 = d1 if d1 >= 0 else ndim + d1
                perm[d0], perm[d1] = perm[d1], perm[d0]
            n = add_node(IntNode(node.name, "transpose", [alias[src.name]], attrs=dict(out_shape=osh, perm=perm)))
            n.out_q = by_name[alias[src.name]].out_q; alias[node.name] = n.name
        elif node.op == "call_method" and node.target == "softmax":
            src = node.args[0]
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else -1)
            osh = shape_of(node); dim = int(dim) if int(dim) >= 0 else len(osh) + 1 + int(dim)
            n = add_node(IntNode(node.name, "softmax", [alias[src.name]], attrs=dict(out_shape=osh, dim=dim)))
            alias[node.name] = n.name
        elif node.op == "call_method" and node.target == "mean":
            src = node.args[0]
            n = add_node(IntNode(node.name, "pool", [alias[src.name]],
                                 attrs=dict(out_shape=shape_of(node), kind="token_mean",
                                            keepdim=bool(node.kwargs.get("keepdim", False)))))
            alias[node.name] = n.name
        elif (node.op == "call_function" and node.target is torch.flatten) or \
             (node.op == "call_method" and node.target in ("flatten", "view", "reshape") and len(shape_of(node)) == 1):
            src = node.args[0]
            n = add_node(IntNode(node.name, "flatten", [alias[src.name]], attrs=dict(out_shape=shape_of(node))))
            n.out_q = by_name[alias[src.name]].out_q; alias[node.name] = n.name
        elif node.op == "call_method" and node.target in ("flatten", "view", "reshape"):
            src = node.args[0]
            n = add_node(IntNode(node.name, "reshape", [alias[src.name]], attrs=dict(out_shape=shape_of(node))))
            n.out_q = by_name[alias[src.name]].out_q; alias[node.name] = n.name
        else:
            raise ValueError(f"export: unsupported fx node {node.op} {node.target}")

    # second pass: derive integer parameters now that every out_q is known
    for n in nodes:
        if n.out_q is not None and n.op in ("conv", "linear", "add", "matmul"):
            # The requantization clamp implements the fused ReLU/ReLU6 only if its lower bound is the code of 0.0
            # (true for uint8 with zp=0, NOT for symmetric int8 where qmin=-127). Derive explicit bounds.
            lo, hi = n.out_q.qmin, n.out_q.qmax
            fused = n.attrs.get("fused_act")
            if fused in ("relu", "relu6"):
                lo = max(lo, n.out_q.zero_point)
            if fused == "relu6":
                hi = min(hi, n.out_q.zero_point + int(round(6.0 / n.out_q.scale)))
            n.attrs["clamp"] = (int(lo), int(hi))
        if n.op in ("conv", "linear"):
            in_q = by_name[n.inputs[0]].out_q
            if n.out_q is None:
                raise ValueError(f"{n.name}: output quantizer missing")
            s_in = in_q.scale
            real_mult = s_in * n.w_scale / n.out_q.scale * n.attrs.get("fold_scale", 1.0)
            qm = [quantize_multiplier(float(m), requant.mult_bits) for m in real_mult]
            n.mult = np.array([q for q, _ in qm], dtype=np.int64)
            n.shift = np.array([s for _, s in qm], dtype=np.int64)
            bias = np.rint(n.bias_float / (s_in * n.w_scale))
            lo, hi = -(1 << (requant.bias_bits - 1)), (1 << (requant.bias_bits - 1)) - 1
            n.bias_int = np.clip(bias, lo, hi).astype(np.int64)
            n.attrs["real_multiplier"] = real_mult
        elif n.op == "lut":
            in_q, out_q = by_name[n.inputs[0]].out_q, n.out_q
            qs = np.arange(in_q.qmin, in_q.qmax + 1)
            x = (qs - in_q.zero_point) * in_q.scale
            y = apply_act(x.astype(np.float32), n.attrs["kind"], n.attrs).astype(np.float64)
            q = np.floor(np.abs(y / out_q.scale) + 0.5) * np.sign(y / out_q.scale)   # round half away from zero
            n.lut = np.clip(q + out_q.zero_point, out_q.qmin, out_q.qmax).astype(np.int64)
            n.attrs["lut_qmin"] = in_q.qmin
        elif n.op == "add":
            q1, q2, qo = by_name[n.inputs[0]].out_q, by_name[n.inputs[1]].out_q, n.out_q
            left_shift = 20
            twice_max = 2.0 * max(q1.scale, q2.scale)
            m1, m2 = q1.scale / twice_max, q2.scale / twice_max
            mo = twice_max / ((1 << left_shift) * qo.scale)
            n.add_params = dict(left_shift=left_shift, m1=quantize_multiplier(m1, requant.mult_bits),
                                m2=quantize_multiplier(m2, requant.mult_bits), mo=quantize_multiplier(mo, requant.mult_bits),
                                real=(m1, m2, mo))
        elif n.op == "pool":
            in_q = by_name[n.inputs[0]].out_q
            if n.out_q is None:
                n.out_q = in_q
            if (n.out_q.scale, n.out_q.zero_point) != (in_q.scale, in_q.zero_point):
                raise ValueError("pool output quantizer must be tied to its input (NPU avgpool keeps the scale)")
        elif n.op == "const":
            q = n.out_q
            v = n.attrs["value"] / q.scale
            n.codes = np.clip(np.floor(np.abs(v) + 0.5) * np.sign(v) + q.zero_point, q.qmin, q.qmax).astype(np.int64)
        elif n.op == "matmul":
            q1, q2, qo = by_name[n.inputs[0]].out_q, by_name[n.inputs[1]].out_q, n.out_q
            real = q1.scale * q2.scale / qo.scale * n.attrs.get("fold_scale", 1.0)
            m0, sh = quantize_multiplier(float(real), requant.mult_bits)
            n.mult = np.array([m0], dtype=np.int64); n.shift = np.array([sh], dtype=np.int64)
            n.attrs["real_multiplier"] = np.array([real])
        elif n.op == "concat":
            qo = n.out_q
            mults, shifts = [], []
            for name in n.inputs:
                qi = by_name[name].out_q
                m0, sh = quantize_multiplier(float(qi.scale / qo.scale), requant.mult_bits)
                mults.append(m0); shifts.append(sh)
            n.in_mult = np.array(mults, dtype=np.int64); n.in_shift = np.array(shifts, dtype=np.int64)
            n.attrs["in_scales"] = [by_name[name].out_q.scale for name in n.inputs]
        elif n.op == "softmax":
            in_q, qo = by_name[n.inputs[0]].out_q, n.out_q
            span = in_q.qmax - in_q.qmin
            d = np.arange(-span, 1)                                  # q - rowmax, always <= 0
            n.lut = np.rint(np.exp(d * in_q.scale) * (1 << 15)).astype(np.int64)   # Q15 exp table
            n.attrs["lut_offset"] = int(span)
            m0, sh = quantize_multiplier(float(2.0 ** -15 / qo.scale), requant.mult_bits)
            n.mult = np.array([m0], dtype=np.int64); n.shift = np.array([sh], dtype=np.int64)
        elif n.op == "layernorm":
            in_q, qo = by_name[n.inputs[0]].out_q, n.out_q
            c = n.attrs["channels"]; gamma = n.attrs["gamma"]; beta = n.attrs["beta"]
            n.attrs["eps_int"] = int(round(n.attrs["eps"] * c * c / (in_q.scale ** 2)))
            real = gamma / (2.0 ** 15 * qo.scale)
            qm = [quantize_multiplier(float(m), requant.mult_bits) for m in real]
            n.mult = np.array([q for q, _ in qm], dtype=np.int64)
            n.shift = np.array([s for _, s in qm], dtype=np.int64)
            n.bias_int = np.rint(beta / qo.scale).astype(np.int64)
            n.attrs["real_multiplier"] = real
    if input_q is None:
        raise ValueError("no input quantizer found")
    return IntGraph(nodes, input_q, requant)
