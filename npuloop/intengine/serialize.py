"""Portable, integer-only file format for an IntGraph: `.npuloop`.

Layout: 8-byte magic "NPULOOP1" | uint32 little-endian header length | UTF-8 JSON header | raw arrays.
The header lists every node (op, inputs, output quantization, JSON-safe attrs) and, for each array the node
carries, a descriptor {field, dtype, shape, offset, nbytes} into the raw section. Weights are int8, everything
else int32, all little-endian, C order. The file is what a compiler would hand to a device: nothing in it is
float except the input/output scales needed to quantize the input and read the logits back.

`save_int_graph` / `load_int_graph` round-trip an IntGraph so that both Python engines run it unchanged, and
`intengine/cpp/int8_runner.cpp` executes the same file with no Python at all.
"""
from __future__ import annotations
import json, os, struct, time
import numpy as np
from .graph import IntGraph, IntNode, QParams
from .requant import RequantConfig

MAGIC = b"NPULOOP1"
FORMAT_VERSION = 1
ARRAY_DTYPES = {"w_int": "int8", "bias_int": "int32", "mult": "int32", "shift": "int32", "lut": "int32",
                "in_mult": "int32", "in_shift": "int32", "codes": "int32"}
DROP_ATTRS = {"value", "gamma", "beta", "real_multiplier", "in_scales", "fq_target", "eps"}   # float-only / provenance


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _q(q: QParams | None):
    return None if q is None else dict(scale=float(q.scale), zero_point=int(q.zero_point), qmin=int(q.qmin), qmax=int(q.qmax))


def _unq(d):
    return None if d is None else QParams(float(d["scale"]), int(d["zero_point"]), int(d["qmin"]), int(d["qmax"]))


def save_int_graph(ig: IntGraph, path: str, provenance: dict | None = None) -> dict:
    """Write `ig` to `path`; returns the header that was written."""
    blobs: list[bytes] = []
    offset = 0
    nodes = []
    for n in ig.nodes:
        arrays = []
        for field, dtype in ARRAY_DTYPES.items():
            a = getattr(n, field)
            if a is None:
                continue
            arr = np.ascontiguousarray(np.asarray(a), dtype=dtype)
            if dtype == "int8" and (np.asarray(a).min() < -128 or np.asarray(a).max() > 127):
                raise ValueError(f"{n.name}.{field} does not fit int8")
            raw = arr.astype(arr.dtype.newbyteorder("<")).tobytes(order="C")
            arrays.append(dict(field=field, dtype=dtype, shape=list(arr.shape), offset=offset, nbytes=len(raw)))
            blobs.append(raw); offset += len(raw)
        attrs = {k: _jsonable(v) for k, v in n.attrs.items() if k not in DROP_ATTRS}
        rec = dict(name=n.name, op=n.op, inputs=list(n.inputs), out_q=_q(n.out_q), attrs=attrs, arrays=arrays)
        if n.add_params is not None:
            p = n.add_params
            rec["add_params"] = dict(left_shift=int(p["left_shift"]), m1=[int(p["m1"][0]), int(p["m1"][1])],
                                     m2=[int(p["m2"][0]), int(p["m2"][1])], mo=[int(p["mo"][0]), int(p["mo"][1])])
        nodes.append(rec)
    header = dict(format="npuloop", version=FORMAT_VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  input_q=_q(ig.input_q), requant=ig.requant.to_dict(), nodes=nodes, provenance=provenance or {})
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC); f.write(struct.pack("<I", len(hj))); f.write(hj)
        for b in blobs:
            f.write(b)
    return header


def load_int_graph(path: str) -> IntGraph:
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a .npuloop file")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12:12 + hlen].decode("utf-8"))
    if header.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format version {header.get('version')}")
    base = 12 + hlen
    nodes = []
    for rec in header["nodes"]:
        attrs = dict(rec["attrs"])
        for k in ("stride", "padding", "clamp", "perm", "out_shape"):
            if k in attrs and isinstance(attrs[k], list):
                attrs[k] = tuple(attrs[k])
        n = IntNode(rec["name"], rec["op"], list(rec["inputs"]), out_q=_unq(rec["out_q"]), attrs=attrs)
        for a in rec["arrays"]:
            raw = data[base + a["offset"]: base + a["offset"] + a["nbytes"]]
            arr = np.frombuffer(raw, dtype=np.dtype(a["dtype"]).newbyteorder("<")).reshape(a["shape"]).astype(np.int64)
            setattr(n, a["field"], arr)
        if "add_params" in rec:
            p = rec["add_params"]
            n.add_params = dict(left_shift=p["left_shift"], m1=tuple(p["m1"]), m2=tuple(p["m2"]), mo=tuple(p["mo"]))
        nodes.append(n)
    rq = header["requant"]
    return IntGraph(nodes, _unq(header["input_q"]), RequantConfig(**rq))


def describe(path: str) -> str:
    """One-line-per-node summary of a .npuloop file (no arrays loaded)."""
    with open(path, "rb") as f:
        head = f.read(12)
        (hlen,) = struct.unpack("<I", head[8:12])
        header = json.loads(f.read(hlen).decode("utf-8"))
    total = sum(a["nbytes"] for n in header["nodes"] for a in n["arrays"])
    lines = [f"{path}: {len(header['nodes'])} nodes, {total / 1024:.1f} KB of integer arrays, requant {header['requant']}",
             f"input: {header['input_q']}"]
    for n in header["nodes"]:
        arrs = " ".join(f"{a['field']}{tuple(a['shape'])}" for a in n["arrays"])
        q = n["out_q"]
        lines.append(f"  {n['name']:26s} {n['op']:9s} <- {','.join(n['inputs']):34s} "
                     f"{'s=%.5g zp=%d' % (q['scale'], q['zero_point']) if q else '':22s} {arrs}")
    return "\n".join(lines)
