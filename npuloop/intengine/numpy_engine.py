"""Bit-exact integer execution of an IntGraph in NumPy.

The multiply-accumulate itself is done with float64 conv/matmul on integer-valued tensors: products
of int8 x uint8 summed over K <= 2^20 terms stay far below 2^53, so float64 arithmetic is exact and
we can use PyTorch's fast conv kernels while keeping integer semantics. Everything after the
accumulator (bias, requantization, clamping, LUTs, adds, pooling) is int64 integer arithmetic.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .graph import IntGraph, IntNode, QParams
from .requant import multiply_by_quantized_multiplier, saturate, INT32_MIN, INT32_MAX


def quantize_input(x: np.ndarray, q: QParams) -> np.ndarray:
    """Float input -> uint8/int8 codes (round half to even, like the fake-quant model)."""
    codes = np.rint(x.astype(np.float64) / q.scale) + q.zero_point
    return np.clip(codes, q.qmin, q.qmax).astype(np.int64)


def dequantize(codes: np.ndarray, q: QParams) -> np.ndarray:
    return (codes.astype(np.float64) - q.zero_point) * q.scale


class NumpyEngine:
    def __init__(self, graph: IntGraph):
        self.g = graph
        self.cfg = graph.requant
        self.saturations: dict[str, int] = {}

    def _mac(self, n: IntNode, x: np.ndarray) -> np.ndarray:
        """Integer conv/linear accumulation (exact via float64)."""
        in_q = self.g[n.inputs[0]].out_q
        xf = torch.from_numpy((x - in_q.zero_point).astype(np.float64))
        wf = torch.from_numpy(n.w_int.astype(np.float64))
        if n.op == "conv":
            acc = F.conv2d(xf, wf, None, n.attrs["stride"], n.attrs["padding"], 1, n.attrs["groups"])
        else:
            acc = xf @ wf.T
        acc = acc.numpy()
        assert np.all(acc == np.round(acc)), "accumulator is not integer-valued (float64 exactness violated)"
        return acc.astype(np.int64)

    def run(self, x_codes: np.ndarray, return_all: bool = False):
        """x_codes: (N,C,H,W) integer input codes. Returns output codes (and all intermediates)."""
        vals: dict[str, np.ndarray] = {"__input__": x_codes.astype(np.int64)}
        for n in self.g.nodes:
            vals[n.name] = self.exec_node(n, vals)
        vals.pop("__input__")
        return vals if return_all else vals["output"]

    def exec_node(self, n: IntNode, vals: dict) -> np.ndarray:
        """Execute one node given a dict of (already computed) input tensors."""
        cfg = self.cfg
        if True:
            if n.op == "input":
                return vals["__input__"]
            elif n.op in ("conv", "linear"):
                acc = self._mac(n, vals[n.inputs[0]])
                if cfg.acc_bits < 32:
                    sat = saturate(acc, cfg.acc_bits); self.saturations[n.name] = int((sat != acc).sum()); acc = sat
                else:
                    self.saturations[n.name] = int(((acc < INT32_MIN) | (acc > INT32_MAX)).sum())
                    acc = np.clip(acc, INT32_MIN, INT32_MAX)
                cshape = (1, -1, 1, 1) if n.op == "conv" else (1, -1)
                acc = acc + n.bias_int.reshape(cshape)
                acc = np.clip(acc, INT32_MIN, INT32_MAX)
                y = multiply_by_quantized_multiplier(acc, n.mult.reshape(cshape), n.shift.reshape(cshape), cfg.rounding)
                y = y + n.out_q.zero_point
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)      # clamp == fused ReLU/ReLU6
            elif n.op == "lut":
                x = vals[n.inputs[0]]
                return n.lut[x - n.attrs["lut_qmin"]]
            elif n.op == "add":
                p = n.add_params
                q1, q2 = self.g[n.inputs[0]].out_q, self.g[n.inputs[1]].out_q
                x1 = (vals[n.inputs[0]] - q1.zero_point) << p["left_shift"]
                x2 = (vals[n.inputs[1]] - q2.zero_point) << p["left_shift"]
                y1 = multiply_by_quantized_multiplier(x1, p["m1"][0], p["m1"][1], cfg.rounding)
                y2 = multiply_by_quantized_multiplier(x2, p["m2"][0], p["m2"][1], cfg.rounding)
                raw = y1 + y2
                y = multiply_by_quantized_multiplier(raw, p["mo"][0], p["mo"][1], cfg.rounding) + n.out_q.zero_point
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)
            elif n.op == "pool":
                x = vals[n.inputs[0]]
                cnt = x.shape[2] * x.shape[3]
                s = x.sum(axis=(2, 3), keepdims=True)
                # TFLite AveragePool: round half away from zero (same scale/zero-point as input)
                y = np.where(s >= 0, (s + cnt // 2) // cnt, -((-s + cnt // 2) // cnt))
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)
            elif n.op == "flatten":
                x = vals[n.inputs[0]]
                return x.reshape(x.shape[0], -1)
            elif n.op == "output":
                return vals[n.inputs[0]]
            else:
                raise ValueError(n.op)

    def predict(self, x_float: np.ndarray) -> np.ndarray:
        codes = quantize_input(x_float, self.g.input_q)
        out = self.run(codes)
        return dequantize(out, self.g["output"].out_q)

    def evaluate(self, ds, batch_size: int = 500, limit: int | None = None) -> float:
        correct = 0; total = 0
        for xb, yb in ds.test_batches(batch_size):
            logits = self.predict(xb.numpy())
            correct += int((logits.argmax(1) == yb.numpy()).sum()); total += len(yb)
            if limit and total >= limit:
                break
        return correct / total
