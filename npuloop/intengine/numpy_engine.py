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
from .requant import multiply_by_quantized_multiplier, saturate, isqrt64, round_div, INT32_MIN, INT32_MAX


def quantize_input(x: np.ndarray, q: QParams) -> np.ndarray:
    """Float input -> uint8/int8 codes (round half to even, like the fake-quant model)."""
    codes = np.rint(x.astype(np.float64) / q.scale) + q.zero_point
    return np.clip(codes, q.qmin, q.qmax).astype(np.int64)


def dequantize(codes: np.ndarray, q: QParams) -> np.ndarray:
    return (codes.astype(np.float64) - q.zero_point) * q.scale


class _Int64View:
    """Dict view that hands out int64 copies of stored integer tensors (stored tensors may be int16)."""

    def __init__(self, d):
        self.d = d

    def __getitem__(self, k):
        v = self.d[k]
        return v if v.dtype == np.int64 else v.astype(np.int64)


class NumpyEngine:
    def __init__(self, graph: IntGraph):
        self.g = graph
        self.cfg = graph.requant
        self.saturations: dict[str, int] = {}

    # float32 has a 24-bit significand: a partial sum is exact while |sum| < 2^24. With |x-zp| <= 255 and
    # |w| <= 127 each product is < 2^15, so K_chunk * 255 * 127 < 2^24  <=>  K_chunk <= 518 terms per chunk.
    _F32_SAFE_K = 512

    def _mac(self, n: IntNode, x: np.ndarray) -> np.ndarray:
        """Integer conv/linear accumulation, exact: float32 kernels on input-channel chunks whose worst-case
        partial sums fit 24 bits, summed in int64 (falls back to float64 for grouped convs)."""
        in_q = self.g[n.inputs[0]].out_q
        xc = (x - in_q.zero_point)
        w = n.w_int
        if n.op == "conv":
            groups = n.attrs["groups"]
            cout, cin_g, kh, kw = w.shape
            k_total = cin_g * kh * kw
            if groups == 1 and k_total > self._F32_SAFE_K:
                ch_per_chunk = max(1, self._F32_SAFE_K // (kh * kw))
                acc = None
                for c0 in range(0, cin_g, ch_per_chunk):
                    xf = torch.from_numpy(xc[:, c0:c0 + ch_per_chunk].astype(np.float32))
                    wf = torch.from_numpy(w[:, c0:c0 + ch_per_chunk].astype(np.float32))
                    part = F.conv2d(xf, wf, None, n.attrs["stride"], n.attrs["padding"], 1, 1).numpy().astype(np.int64)
                    acc = part if acc is None else acc + part
                return acc
            dtype = np.float32 if k_total <= self._F32_SAFE_K else np.float64
            xf = torch.from_numpy(xc.astype(dtype)); wf = torch.from_numpy(w.astype(dtype))
            acc = F.conv2d(xf, wf, None, n.attrs["stride"], n.attrs["padding"], 1, groups).numpy()
        else:
            xf = torch.from_numpy(xc.astype(np.float64)); wf = torch.from_numpy(w.astype(np.float64))
            acc = (xf @ wf.T).numpy()
        return acc.astype(np.int64)

    def run(self, x_codes: np.ndarray, return_all: bool = False):
        """x_codes: (N,C,H,W) integer input codes. Returns output codes (and all intermediates).

        Memory: intermediate tensors are released as soon as every consumer has run (unless return_all), and
        captured tensors are stored as int16 (codes fit) — a 500-image MobileNetV2 batch otherwise needs >10 GB.
        """
        remaining = {n.name: len(self.g_users[n.name]) for n in self.g.nodes}
        vals: dict[str, np.ndarray] = {"__input__": x_codes.astype(np.int64)}
        keep: dict[str, np.ndarray] = {}
        for n in self.g.nodes:
            out = self.exec_node(n, vals)
            vals[n.name] = out
            if return_all:
                keep[n.name] = out.astype(np.int16)
            for src in n.inputs:
                remaining[src] -= 1
                if remaining[src] <= 0 and src in vals and not (return_all and src == "output"):
                    del vals[src]
        result = vals["output"]
        return keep if return_all else result

    @property
    def g_users(self) -> dict[str, list]:
        if not hasattr(self, "_users"):
            self._users = {n.name: [] for n in self.g.nodes}
            for n in self.g.nodes:
                for src in n.inputs:
                    self._users[src].append(n.name)
        return self._users

    def exec_node(self, n: IntNode, vals: dict) -> np.ndarray:
        """Execute one node given a dict of (already computed) input tensors (any integer dtype)."""
        cfg = self.cfg
        if "rounding" in n.attrs and n.attrs["rounding"] != cfg.rounding:     # per-node override (E11: TFLite's FC rounds once)
            from dataclasses import replace
            cfg = replace(cfg, rounding=n.attrs["rounding"])
        vals = _Int64View(vals)
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
                lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
                return np.clip(y, lo, hi)                          # clamp bounds implement the fused ReLU/ReLU6
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
                lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
                return np.clip(y, lo, hi)
            elif n.op == "const":
                return n.codes
            elif n.op == "matmul":
                q1, q2 = self.g[n.inputs[0]].out_q, self.g[n.inputs[1]].out_q
                a = (vals[n.inputs[0]] - q1.zero_point).astype(np.float64)
                b = (vals[n.inputs[1]] - q2.zero_point).astype(np.float64)
                acc = (torch.from_numpy(a) @ torch.from_numpy(b)).numpy().astype(np.int64)
                if cfg.acc_bits < 32:
                    sat = saturate(acc, cfg.acc_bits); self.saturations[n.name] = int((sat != acc).sum()); acc = sat
                else:
                    self.saturations[n.name] = int(((acc < INT32_MIN) | (acc > INT32_MAX)).sum())
                    acc = np.clip(acc, INT32_MIN, INT32_MAX)
                y = multiply_by_quantized_multiplier(acc, n.mult[0], n.shift[0], cfg.rounding) + n.out_q.zero_point
                lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
                return np.clip(y, lo, hi)
            elif n.op == "concat":
                parts = []
                for i, name in enumerate(n.inputs):
                    qi = self.g[name].out_q
                    y = multiply_by_quantized_multiplier(vals[name] - qi.zero_point, n.in_mult[i], n.in_shift[i], cfg.rounding)
                    parts.append(np.clip(y + n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax))
                return np.concatenate(parts, axis=n.attrs["dim"])
            elif n.op == "softmax":
                d = n.attrs["dim"]
                x = vals[n.inputs[0]]
                diff = x - x.max(axis=d, keepdims=True)              # in [-(qmax-qmin), 0]
                e = n.lut[diff + n.attrs["lut_offset"]]              # Q15 exp table
                total = e.sum(axis=d, keepdims=True)
                v = round_div(e << 15, np.maximum(total, 1))         # Q15 probability
                y = multiply_by_quantized_multiplier(v, n.mult[0], n.shift[0], cfg.rounding) + n.out_q.zero_point
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)
            elif n.op == "layernorm":
                in_q = self.g[n.inputs[0]].out_q
                c = n.attrs["channels"]
                d = vals[n.inputs[0]] - in_q.zero_point
                ssum = d.sum(axis=-1, keepdims=True)
                ssq = (d * d).sum(axis=-1, keepdims=True)
                var_num = c * ssq - ssum * ssum                      # = C^2 * variance(d), always >= 0
                denom = np.maximum(isqrt64(var_num + n.attrs["eps_int"]), 1)
                t = round_div((d * c - ssum) << 15, denom)           # Q15 normalized value
                y = multiply_by_quantized_multiplier(t, n.mult.reshape(1, -1), n.shift.reshape(1, -1), cfg.rounding)
                y = y + n.bias_int.reshape(1, -1) + n.out_q.zero_point
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)
            elif n.op == "transpose":
                return np.transpose(vals[n.inputs[0]], n.attrs["perm"])
            elif n.op == "reshape":
                x = vals[n.inputs[0]]
                return x.reshape((-1, *n.attrs["out_shape"]))
            elif n.op == "pool" and n.attrs.get("kind") == "token_mean":
                x = vals[n.inputs[0]]
                cnt = x.shape[1]
                s = x.sum(axis=1, keepdims=n.attrs.get("keepdim", False))
                y = np.where(s >= 0, (s + cnt // 2) // cnt, -((-s + cnt // 2) // cnt))
                return np.clip(y, n.out_q.qmin, n.out_q.qmax)
            elif n.op == "pool" and n.attrs.get("kind") == "global_avg_requant":
                # TFLite MEAN over H,W: sum of zero-point-centred codes, one requantization by s_in/(s_out*HW)
                in_q = self.g[n.inputs[0]].out_q
                x = vals[n.inputs[0]]
                acc = (x - in_q.zero_point).sum(axis=(2, 3), keepdims=True)
                y = multiply_by_quantized_multiplier(acc, n.mult[0], n.shift[0], cfg.rounding) + n.out_q.zero_point
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

    def evaluate(self, ds, batch_size: int = 200, limit: int | None = None, split: str = "test") -> float:
        """Top-1 accuracy of the integer graph on the 'test' (default) or 'val' split."""
        correct = 0; total = 0
        for xb, yb in ds.batches(split, batch_size):
            logits = self.predict(xb.numpy())
            correct += int((logits.argmax(1) == yb.numpy()).sum()); total += len(yb)
            if limit and total >= limit:
                break
        return correct / total
