"""C++ integer kernels (compiled on first use) driving the same IntGraph as the NumPy engine."""
from __future__ import annotations
import ctypes
import hashlib
import os
import subprocess
import numpy as np
from .graph import IntGraph, IntNode
from .numpy_engine import NumpyEngine, quantize_input, dequantize

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "cpp", "int8_engine.cpp")
ROUNDING = {"tflite": 0, "half_even": 1, "truncate": 2, "floor": 3, "single": 4}
_lib = None


def build_library(force: bool = False) -> str:
    """Compile int8_engine.cpp into a shared library (cached by source hash)."""
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src).hexdigest()[:10]
    out_dir = os.path.join(_HERE, "cpp", "build")
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, f"libint8engine_{tag}.so")
    if force or not os.path.exists(so):
        cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-o", so, _SRC]
        subprocess.run(cmd, check=True)
    return so


def load_library():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build_library())
        i32p = ctypes.POINTER(ctypes.c_int32); i8p = ctypes.POINTER(ctypes.c_int8); i64p = ctypes.POINTER(ctypes.c_int64)
        c = ctypes.c_int
        _lib.conv_requant.argtypes = [i32p, c, c, c, c, i8p, c, c, c, c, c, c, c, c, c, i32p, i32p, i32p, c, c, c, c, c, i32p, c, c, i64p]
        _lib.linear_requant.argtypes = [i32p, c, c, i8p, c, c, i32p, i32p, i32p, c, c, c, c, c, i32p]
        _lib.add_requant.argtypes = [i32p, i32p, ctypes.c_int64, c, c, c, ctypes.c_int32, c, ctypes.c_int32, c, ctypes.c_int32, c, c, c, c, c, i32p]
        _lib.global_avgpool.argtypes = [i32p, c, c, c, c, c, i32p]
        _lib.lut_apply.argtypes = [i32p, ctypes.c_int64, i32p, c, i32p]
        _lib.test_mbqm.argtypes = [ctypes.c_int32, ctypes.c_int32, c, c]; _lib.test_mbqm.restype = ctypes.c_int32
        _lib.test_srdhm.argtypes = [ctypes.c_int32, ctypes.c_int32]; _lib.test_srdhm.restype = ctypes.c_int32
        _lib.test_rdbpot.argtypes = [ctypes.c_int32, c, c]; _lib.test_rdbpot.restype = ctypes.c_int32
    return _lib


def _i32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _ptr(a: np.ndarray, t=ctypes.c_int32):
    return a.ctypes.data_as(ctypes.POINTER(t))


class CppEngine(NumpyEngine):
    """Same graph walk as NumpyEngine, every op executed by the C++ kernels."""

    def __init__(self, graph: IntGraph):
        super().__init__(graph)
        self.lib = load_library()
        self.rounding = ROUNDING[graph.requant.rounding]

    def exec_node(self, n: IntNode, vals: dict) -> np.ndarray:
        lib, cfg = self.lib, self.cfg
        if n.op == "input":
            return vals["__input__"]
        if n.op == "conv":
            x = _i32(vals[n.inputs[0]]); N, C, H, W = x.shape
            w = np.ascontiguousarray(n.w_int, dtype=np.int8); Cout, cin_g, kh, kw = w.shape
            sh, sw = n.attrs["stride"]; ph, pw = n.attrs["padding"]
            Ho = (H + 2 * ph - kh) // sh + 1; Wo = (W + 2 * pw - kw) // sw + 1
            out = np.empty((N, Cout, Ho, Wo), dtype=np.int32)
            in_q = self.g[n.inputs[0]].out_q
            bias, mult, shift = _i32(n.bias_int), _i32(n.mult), _i32(n.shift)
            nsat = np.zeros(1, dtype=np.int64)
            lib.conv_requant(_ptr(x), N, C, H, W, _ptr(w, ctypes.c_int8), Cout, kh, kw, sh, sw, ph, pw, n.attrs["groups"],
                             in_q.zero_point, _ptr(bias), _ptr(mult), _ptr(shift), n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax,
                             self.rounding, cfg.acc_bits, _ptr(out), Ho, Wo, _ptr(nsat, ctypes.c_int64))
            self.saturations[n.name] = int(nsat[0])
            return out.astype(np.int64)
        if n.op == "linear":
            x = _i32(vals[n.inputs[0]]); N, K = x.shape
            w = np.ascontiguousarray(n.w_int, dtype=np.int8); M = w.shape[0]
            out = np.empty((N, M), dtype=np.int32)
            in_q = self.g[n.inputs[0]].out_q
            bias, mult, shift = _i32(n.bias_int), _i32(n.mult), _i32(n.shift)
            lib.linear_requant(_ptr(x), N, K, _ptr(w, ctypes.c_int8), M, in_q.zero_point, _ptr(bias), _ptr(mult), _ptr(shift),
                               n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax, self.rounding, cfg.acc_bits, _ptr(out))
            return out.astype(np.int64)
        if n.op == "add":
            a, b = _i32(vals[n.inputs[0]]), _i32(vals[n.inputs[1]])
            p = n.add_params; q1, q2 = self.g[n.inputs[0]].out_q, self.g[n.inputs[1]].out_q
            out = np.empty_like(a)
            lib.add_requant(_ptr(a), _ptr(b), a.size, q1.zero_point, q2.zero_point, p["left_shift"],
                            int(p["m1"][0]), int(p["m1"][1]), int(p["m2"][0]), int(p["m2"][1]), int(p["mo"][0]), int(p["mo"][1]),
                            n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax, self.rounding, _ptr(out))
            return out.astype(np.int64)
        if n.op == "pool":
            x = _i32(vals[n.inputs[0]]); N, C, H, W = x.shape
            out = np.empty((N, C, 1, 1), dtype=np.int32)
            lib.global_avgpool(_ptr(x), N, C, H * W, n.out_q.qmin, n.out_q.qmax, _ptr(out))
            return out.astype(np.int64)
        if n.op == "lut":
            x = _i32(vals[n.inputs[0]]); table = _i32(n.lut)
            out = np.empty_like(x)
            lib.lut_apply(_ptr(x), x.size, _ptr(table), n.attrs["lut_qmin"], _ptr(out))
            return out.astype(np.int64)
        return super().exec_node(n, vals)
