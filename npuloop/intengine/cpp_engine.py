"""C++ integer kernels (compiled on first use) driving the same IntGraph as the NumPy engine."""
from __future__ import annotations
import ctypes, platform
import hashlib
import os
import subprocess
import numpy as np
from .graph import IntGraph, IntNode
from .numpy_engine import NumpyEngine

_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "cpp", "int8_engine.cpp")
ROUNDING = {"tflite": 0, "half_even": 1, "truncate": 2, "floor": 3, "single": 4}
_lib = None


def _cpu_tag() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def build_flags() -> list[str]:
    """-O3 plus the host ISA (-march=native) unless NPULOOP_CPP_NATIVE=0; the build is cached per source+flags+CPU."""
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC"]
    if os.environ.get("NPULOOP_CPP_NATIVE", "1") != "0":
        flags.append("-march=native")
    return flags


def build_library(force: bool = False) -> str:
    """Compile int8_engine.cpp into a shared library (cached by source hash, compiler flags and CPU model)."""
    src = open(_SRC, "rb").read()
    flags = build_flags()
    tag = hashlib.sha1(src + " ".join(flags).encode() + _cpu_tag().encode()).hexdigest()[:10]
    out_dir = os.path.join(_HERE, "cpp", "build")
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, f"libint8engine_{tag}.so")
    if force or not os.path.exists(so):
        try:
            subprocess.run(["g++", *flags, "-o", so, _SRC], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if "-march=native" not in flags:
                raise RuntimeError(e.stderr.decode(errors="replace")) from e
            flags.remove("-march=native")            # compilers without -march=native: fall back to generic
            subprocess.run(["g++", *flags, "-o", so, _SRC], check=True)
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
        _lib.global_avgpool_requant.argtypes = [i32p, c, c, c, c, ctypes.c_int32, c, c, c, c, c, i32p]
        _lib.lut_apply.argtypes = [i32p, ctypes.c_int64, i32p, c, i32p]
        _lib.token_mean.argtypes = [i32p, c, c, c, c, c, i32p]
        _lib.matmul_requant.argtypes = [i32p, i32p, c, c, c, c, c, c, ctypes.c_int32, c, c, c, c, c, c, i32p, i64p]
        _lib.concat_requant.argtypes = [i32p, ctypes.c_int64, c, ctypes.c_int32, c, c, c, c, c, i32p]
        _lib.softmax_int.argtypes = [i32p, ctypes.c_int64, c, ctypes.c_int64, i32p, c, ctypes.c_int32, c, c, c, c, c, i32p]
        _lib.layernorm_int.argtypes = [i32p, ctypes.c_int64, c, c, ctypes.c_int64, i32p, i32p, i32p, c, c, c, c, i32p]
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
        if n.op == "input":
            return vals["__input__"]
        rounding_saved = self.rounding
        if "rounding" in n.attrs:
            self.rounding = ROUNDING[n.attrs["rounding"]]
        try:
            return self._exec(n, vals)
        finally:
            self.rounding = rounding_saved

    def _exec(self, n: IntNode, vals: dict) -> np.ndarray:
        lib, cfg = self.lib, self.cfg
        if n.op == "conv":
            x = _i32(vals[n.inputs[0]]); N, C, H, W = x.shape
            w = np.ascontiguousarray(n.w_int, dtype=np.int8); Cout, cin_g, kh, kw = w.shape
            sh, sw = n.attrs["stride"]; ph, pw = n.attrs["padding"]
            Ho = (H + 2 * ph - kh) // sh + 1; Wo = (W + 2 * pw - kw) // sw + 1
            out = np.empty((N, Cout, Ho, Wo), dtype=np.int32)
            in_q = self.g[n.inputs[0]].out_q
            bias, mult, shift = _i32(n.bias_int), _i32(n.mult), _i32(n.shift)
            nsat = np.zeros(1, dtype=np.int64)
            lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
            lib.conv_requant(_ptr(x), N, C, H, W, _ptr(w, ctypes.c_int8), Cout, kh, kw, sh, sw, ph, pw, n.attrs["groups"],
                             in_q.zero_point, _ptr(bias), _ptr(mult), _ptr(shift), n.out_q.zero_point, lo, hi,
                             self.rounding, cfg.acc_bits, _ptr(out), Ho, Wo, _ptr(nsat, ctypes.c_int64))
            self.saturations[n.name] = int(nsat[0])
            return out.astype(np.int64)
        if n.op == "matmul":
            a = _i32(vals[n.inputs[0]]); b = _i32(vals[n.inputs[1]])
            M, K = a.shape[-2], a.shape[-1]; N = b.shape[-1]
            batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1
            a3 = np.ascontiguousarray(a.reshape(batch, M, K)); b3 = np.ascontiguousarray(b.reshape(batch, K, N))
            out = np.empty((batch, M, N), dtype=np.int32)
            q1, q2 = self.g[n.inputs[0]].out_q, self.g[n.inputs[1]].out_q
            nsat = np.zeros(1, dtype=np.int64)
            lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
            lib.matmul_requant(_ptr(a3), _ptr(b3), batch, M, K, N, q1.zero_point, q2.zero_point,
                               int(n.mult[0]), int(n.shift[0]), n.out_q.zero_point, lo, hi,
                               self.rounding, cfg.acc_bits, _ptr(out), _ptr(nsat, ctypes.c_int64))
            self.saturations[n.name] = int(nsat[0])
            return out.astype(np.int64).reshape(*a.shape[:-2], M, N)
        if n.op == "concat":
            parts = []
            for i, name in enumerate(n.inputs):
                x = _i32(vals[name]); out = np.empty_like(x)
                qi = self.g[name].out_q
                lib.concat_requant(_ptr(x), x.size, qi.zero_point, int(n.in_mult[i]), int(n.in_shift[i]),
                                   n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax, self.rounding, _ptr(out))
                parts.append(out.astype(np.int64))
            return np.concatenate(parts, axis=n.attrs["dim"])
        if n.op == "softmax":
            x = _i32(vals[n.inputs[0]]); d = n.attrs["dim"]
            outer = int(np.prod(x.shape[:d])) if d > 0 else 1
            inner = int(np.prod(x.shape[d + 1:])) if d + 1 < x.ndim else 1
            table = _i32(n.lut); out = np.empty_like(x)
            lib.softmax_int(_ptr(x), outer, x.shape[d], inner, _ptr(table), n.attrs["lut_offset"],
                            int(n.mult[0]), int(n.shift[0]), n.out_q.zero_point, n.out_q.qmin, n.out_q.qmax,
                            self.rounding, _ptr(out))
            return out.astype(np.int64)
        if n.op == "layernorm":
            x = _i32(vals[n.inputs[0]]); c = n.attrs["channels"]
            rows = x.size // c
            out = np.empty_like(x)
            in_q = self.g[n.inputs[0]].out_q
            mult, shift, bias = _i32(n.mult), _i32(n.shift), _i32(n.bias_int)
            lib.layernorm_int(_ptr(x), rows, c, in_q.zero_point, int(n.attrs["eps_int"]),
                              _ptr(mult), _ptr(shift), _ptr(bias), n.out_q.zero_point,
                              n.out_q.qmin, n.out_q.qmax, self.rounding, _ptr(out))
            return out.astype(np.int64)
        if n.op == "pool" and n.attrs.get("kind") == "token_mean":
            x = _i32(vals[n.inputs[0]]); N, T, C = x.shape
            out = np.empty((N, C), dtype=np.int32)
            lib.token_mean(_ptr(x), N, T, C, n.out_q.qmin, n.out_q.qmax, _ptr(out))
            y = out.astype(np.int64)
            return y.reshape(N, 1, C) if n.attrs.get("keepdim") else y
        if n.op == "linear":
            x = _i32(vals[n.inputs[0]])
            if x.ndim > 2:                       # token-wise linear: fold the token axis into rows
                lead, K = x.shape[:-1], x.shape[-1]
                x = np.ascontiguousarray(x.reshape(-1, K))
                w = np.ascontiguousarray(n.w_int, dtype=np.int8); M = w.shape[0]
                out = np.empty((x.shape[0], M), dtype=np.int32)
                in_q = self.g[n.inputs[0]].out_q
                bias, mult, shift = _i32(n.bias_int), _i32(n.mult), _i32(n.shift)
                lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
                lib.linear_requant(_ptr(x), x.shape[0], K, _ptr(w, ctypes.c_int8), M, in_q.zero_point,
                                   _ptr(bias), _ptr(mult), _ptr(shift), n.out_q.zero_point, lo, hi,
                                   self.rounding, cfg.acc_bits, _ptr(out))
                return out.astype(np.int64).reshape(*lead, M)
            N, K = x.shape
            w = np.ascontiguousarray(n.w_int, dtype=np.int8); M = w.shape[0]
            out = np.empty((N, M), dtype=np.int32)
            in_q = self.g[n.inputs[0]].out_q
            bias, mult, shift = _i32(n.bias_int), _i32(n.mult), _i32(n.shift)
            lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
            lib.linear_requant(_ptr(x), N, K, _ptr(w, ctypes.c_int8), M, in_q.zero_point, _ptr(bias), _ptr(mult), _ptr(shift),
                               n.out_q.zero_point, lo, hi, self.rounding, cfg.acc_bits, _ptr(out))
            return out.astype(np.int64)
        if n.op == "add":
            a, b = _i32(vals[n.inputs[0]]), _i32(vals[n.inputs[1]])
            if a.shape != b.shape:               # a constant operand (e.g. a positional embedding) broadcasts
                shape = np.broadcast_shapes(a.shape, b.shape)
                a = np.ascontiguousarray(np.broadcast_to(a, shape)); b = np.ascontiguousarray(np.broadcast_to(b, shape))
            p = n.add_params; q1, q2 = self.g[n.inputs[0]].out_q, self.g[n.inputs[1]].out_q
            out = np.empty_like(a)
            lo, hi = n.attrs.get("clamp", (n.out_q.qmin, n.out_q.qmax))
            lib.add_requant(_ptr(a), _ptr(b), a.size, q1.zero_point, q2.zero_point, p["left_shift"],
                            int(p["m1"][0]), int(p["m1"][1]), int(p["m2"][0]), int(p["m2"][1]), int(p["mo"][0]), int(p["mo"][1]),
                            n.out_q.zero_point, lo, hi, self.rounding, _ptr(out))
            return out.astype(np.int64)
        if n.op == "pool" and n.attrs.get("kind") == "global_avg_requant":
            x = _i32(vals[n.inputs[0]]); N, C, H, W = x.shape
            out = np.empty((N, C, 1, 1), dtype=np.int32)
            in_q = self.g[n.inputs[0]].out_q
            lib.global_avgpool_requant(_ptr(x), N, C, H * W, in_q.zero_point, int(n.mult[0]), int(n.shift[0]),
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
