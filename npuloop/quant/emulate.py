"""Integer-faithful LayerNorm inside the fake-quant graph.

A fake-quant graph computes LayerNorm in float32 and then snaps the result to the output grid; the integer
engine computes it with int64 sums, an exact integer square root, a Q15 normalized value and a fixed-point
requantization (`numpy_engine.py`, op "layernorm"). The two land on opposite sides of a rounding boundary in
about one element out of four (E9), which is where most of the transformer's fake-vs-integer gap is born.

`emulate_integer_layernorm(gm)` swaps every LayerNorm of a calibrated graph for `IntLayerNormEmu`: it recovers
the integer codes of its input (the tensor is the dequantized output of the preceding quantizer), runs the
engine's own arithmetic — the same functions, so the result is bit-exact by construction — and dequantizes
with the output quantizer's parameters. The quantizer that follows then re-quantizes idempotently.

Inference only: the integer path has no gradient, so the module falls back to float LayerNorm while the
quantizers are calibrating or disabled (and for QAT).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from .fake import FakeQuantAct
from .fold import _set_module
from ..intengine.requant import RequantConfig, quantize_multiplier, multiply_by_quantized_multiplier, isqrt64, round_div


class IntLayerNormEmu(nn.LayerNorm):
    """nn.LayerNorm whose forward reproduces the integer engine's LayerNorm bit for bit (see module doc)."""

    def __init__(self, ln: nn.LayerNorm, in_fq: FakeQuantAct, out_fq: FakeQuantAct, requant: RequantConfig = RequantConfig()):
        super().__init__(ln.normalized_shape, ln.eps, elementwise_affine=ln.elementwise_affine)
        if ln.elementwise_affine:
            self.weight.data.copy_(ln.weight.data); self.bias.data.copy_(ln.bias.data)
        # plain attributes (not submodules): the quantizers already live elsewhere in the graph
        object.__setattr__(self, "in_fq", in_fq)
        object.__setattr__(self, "out_fq", out_fq)
        self.requant = requant
        self.active = True
        self._cache_key = None
        self._params = None

    def extra_repr(self) -> str:
        return super().extra_repr() + f", integer_emulation={'on' if self.active else 'off'}"

    def _integer_params(self, s_in: float, s_out: float):
        w, b = self.weight, self.bias
        # gamma/beta/eps enter the multipliers and the bias, so a change to any of them must rebuild them
        key = (s_in, s_out, self.requant, self.eps,
               None if w is None else (w.data_ptr(), w._version, int(w.numel())),
               None if b is None else (b.data_ptr(), b._version, int(b.numel())))
        if self._cache_key != key:
            c = int(self.normalized_shape[0])
            gamma = self.weight.detach().double().numpy() if self.weight is not None else np.ones(c)
            beta = self.bias.detach().double().numpy() if self.bias is not None else np.zeros(c)
            eps_int = int(round(self.eps * c * c / (s_in ** 2)))
            qm = [quantize_multiplier(float(m), self.requant.mult_bits) for m in gamma / (2.0 ** 15 * s_out)]
            self._params = dict(c=c, eps_int=eps_int,
                                mult=np.array([q for q, _ in qm], dtype=np.int64).reshape(1, -1),
                                shift=np.array([s for _, s in qm], dtype=np.int64).reshape(1, -1),
                                bias_int=np.rint(beta / s_out).astype(np.int64).reshape(1, -1))
            self._cache_key = key
        return self._params

    def integer_forward(self, codes_minus_zp: np.ndarray, s_in: float, s_out: float, zp_out: int) -> np.ndarray:
        """Engine arithmetic on zero-point-centred int64 input codes; returns output codes."""
        p = self._integer_params(s_in, s_out)
        d = codes_minus_zp.astype(np.int64)
        ssum = d.sum(axis=-1, keepdims=True)
        ssq = (d * d).sum(axis=-1, keepdims=True)
        var_num = p["c"] * ssq - ssum * ssum
        denom = np.maximum(isqrt64(var_num + p["eps_int"]), 1)
        t = round_div((d * p["c"] - ssum) << 15, denom)
        y = multiply_by_quantized_multiplier(t, p["mult"], p["shift"], self.requant.rounding)
        y = y + p["bias_int"] + zp_out
        return np.clip(y, self.out_fq.qmin, self.out_fq.qmax)

    def forward(self, x):
        # inference only: the integer path is a NumPy round trip, so it carries no gradient. Training,
        # calibration and disabled quantizers all fall back to float LayerNorm (see the module docstring).
        ready = (self.active and not self.training and self.in_fq.enabled and self.out_fq.enabled
                 and not self.in_fq.calibrating and not self.out_fq.calibrating and bool(self.out_fq.initialized))
        if not ready or x.shape[-1] != self.normalized_shape[0]:
            return super().forward(x)
        s_in, _ = self.in_fq.qparams()
        s_out, zp_out = self.out_fq.qparams()
        d = np.rint(x.detach().double().numpy() / s_in)          # the input is (q - zp) * s_in: recover q - zp exactly
        y = self.integer_forward(d, s_in, s_out, zp_out)
        return torch.from_numpy(((y - zp_out) * s_out).astype(np.float32)).to(x.dtype)


def emulate_integer_layernorm(gm: fx.GraphModule, requant: RequantConfig = RequantConfig()) -> int:
    """Swap every LayerNorm of a prepared+calibrated graph for IntLayerNormEmu. Returns the number swapped."""
    modules = dict(gm.named_modules())
    swapped = 0
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(modules[node.target], nn.LayerNorm):
            continue
        if isinstance(modules[node.target], IntLayerNormEmu):
            continue
        src = node.args[0]
        users = list(node.users)
        if not (src.op == "call_module" and isinstance(modules[src.target], FakeQuantAct)):
            raise ValueError(f"{node.name}: LayerNorm input {src.name} is not a quantizer output")
        if not (len(users) == 1 and users[0].op == "call_module" and isinstance(modules[users[0].target], FakeQuantAct)):
            raise ValueError(f"{node.name}: LayerNorm output must feed exactly one quantizer")
        emu = IntLayerNormEmu(modules[node.target], modules[src.target], modules[users[0].target], requant)
        emu.train(modules[node.target].training)   # a freshly constructed module defaults to train(); inherit the graph's mode
        _set_module(gm, node.target, emu)
        swapped += 1
    return swapped


def set_layernorm_emulation(gm: fx.GraphModule, active: bool) -> None:
    for m in gm.modules():
        if isinstance(m, IntLayerNormEmu):
            m.active = active
