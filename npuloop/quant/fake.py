"""Fake-quantization modules with straight-through estimators (optionally LSQ-learnable scales)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .observers import make_observer, affine_qparams, weight_qparams, _qrange


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


def fake_quant(x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, qmin: int, qmax: int, ch_axis: int | None = None):
    """Quantize-dequantize with STE on rounding; gradients also flow to `scale` (LSQ-style) if it requires grad."""
    if ch_axis is None:
        q = torch.clamp(round_ste(x / scale) + zp, qmin, qmax)
        return (q - zp) * scale
    shape = [1] * x.ndim; shape[ch_axis] = -1
    s = scale.reshape(shape); z = zp.reshape(shape)
    q = torch.clamp(round_ste(x / s) + z, qmin, qmax)
    return (q - z) * s


class FakeQuantAct(nn.Module):
    """Per-tensor activation fake quantizer.

    Modes: calibrating (observe only, pass through), enabled (fake quantize), disabled (pass through).
    `tied_to` lets a pool output reuse its input's quantization parameters (NPU avgpool keeps the scale).
    """

    def __init__(self, bits: int = 8, symmetric: bool = False, observer: str = "minmax", ema: float | None = None,
                 learnable: bool = False, pow2: bool = False, name: str = ""):
        super().__init__()
        self.bits, self.symmetric, self.pow2, self.name = bits, symmetric, pow2, name
        self.qmin, self.qmax = _qrange(bits, symmetric)
        self.observer = make_observer(observer, bits, symmetric, ema)
        self.observer_kind = observer
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=learnable)
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrating = False
        self.enabled = False
        self.learnable = learnable
        self.tied_to: FakeQuantAct | None = None
        self.register_buffer("initialized", torch.tensor(False))

    def extra_repr(self) -> str:
        return f"bits={self.bits} sym={self.symmetric} obs={self.observer_kind} scale={float(self.scale):.5g} zp={int(self.zero_point)} on={self.enabled}"

    def compute_qparams(self):
        if self.tied_to is not None:
            self.scale.data.copy_(self.tied_to.scale.data); self.zero_point.copy_(self.tied_to.zero_point)
        else:
            lo, hi = self.observer.range()
            if torch.isinf(lo) or torch.isinf(hi):
                return
            s, z = affine_qparams(lo, hi, self.bits, self.symmetric, self.pow2)
            self.scale.data.copy_(s); self.zero_point.copy_(z)
        self.initialized.fill_(True)

    def forward(self, x):
        if self.calibrating and self.tied_to is None:
            if self.enabled and self.observer_kind != "minmax":
                pass          # percentile/MSE searches are far too expensive per training step; ranges stay frozen
            else:
                self.observer(x)
                if self.enabled or (self.observer_kind == "minmax" and self.observer.ema is not None):
                    self.compute_qparams()      # QAT: track ranges as the weights move
        if not self.enabled:
            return x
        if self.tied_to is not None:
            return fake_quant(x, self.tied_to.scale, self.tied_to.zero_point, self.qmin, self.qmax)
        return fake_quant(x, self.scale, self.zero_point, self.qmin, self.qmax)

    def qparams(self) -> tuple[float, int]:
        src = self.tied_to if self.tied_to is not None else self
        return float(src.scale), int(src.zero_point)


class QConv2d(nn.Conv2d):
    """Conv2d with per-output-channel symmetric fake-quantized weights (BN already folded)."""

    def __init__(self, *args, w_bits: int = 8, per_channel: bool = True, w_method: str = "minmax",
                 learnable: bool = False, pow2: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits, self.per_channel, self.w_method, self.pow2 = w_bits, per_channel, w_method, pow2
        self.qmin, self.qmax = _qrange(w_bits, True)
        n = self.out_channels if per_channel else 1
        self.w_scale = nn.Parameter(torch.ones(n), requires_grad=learnable)
        self.w_enabled = False
        self.learnable = learnable

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, **q):
        m = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                conv.groups, bias=True, **q)
        m.weight.data.copy_(conv.weight.data)
        m.bias.data.copy_(conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels))
        return m

    def init_w_scale(self):
        s, _ = weight_qparams(self.weight, self.w_bits, self.per_channel, self.w_method, self.pow2)
        self.w_scale.data.copy_(s)

    def quantized_weight(self) -> torch.Tensor:
        if not self.w_enabled:
            return self.weight
        ch = 0 if self.per_channel else None
        z = torch.zeros_like(self.w_scale)
        return fake_quant(self.weight, self.w_scale, z, self.qmin, self.qmax, ch_axis=ch)

    def int_weight(self) -> torch.Tensor:
        """int8 weights (as int32 tensor) and matching scale."""
        s = self.w_scale if self.per_channel else self.w_scale.expand(self.out_channels)
        q = torch.clamp(torch.round(self.weight.detach() / s.reshape(-1, 1, 1, 1)), self.qmin, self.qmax)
        return q.to(torch.int32)

    def forward(self, x):
        return F.conv2d(x, self.quantized_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class QLinear(nn.Linear):
    def __init__(self, *args, w_bits: int = 8, per_channel: bool = True, w_method: str = "minmax",
                 learnable: bool = False, pow2: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits, self.per_channel, self.w_method, self.pow2 = w_bits, per_channel, w_method, pow2
        self.qmin, self.qmax = _qrange(w_bits, True)
        n = self.out_features if per_channel else 1
        self.w_scale = nn.Parameter(torch.ones(n), requires_grad=learnable)
        self.w_enabled = False
        self.learnable = learnable

    @classmethod
    def from_linear(cls, lin: nn.Linear, **q):
        m = cls(lin.in_features, lin.out_features, bias=True, **q)
        m.weight.data.copy_(lin.weight.data)
        m.bias.data.copy_(lin.bias.data if lin.bias is not None else torch.zeros(lin.out_features))
        return m

    def init_w_scale(self):
        s, _ = weight_qparams(self.weight, self.w_bits, self.per_channel, self.w_method, self.pow2)
        self.w_scale.data.copy_(s)

    def quantized_weight(self):
        if not self.w_enabled:
            return self.weight
        ch = 0 if self.per_channel else None
        return fake_quant(self.weight, self.w_scale, torch.zeros_like(self.w_scale), self.qmin, self.qmax, ch_axis=ch)

    def int_weight(self):
        s = self.w_scale if self.per_channel else self.w_scale.expand(self.out_features)
        return torch.clamp(torch.round(self.weight.detach() / s.reshape(-1, 1)), self.qmin, self.qmax).to(torch.int32)

    def forward(self, x):
        return F.linear(x, self.quantized_weight(), self.bias)
