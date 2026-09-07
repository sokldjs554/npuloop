"""Range observers -> (scale, zero_point) for uniform affine quantization.

Conventions (match TFLite / typical INT8 NPUs):
  * activations: per-tensor, asymmetric uint8 by default (zp in [0,255]); symmetric int8 optional
  * weights: per-output-channel symmetric int8 (zp = 0)
Observers only track statistics; `qparams()` turns them into (scale, zero_point).
"""
from __future__ import annotations
import torch


def _qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1     # int8: [-127, 127] (no -128, TFLite style)
    return 0, 2 ** bits - 1                                    # uint8: [0, 255]


def affine_qparams(lo: torch.Tensor, hi: torch.Tensor, bits: int, symmetric: bool, pow2: bool = False):
    """Return (scale, zero_point) tensors for range [lo, hi] (elementwise over tensors)."""
    qmin, qmax = _qrange(bits, symmetric)
    lo = torch.minimum(lo, torch.zeros_like(lo))   # the range must contain 0 so that padding is exact
    hi = torch.maximum(hi, torch.zeros_like(hi))
    if symmetric:
        amax = torch.maximum(-lo, hi)
        scale = amax / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        if pow2:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        zp = torch.zeros_like(scale)
    else:
        scale = (hi - lo) / (qmax - qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        if pow2:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        zp = torch.clamp(torch.round(qmin - lo / scale), qmin, qmax)
    return scale, zp


class MinMaxObserver(torch.nn.Module):
    """Running min/max (plain for PTQ calibration, EMA for QAT)."""

    def __init__(self, ema: float | None = None):
        super().__init__()
        self.ema = ema
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor):
        x = x.detach()
        lo, hi = x.min(), x.max()
        if torch.isinf(self.lo) or self.ema is None:
            self.lo = torch.minimum(self.lo, lo); self.hi = torch.maximum(self.hi, hi)
        else:
            self.lo = self.ema * self.lo + (1 - self.ema) * lo
            self.hi = self.ema * self.hi + (1 - self.ema) * hi

    def range(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lo, self.hi

    def reset(self):
        self.lo.fill_(float("inf")); self.hi.fill_(float("-inf"))


class PercentileObserver(torch.nn.Module):
    """Clips to the [p, 1-p] quantiles of a sample of activations (robust to rare outliers)."""

    def __init__(self, percentile: float = 99.99, max_samples: int = 2_000_000):
        super().__init__()
        self.p = percentile / 100.0
        self.max_samples = max_samples
        self.samples: list[torch.Tensor] = []
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor):
        x = x.detach().flatten()
        if x.numel() > 200_000:
            idx = torch.randint(0, x.numel(), (200_000,), generator=torch.Generator().manual_seed(len(self.samples)))
            x = x[idx]
        self.samples.append(x)
        total = sum(s.numel() for s in self.samples)
        while total > self.max_samples and len(self.samples) > 1:
            total -= self.samples.pop(0).numel()
        allx = torch.cat(self.samples)
        self.lo = torch.quantile(allx, 1 - self.p); self.hi = torch.quantile(allx, self.p)

    def range(self):
        return self.lo, self.hi

    def reset(self):
        self.samples = []; self.lo.fill_(float("inf")); self.hi.fill_(float("-inf"))


class MSEObserver(torch.nn.Module):
    """Searches the clipping ratio that minimises the quantization MSE on the calibration sample."""

    def __init__(self, bits: int = 8, symmetric: bool = False, steps: int = 40, max_samples: int = 1_000_000):
        super().__init__()
        self.bits, self.symmetric, self.steps, self.max_samples = bits, symmetric, steps, max_samples
        self.samples: list[torch.Tensor] = []
        self.register_buffer("lo", torch.tensor(float("inf")))
        self.register_buffer("hi", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor):
        x = x.detach().flatten()
        if x.numel() > 100_000:
            idx = torch.randint(0, x.numel(), (100_000,), generator=torch.Generator().manual_seed(len(self.samples)))
            x = x[idx]
        self.samples.append(x)
        total = sum(s.numel() for s in self.samples)
        while total > self.max_samples and len(self.samples) > 1:
            total -= self.samples.pop(0).numel()
        allx = torch.cat(self.samples)
        lo0, hi0 = allx.min(), allx.max()
        best, best_err = (lo0, hi0), float("inf")
        qmin, qmax = _qrange(self.bits, self.symmetric)
        for i in range(self.steps):
            r = 1.0 - 0.6 * i / (self.steps - 1)          # clip ratios 1.0 -> 0.4
            lo, hi = lo0 * r, hi0 * r
            scale, zp = affine_qparams(lo, hi, self.bits, self.symmetric)
            q = torch.clamp(torch.round(allx / scale) + zp, qmin, qmax)
            err = ((q - zp) * scale - allx).pow(2).mean().item()
            if err < best_err:
                best, best_err = (lo, hi), err
        self.lo, self.hi = best[0].clone(), best[1].clone()

    def range(self):
        return self.lo, self.hi

    def reset(self):
        self.samples = []; self.lo.fill_(float("inf")); self.hi.fill_(float("-inf"))


def make_observer(kind: str, bits: int = 8, symmetric: bool = False, ema: float | None = None):
    if kind == "minmax":
        return MinMaxObserver(ema=ema)
    if kind.startswith("percentile"):
        p = float(kind.split(":")[1]) if ":" in kind else 99.99
        return PercentileObserver(p)
    if kind == "mse":
        return MSEObserver(bits=bits, symmetric=symmetric)
    raise ValueError(f"unknown observer {kind}")


def weight_qparams(w: torch.Tensor, bits: int = 8, per_channel: bool = True, method: str = "minmax", pow2: bool = False):
    """Per-output-channel (or per-tensor) symmetric weight scales. Returns (scale[C] or scale[1], zp zeros)."""
    qmax = 2 ** (bits - 1) - 1
    wf = w.detach().reshape(w.shape[0], -1) if per_channel else w.detach().reshape(1, -1)
    amax = wf.abs().max(dim=1).values
    if method == "mse":
        best = amax.clone(); best_err = torch.full_like(amax, float("inf"))
        for i in range(30):
            r = 1.0 - 0.5 * i / 29
            s = (amax * r / qmax).clamp_min(1e-12)
            q = torch.clamp(torch.round(wf / s[:, None]), -qmax, qmax) * s[:, None]
            err = (q - wf).pow(2).mean(dim=1)
            better = err < best_err
            best = torch.where(better, amax * r, best); best_err = torch.where(better, err, best_err)
        amax = best
    # an all-zero channel (e.g. BN gamma = 0) has no meaningful scale; give it the largest channel's scale so the
    # requantization multiplier stays representable and its int32 bias survives (a 1e-12 scale would make the
    # multiplier underflow to zero and silently drop the bias in the integer engine)
    fallback = amax.max() if float(amax.max()) > 0 else torch.tensor(1.0, dtype=amax.dtype)
    amax = torch.where(amax > 0, amax, fallback)
    scale = (amax / qmax).clamp_min(1e-12)
    if pow2:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return scale, torch.zeros_like(scale)
