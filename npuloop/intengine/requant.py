"""Fixed-point requantization primitives, bit-exact with gemmlowp / TFLite reference kernels.

  QuantizeMultiplier(M)                -> (q31 multiplier, shift)  with M = q * 2^shift, q in [0.5, 1)
  SaturatingRoundingDoublingHighMul    -> (a*b + nudge) >> 31 with saturation (int32 x int32 -> int32)
  RoundingDivideByPOT                  -> rounding arithmetic right shift (round half away from zero)
  MultiplyByQuantizedMultiplier        -> the requant op used by every conv/linear/add output

All functions are vectorised NumPy on int64 arrays and return int64 arrays holding int32 values.
`RequantConfig` exposes deliberate deviations (fewer multiplier bits, truncation, narrow accumulators)
so the accuracy cost of *sloppy* integer implementations can be measured.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

INT32_MIN, INT32_MAX = -(2 ** 31), 2 ** 31 - 1


@dataclass(frozen=True)
class RequantConfig:
    rounding: str = "tflite"      # tflite (round half away from zero) | half_even | truncate | floor
    mult_bits: int = 31           # bits of the fixed-point multiplier (31 = TFLite; 15/7 = cheap hardware)
    acc_bits: int = 32            # accumulator width; narrower accumulators saturate
    bias_bits: int = 32           # bias width (int32 default; 16 = cheap hardware, saturates)

    def to_dict(self):
        return asdict(self)

    @property
    def tag(self):
        return f"{self.rounding}-m{self.mult_bits}-acc{self.acc_bits}-b{self.bias_bits}"


def quantize_multiplier(m: float, mult_bits: int = 31) -> tuple[int, int]:
    """TFLite QuantizeMultiplier: m = q * 2^shift with q a (mult_bits+1)-bit fixed-point in [0.5, 1)."""
    if m == 0.0:
        return 0, 0
    if m < 0:
        raise ValueError("negative multiplier")
    q, shift = math.frexp(m)                    # m = q * 2^shift, q in [0.5, 1)
    q_fixed = int(round(q * (1 << 31)))
    if mult_bits < 31:                          # cheap hardware: fewer significant bits, zero the rest
        drop = 31 - mult_bits
        q_fixed = int(round(q_fixed / (1 << drop))) << drop
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:                             # underflow -> multiplier 0
        return 0, 0
    return q_fixed, shift


def _srdhm(a: np.ndarray, b: np.ndarray | int) -> np.ndarray:
    """SaturatingRoundingDoublingHighMul on int64 arrays holding int32 values."""
    a = a.astype(np.int64); b = np.asarray(b, dtype=np.int64)
    ab = a * b                                                    # |ab| < 2^62 -> fits int64
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30)).astype(np.int64)
    v = ab + nudge
    hi = np.where(v >= 0, v >> 31, -((-v) >> 31))                 # truncating division by 2^31
    overflow = (a == INT32_MIN) & (b == INT32_MIN)
    return np.where(overflow, INT32_MAX, hi)


def _rdbpot(x: np.ndarray, exponent: np.ndarray | int, rounding: str = "tflite") -> np.ndarray:
    """RoundingDivideByPOT: divide by 2^exponent (exponent >= 0)."""
    x = x.astype(np.int64); e = np.asarray(exponent, dtype=np.int64)
    if rounding == "truncate":
        return np.where(x >= 0, x >> e, -((-x) >> e))
    if rounding == "floor":
        return x >> e
    mask = (np.int64(1) << e) - 1
    remainder = x & mask
    if rounding == "half_even":
        half = (mask + 1) >> 1
        q = x >> e
        up = (remainder > half) | ((remainder == half) & (q & 1 == 1))
        return q + up.astype(np.int64)
    threshold = (mask >> 1) + (x < 0).astype(np.int64)            # tflite: round half away from zero
    return (x >> e) + (remainder > threshold).astype(np.int64)


def multiply_by_quantized_multiplier(x: np.ndarray, q: np.ndarray | int, shift: np.ndarray | int,
                                     rounding: str = "tflite") -> np.ndarray:
    """MultiplyByQuantizedMultiplier(x, q, shift) = round(x * q * 2^shift / 2^31), bit-exact with TFLite."""
    q = np.asarray(q, dtype=np.int64); shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0); right = np.maximum(-shift, 0)
    xs = x.astype(np.int64) * (np.int64(1) << left)
    hi = _srdhm(xs, q)
    return _rdbpot(hi, right, rounding)


def saturate(x: np.ndarray, bits: int) -> np.ndarray:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(x, lo, hi)


def requant_reference(x: np.ndarray, real_multiplier: float) -> np.ndarray:
    """Float reference (round half to even) — what a fake-quant model computes for the same tensor."""
    return np.rint(x.astype(np.float64) * real_multiplier).astype(np.int64)
