"""Quantization scheme description (what the target NPU supports)."""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class QScheme:
    w_bits: int = 8
    a_bits: int = 8
    w_per_channel: bool = True        # per-output-channel symmetric weights (False -> per-tensor)
    w_method: str = "minmax"          # minmax | mse
    a_symmetric: bool = False         # uint8 asymmetric activations by default
    a_observer: str = "minmax"        # minmax | percentile[:p] | mse
    pow2: bool = False                # restrict all scales to powers of two
    learnable: bool = False           # LSQ-style learnable scales for QAT
    ema: float | None = None          # EMA min/max during QAT (None = plain min/max)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tag(self) -> str:
        return (f"w{self.w_bits}{'pc' if self.w_per_channel else 'pt'}-{self.w_method}"
                f"_a{self.a_bits}{'s' if self.a_symmetric else 'u'}-{self.a_observer}" + ("_pow2" if self.pow2 else ""))


PRESET_SCHEMES = {
    "npu-default": QScheme(),                                              # per-channel W, uint8 A, min-max
    "npu-percentile": QScheme(a_observer="percentile:99.99"),
    "npu-mse": QScheme(w_method="mse", a_observer="mse"),
    "per-tensor": QScheme(w_per_channel=False),                            # NPUs without per-channel support
    "per-tensor-mse": QScheme(w_per_channel=False, w_method="mse", a_observer="mse"),
    "pow2": QScheme(pow2=True),                                            # shift-only requantization NPUs
    "sym-act": QScheme(a_symmetric=True),
}
