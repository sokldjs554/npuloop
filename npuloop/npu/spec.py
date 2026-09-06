"""Parametric description of an INT8 edge NPU ("virtual NPU").

The presets are *assumptions* matched to publicly quoted headline numbers (TOPS, power,
DRAM bandwidth) of commercial edge NPUs. They are not a description of any vendor's
real microarchitecture. Provenance label for everything derived from them: `simulated`.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class NPUSpec:
    name: str
    pe_rows: int = 64            # reduction dimension (Cin*kh*kw) of the weight-stationary array
    pe_cols: int = 64            # output-channel dimension of the array
    cores: int = 1               # identical cores; the mapper splits M (spatial) or N (Cout) across them
    freq_mhz: float = 1000.0
    sram_kb: float = 2048.0      # on-chip activation buffer (per chip)
    dram_gbps: float = 12.8      # DRAM bandwidth shared by all cores
    vector_lanes: int = 256      # elements / cycle / core for add, pool, LUT activations
    dw_lanes: int = 256          # MACs / cycle / core of the depthwise engine (0 => depthwise runs on the array)
    lut_acts: frozenset = frozenset({"silu", "gelu", "hswish", "hsigmoid", "sigmoid", "tanh", "lrelu"})
    fused_acts: frozenset = frozenset({"relu", "relu6"})   # folded into the requantization clamp (free)
    unsupported_ops: frozenset = frozenset()              # op kinds that fall back to the host CPU
    fallback_cycles_per_elem: float = 8.0                 # cost of a host-side elementwise op (very rough)
    fill_drain: bool = True                               # count systolic pipeline fill/drain per weight tile

    @property
    def macs_per_cycle(self) -> int:
        return self.pe_rows * self.pe_cols * self.cores

    @property
    def peak_tops(self) -> float:
        return 2 * self.macs_per_cycle * self.freq_mhz * 1e6 / 1e12

    @property
    def dram_bytes_per_cycle(self) -> float:
        return self.dram_gbps * 1e9 / (self.freq_mhz * 1e6)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("lut_acts", "fused_acts", "unsupported_ops"):
            d[k] = sorted(d[k])
        d["peak_tops"] = round(self.peak_tops, 2)
        return d


PRESETS: dict[str, NPUSpec] = {
    # ~1 TOPS microNPU class (Ethos-U-like): single 32x32 array, tiny SRAM, slow memory.
    "tiny-1tops": NPUSpec("tiny-1tops", pe_rows=32, pe_cols=32, cores=1, freq_mhz=500, sram_kb=512,
                          dram_gbps=3.2, vector_lanes=64, dw_lanes=64),
    # ~10 TOPS @ few watts edge SoC class (REGULUS-like headline numbers): 2 cores x 64x64 @ 600 MHz = 9.8 TOPS.
    "edge-10tops": NPUSpec("edge-10tops", pe_rows=64, pe_cols=64, cores=2, freq_mhz=600, sram_kb=2048,
                           dram_gbps=12.8, vector_lanes=256, dw_lanes=256),
    # ~80 TOPS PCIe/MXM accelerator class (ARIES/MLA100-like headline numbers: 8 cores, 66.7 GB/s LPDDR4X):
    # 8 cores x 64x64 @ 1.2 GHz = 78.6 TOPS.
    "pcie-80tops": NPUSpec("pcie-80tops", pe_rows=64, pe_cols=64, cores=8, freq_mhz=1200, sram_kb=8192,
                           dram_gbps=66.7, vector_lanes=1024, dw_lanes=1024),
    # Same as edge-10tops but with no depthwise engine and no LUT unit: "strict" NPU where
    # depthwise runs on the array and non-ReLU activations fall back to the host.
    "edge-10tops-strict": NPUSpec("edge-10tops-strict", pe_rows=64, pe_cols=64, cores=2, freq_mhz=600, sram_kb=2048,
                                  dram_gbps=12.8, vector_lanes=256, dw_lanes=0,
                                  lut_acts=frozenset(), unsupported_ops=frozenset({"silu", "gelu", "hswish", "hsigmoid", "sigmoid", "tanh", "lrelu"})),
}


def get_spec(name_or_spec) -> NPUSpec:
    if isinstance(name_or_spec, NPUSpec):
        return name_or_spec
    if name_or_spec not in PRESETS:
        raise KeyError(f"unknown NPU preset {name_or_spec!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name_or_spec]
