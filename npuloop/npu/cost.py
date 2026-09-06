"""Analytical cost model: StaticGraph x NPUSpec -> per-layer cycles, utilization, roofline bound.

Mapping assumptions (documented, deliberately simple, deterministic):
  * conv/linear are lowered to GEMMs (im2col): M = Hout*Wout, K = Cin/groups*kh*kw, N = Cout.
  * Weight-stationary systolic array of pe_rows x pe_cols: a weight tile of [rows x cols] is held in
    the array while all M input rows stream through. Tiles: Kt = ceil(K/rows), Nt = ceil(N/cols).
    cycles_tile = M + (rows + cols) fill/drain  -> compute_cycles = Kt * Nt * cycles_tile.
    This is the same first-order model SCALE-Sim uses for its WS dataflow.
  * Multi-core: the mapper tries splitting N (output channels) or M (spatial) across cores and keeps
    the faster one — exactly the decision a compiler makes for small-Cout layers.
  * Depthwise conv: on the depthwise engine (dw_lanes MACs/cycle/core) when present, else on the array
    where each channel is a K=kh*kw, N=1 GEMM (catastrophic utilization: that is the point).
  * Elementwise add / global pool / LUT activations: vector unit, vector_lanes elem/cycle/core.
    ReLU-family activations are fused into the requantization clamp and cost nothing.
  * Memory: int8 weights (+int32 bias) are streamed from DRAM once per layer. Activations stay in SRAM
    when producer output + consumer input fit in sram_kb, otherwise they round-trip through DRAM.
    Roofline per layer: cycles = max(compute, dram). No cross-layer overlap.
  * Unsupported ops fall back to the host: activation round-trips through DRAM plus
    fallback_cycles_per_elem per element.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from ..graph.ir import StaticGraph, Node
from .spec import NPUSpec, get_spec


@dataclass
class LayerCost:
    name: str
    op: str
    kind: str                 # conv|dwconv|linear|add|pool|act-fused|act-lut|act-fallback|flatten|input|output
    macs: int
    m: int = 0
    k: int = 0
    n: int = 0
    compute_cycles: float = 0.0
    vector_cycles: float = 0.0
    dram_bytes: int = 0
    dram_cycles: float = 0.0
    cycles: float = 0.0
    bound: str = "none"       # compute|memory|vector|fallback|none
    array_util: float = 0.0   # MACs / (cycles * array MACs/cycle) for array ops
    split: str = ""           # multi-core split strategy chosen
    weight_tiles: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostReport:
    spec: NPUSpec
    layers: list[LayerCost]

    @property
    def total_cycles(self) -> float:
        return sum(l.cycles for l in self.layers)

    @property
    def total_macs(self) -> int:
        return sum(l.macs for l in self.layers)

    @property
    def latency_ms(self) -> float:
        return self.total_cycles / (self.spec.freq_mhz * 1e3)

    @property
    def ideal_cycles(self) -> float:
        return self.total_macs / self.spec.macs_per_cycle

    @property
    def array_utilization(self) -> float:
        """Whole-network MAC-array utilization = ideal cycles / actual cycles."""
        return self.ideal_cycles / self.total_cycles if self.total_cycles else 0.0

    @property
    def dram_bytes(self) -> int:
        return sum(l.dram_bytes for l in self.layers)

    def breakdown(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for l in self.layers:
            out[l.kind] = out.get(l.kind, 0.0) + l.cycles
        return out

    def to_dict(self) -> dict:
        return dict(spec=self.spec.to_dict(), total_cycles=self.total_cycles, latency_ms=self.latency_ms,
                    total_macs=self.total_macs, ideal_cycles=self.ideal_cycles, array_utilization=self.array_utilization,
                    dram_bytes=self.dram_bytes, breakdown=self.breakdown(), layers=[l.to_dict() for l in self.layers])

    def table(self) -> str:
        rows = [f"{'layer':26s} {'kind':12s} {'MACs':>11s} {'M':>5s} {'K':>5s} {'N':>5s} {'cyc':>10s} {'bound':8s} {'util':>6s} {'split':6s} {'dramKB':>8s}"]
        for l in self.layers:
            if l.op in ("input", "output", "flatten"):
                continue
            rows.append(f"{l.name:26s} {l.kind:12s} {l.macs:11,d} {l.m:5d} {l.k:5d} {l.n:5d} {l.cycles:10,.0f} {l.bound:8s} {l.array_util*100:5.1f}% {l.split:6s} {l.dram_bytes/1024:8.1f}")
        rows.append(f"TOTAL cycles={self.total_cycles:,.0f}  latency={self.latency_ms:.3f} ms  "
                    f"array util={self.array_utilization*100:.1f}%  DRAM={self.dram_bytes/1024:.0f} KB  "
                    f"(spec {self.spec.name}: {self.spec.peak_tops:.1f} TOPS peak)")
        return "\n".join(rows)


def gemm_cycles(m: int, k: int, n: int, spec: NPUSpec) -> tuple[float, str, int]:
    """Weight-stationary systolic cycles for a GEMM with multi-core split. Returns (cycles, split, tiles)."""
    R, C = spec.pe_rows, spec.pe_cols
    fd = (R + C) if spec.fill_drain else 0
    kt = math.ceil(k / R)
    # split N across cores
    n_per = math.ceil(n / spec.cores)
    cyc_n = kt * math.ceil(n_per / C) * (m + fd)
    # split M across cores
    m_per = math.ceil(m / spec.cores)
    cyc_m = kt * math.ceil(n / C) * (m_per + fd)
    if spec.cores == 1 or cyc_n <= cyc_m:
        return float(cyc_n), "N" if spec.cores > 1 else "-", kt * math.ceil(n_per / C)
    return float(cyc_m), "M", kt * math.ceil(n / C)


def estimate(graph: StaticGraph, spec="edge-10tops") -> CostReport:
    spec = get_spec(spec)
    layers: list[LayerCost] = []
    sram_bytes = spec.sram_kb * 1024
    resident: dict[str, bool] = {}   # node name -> output lives in SRAM (True) or DRAM (False)
    out_bytes = {n.name: n.n_elements for n in graph.nodes}  # int8 activations

    def in_traffic(node: Node) -> int:
        """DRAM bytes to read the inputs of `node` (0 if the producers left them in SRAM)."""
        b = 0
        for src in node.inputs:
            if not resident.get(src, False):
                b += out_bytes[src]
        return b

    def place_output(node: Node, extra_live: int = 0) -> int:
        """Decide whether the output stays on-chip; return DRAM bytes written."""
        fits = (out_bytes[node.name] + extra_live) <= sram_bytes
        resident[node.name] = fits
        return 0 if fits else out_bytes[node.name]

    for node in graph.nodes:
        lc = LayerCost(node.name, node.op, node.op, node.macs)
        if node.op == "input":
            resident[node.name] = False   # the input frame arrives from DRAM
            lc.kind = "input"; layers.append(lc); continue
        if node.op in ("output", "flatten"):
            resident[node.name] = resident.get(node.inputs[0], False)
            lc.kind = node.op; layers.append(lc); continue
        if node.op in ("conv", "linear"):
            if node.op == "conv":
                cout, cin_g, kh, kw = node.weight.shape
                _, ho, wo = node.out_shape
                m, k, n = ho * wo, cin_g * kh * kw, cout
                groups = node.attrs["groups"]
            else:
                n, k = node.weight.shape; m = 1; groups = 1
            lc.m, lc.k, lc.n = m, k, n
            live_in = sum(out_bytes[s] for s in node.inputs)
            if node.op == "conv" and node.attrs.get("depthwise"):
                lc.kind = "dwconv"
                if spec.dw_lanes > 0:
                    ch_per_core = math.ceil(cout / spec.cores)
                    lc.compute_cycles = math.ceil(ch_per_core / spec.dw_lanes) * kh * kw * m
                    lc.array_util = 0.0
                    lc.note = "depthwise engine"
                    dw_peak = spec.dw_lanes * spec.cores
                    lc.array_util = node.macs / (lc.compute_cycles * dw_peak) if lc.compute_cycles else 0
                    lc.split = "C"
                else:
                    # each channel is its own GEMM: M x (kh*kw) x 1 -> one weight tile per channel
                    per_ch, split, tiles = gemm_cycles(m, kh * kw, 1, NPUSpec("_", spec.pe_rows, spec.pe_cols, 1, spec.freq_mhz, fill_drain=spec.fill_drain))
                    lc.compute_cycles = math.ceil(cout / spec.cores) * per_ch
                    lc.weight_tiles = cout
                    lc.split = "C"
                    lc.array_util = node.macs / (lc.compute_cycles * spec.macs_per_cycle)
                    lc.note = "depthwise on array"
            elif groups > 1:
                lc.kind = "gconv"
                per_g, split, tiles = gemm_cycles(m, k, n // groups, spec)
                lc.compute_cycles = per_g * groups; lc.split = split; lc.weight_tiles = tiles * groups
                lc.array_util = node.macs / (lc.compute_cycles * spec.macs_per_cycle)
            else:
                lc.kind = "conv" if node.op == "conv" else "linear"
                lc.compute_cycles, lc.split, lc.weight_tiles = gemm_cycles(m, k, n, spec)
                lc.array_util = node.macs / (lc.compute_cycles * spec.macs_per_cycle)
            weight_bytes = int(node.weight.size) + 4 * int(node.weight.shape[0])
            lc.dram_bytes = weight_bytes + in_traffic(node) + place_output(node, extra_live=live_in)
            lc.dram_cycles = lc.dram_bytes / spec.dram_bytes_per_cycle
            lc.cycles = max(lc.compute_cycles, lc.dram_cycles)
            lc.bound = "compute" if lc.compute_cycles >= lc.dram_cycles else "memory"
            layers.append(lc); continue
        if node.op == "act":
            kind = node.attrs["kind"]
            if kind in spec.fused_acts:
                lc.kind = "act-fused"; lc.cycles = 0.0; lc.bound = "none"
                resident[node.name] = resident.get(node.inputs[0], False)
                lc.note = "fused into requant clamp"
            elif kind in spec.unsupported_ops or kind not in spec.lut_acts:
                lc.kind = "act-fallback"
                elems = node.n_elements
                # write activation to DRAM, host computes, read back
                lc.dram_bytes = 2 * elems
                lc.dram_cycles = lc.dram_bytes / spec.dram_bytes_per_cycle
                lc.vector_cycles = elems * spec.fallback_cycles_per_elem
                lc.cycles = lc.vector_cycles + lc.dram_cycles
                lc.bound = "fallback"
                resident[node.name] = False
                lc.note = f"{kind} unsupported -> host fallback"
            else:
                lc.kind = "act-lut"
                lc.vector_cycles = math.ceil(node.n_elements / (spec.vector_lanes * spec.cores))
                lc.dram_bytes = in_traffic(node) + place_output(node)
                lc.dram_cycles = lc.dram_bytes / spec.dram_bytes_per_cycle
                lc.cycles = max(lc.vector_cycles, lc.dram_cycles)
                lc.bound = "vector" if lc.vector_cycles >= lc.dram_cycles else "memory"
                lc.note = f"{kind} via int8 LUT"
            layers.append(lc); continue
        if node.op in ("add", "pool"):
            lc.kind = node.op
            elems = sum(out_bytes[s] for s in node.inputs) if node.op == "add" else out_bytes[node.inputs[0]]
            lc.vector_cycles = math.ceil(elems / (spec.vector_lanes * spec.cores))
            lc.dram_bytes = in_traffic(node) + place_output(node)
            lc.dram_cycles = lc.dram_bytes / spec.dram_bytes_per_cycle
            lc.cycles = max(lc.vector_cycles, lc.dram_cycles)
            lc.bound = "vector" if lc.vector_cycles >= lc.dram_cycles else "memory"
            layers.append(lc); continue
        raise ValueError(f"cost model: unhandled op {node.op}")
    return CostReport(spec, layers)


def alignment_loss(cin_k: int, cout: int, spec: NPUSpec) -> dict:
    """Utilization loss purely from channel counts not being multiples of the array dims."""
    R, C = spec.pe_rows, spe_cols_of(spec)
    k_eff = math.ceil(cin_k / R) * R
    n_eff = math.ceil(cout / C) * C
    return dict(k_util=cin_k / k_eff, n_util=cout / n_eff, util=(cin_k / k_eff) * (cout / n_eff),
                k_padded=k_eff, n_padded=n_eff)


def spe_cols_of(spec: NPUSpec) -> int:
    return spec.pe_cols
