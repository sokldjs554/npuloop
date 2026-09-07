"""NPU readiness lint: static (weights + shapes only) and dynamic (needs a calibration batch) checks.

A finding is a fact about the *architecture* that predicts INT8-NPU trouble *before* any quantization
is attempted. The readiness score is a heuristic aggregate meant for ranking candidate models, not a
guarantee — the experiments in this repo measure how well it predicts the real INT8 accuracy drop.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import math
import numpy as np
from ..graph.ir import StaticGraph, Node, RELU_FAMILY, run_reference
from ..npu.spec import NPUSpec, get_spec
from ..npu.cost import estimate

SEVERITY_WEIGHT = {"high": 12.0, "medium": 5.0, "low": 1.5, "info": 0.0}


@dataclass
class Finding:
    check: str
    severity: str          # high|medium|low|info
    layer: str
    message: str
    metric: float = 0.0
    suggestion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LintReport:
    spec_name: str
    findings: list[Finding]
    stats: dict = field(default_factory=dict)

    @property
    def scores(self) -> dict[str, float]:
        """Two 0-100 sub-scores computed from *fractions* of affected layers (so depth does not saturate them).

        efficiency       : will the NPU's MAC array be busy? (alignment, depthwise, activation execution cost)
        quant_robustness : will INT8 quantization hurt? (weight-range disparity, outliers, asymmetry, residual scales, LUT acts)
        """
        bc = self.by_check()
        n_compute = max(self.stats.get("n_compute", 1), 1)
        n_act = max(self.stats.get("n_act", 1), 1)
        n_add = max(self.stats.get("n_add", 1), 1)
        n_tensors = max(self.stats.get("n_tensors", n_compute), 1)   # nodes the dynamic checks can flag

        def frac(check, sev=None):
            fs = bc.get(check, [])
            if sev:
                fs = [f for f in fs if f.severity == sev]
            return len(fs)

        eff = 100.0
        util = self.stats.get("alignment_util_weighted", 1.0)
        eff -= 35.0 * (1.0 - util)                                   # MAC-weighted alignment utilization bound
        eff -= 30.0 * min(1.0, frac("activation-support", "high") / n_act)
        eff -= 8.0 * min(1.0, frac("activation-support", "low") / n_act)
        eff -= 20.0 * min(1.0, frac("depthwise", "high") / n_compute) + 8.0 * min(1.0, frac("depthwise", "low") / n_compute)
        eff -= 3.0 * min(1.0, frac("stem-underutilization"))
        qr = 100.0
        qr -= 35.0 * min(1.0, frac("weight-range-disparity", "high") / n_compute) + 15.0 * min(1.0, frac("weight-range-disparity", "medium") / n_compute)
        qr -= 15.0 * min(1.0, frac("activation-outliers", "medium") / n_tensors) + 6.0 * min(1.0, frac("activation-outliers", "low") / n_tensors)
        qr -= 10.0 * min(1.0, frac("activation-asymmetry") / n_act)
        qr -= 12.0 * min(1.0, frac("activation-support", "low") / n_act)      # LUT activation = extra quantization point
        qr -= 10.0 * min(1.0, frac("residual-scale-mismatch", "medium") / n_add) + 4.0 * min(1.0, frac("residual-scale-mismatch", "low") / n_add)
        qr -= 6.0 * min(1.0, frac("depthwise") / n_compute)
        eff, qr = max(0.0, eff), max(0.0, qr)
        return dict(efficiency=round(eff, 1), quant_robustness=round(qr, 1), overall=round(min(eff, qr), 1))

    @property
    def score(self) -> float:
        return self.scores["overall"]

    def counts(self) -> dict[str, int]:
        c = {"high": 0, "medium": 0, "low": 0, "info": 0}
        for f in self.findings:
            c[f.severity] += 1
        return c

    def by_check(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.check, []).append(f)
        return out

    def to_dict(self) -> dict:
        return dict(spec=self.spec_name, score=self.score, scores=self.scores, counts=self.counts(), stats=self.stats,
                    findings=[f.to_dict() for f in self.findings])

    def markdown(self) -> str:
        lines = [f"## NPU readiness report (spec: {self.spec_name})", "",
                 f"**Efficiency {self.scores['efficiency']:.0f}/100 · Quant-robustness {self.scores['quant_robustness']:.0f}/100**  — "
                 + ", ".join(f"{k}: {v}" for k, v in self.counts().items()), ""]
        for check, fs in self.by_check().items():
            lines.append(f"### {check}")
            lines.append("| severity | layer | metric | message | suggestion |")
            lines.append("|---|---|---|---|---|")
            for f in fs:
                lines.append(f"| {f.severity} | `{f.layer}` | {f.metric:.3g} | {f.message} | {f.suggestion} |")
            lines.append("")
        return "\n".join(lines)


def _sev(util: float) -> str:
    if util < 0.35:
        return "high"
    if util < 0.6:
        return "medium"
    if util < 0.9:
        return "low"
    return "info"


def check_array_alignment(graph: StaticGraph, spec: NPUSpec) -> list[Finding]:
    out = []
    R, C = spec.pe_rows, spec.pe_cols
    for n in graph.compute_nodes():
        if n.op == "conv" and n.attrs.get("depthwise"):
            continue
        if n.op == "conv":
            cout, cin_g, kh, kw = n.weight.shape
            k = cin_g * kh * kw
            cout = cout // n.attrs.get("groups", 1)        # grouped conv: each group is its own GEMM
        else:
            cout, k = n.weight.shape
        n_util = cout / (math.ceil(cout / C) * C)
        k_util = k / (math.ceil(k / R) * R)
        util = n_util * k_util
        sev = _sev(util)
        if sev == "info":
            continue
        sug = []
        if n_util < 0.9:
            sug.append(f"Cout={cout} fills {n_util*100:.0f}% of {C} columns -> prune/pad Cout to a multiple of {C} (or pick a narrower array)")
        if k_util < 0.9:
            sug.append(f"K={k} fills {k_util*100:.0f}% of {R} rows -> Cin*k*k should be a multiple of {R}")
        out.append(Finding("array-alignment", sev, n.name,
                           f"MAC-array tile utilization bound {util*100:.0f}% (Cout={cout}, K={k}, array {R}x{C})", util, "; ".join(sug)))
    return out


def check_activation_support(graph: StaticGraph, spec: NPUSpec) -> list[Finding]:
    out = []
    for n in graph.nodes:
        if n.op != "act":
            continue
        kind = n.attrs["kind"]
        if kind in spec.fused_acts:
            continue
        if kind in spec.unsupported_ops or kind not in spec.lut_acts:
            out.append(Finding("activation-support", "high", n.name, f"{kind} is not executable on the NPU -> host fallback (DRAM round trip)", 0.0,
                               "replace with ReLU/ReLU6 (fused) or a LUT-able activation, then heal with a short fine-tune"))
        else:
            out.append(Finding("activation-support", "low", n.name,
                               f"{kind} runs as an int8 LUT: the pre-activation must be materialised as int8 first (extra quantization point)", 1.0,
                               "expect a larger fake-quant vs integer gap than for ReLU; consider ReLU/ReLU6 if accuracy allows"))
    return out


def weight_channel_ranges(n: Node) -> np.ndarray:
    w = n.weight.reshape(n.weight.shape[0], -1)
    return np.abs(w).max(axis=1)


def check_weight_range_disparity(graph: StaticGraph, ratio_medium: float = 8.0, ratio_high: float = 32.0,
                                 crushed_frac: float = 1 / 32) -> list[Finding]:
    """After BN folding, per-output-channel weight ranges can differ by orders of magnitude.
    Per-tensor weight quantization then rounds the small channels to (almost) zero."""
    out = []
    for n in graph.compute_nodes():
        r = weight_channel_ranges(n)
        r = r[r > 0] if (r > 0).any() else r
        if r.size < 2:
            continue
        ratio = float(r.max() / np.median(r))
        crushed = float((r < r.max() * crushed_frac).mean())   # channels whose range < 4 int8 steps under per-tensor
        if ratio >= ratio_high or crushed > 0.25:
            sev = "high"
        elif ratio >= ratio_medium or crushed > 0.1:
            sev = "medium"
        else:
            continue
        dw = " (depthwise)" if n.attrs.get("depthwise") else ""
        out.append(Finding("weight-range-disparity", sev, n.name,
                           f"max/median per-channel |w| ratio {ratio:.1f}; {crushed*100:.0f}% of channels would get <4 int8 levels under per-tensor weights{dw}",
                           ratio, "use per-output-channel weight scales; if the NPU only supports per-tensor, apply cross-layer equalization (CLE)"))
    return out


def check_depthwise(graph: StaticGraph, spec: NPUSpec) -> list[Finding]:
    out = []
    for n in graph.compute_nodes():
        if n.op == "conv" and n.attrs.get("depthwise"):
            if spec.dw_lanes == 0:
                out.append(Finding("depthwise", "high", n.name, "depthwise conv on a plain MAC array: each channel is a K=9,N=1 GEMM (utilization ~0.2%)", 0.0,
                                   "avoid depthwise on this NPU (use regular/grouped convs) or target an NPU with a depthwise engine"))
            else:
                out.append(Finding("depthwise", "low", n.name, "depthwise conv runs on the depthwise engine (lower throughput than the array); ranges after BN folding tend to be wide", 0.0,
                                   "check weight-range-disparity findings for this layer; prefer per-channel weights"))
    return out


def check_stem(graph: StaticGraph, spec: NPUSpec) -> list[Finding]:
    out = []
    for n in graph.compute_nodes():
        if n.op == "conv" and n.attrs["in_channels"] <= 4:
            cout, cin_g, kh, kw = n.weight.shape
            k = cin_g * kh * kw
            util = k / spec.pe_rows
            if util < 0.6:
                out.append(Finding("stem-underutilization", "low", n.name,
                                   f"first conv has K={k} (Cin={cin_g}) -> only {util*100:.0f}% of {spec.pe_rows} array rows are busy", util,
                                   "space-to-depth the input (e.g. 2x2 -> Cin=12, K=108) or accept it: it is usually a small share of total cycles"))
    return out


def check_fc_head(graph: StaticGraph) -> list[Finding]:
    out = []
    for n in graph.compute_nodes():
        if n.op == "linear":
            out.append(Finding("fc-head", "info", n.name, "batch-1 fully-connected layer is a GEMV: memory-bound on weights, array mostly idle", 0.0,
                               "fine for small heads; for large heads keep weights on-chip or batch requests"))
    return out


def collect_activation_stats(graph: StaticGraph, x: np.ndarray) -> dict[str, dict]:
    """Per-node activation statistics of the float (BN-folded) graph on a calibration batch."""
    vals = run_reference(graph, x)
    stats = {}
    for n in graph.nodes:
        v = vals[n.name].astype(np.float32).ravel()
        if v.size == 0:
            continue
        q = np.quantile(v, [0.0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1.0])
        stats[n.name] = dict(min=float(q[0]), p0001=float(q[1]), p001=float(q[2]), p01=float(q[3]), median=float(q[4]),
                             p99=float(q[5]), p999=float(q[6]), p9999=float(q[7]), max=float(q[8]),
                             mean=float(v.mean()), std=float(v.std()), neg_frac=float((v < 0).mean()))
    return stats


def check_dynamic(graph: StaticGraph, stats: dict[str, dict]) -> list[Finding]:
    out = []
    for n in graph.nodes:
        s = stats.get(n.name)
        if s is None:
            continue
        rng = s["max"] - s["min"]
        if n.op == "add":
            a, b = (stats.get(i) for i in n.inputs)
            if a and b:
                ra, rb = a["max"] - a["min"], b["max"] - b["min"]
                ratio = max(ra, rb) / max(min(ra, rb), 1e-9)
                if ratio > 4:
                    out.append(Finding("residual-scale-mismatch", "medium" if ratio > 8 else "low", n.name,
                                       f"residual add inputs have range ratio {ratio:.1f}: the smaller branch is rescaled into the larger grid and loses ~{math.log2(ratio):.1f} bits", ratio,
                                       "QAT usually absorbs this; otherwise check that the shortcut branch has its own quantizer"))
        if n.op == "act" and n.attrs["kind"] not in RELU_FAMILY:
            if s["min"] < 0 and rng > 0:
                neg_share = (0 - s["min"]) / rng
                out.append(Finding("activation-asymmetry", "low", n.name,
                                   f"{n.attrs['kind']} output spans [{s['min']:.2f}, {s['max']:.2f}]: {neg_share*100:.0f}% of the int8 grid is spent on the small negative tail", neg_share,
                                   "asymmetric (uint8 + zero-point) activation quantization is required; symmetric int8 wastes almost a bit"))
        if n.op in ("conv", "linear", "add", "act", "pool") and rng > 0:
            tail = (s["max"] - s["p9999"]) / rng if n.op != "act" or n.attrs["kind"] not in RELU_FAMILY else (s["max"] - s["p9999"]) / max(s["max"], 1e-9)
            if tail > 0.3:
                out.append(Finding("activation-outliers", "medium" if tail > 0.5 else "low", n.name,
                                   f"top 0.01% of activations occupy {tail*100:.0f}% of the min-max range -> min-max calibration wastes {math.log2(1/(1-tail+1e-9)):.1f} bits", tail,
                                   "calibrate with a percentile/MSE observer instead of min-max"))
    return out


def lint(graph: StaticGraph, spec="edge-10tops", calib: np.ndarray | None = None) -> LintReport:
    spec = get_spec(spec)
    findings: list[Finding] = []
    findings += check_activation_support(graph, spec)
    findings += check_depthwise(graph, spec)
    findings += check_weight_range_disparity(graph)
    findings += check_array_alignment(graph, spec)
    findings += check_stem(graph, spec)
    findings += check_fc_head(graph)
    stats: dict = {}
    if calib is not None:
        act_stats = collect_activation_stats(graph, calib)
        findings += check_dynamic(graph, act_stats)
        stats["activations"] = act_stats
    cost = estimate(graph, spec)
    stats["cost"] = dict(total_cycles=cost.total_cycles, latency_ms=cost.latency_ms, array_utilization=cost.array_utilization,
                         dram_bytes=cost.dram_bytes, breakdown=cost.breakdown())
    stats["n_compute"] = len(graph.compute_nodes())
    stats["n_act"] = sum(1 for n in graph.nodes if n.op == "act")
    stats["n_add"] = sum(1 for n in graph.nodes if n.op == "add")
    stats["n_tensors"] = sum(1 for n in graph.nodes if n.op in ("conv", "linear", "add", "act", "pool"))
    stats["alignment_util_weighted"] = alignment_util_weighted(graph, spec)
    stats["weights"] = {n.name: dict(ratio=float(weight_channel_ranges(n).max() / max(np.median(weight_channel_ranges(n)), 1e-12)))
                        for n in graph.compute_nodes()}
    return LintReport(spec.name, findings, stats)


def alignment_util_weighted(graph: StaticGraph, spec: NPUSpec) -> float:
    """MAC-weighted upper bound on array utilization from channel alignment alone (depthwise excluded)."""
    num = den = 0.0
    for n in graph.compute_nodes():
        if n.op == "conv" and n.attrs.get("depthwise"):
            continue
        if n.op == "conv":
            cout, cin_g, kh, kw = n.weight.shape; k = cin_g * kh * kw
        else:
            cout, k = n.weight.shape
        util = (cout / (math.ceil(cout / spec.pe_cols) * spec.pe_cols)) * (k / (math.ceil(k / spec.pe_rows) * spec.pe_rows))
        num += util * n.macs; den += n.macs
    return num / den if den else 1.0
