"""Customer model intake: can this model run on this NPU, what will it cost, and what should change?

This is the report an NPU vendor's model team writes when a customer hands over a checkpoint. It answers
three questions with numbers rather than opinions:

  1. RECEIVE   - which ops does the chip actually execute, and which fall back to the host?
  2. DIAGNOSE  - how many cycles, where do they go, and what does the static lint expect INT8 to cost?
  3. PRESCRIBE - which changes are worth making, each one priced by re-running the cost model on the
                 transformed graph (no retraining needed to get the cycle number).

`apply_prescriptions` then actually performs the accepted changes and measures the accuracy, so the
predicted saving and the real one can be compared side by side.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .graph.ir import LUT_ACTS, RELU_FAMILY, StaticGraph, UnsupportedOpError, trace
from .lint import lint
from .npu import estimate, get_spec
from .prune import prune
from .quant.surgery import swap_activations

# How each IR op reaches the silicon. "host" means a DRAM round trip and a CPU kernel.
EXEC_UNITS = {
    "conv": "mac-array", "linear": "mac-array", "matmul": "mac-array",
    "add": "vector", "concat": "vector", "pool": "vector", "mul": "vector",
    "softmax": "vector", "layernorm": "vector", "transpose": "vector",
    "act": "vector", "reshape": "free", "flatten": "free", "const": "free",
    "input": "free", "output": "free",
}


def op_support(graph: StaticGraph, spec) -> list[dict]:
    """Per op kind: how many nodes, where they run, and whether the chip supports them at all."""
    spec = get_spec(spec)
    rows: dict[str, dict] = {}
    for n in graph.nodes:
        unit = EXEC_UNITS.get(n.op, "unknown")
        detail = ""
        if n.op == "act":
            kind = n.attrs["kind"]
            if kind in spec.fused_acts:
                unit, detail = "fused", f"{kind} folded into the requantization clamp"
            elif kind in spec.unsupported_ops or kind not in spec.lut_acts:
                unit, detail = "host", f"{kind} has no int8 LUT on this NPU"
            else:
                unit, detail = "lut", f"{kind} via a 256-entry int8 LUT"
            key = f"act:{kind}"
        elif n.op in ("softmax", "layernorm"):
            key = n.op
            if n.op in spec.unsupported_ops:
                unit, detail = "host", f"{n.op} needs exp/rsqrt this NPU does not have"
            else:
                detail = f"{spec.softmax_passes if n.op == 'softmax' else spec.layernorm_passes} vector passes"
        elif n.op == "conv" and n.attrs.get("depthwise"):
            key = "conv:depthwise"
            unit = "dw-engine" if spec.dw_lanes > 0 else "mac-array"
            detail = "depthwise engine" if spec.dw_lanes > 0 else "one K=9,N=1 GEMM per channel"
        elif n.op == "matmul":
            key, detail = "matmul", "activation x activation: no static weights to reuse"
        elif n.op == "mul":
            key = "mul"
            if n.attrs.get("kind") == "scalar":
                unit, detail = "fused", "scalar folded into the requantization multiplier"
        else:
            key = n.op
        r = rows.setdefault(key, dict(op=key, count=0, unit=unit, detail=detail))
        r["count"] += 1
    order = {"host": 0, "mac-array": 1, "dw-engine": 2, "lut": 3, "vector": 4, "fused": 5, "free": 6}
    return sorted(rows.values(), key=lambda r: (order.get(r["unit"], 9), -r["count"]))


def can_run(graph: StaticGraph, spec) -> tuple[bool, list[str]]:
    """Does the whole graph execute on-chip? Returns (verdict, reasons it does not)."""
    reasons = [f"{r['count']}x {r['op']} ({r['detail']})" for r in op_support(graph, spec) if r["unit"] == "host"]
    return (not reasons), reasons


def bottlenecks(report, k: int = 3) -> list[dict]:
    layers = [l for l in report.layers if l.cycles > 0]
    layers.sort(key=lambda l: -l.cycles)
    total = report.total_cycles or 1
    return [dict(name=l.name, kind=l.kind, cycles=l.cycles, share=l.cycles / total, bound=l.bound,
                 util=l.array_util if l.on_array else l.engine_util, note=l.note) for l in layers[:k]]


def _cycles(model: nn.Module, spec, input_shape=(3, 32, 32)) -> float:
    return estimate(trace(model, input_shape), spec).total_cycles


UNIT_OF_KIND = {"conv": "mac-array", "gconv": "mac-array", "linear": "mac-array", "matmul": "mac-array",
                "dwconv": "dw-engine", "act-lut": "vector", "add": "vector", "concat": "vector", "pool": "vector",
                "softmax": "vector", "layernorm": "vector", "transpose": "vector", "mul": "vector"}


def cycles_by_unit(report) -> dict[str, float]:
    """Where the cycles go: the MAC array, the depthwise engine, the vector unit, or the host CPU."""
    out: dict[str, float] = {}
    for l in report.layers:
        if l.cycles <= 0:
            continue
        unit = "host" if l.kind.endswith("-fallback") else UNIT_OF_KIND.get(l.kind, "other")
        out[unit] = out.get(unit, 0.0) + l.cycles
    return out


def prescriptions(model: nn.Module, spec, input_shape=(3, 32, 32)) -> list[dict]:
    """Candidate changes, each priced by re-running the cost model on the transformed model.

    Only the cycle number is predicted here; accuracy needs the fine-tuning that `apply_prescriptions`
    (or the E9 experiment) actually runs.
    """
    spec = get_spec(spec)
    base_graph = trace(model, input_shape)
    base = estimate(base_graph, spec).total_cycles
    kinds = {n.attrs["kind"] for n in base_graph.nodes if n.op == "act"}
    out: list[dict] = []

    for kind in sorted(kinds - RELU_FAMILY):
        if kind not in LUT_ACTS:
            continue
        for target in ("relu", "hswish"):
            if target == "hswish" and (target in spec.unsupported_ops or target not in spec.lut_acts):
                continue
            try:
                swapped = swap_activations(model, {kind: target})
                cyc = _cycles(swapped, spec, input_shape)
            except (UnsupportedOpError, ValueError, KeyError):
                continue
            if cyc >= base:
                continue
            out.append(dict(action=f"swap {kind} -> {target}", kind="activation-surgery",
                            why=("the host fallback disappears" if kind in spec.unsupported_ops
                                 else "one int8 LUT pass per activation disappears"),
                            cycles=cyc, saving=1 - cyc / base,
                            risk="needs a short healing fine-tune (E4: 3 epochs recovered the drop)"))

    for ratio in (0.75, 0.5):
        try:
            pruned, _ = prune(copy.deepcopy(model), ratio=ratio, strategy="uniform")
            cyc = _cycles(pruned, spec, input_shape)
        except (UnsupportedOpError, ValueError, IndexError, RuntimeError):
            continue
        if cyc < base:
            out.append(dict(action=f"uniform structured pruning, ratio {ratio}", kind="pruning",
                            why="fewer channels per block", cycles=cyc, saving=1 - cyc / base,
                            risk="needs fine-tuning; E6 measured -1.1 to -2.4%p at these ratios"))

    out.sort(key=lambda r: -(r["saving"] or 0))
    return out


def alternatives(model: nn.Module, spec, input_shape=(3, 32, 32)) -> list[dict]:
    """The same model on the other presets — a hardware answer, kept apart from the model changes."""
    from .npu.spec import PRESETS
    spec = get_spec(spec)
    g = trace(model, input_shape)
    base = estimate(g, spec).total_cycles
    rows = []
    for name in PRESETS:
        if name == spec.name:
            continue
        ok, reasons = can_run(g, PRESETS[name])
        cyc = estimate(g, name).total_cycles
        rows.append(dict(preset=name, cycles=cyc, runs_on_chip=ok, vs_current=cyc / base,
                         host_fallbacks=reasons))
    return sorted(rows, key=lambda r: r["cycles"])


def intake_report(model: nn.Module, spec="edge-10tops", calib: np.ndarray | None = None,
                  input_shape=(3, 32, 32), with_prescriptions: bool = True) -> dict[str, Any]:
    """The full receive -> diagnose -> prescribe report as a JSON-friendly dict."""
    spec = get_spec(spec)
    graph = trace(model, input_shape)
    cost = estimate(graph, spec)
    lr = lint(graph, spec, calib)
    ok, reasons = can_run(graph, spec)
    return dict(
        spec=spec.name,
        model=dict(config=getattr(model, "config", None),
                   params=int(sum(p.numel() for p in model.parameters())),
                   macs=int(graph.total_macs), nodes=len(graph.nodes)),
        receive=dict(runs_on_chip=ok, host_fallbacks=reasons, ops=op_support(graph, spec)),
        diagnose=dict(cycles=cost.total_cycles, latency_ms=cost.latency_ms,
                      array_utilization=cost.array_utilization, dram_bytes=cost.dram_bytes,
                      breakdown=cost.breakdown(), by_unit=cycles_by_unit(cost), bottlenecks=bottlenecks(cost),
                      lint=dict(scores=lr.scores, findings=[f.to_dict() for f in lr.findings[:12]],
                                counts=lr.counts())),
        prescribe=prescriptions(model, spec, input_shape) if with_prescriptions else [],
        alternatives=alternatives(model, spec, input_shape) if with_prescriptions else [],
    )


def render(report: dict) -> str:
    """Human-readable intake sheet."""
    m, r, d = report["model"], report["receive"], report["diagnose"]
    lines = [f"# Intake report — {m['config']} on `{report['spec']}`", "",
             f"params {m['params']:,} · MACs {m['macs'] / 1e6:.1f}M · {m['nodes']} graph nodes", "",
             "## 1. Receive", ""]
    lines.append("**runs entirely on-chip: " + ("yes" if r["runs_on_chip"] else "NO**") + ("**" if r["runs_on_chip"] else ""))
    for reason in r["host_fallbacks"]:
        lines.append(f"  - host fallback: {reason}")
    lines += ["", "| op | count | executes on | note |", "|---|---|---|---|"]
    for row in r["ops"]:
        if row["unit"] == "free":
            continue
        lines.append(f"| `{row['op']}` | {row['count']} | {row['unit']} | {row['detail']} |")
    unit_txt = " · ".join(f"{k} {v / max(d['cycles'], 1) * 100:.0f}%"
                          for k, v in sorted(d["by_unit"].items(), key=lambda kv: -kv[1]))
    lines += ["", "## 2. Diagnose", "",
              f"cycles **{d['cycles']:,.0f}** · latency {d['latency_ms']:.3f} ms · array utilization "
              f"{d['array_utilization'] * 100:.1f}% · DRAM {d['dram_bytes'] / 1024:.0f} KB",
              f"cycle budget: {unit_txt}",
              f"lint: efficiency {d['lint']['scores']['efficiency']:.0f}/100 · "
              f"quant-robustness {d['lint']['scores']['quant_robustness']:.0f}/100", "",
              "| bottleneck | kind | cycles | share | bound |", "|---|---|---|---|---|"]
    for b in d["bottlenecks"]:
        lines.append(f"| `{b['name']}` | {b['kind']} | {b['cycles']:,.0f} | {b['share'] * 100:.0f}% | {b['bound']} |")
    lines += ["", "## 3. Prescribe", ""]
    if not report["prescribe"]:
        lines.append("nothing worth changing: no candidate beat the baseline cycle count.")
    else:
        lines += ["| action | predicted cycles | saving | why | risk |", "|---|---|---|---|---|"]
        for p in report["prescribe"]:
            cyc = f"{p['cycles']:,.0f}" if p["cycles"] else "—"
            sav = f"{p['saving'] * 100:.0f}%" if p["saving"] else "—"
            lines.append(f"| {p['action']} | {cyc} | {sav} | {p['why']} | {p['risk']} |")
    if report.get("alternatives"):
        lines += ["", "Same model, other presets (a hardware answer, not a model change):", "",
                  "| preset | cycles | vs current | runs on-chip |", "|---|---|---|---|"]
        for a in report["alternatives"]:
            lines.append(f"| `{a['preset']}` | {a['cycles']:,.0f} | {a['vs_current'] * 100:.0f}% | "
                         f"{'yes' if a['runs_on_chip'] else 'no'} |")
    return "\n".join(lines)
