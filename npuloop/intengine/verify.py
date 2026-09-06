"""Fake-quant (PyTorch) vs integer-engine agreement, with per-layer attribution.

Two views of every tensor:
  * propagated: the integer engine runs end-to-end, so an early ±1 LSB flip can change later roundings
  * local (teacher-forced): each integer op is fed the *fake-quant* codes of its inputs, isolating the
    mismatch caused by that op's own integer arithmetic (bias rounding, fixed-point multiplier, rounding mode)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from ..quant.fake import FakeQuantAct
from .graph import IntGraph
from .numpy_engine import NumpyEngine, quantize_input


@dataclass
class NodeAgreement:
    name: str
    op: str
    n: int
    mismatch_frac: float       # propagated
    max_abs: int
    mean_abs: float
    local_mismatch_frac: float # teacher-forced
    local_max_abs: int

    def to_dict(self):
        return asdict(self)


def fake_codes(gm, ig: IntGraph, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Run the fake-quant model, capture every quantizer output and convert it to integer codes."""
    fq_by_target = {n.attrs["fq_target"]: n.name for n in ig.nodes if "fq_target" in n.attrs}
    captured: dict[str, np.ndarray] = {}
    hooks = []
    for target, node_name in fq_by_target.items():
        fq: FakeQuantAct = gm.get_submodule(target)
        s, z = fq.qparams()

        def mk(node_name, s, z, fq):
            def hook(mod, inp, out):
                codes = np.rint(out.detach().double().numpy() / s) + z
                captured[node_name] = np.clip(codes, fq.qmin, fq.qmax).astype(np.int64)
            return hook
        hooks.append(fq.register_forward_hook(mk(node_name, s, z, fq)))
    gm.eval()
    with torch.no_grad():
        logits = gm(x)
    for h in hooks:
        h.remove()
    # flatten / output alias the tensors of their inputs
    for n in ig.nodes:
        if n.op in ("flatten", "output") and n.name not in captured and n.inputs[0] in captured:
            src = captured[n.inputs[0]]
            captured[n.name] = src.reshape(src.shape[0], -1) if n.op == "flatten" else src
    return captured, logits.numpy()


def compare(gm, ig: IntGraph, x: torch.Tensor, engine: NumpyEngine | None = None):
    """Return (per-node agreement list, summary dict)."""
    engine = engine or NumpyEngine(ig)
    fcodes, flogits = fake_codes(gm, ig, x)
    codes_in = quantize_input(x.numpy(), ig.input_q)
    ivals = engine.run(codes_in, return_all=True)
    # teacher forcing: run every node on the fake-quant codes of its inputs
    tf_vals = dict(fcodes); tf_vals["__input__"] = codes_in
    rows = []
    for n in ig.nodes:
        if n.name not in fcodes or n.name not in ivals:
            continue
        a, b = fcodes[n.name], ivals[n.name]
        d = np.abs(a - b)
        local = engine.exec_node(n, tf_vals) if n.op != "input" else b
        dl = np.abs(a - local)
        rows.append(NodeAgreement(n.name, n.op, int(a.size), float((d > 0).mean()), int(d.max()), float(d.mean()),
                                  float((dl > 0).mean()), int(dl.max())))
    from .numpy_engine import dequantize
    ilogits = dequantize(ivals["output"], ig["output"].out_q)
    summary = dict(top1_agreement=float((flogits.argmax(1) == ilogits.argmax(1)).mean()),
                   logit_max_abs_diff=float(np.abs(flogits - ilogits).max()),
                   first_divergence=next((r.name for r in rows if r.mismatch_frac > 0), None),
                   output_mismatch_frac=rows[-1].mismatch_frac if rows else None)
    return rows, summary


def agreement_table(rows: list[NodeAgreement]) -> str:
    out = [f"{'node':26s} {'op':7s} {'prop.mism%':>10s} {'prop.max':>8s} {'local.mism%':>11s} {'local.max':>9s}"]
    for r in rows:
        out.append(f"{r.name:26s} {r.op:7s} {r.mismatch_frac*100:10.3f} {r.max_abs:8d} {r.local_mismatch_frac*100:11.3f} {r.local_max_abs:9d}")
    return "\n".join(out)
