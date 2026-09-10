"""Wall-clock of the two reference engines, per node, on the host CPU.

These are host-CPU numbers for the *verification* engines — the single-threaded NumPy int64 walk and the
single-threaded C++ kernels — not NPU latency. The cost model's cycles describe a systolic array that does not
exist on this machine, so the two columns answer different questions: "how long does bit-exact verification
take here" (measured) and "where would the time go on the target" (modelled). Putting them side by side keeps
the modelled column honest about what it is.
"""
from __future__ import annotations
import platform, statistics, time
import numpy as np
from .graph import IntGraph
from .numpy_engine import NumpyEngine, quantize_input


def _engine(name: str, ig: IntGraph):
    if name == "numpy":
        return NumpyEngine(ig)
    if name == "cpp":
        from .cpp_engine import CppEngine
        return CppEngine(ig)
    raise ValueError(f"unknown engine {name!r}")


def timed_pass(engine, x_codes: np.ndarray) -> dict[str, float]:
    """One forward pass, returning seconds spent in every node (same walk and memory release as NumpyEngine.run)."""
    ig = engine.g
    remaining = {n.name: len(engine.g_users[n.name]) for n in ig.nodes}
    vals: dict[str, np.ndarray] = {"__input__": x_codes.astype(np.int64)}
    secs: dict[str, float] = {}
    for n in ig.nodes:
        t = time.perf_counter()
        vals[n.name] = engine.exec_node(n, vals)
        secs[n.name] = time.perf_counter() - t
        for src in n.inputs:
            remaining[src] -= 1
            if remaining[src] <= 0 and src in vals:
                del vals[src]
    return secs


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def bench_engines(ig: IntGraph, x_float: np.ndarray, repeats: int = 3, warmup: int = 1,
                  engines: tuple[str, ...] = ("numpy", "cpp")) -> dict:
    """Per-node wall-clock (median over `repeats` passes after `warmup`) for each engine on one batch.

    Returns {batch, repeats, warmup, cpu, engines: {name: {total_ms, per_image_ms, images_per_s, nodes: {node: ms}}},
    speedup: numpy_total / cpp_total (when both ran)}.
    """
    codes = quantize_input(x_float, ig.input_q)
    out: dict = dict(batch=int(len(x_float)), repeats=repeats, warmup=warmup, cpu=cpu_name(), threads=1, engines={})
    for name in engines:
        eng = _engine(name, ig)
        for _ in range(warmup):
            timed_pass(eng, codes)
        passes = [timed_pass(eng, codes) for _ in range(repeats)]
        nodes = {n.name: statistics.median(p[n.name] for p in passes) * 1e3 for n in ig.nodes}
        total = statistics.median(sum(p.values()) for p in passes) * 1e3
        out["engines"][name] = dict(total_ms=total, per_image_ms=total / len(x_float),
                                    images_per_s=len(x_float) / (total / 1e3), nodes=nodes)
    if "numpy" in out["engines"] and "cpp" in out["engines"]:
        out["speedup"] = out["engines"]["numpy"]["total_ms"] / max(out["engines"]["cpp"]["total_ms"], 1e-9)
    return out


def layer_table(ig: IntGraph, bench: dict, cost=None, top: int | None = None) -> list[dict]:
    """Rows joining measured ms per node with the cost model's cycles for the same node (by name), sorted by C++ time."""
    cyc = {l.name: l for l in cost.layers} if cost is not None else {}
    total_cycles = sum(l.cycles for l in cost.layers) if cost is not None else 0
    rows = []
    for n in ig.nodes:
        if n.op in ("input", "output"):
            continue
        row = dict(name=n.name, op=n.op)
        for name, e in bench["engines"].items():
            row[f"{name}_ms"] = e["nodes"][n.name]
        lc = cyc.get(n.name)
        row["cycles"] = lc.cycles if lc else None
        row["cycle_share"] = (lc.cycles / total_cycles) if (lc and total_cycles) else None
        row["kind"] = lc.kind if lc else None
        rows.append(row)
    key = "cpp_ms" if "cpp" in bench["engines"] else next(iter(bench["engines"])) + "_ms"
    rows.sort(key=lambda r: -r[key])
    return rows[:top] if top else rows


def render_bench(bench: dict, rows: list[dict] | None = None) -> str:
    e = bench["engines"]
    lines = [f"batch {bench['batch']} · median of {bench['repeats']} passes after {bench['warmup']} warm-up · "
             f"single thread on {bench['cpu']}"]
    for name, r in e.items():
        lines.append(f"  {name:6s} {r['total_ms']:9.1f} ms/batch  {r['per_image_ms']:8.3f} ms/image  {r['images_per_s']:8.1f} images/s")
    if "speedup" in bench:
        lines.append(f"  C++ / NumPy speed-up: {bench['speedup']:.1f}x")
    if rows:
        lines += ["", "| node | op | " + " | ".join(f"{n} ms" for n in e) + " | modelled cycles | share |", "|---|---|" + "---|" * (len(e) + 2)]
        for r in rows:
            cyc = f"{r['cycles']:,.0f}" if r["cycles"] is not None else "—"
            share = f"{r['cycle_share'] * 100:.1f}%" if r["cycle_share"] is not None else "—"
            lines.append(f"| `{r['name']}` | {r['op']} | " + " | ".join(f"{r[f'{n}_ms']:.3f}" for n in e) + f" | {cyc} | {share} |")
    return "\n".join(lines)
