"""E10: wall-clock of the two bit-exact reference engines, per node, on the host CPU.

The cost model's cycles describe a virtual NPU; nothing in this repository measures that chip. What can be
measured is how long bit-exact verification takes on this machine: the single-threaded NumPy int64 walk and the
single-threaded C++ kernels, node by node, on the same 64-image batch. The table puts those numbers next to the
modelled cycles of the same node so the two columns are never confused — one is measured, one is modelled.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.npu import estimate
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, bench_engines, layer_table

MODELS = os.environ.get("NPULOOP_MODELS", "resnet20_relu,resnet20_silu,mnv2_050_relu6,cust_vit,cust_inception").split(",")
BATCH = int(os.environ.get("NPULOOP_E10_BATCH", 64))
REPEATS = int(os.environ.get("NPULOOP_E10_REPEATS", 5))
SPEC = "edge-10tops"


def main():
    torch.set_num_threads(1)
    ds = dataset()
    res = Results("e10_engine_timing", meta=dict(batch=BATCH, repeats=REPEATS, warmup=1, threads=1, spec=SPEC, scheme="npu-default",
                                                  note="host-CPU wall-clock of the verification engines; not NPU latency"))
    calib = calib_batches(ds, 512, seed=0)
    x, _ = next(ds.test_batches(BATCH))
    for name in MODELS:
        if name not in available_baselines() or res.has(model=name):
            continue
        t0 = time.time()
        m = load_model(name)
        qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
        ig = export_int_graph(qm)
        cost = estimate(trace(m), SPEC)
        b = bench_engines(ig, x.numpy(), repeats=REPEATS)
        rows = layer_table(ig, b, cost)
        by_op = {}
        for r in rows:
            d = by_op.setdefault(r["op"], dict(numpy_ms=0.0, cpp_ms=0.0, cycles=0.0, count=0))
            d["numpy_ms"] += r["numpy_ms"]; d["cpp_ms"] += r["cpp_ms"]; d["cycles"] += (r["cycles"] or 0); d["count"] += 1
        res.add(dict(model=name, macs=int(trace(m).total_macs), nodes=len(ig.nodes), cpu=b["cpu"], batch=b["batch"],
                     numpy=b["engines"]["numpy"], cpp=b["engines"]["cpp"], speedup=b["speedup"],
                     modelled_cycles=cost.total_cycles, modelled_latency_ms=cost.latency_ms,
                     by_op=by_op, top_nodes=rows[:10], minutes=(time.time() - t0) / 60), provenance="measured+simulated")
        log(f"{name}: numpy {b['engines']['numpy']['per_image_ms']:.2f} ms/img, cpp {b['engines']['cpp']['per_image_ms']:.2f} ms/img "
            f"({b['speedup']:.1f}x), modelled {cost.latency_ms:.3f} ms on {SPEC}")


if __name__ == "__main__":
    main()
