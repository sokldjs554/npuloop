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
EQUAL_IMAGES = int(os.environ.get("NPULOOP_E10_EQUAL", 2000))   # test images on which both engines must agree code for code
SPEC = "edge-10tops"


def output_equality(ig, ds, images: int, batch: int = 250) -> dict:
    """Run both engines on `images` test images and count images whose output codes differ anywhere."""
    from npuloop.intengine import NumpyEngine, quantize_input
    from npuloop.intengine.cpp_engine import CppEngine
    np_eng, cpp_eng = NumpyEngine(ig), CppEngine(ig)
    seen = differ = 0
    for xb, _ in ds.test_batches(batch):
        codes = quantize_input(xb.numpy(), ig.input_q)
        a, b = np_eng.run(codes), cpp_eng.run(codes)
        differ += int((a.reshape(len(xb), -1) != b.reshape(len(xb), -1)).any(axis=1).sum()); seen += len(xb)
        if seen >= images:
            break
    return dict(images=seen, images_with_any_mismatch=differ)


def main():
    torch.set_num_threads(1)
    ds = dataset()
    res = Results("e10_engine_timing", meta=dict(batch=BATCH, repeats=REPEATS, warmup=1, threads=1, spec=SPEC, scheme="npu-default",
                                                  malloc=("mmap threshold raised (MALLOC_MMAP_THRESHOLD_)" if "MALLOC_MMAP_THRESHOLD_" in os.environ else "glibc default"),
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
        eq = output_equality(ig, ds, EQUAL_IMAGES)
        by_op = {}
        for r in rows:
            d = by_op.setdefault(r["op"], dict(numpy_ms=0.0, cpp_ms=0.0, cycles=0.0, count=0))
            d["numpy_ms"] += r["numpy_ms"]; d["cpp_ms"] += r["cpp_ms"]; d["cycles"] += (r["cycles"] or 0); d["count"] += 1
        res.add(dict(model=name, macs=int(trace(m).total_macs), nodes=len(ig.nodes), cpu=b["cpu"], batch=b["batch"],
                     numpy=b["engines"]["numpy"], cpp=b["engines"]["cpp"], speedup=b["speedup"],
                     modelled_cycles=cost.total_cycles, modelled_latency_ms=cost.latency_ms,
                     by_op=by_op, top_nodes=rows[:10], output_equality=eq, minutes=(time.time() - t0) / 60), provenance="measured+simulated")
        log(f"{name}: numpy {b['engines']['numpy']['per_image_ms']:.2f} ms/img, cpp {b['engines']['cpp']['per_image_ms']:.2f} ms/img "
            f"({b['speedup']:.1f}x), modelled {cost.latency_ms:.3f} ms on {SPEC}; output codes differ on "
            f"{eq['images_with_any_mismatch']}/{eq['images']} images")


if __name__ == "__main__":
    main()
