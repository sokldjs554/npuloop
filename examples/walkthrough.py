"""End-to-end walkthrough on one checkpoint (~1-2 min on CPU):
cost model -> lint -> PTQ -> export IntGraph -> NumPy/C++ bit-exact check -> fake-vs-int agreement -> requant ablation.

python examples/walkthrough.py --ckpt runs/resnet20_relu/best.pt --data data/cifar10.npz
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--ckpt", default=os.environ.get("NPULOOP_CKPT", os.path.join(ROOT, "runs", "resnet20_relu", "best.pt")))
    p.add_argument("--data", default=os.environ.get("NPULOOP_DATA", os.path.join(ROOT, "data", "cifar10.npz")))
    p.add_argument("--spec", default="edge-10tops")
    p.add_argument("--images", type=int, default=1000)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    from npuloop.zoo import load_checkpoint, CIFAR10NPZ
    from npuloop.graph import trace
    from npuloop.npu import estimate
    from npuloop.lint import lint
    from npuloop.quant import prepare, calibrate, evaluate, QScheme
    from npuloop.intengine import export_int_graph, NumpyEngine, RequantConfig
    from npuloop.intengine.cpp_engine import CppEngine
    from npuloop.intengine.numpy_engine import quantize_input
    from npuloop.intengine.verify import compare, agreement_table

    m = load_checkpoint(a.ckpt); ds = CIFAR10NPZ(a.data)
    g = trace(m)
    print(f"== model {m.config}  MACs={g.total_macs/1e6:.1f}M params={g.total_params:,}\n")

    print(f"== cost model on {a.spec} (simulated)")
    r = estimate(g, a.spec); print(r.table()); print()

    print("== NPU readiness lint (static + 64-image dynamic checks)")
    lr = lint(g, a.spec, ds.calib_batch(64).numpy())
    print(f"scores: {lr.scores}  counts: {lr.counts()}")
    for f in lr.findings[:6]:
        print(f"  [{f.severity:6s}] {f.check:24s} {f.layer:18s} {f.message}")
    print()

    print("== PTQ (per-channel W, uint8 A, min-max, 512 calibration images)")
    calib = [ds.calib_batch(256, seed=s) for s in range(2)]
    qm = prepare(m, QScheme()); calibrate(qm, calib)
    print(f"float acc {evaluate(m, ds, limit=a.images):.4f} | fake-quant acc {evaluate(qm, ds, limit=a.images):.4f}  ({a.images} images)\n")

    print("== export IntGraph and run the bit-exact engines")
    ig = export_int_graph(qm)
    x, y = next(ds.test_batches(200))
    codes = quantize_input(x.numpy(), ig.input_q)
    t = time.time(); vn = NumpyEngine(ig).run(codes, return_all=True); tn = time.time() - t
    t = time.time(); vc = CppEngine(ig).run(codes, return_all=True); tc = time.time() - t
    mism = sum(int((vn[k] != vc[k]).sum()) for k in vn)
    print(f"NumPy vs C++ over {len(vn)} tensors x 200 images: {mism} differing elements (numpy {tn:.1f}s, cpp {tc:.1f}s)")
    rows, summ = compare(qm, ig, x)
    print(agreement_table(rows)); print(summ)
    print(f"integer engine acc {NumpyEngine(ig).evaluate(ds, limit=a.images):.4f} ({a.images} images)\n")

    print("== requant ablation (same quantization parameters, different integer implementations)")
    for cfg in [RequantConfig(), RequantConfig(rounding="single"), RequantConfig(rounding="truncate"), RequantConfig(mult_bits=3), RequantConfig(acc_bits=16)]:
        eng = NumpyEngine(export_int_graph(qm, cfg))
        acc = eng.evaluate(ds, limit=a.images)
        print(f"  {cfg.tag:26s} acc {acc:.4f}  saturations {sum(eng.saturations.values())}")


if __name__ == "__main__":
    main()
