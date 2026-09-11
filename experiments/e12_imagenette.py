"""E12: a second dataset and input size — Imagenette (10 ImageNet classes) at 128x128.

Everything the CIFAR pipeline does, on a larger input: intake on three presets (cycles, energy, utilization),
PTQ with three schemes measured by the bit-exact engines on the whole 3,925-image test split (C++ engine),
NumPy/C++ output equality, and the measured engine wall-clock. ImageNet-pretrained weights are not used —
the model is trained from scratch here on the 45k/5k-style split of the Imagenette train set (100 hold-out
images per class), because the pretrained-weight hosts are not reachable from this environment.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.zoo import CIFAR10NPZ, load_checkpoint, evaluate as evaluate_float
from npuloop.graph import trace
from npuloop.npu import estimate, PRESETS
from npuloop.lint import lint
from npuloop.intake import intake_report
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, NumpyEngine, quantize_input, bench_engines, layer_table
from npuloop.intengine.cpp_engine import CppEngine

DATA_IMAGENETTE = os.environ.get("NPULOOP_IMAGENETTE", "/home/user/data/imagenette128.npz")
RUN = os.path.join(RUNS, "imagenette_resnet20")
SCHEMES = ["npu-default", "per-tensor", "pow2"]
EQUAL_IMAGES = int(os.environ.get("NPULOOP_E12_EQUAL", 500))


def int_eval(ig, ds, engine="cpp"):
    eng = CppEngine(ig) if engine == "cpp" else NumpyEngine(ig)
    return eng.evaluate(ds, batch_size=100)


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    if not os.path.exists(os.path.join(RUN, "log.json")):
        log("no Imagenette checkpoint yet"); return
    ds = CIFAR10NPZ(DATA_IMAGENETTE)
    shape = (3, ds.img_size, ds.img_size)
    res = Results("e12_imagenette", meta=dict(dataset="Imagenette-160 -> 128x128 center crop", train=len(ds.x_train), val=len(ds.x_val),
                                              test=len(ds.x_test), model="ResNet-20 (stem stride 2)", schemes=SCHEMES))
    logj = json.load(open(os.path.join(RUN, "log.json")))
    m = load_checkpoint(os.path.join(RUN, "best.pt")).eval()
    calib = calib_batches(ds, 256, seed=0, per_batch=64)
    calib_np = calib[0].numpy()
    if not res.has(kind="baseline"):
        g = trace(m, shape)
        rec = dict(kind="baseline", macs=int(g.total_macs), params=int(sum(p.numel() for p in m.parameters())),
                   val_acc=evaluate_float(m, ds, split="val"), test_acc=evaluate_float(m, ds), selected_epoch=logj["selected_epoch"],
                   train_minutes=logj["train_minutes"], epochs=len(logj["epochs"]), input_shape=shape, cost={}, lint={})
        for spec in PRESETS:
            r = estimate(g, spec)
            rec["cost"][spec] = dict(total_cycles=r.total_cycles, latency_ms=r.latency_ms, array_utilization=r.array_utilization,
                                     dram_bytes=r.dram_bytes, energy_uj=r.energy_uj, energy_breakdown=r.energy_breakdown(), breakdown=r.breakdown())
            lr = lint(g, spec, calib_np)
            rec["lint"][spec] = dict(scores=lr.scores, counts=lr.counts())
        rec["intake"] = intake_report(m, "edge-10tops", calib_np, input_shape=shape)
        res.add(rec, provenance="measured+simulated")
        log(f"baseline: test {rec['test_acc']:.4f} val {rec['val_acc']:.4f} MACs {g.total_macs/1e6:.1f}M edge cycles {rec['cost']['edge-10tops']['total_cycles']:,.0f} energy {rec['cost']['edge-10tops']['energy_uj']:.1f} uJ")
    float_acc = evaluate_float(m, ds)
    for sname in SCHEMES:
        if res.has(kind="ptq", scheme=sname):
            continue
        t = time.time()
        qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
        fake = evaluate(qm, ds)
        ig = export_int_graph(qm, input_shape=shape)
        int_acc = int_eval(ig, ds, "cpp")
        rec = dict(kind="ptq", scheme=sname, float_acc=float_acc, fake_acc=fake, int_acc=int_acc, int_eval_images=len(ds.x_test),
                   int_engine="cpp", seconds=time.time() - t)
        if sname == "npu-default":
            x, _ = next(ds.test_batches(EQUAL_IMAGES))
            codes = quantize_input(x.numpy(), ig.input_q)
            a = NumpyEngine(ig).run(codes); b = CppEngine(ig).run(codes)
            rec["output_equality"] = dict(images=int(len(x)), images_with_any_mismatch=int((a != b).any(axis=1).sum()))
            xb, _ = next(ds.test_batches(16))
            bch = bench_engines(ig, xb.numpy(), repeats=3)
            rows = layer_table(ig, bch, estimate(trace(m, shape), "edge-10tops"), top=6)
            rec["bench"] = dict(batch=bch["batch"], cpu=bch["cpu"], engines={k: dict(per_image_ms=v["per_image_ms"]) for k, v in bch["engines"].items()},
                                speedup=bch["speedup"], top_nodes=rows)
        res.add(rec, provenance="measured")
        log(f"ptq {sname}: fp32 {float_acc:.4f} fake {fake:.4f} int {int_acc:.4f} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
