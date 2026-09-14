"""E14: the requantization rounding-mode cost of E7, now across three training seeds and the whole test split.

Seed 0 is the E1 checkpoint (runs/<name>); seeds 1 and 2 come from experiments/run_seeds.sh (runs/<name>_s1, _s2).
The quantization parameters (npu-default, 512 calibration images, seed 0) are the same for every rounding mode;
only the integer engine's second-stage rounding changes. Every configuration is compared with the reference
implementation (gemmlowp double rounding) and with fake-quant on the same 10,000 images, with exact paired
standard errors.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.zoo import load_checkpoint
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, RequantConfig
from npuloop.intengine.cpp_engine import CppEngine

MODELS = os.environ.get("NPULOOP_E14_MODELS", "resnet20_relu,resnet20_silu,mnv2_050_relu6").split(",")
SEEDS = [int(s) for s in os.environ.get("NPULOOP_E14_SEEDS", "0,1,2").split(",")]
ROUNDINGS = ["tflite", "single", "half_even", "truncate", "floor"]


def checkpoint(name: str, seed: int) -> str | None:
    d = os.path.join(RUNS, name if seed == 0 else f"{name}_s{seed}")
    p = os.path.join(d, "log.json")
    if os.path.exists(p) and "final_test_acc" in json.load(open(p)):
        return os.path.join(d, "best.pt")
    return None


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e14_rounding_seeds", meta=dict(scheme="npu-default", calib="512 random train images (seed 0)", int_engine="cpp",
                                                  roundings=ROUNDINGS, seeds=SEEDS, n_test=int(len(ds.y_test))))
    calib = calib_batches(ds, 512, seed=0)
    y = ds.y_test
    for name in MODELS:
        for seed in SEEDS:
            ck = checkpoint(name, seed)
            if ck is None:
                log(f"{name} seed {seed}: no checkpoint yet"); continue
            if all(res.has(model=name, seed=seed, rounding=r) for r in ROUNDINGS):
                continue
            m = load_checkpoint(ck)
            qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
            fake_labels, _ = predict_labels(torch_predict(qm), ds, 500)
            float_labels, _ = predict_labels(torch_predict(m), ds, 500)
            labels = {}
            for r in ROUNDINGS:
                t = time.time()
                eng = CppEngine(export_int_graph(qm, RequantConfig(rounding=r)))
                labels[r], _ = predict_labels(eng.predict_batch if hasattr(eng, "predict_batch") else (lambda xb, e=eng: e.predict(xb.numpy())), ds, 500)
                if res.has(model=name, seed=seed, rounding=r):
                    continue
                rec = dict(model=name, seed=seed, rounding=r, config=RequantConfig(rounding=r).tag, n_test=int(len(y)),
                           float_acc=float((float_labels == y).mean()), fake_acc=float((fake_labels == y).mean()), int_acc=float((labels[r] == y).mean()),
                           vs_reference=paired_stats(labels["tflite"], labels[r], y), vs_fake=paired_stats(fake_labels, labels[r], y),
                           seconds=time.time() - t)
                res.add(rec)
                v = rec["vs_reference"]
                log(f"{name:15s} s{seed} {r:9s} int={rec['int_acc']:.4f} vs-ref={v['delta']*100:+.2f}±{v['se']*100:.2f}%p agree={v['top1_agreement']:.4f} ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
