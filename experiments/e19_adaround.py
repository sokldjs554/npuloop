"""E19: does AdaRound's accuracy gain survive bit-exact integer execution?

AdaRound (Nagel et al., ICML 2020) chooses each weight's rounding direction by optimizing the layer's own
output error instead of rounding to nearest. It is evaluated, everywhere it is reported, on a fake-quantization
graph. This repository can ask the question that setting cannot: the learned rounding is written to the
layer's `w_round` buffer, which `int_weight()` also reads, so the exported integer program executes exactly
the rounding that was optimized -- and E15 has already shown that fake-quantization and integer execution
agree on accuracy but not on tensors.

For every (model, scheme) we report four accuracies on the complete test split -- fake and integer, with
round-to-nearest and with AdaRound -- and the paired difference of each, so a gain in the simulator can be
compared against the gain on the device.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *                                        # noqa: F403
from common import Results, log, calib_batches, dataset, paired_stats, predict_labels, torch_predict, available_baselines, load_model
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.quant.adaround import adaround
from npuloop.intengine import export_int_graph
from npuloop.intengine.cpp_engine import CppEngine

MODELS = os.environ.get("NPULOOP_E19_MODELS", "resnet20_relu,resnet20_silu,mnv2_050_relu6").split(",")
SCHEMES = os.environ.get("NPULOOP_E19_SCHEMES", "npu-default,per-tensor").split(",")
ITERS = int(os.environ.get("NPULOOP_E19_ITERS", "1500"))


def accuracies(qm, ds, shape):
    """(fake labels, integer labels, true labels) over the complete test split."""
    fake, y = predict_labels(torch_predict(qm), ds, 500)
    ig = export_int_graph(qm, input_shape=shape)
    intr, _ = predict_labels(CppEngine(ig).predict, ds, 500)
    return fake, intr, y


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    shape = (3, ds.img_size, ds.img_size)
    res = Results("e19_adaround", meta=dict(
        method="AdaRound (Nagel et al., ICML 2020), layer-wise, quantized input / float output",
        iters=ITERS, calib="512 random train images (seed 0)", schemes=SCHEMES, int_engine="cpp",
        n_test=int(len(ds.y_test)),
        note="the learned rounding is written to w_round, which int_weight() reads, so the integer program "
             "executes the rounding that was optimized"))
    calib = calib_batches(ds, 512, seed=0)
    for name in MODELS:
        if name not in available_baselines():
            log(f"{name}: no checkpoint"); continue
        for sname in SCHEMES:
            if res.has(model=name, scheme=sname):
                continue
            t = time.time()
            m = load_model(name)
            base = prepare(m, PRESET_SCHEMES[sname]); calibrate(base, calib)
            f0, i0, y = accuracies(base, ds, shape)

            ada = prepare(m, PRESET_SCHEMES[sname]); calibrate(ada, calib)
            report = adaround(ada, calib, iters=ITERS, log=None)
            f1, i1, _ = accuracies(ada, ds, shape)

            rec = dict(model=name, scheme=sname, n_test=int(len(y)), adaround_iters=ITERS,
                       fake_rtn=float((f0 == y).mean()), int_rtn=float((i0 == y).mean()),
                       fake_ada=float((f1 == y).mean()), int_ada=float((i1 == y).mean()),
                       fake_gain=paired_stats(f0, f1, y),      # AdaRound - round-to-nearest, in the simulator
                       int_gain=paired_stats(i0, i1, y),       # the same comparison on the integer program
                       int_vs_fake_rtn=paired_stats(f0, i0, y),
                       int_vs_fake_ada=paired_stats(f1, i1, y),
                       flipped_frac=float(sum(L["flipped_frac"] * L["weights"] for L in report["layers"])
                                          / max(1, sum(L["weights"] for L in report["layers"]))),
                       layers=report["layers"], seconds=time.time() - t)
            res.add(rec)
            fg, ig_ = rec["fake_gain"], rec["int_gain"]
            log(f"{name:15s} {sname:12s} fake {rec['fake_rtn']*100:.2f}->{rec['fake_ada']*100:.2f} "
                f"({fg['delta']*100:+.2f}±{fg['se']*100:.2f}%p)  int {rec['int_rtn']*100:.2f}->{rec['int_ada']*100:.2f} "
                f"({ig_['delta']*100:+.2f}±{ig_['se']*100:.2f}%p)  flipped {rec['flipped_frac']*100:.1f}% "
                f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
