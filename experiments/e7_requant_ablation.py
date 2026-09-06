"""E7: what does a sloppy integer implementation cost? RequantConfig ablation on the bit-exact engine."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, NumpyEngine, RequantConfig

CONFIGS = [
    RequantConfig(),                                   # reference: TFLite semantics
    RequantConfig(rounding="half_even"),
    RequantConfig(rounding="truncate"),
    RequantConfig(rounding="floor"),
    RequantConfig(mult_bits=15),
    RequantConfig(mult_bits=7),
    RequantConfig(mult_bits=3),
    RequantConfig(bias_bits=16),
    RequantConfig(bias_bits=12),
    RequantConfig(acc_bits=24),
    RequantConfig(acc_bits=20),
    RequantConfig(acc_bits=16),
]
INT_EVAL = int(os.environ.get("NPULOOP_INT_EVAL", "5000"))
MODELS = os.environ.get("NPULOOP_MODELS", "resnet20_relu,mnv2_050_relu6,resnet20_silu").split(",")


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e7_requant_ablation", meta=dict(int_eval_images=INT_EVAL, scheme="npu-default", calib="512 random (seed 0)"))
    calib = calib_batches(ds, 512, seed=0)
    for name in MODELS:
        if name not in available_baselines():
            continue
        m = load_model(name)
        qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
        fake_acc = None
        for cfg in CONFIGS:
            if res.has(model=name, config=cfg.tag):
                continue
            if fake_acc is None:
                fake_acc = evaluate(qm, ds, limit=INT_EVAL)
            t = time.time()
            ig = export_int_graph(qm, cfg)
            eng = NumpyEngine(ig)
            acc = eng.evaluate(ds, limit=INT_EVAL)
            rec = dict(model=name, config=cfg.tag, requant=cfg.to_dict(), fake_acc=fake_acc, int_acc=acc, drop_vs_fake=fake_acc - acc,
                       saturations=dict(eng.saturations), seconds=time.time() - t)
            res.add(rec)
            log(f"{name:16s} {cfg.tag:28s} fake={fake_acc:.4f} int={acc:.4f} sat={sum(eng.saturations.values())} ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
