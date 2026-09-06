"""E5: calibration-set science — how many images, random vs class-balanced, how much seed variance?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES

SIZES = [8, 32, 128, 512, 2048]
SEEDS = [0, 1, 2]
MODELS = os.environ.get("NPULOOP_MODELS", "resnet20_relu,mnv2_050_relu6").split(",")


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e5_calibration", meta=dict(sizes=SIZES, seeds=SEEDS, schemes=["npu-default", "per-tensor"]))
    for name in MODELS:
        if name not in available_baselines():
            continue
        m = load_model(name)
        float_acc = evaluate(m, ds)
        for sname in ["npu-default", "per-tensor"]:
            for n in SIZES:
                for balanced in (False, True):
                    if balanced and n < 32:
                        continue
                    for seed in SEEDS:
                        if res.has(model=name, scheme=sname, n=n, balanced=balanced, seed=seed):
                            continue
                        t = time.time()
                        cb = calib_batches(ds, n, seed=seed, per_batch=min(n, 256), balanced=balanced)
                        qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, cb)
                        acc = evaluate(qm, ds)
                        res.add(dict(model=name, scheme=sname, n=n, balanced=balanced, seed=seed, float_acc=float_acc, fake_acc=acc, drop=float_acc - acc, seconds=time.time() - t))
                        log(f"{name:16s} {sname:12s} n={n:5d} bal={int(balanced)} seed={seed} acc={acc:.4f} drop={float_acc-acc:+.4f}")


if __name__ == "__main__":
    main()
