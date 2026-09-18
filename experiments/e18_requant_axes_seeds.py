"""E18: the requantization width axes of E7, now across three training seeds and the whole test split.

E7 measured multiplier width, accumulator width and bias width on one checkpoint and 2,000 images. E14 then
showed, on the rounding axis, that the *size* of a requantization effect moves by up to 37% of itself across
three seeds of the same architecture while its sign does not. That makes E7's single-seed magnitudes
unquotable, which is exactly what this experiment fixes: the same axes, three independently trained
checkpoints per network, the complete test split, and a paired standard error against the reference config.

Seed 0 is the E1 checkpoint (runs/<name>); seeds 1 and 2 come from experiments/run_seeds.sh.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *                                        # noqa: F403
from common import RUNS, Results, log, calib_batches, dataset, paired_stats, predict_labels
from npuloop.zoo import load_checkpoint
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, RequantConfig
from npuloop.intengine.cpp_engine import CppEngine

MODELS = os.environ.get("NPULOOP_E18_MODELS", "resnet20_relu,resnet20_silu,mnv2_050_relu6").split(",")
SEEDS = [int(s) for s in os.environ.get("NPULOOP_E18_SEEDS", "0,1,2").split(",")]
# The reference first: every other config is scored against it on the same images.
CONFIGS = [RequantConfig(),
           RequantConfig(mult_bits=15), RequantConfig(mult_bits=7),
           RequantConfig(acc_bits=24), RequantConfig(acc_bits=20), RequantConfig(acc_bits=16),
           RequantConfig(bias_bits=16), RequantConfig(bias_bits=12)]


def checkpoint(name: str, seed: int) -> str | None:
    d = os.path.join(RUNS, name if seed == 0 else f"{name}_s{seed}")
    p = os.path.join(d, "best.pt")
    return p if os.path.exists(p) else None


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e18_requant_axes_seeds", meta=dict(
        scheme="npu-default", calib="512 random train images (seed 0)", int_engine="cpp",
        configs=[c.tag for c in CONFIGS], seeds=SEEDS, n_test=int(len(ds.y_test)),
        note="paired SE against the reference config on the same images; E7 measured these axes on one seed "
             "and 2,000 images"))
    calib = calib_batches(ds, 512, seed=0)
    for name in MODELS:
        for seed in SEEDS:
            ck = checkpoint(name, seed)
            if ck is None:
                log(f"{name} seed {seed}: no checkpoint"); continue
            if all(res.has(model=name, seed=seed, config=c.tag) for c in CONFIGS):
                continue
            m = load_checkpoint(ck).eval()
            qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
            ref_labels = y = None
            for cfg in CONFIGS:
                t = time.time()
                if res.has(model=name, seed=seed, config=cfg.tag) and ref_labels is not None:
                    continue
                ig = export_int_graph(qm, cfg)
                eng = CppEngine(ig)
                labels, y = predict_labels(eng.predict, ds, 500)
                if ref_labels is None:                      # CONFIGS[0] is the reference
                    ref_labels = labels
                v = paired_stats(ref_labels, labels, y)
                if not res.has(model=name, seed=seed, config=cfg.tag):
                    res.add(dict(model=name, seed=seed, config=cfg.tag, requant=cfg.__dict__.copy(),
                                 n_test=int(len(y)), int_acc=float((labels == y).mean()),
                                 vs_reference=v, seconds=time.time() - t))
                log(f"{name:15s} s{seed} {cfg.tag:22s} int={float((labels == y).mean()):.4f} "
                    f"vs-ref={v['delta']*100:+.2f}±{v['se']*100:.2f}%p agree={v['top1_agreement']:.4f} "
                    f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
