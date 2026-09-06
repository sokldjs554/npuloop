"""E7: what does a sloppy integer implementation cost? RequantConfig ablation on the bit-exact engine."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph, NumpyEngine, RequantConfig

CONFIGS = [
    RequantConfig(),                                   # reference: legacy TFLite/gemmlowp double rounding
    RequantConfig(rounding="single"),                  # TFLITE_SINGLE_ROUNDING
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
INT_EVAL = int(os.environ.get("NPULOOP_INT_EVAL_ABLATION", "2000"))
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
        todo = [cfg for cfg in CONFIGS if not res.has(model=name, config=cfg.tag)]
        if not todo:
            continue
        xs, ys = [], []
        for xb, yb in ds.test_batches(500):
            xs.append(xb); ys.append(yb)
            if sum(len(y) for y in ys) >= INT_EVAL:
                break
        x = torch.cat(xs)[:INT_EVAL]; y = torch.cat(ys)[:INT_EVAL].numpy()
        with torch.no_grad():
            fake_logits = torch.cat([qm(x[i:i + 500]) for i in range(0, len(x), 500)]).numpy()
        fake_acc = float((fake_logits.argmax(1) == y).mean())
        ref_logits = None
        for cfg in CONFIGS:
            t = time.time()
            ig = export_int_graph(qm, cfg)
            eng = NumpyEngine(ig)
            logits = np.concatenate([eng.predict(x[i:i + 500].numpy()) for i in range(0, len(x), 500)])
            if cfg == CONFIGS[0]:
                ref_logits = logits
            if res.has(model=name, config=cfg.tag):
                continue
            acc = float((logits.argmax(1) == y).mean())
            rec = dict(model=name, config=cfg.tag, requant=cfg.to_dict(), n_images=INT_EVAL, fake_acc=fake_acc, int_acc=acc, drop_vs_fake=fake_acc - acc,
                       top1_agreement_vs_reference=float((logits.argmax(1) == ref_logits.argmax(1)).mean()),
                       top1_agreement_vs_fake=float((logits.argmax(1) == fake_logits.argmax(1)).mean()),
                       mean_abs_logit_diff_vs_reference=float(np.abs(logits - ref_logits).mean()),
                       saturations=dict(eng.saturations), seconds=time.time() - t)
            res.add(rec)
            log(f"{name:16s} {cfg.tag:28s} fake={fake_acc:.4f} int={acc:.4f} agree(ref)={rec['top1_agreement_vs_reference']:.4f} sat={sum(eng.saturations.values())} ({time.time()-t:.0f}s)")

if __name__ == "__main__":
    main()
