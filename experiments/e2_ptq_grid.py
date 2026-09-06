"""E2: PTQ scheme grid — fake-quant accuracy, bit-exact integer accuracy, fake-vs-int agreement, per-layer sensitivity."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES, layer_sensitivity
from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.verify import compare

SCHEMES = ["npu-default", "npu-percentile", "npu-mse", "per-tensor", "per-tensor-mse", "pow2", "sym-act"]
INT_EVAL = int(os.environ.get("NPULOOP_INT_EVAL", "10000"))


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e2_ptq_grid", meta=dict(calib="512 random train images (seed 0)", int_eval_images=INT_EVAL, schemes={k: v.to_dict() for k, v in PRESET_SCHEMES.items()}))
    calib = calib_batches(ds, 512, seed=0)
    x_agree, y_agree = next(ds.test_batches(500))
    for name in available_baselines():
        m = load_model(name)
        float_acc = None
        for sname in SCHEMES:
            if res.has(model=name, scheme=sname):
                continue
            if float_acc is None:
                float_acc = evaluate(m, ds)
            t = time.time()
            qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
            fq_acc = evaluate(qm, ds)
            ig = export_int_graph(qm)
            rows, summ = compare(qm, ig, x_agree)
            int_acc = NumpyEngine(ig).evaluate(ds, limit=INT_EVAL)
            rec = dict(model=name, scheme=sname, float_acc=float_acc, fake_acc=fq_acc, int_acc=int_acc, int_eval_images=INT_EVAL,
                       drop_fake=float_acc - fq_acc, drop_int=float_acc - int_acc, agreement=summ,
                       per_layer_agreement=[r.to_dict() for r in rows], seconds=time.time() - t)
            if sname in ("npu-default", "per-tensor"):
                sens = layer_sensitivity(qm, [(x_agree, y_agree)])
                rec["sensitivity"] = sens
            res.add(rec)
            log(f"{name:16s} {sname:16s} float={float_acc:.4f} fake={fq_acc:.4f} int={int_acc:.4f} top1-agree={summ['top1_agreement']:.3f} ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
