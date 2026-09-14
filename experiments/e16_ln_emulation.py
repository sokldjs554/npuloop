"""E16: make the fake-quant graph integer-faithful at LayerNorm and re-measure the transformer's gap.

E9 found that the integer LayerNorm and the float LayerNorm of the fake-quant graph disagree in about one
element out of four (all ±1 LSB) and that this is where most of the ViT's fake-vs-integer divergence starts.
`npuloop.quant.emulate` replaces the float LayerNorm with the engine's arithmetic (bit-exact by construction,
export unchanged). This experiment measures, on the whole test split, how much of the output-code mismatch,
the top-1 disagreement and the accuracy gap that one change removes.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.quant.emulate import emulate_integer_layernorm
from npuloop.intengine import export_int_graph
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare

MODELS = os.environ.get("NPULOOP_E16_MODELS", "cust_vit").split(",")
SCHEMES = os.environ.get("NPULOOP_E16_SCHEMES", "npu-default,per-tensor").split(",")
AGREE_IMAGES = int(os.environ.get("NPULOOP_AGREE_IMAGES", "250"))


def output_codes(logits, out_q):
    return np.clip(np.rint(logits.astype(np.float64) / out_q.scale) + out_q.zero_point, out_q.qmin, out_q.qmax).astype(np.int64)


def per_op(rows):
    ops = {}
    for r in rows:
        if r["op"] in ("input", "output", "flatten", "transpose", "reshape"):
            continue
        o = ops.setdefault(r["op"], dict(nodes=0, local=[], prop=[]))
        o["nodes"] += 1; o["local"].append(r["local_mismatch_frac"]); o["prop"].append(r["mismatch_frac"])
    return {k: dict(nodes=v["nodes"], local_mean=float(np.mean(v["local"])), local_max=float(np.max(v["local"])), prop_max=float(np.max(v["prop"])))
            for k, v in ops.items()}


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ds = dataset()
    res = Results("e16_ln_emulation", meta=dict(calib="512 random train images (seed 0)", int_engine="cpp", agree_images=AGREE_IMAGES))
    calib = calib_batches(ds, 512, seed=0)
    x_agree, _ = next(ds.batches("test", AGREE_IMAGES))
    for name in MODELS:
        if name not in available_baselines():
            continue
        m = load_model(name)
        for sname in SCHEMES:
            if all(res.has(model=name, scheme=sname, variant=v) for v in ("float-ln", "int-ln")):
                continue                      # both rows of this (model, scheme) are already stored
            t = time.time()
            qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
            ig = export_int_graph(qm)
            eng = CppEngine(ig); out_q = ig["output"].out_q
            int_logits = []
            for xb, _ in ds.batches("test", 500):
                int_logits.append(eng.predict(xb.numpy()))
            int_logits = np.concatenate(int_logits); y = ds.y_test
            int_labels = int_logits.argmax(1); int_codes = output_codes(int_logits, out_q)
            for variant in ("float-ln", "int-ln"):
                if variant == "int-ln":
                    n_ln = emulate_integer_layernorm(qm)
                    ig2 = export_int_graph(qm)
                    for a, b in zip(ig.nodes, ig2.nodes):        # the export must not move: same integer program
                        if a.op == "layernorm":
                            assert np.array_equal(a.mult, b.mult) and np.array_equal(a.bias_int, b.bias_int) and a.attrs["eps_int"] == b.attrs["eps_int"]
                else:
                    n_ln = 0
                fake_logits = []
                with torch.no_grad():
                    for xb, _ in ds.batches("test", 500):
                        fake_logits.append(qm(xb).numpy())
                fake_logits = np.concatenate(fake_logits)
                fake_labels = fake_logits.argmax(1); fake_codes = output_codes(fake_logits, out_q)
                rows, summ = compare(qm, ig, x_agree)
                rec = dict(model=name, scheme=sname, variant=variant, layernorms_emulated=n_ln, n_test=int(len(y)),
                           fake_acc=float((fake_labels == y).mean()), int_acc=float((int_labels == y).mean()),
                           int_vs_fake=paired_stats(fake_labels, int_labels, y),
                           output_codes=dict(mismatch_frac=float((fake_codes != int_codes).mean()),
                                             images_with_any_mismatch=int((fake_codes != int_codes).any(axis=1).sum()), images=int(len(y))),
                           logit_mean_abs_diff=float(np.abs(fake_logits - int_logits).mean()),
                           agreement_batch=dict(images=int(len(x_agree)), **summ), per_op=per_op([r.to_dict() for r in rows]),
                           per_layer_agreement=[r.to_dict() for r in rows], seconds=time.time() - t)
                if res.has(model=name, scheme=sname, variant=variant):
                    continue                  # resumed run: this row already exists, do not duplicate it
                res.add(rec)
                p = rec["int_vs_fake"]
                log(f"{name} {sname:12s} {variant:8s} fake={rec['fake_acc']:.4f} int={rec['int_acc']:.4f} int-fake={p['delta']*100:+.2f}±{p['se']*100:.2f}%p "
                    f"agree={p['top1_agreement']:.4f} codes≠={rec['output_codes']['mismatch_frac']*100:.1f}% LN local={rec['per_op'].get('layernorm',{}).get('local_mean',0)*100:.2f}% ({time.time()-t:.0f}s)")
                t = time.time()


if __name__ == "__main__":
    main()
