"""E15: fake-quant vs integer fidelity with exact paired standard errors, on the whole test split.

For every baseline (five CIFAR-10 models and the Imagenette-128 ResNet-20) and the two primary PTQ schemes:
  * float, fake-quant and integer (C++ engine) top-1 on the same images, so accuracy differences get an exact
    paired standard error (per-image correctness differences) instead of the two-sample bound;
  * the fraction of output codes that differ between fake-quant and integer execution over the full split;
  * the local (teacher-forced) / propagated decomposition of E2/E9 on a fixed batch.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.zoo import CIFAR10NPZ, load_checkpoint
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare

SCHEMES = os.environ.get("NPULOOP_E15_SCHEMES", "npu-default,per-tensor").split(",")
DATA_IMAGENETTE = os.environ.get("NPULOOP_IMAGENETTE", os.path.join(ROOT, "data", "imagenette128.npz"))
IMAGENETTE_RUN = os.path.join(RUNS, "imagenette_resnet20")
AGREE_IMAGES = int(os.environ.get("NPULOOP_AGREE_IMAGES", "250"))


def output_codes(logits: np.ndarray, out_q) -> np.ndarray:
    return np.clip(np.rint(logits.astype(np.float64) / out_q.scale) + out_q.zero_point, out_q.qmin, out_q.qmax).astype(np.int64)


def models():
    for name in available_baselines():
        yield name, "CIFAR-10", load_model(name), None
    if os.path.exists(DATA_IMAGENETTE) and os.path.exists(os.path.join(IMAGENETTE_RUN, "log.json")):
        yield "imagenette_resnet20", "Imagenette-128", load_checkpoint(os.path.join(IMAGENETTE_RUN, "best.pt")).eval(), DATA_IMAGENETTE


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    cifar = dataset()
    res = Results("e15_fidelity", meta=dict(calib="512 random train images (seed 0), re-collected per model — E12 used 256 for Imagenette",
                                            int_engine="cpp", agree_images=f"{AGREE_IMAGES} (half of it for inputs larger than 64px; the per-record value is agreement_batch.images)",
                                            schemes=SCHEMES, note="paired SE = std of per-image correctness differences / sqrt(n)"))
    for name, dsname, m, data_path in models():
        ds = cifar if data_path is None else CIFAR10NPZ(data_path)
        shape = (3, ds.img_size, ds.img_size)
        big = ds.img_size > 64
        calib = calib_batches(ds, 512, seed=0, per_batch=64 if big else 256)
        bs = 100 if big else 500
        float_labels = None
        for sname in SCHEMES:
            if res.has(model=name, scheme=sname):
                continue
            t = time.time()
            if float_labels is None:
                float_labels, y = predict_labels(torch_predict(m), ds, bs)
            qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
            ig = export_int_graph(qm, input_shape=shape)
            eng = CppEngine(ig)
            out_q = ig["output"].out_q
            fake_labels, int_labels = [], []
            mism = 0; any_mism = 0; total = 0
            for xb, _ in ds.batches("test", bs):
                with torch.no_grad():
                    fl = qm(xb).numpy()
                il = eng.predict(xb.numpy())
                fake_labels.append(fl.argmax(1)); int_labels.append(il.argmax(1))
                cf, ci = output_codes(fl, out_q), output_codes(il, out_q)
                mism += int((cf != ci).sum()); any_mism += int((cf != ci).any(axis=1).sum()); total += cf.size
            fake_labels, int_labels = np.concatenate(fake_labels), np.concatenate(int_labels)
            x_agree, _ = next(ds.batches("test", AGREE_IMAGES // 2 if big else AGREE_IMAGES))
            rows, summ = compare(qm, ig, x_agree)
            rec = dict(model=name, dataset=dsname, scheme=sname, n_test=int(len(y)), input_shape=list(shape),
                       float_acc=float((float_labels == y).mean()), fake_acc=float((fake_labels == y).mean()), int_acc=float((int_labels == y).mean()),
                       int_vs_fake=paired_stats(fake_labels, int_labels, y), int_vs_float=paired_stats(float_labels, int_labels, y),
                       fake_vs_float=paired_stats(float_labels, fake_labels, y),
                       output_codes=dict(mismatch_frac=mism / total, images_with_any_mismatch=any_mism, images=int(len(y)), classes=int(total // len(y))),
                       agreement_batch=dict(images=int(len(x_agree)), **summ), per_layer_agreement=[r.to_dict() for r in rows],
                       seconds=time.time() - t)
            res.add(rec)
            p = rec["int_vs_fake"]
            log(f"{name:20s} {sname:12s} float={rec['float_acc']:.4f} fake={rec['fake_acc']:.4f} int={rec['int_acc']:.4f} "
                f"int-fake={p['delta']*100:+.2f}%p ± {p['se']*100:.2f} agree={p['top1_agreement']:.4f} codes≠={mism/total*100:.1f}% ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
