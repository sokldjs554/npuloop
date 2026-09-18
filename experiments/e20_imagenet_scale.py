"""E20: the same fake-vs-integer measurement on an ImageNet-scale graph with ImageNet weights.

Every other experiment here runs on a network this project trained: CIFAR-10 scale, or Imagenette at 128px.
The standing limitation was that the conclusions might be an artifact of that scale. E17 answered the depth
half of it by measuring one task at two depths; this answers the rest by taking a network nobody here trained.

ResNet-50 with Keras's published ImageNet weights, ported to torch in npuloop/zoo/resnet50_keras.py and checked
against Keras itself to 1.1e-6 max probability difference. 53 convolutions, a 1000-way head, reductions up to
K = 4608 -- an order of magnitude past ResNet-20's 576, which is the regime E7/E18's accumulator-width results
explicitly did not cover.

What this is: an ImageNet-scale graph, ImageNet-trained weights, scored through the full 1000-way head.
What this is not: ImageNet-1k validation accuracy. The images are Imagenette's test split (3,925 images of ten
ImageNet classes, shortest side 160), because the ImageNet validation set is not distributable. Top-1 here is
therefore over ten of the thousand classes; the quantities this experiment is about -- the fake-to-integer
difference and the output-code mismatch -- are unaffected by which subset of classes the images come from.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *                                        # noqa: F403
from common import ROOT, Results, log, paired_stats
from npuloop.zoo.resnet50_keras import resnet50_keras, preprocess, WEIGHTS_PATH
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare

SCHEMES = os.environ.get("NPULOOP_E20_SCHEMES", "npu-default,per-tensor").split(",")
DATA = os.environ.get("NPULOOP_IMAGENETTE160", os.path.join(ROOT, "data", "imagenette160.npz"))
LIMIT = int(os.environ.get("NPULOOP_E20_IMAGES", "0")) or None       # 0 = the whole test split
AGREE_IMAGES = int(os.environ.get("NPULOOP_E20_AGREE", "32"))
BATCH = int(os.environ.get("NPULOOP_E20_BATCH", "25"))
# Imagenette's ten folders, in the sorted-wnid order the prep script used, as ImageNet-1k indices.
IMAGENET_IDX = np.array([0, 217, 482, 491, 497, 566, 569, 571, 574, 701])


def output_codes(logits, out_q):
    return np.clip(np.rint(logits.astype(np.float64) / out_q.scale) + out_q.zero_point,
                   out_q.qmin, out_q.qmax).astype(np.int64)


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "4")))
    if not os.path.exists(WEIGHTS_PATH):
        raise SystemExit(f"{WEIGHTS_PATH} missing; see npuloop/zoo/resnet50_keras.py for the URL")
    z = np.load(DATA)
    x_test, y_test = z["x_test"], IMAGENET_IDX[z["y_test"]]
    x_train = z["x_train"]
    if LIMIT:
        x_test, y_test = x_test[:LIMIT], y_test[:LIMIT]
    size = int(x_test.shape[1])
    m = resnet50_keras()
    convs = sum(1 for mod in m.modules() if isinstance(mod, torch.nn.Conv2d))
    res = Results("e20_imagenet_scale", meta=dict(
        model="ResNet-50 (keras.applications weights, ImageNet-1k)", convolutions=convs,
        dataset=f"Imagenette test split at {size}px, scored through the 1000-way ImageNet head",
        classes=1000, scored_classes=10, n_test=int(len(y_test)), int_engine="cpp", schemes=SCHEMES,
        preprocessing="keras 'caffe': RGB->BGR, minus per-channel mean, no scaling",
        calib="512 random Imagenette training images (seed 0)",
        note="ImageNet-scale graph and ImageNet weights; not ImageNet-1k validation accuracy"))

    rng = np.random.default_rng(0)
    pick = rng.choice(len(x_train), 512, replace=False)
    calib = [preprocess(x_train[pick[i:i + 16]]) for i in range(0, 512, 16)]
    shape = (3, size, size)

    def batches(arr, bs):
        for i in range(0, len(arr), bs):
            yield preprocess(arr[i:i + bs])

    float_labels = None
    for sname in SCHEMES:
        if res.has(scheme=sname):
            continue
        t = time.time()
        if float_labels is None:
            with torch.no_grad():
                float_labels = np.concatenate([m(xb).argmax(1).numpy() for xb in batches(x_test, BATCH)])
            log(f"float32 top-1 {float((float_labels == y_test).mean()):.4f} ({time.time() - t:.0f}s)")
        qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
        ig = export_int_graph(qm, input_shape=shape)
        eng = CppEngine(ig)
        out_q = ig["output"].out_q
        fake_labels, int_labels = [], []
        mism = any_mism = total = 0
        max_dcode = 0
        for xb in batches(x_test, BATCH):
            with torch.no_grad():
                fl = qm(xb).numpy()
            il = eng.predict(xb.numpy())
            fake_labels.append(fl.argmax(1)); int_labels.append(il.argmax(1))
            cf, ci = output_codes(fl, out_q), output_codes(il, out_q)
            d = np.abs(cf - ci)
            mism += int((d != 0).sum()); any_mism += int((d != 0).any(axis=1).sum()); total += cf.size
            max_dcode = max(max_dcode, int(d.max()))
        fake_labels, int_labels = np.concatenate(fake_labels), np.concatenate(int_labels)
        rows, summ = compare(qm, ig, preprocess(x_test[:AGREE_IMAGES]))
        rec = dict(model="resnet50_imagenet", scheme=sname, n_test=int(len(y_test)), input_shape=list(shape),
                   convolutions=convs, max_reduction_k=max(int(np.prod(n.w_int.shape[1:])) for n in ig.nodes if n.op == "conv"),
                   float_acc=float((float_labels == y_test).mean()), fake_acc=float((fake_labels == y_test).mean()),
                   int_acc=float((int_labels == y_test).mean()),
                   int_vs_fake=paired_stats(fake_labels, int_labels, y_test),
                   int_vs_float=paired_stats(float_labels, int_labels, y_test),
                   fake_vs_float=paired_stats(float_labels, fake_labels, y_test),
                   output_codes=dict(mismatch_frac=mism / total, images_with_any_mismatch=any_mism,
                                     images=int(len(y_test)), classes=int(total // len(y_test)),
                                     max_abs_code_diff=max_dcode),
                   agreement_batch=dict(images=AGREE_IMAGES, **summ),
                   per_layer_agreement=[r.to_dict() for r in rows], seconds=time.time() - t)
        res.add(rec)
        p = rec["int_vs_fake"]
        log(f"resnet50 {sname:12s} float={rec['float_acc']:.4f} fake={rec['fake_acc']:.4f} int={rec['int_acc']:.4f} "
            f"int-fake={p['delta'] * 100:+.2f}%p ± {p['se'] * 100:.2f} agree={p['top1_agreement']:.4f} "
            f"codes≠={mism / total * 100:.1f}% max|Δc|={max_dcode} K={rec['max_reduction_k']} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
