"""E17: the fake-quant vs integer gap on a task whose deliverable is the output tensor.

E15 measures six classifiers, and on all of them an argmax stands between the tensor and the answer: tens of
percent of the output codes differ while top-1 barely moves. This experiment removes the argmax. The ESPCN x2
super-resolution network outputs the picture itself, so a code that lands one LSB off *is* the result.

For each PTQ scheme we report, on the full Imagenette test split:
  * float, fake-quant and integer PSNR on the same images, with the paired standard error of (integer - fake);
  * the fraction of output *pixel* codes that differ, and the largest such difference in 8-bit levels;
  * the same local (teacher-forced) / propagated per-node decomposition E15 uses.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *                                        # noqa: F403
from common import ROOT, RUNS, Results, log, paired_delta
from npuloop.zoo.data import SRPairs, psnr
from npuloop.zoo.train import load_checkpoint
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.intengine import export_int_graph
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.verify import compare

SCHEMES = os.environ.get("NPULOOP_E17_SCHEMES", "npu-default,per-tensor").split(",")
DATA = os.environ.get("NPULOOP_IMAGENETTE", os.path.join(ROOT, "data", "imagenette128.npz"))
RUN = os.path.join(RUNS, "espcn_x2")
AGREE_IMAGES = int(os.environ.get("NPULOOP_E17_AGREE_IMAGES", "64"))
BS = int(os.environ.get("NPULOOP_E17_BS", "50"))


def out_codes(y: np.ndarray, q) -> np.ndarray:
    return np.clip(np.rint(y.astype(np.float64) / q.scale) + q.zero_point, q.qmin, q.qmax).astype(np.int64)


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "2")))
    ckpt = os.path.join(RUN, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint at {ckpt} -- run python -m npuloop.zoo.train_sr --out {RUN} first")
    m = load_checkpoint(ckpt).eval()
    ds = SRPairs(DATA, scale=m.config["scale"])
    shape = (3, ds.lr_size, ds.lr_size)
    calib = [ds.calib_batch(512, seed=0)[i:i + 64] for i in range(0, 512, 64)]
    res = Results("e17_dense_output", meta=dict(
        model="espcn_x2", dataset="Imagenette-128", task="x2 super-resolution",
        metric="per-image PSNR (dB) on [0,1], prediction clipped as a deployment would",
        calib="512 random LR training images (seed 0)", int_engine="cpp", schemes=SCHEMES,
        agree_images=AGREE_IMAGES,
        note="output codes here are pixels, not logits: 3x128x128 per image instead of 10"))

    float_psnr = None
    for sname in SCHEMES:
        if res.has(scheme=sname):
            continue
        t = time.time()
        qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
        ig = export_int_graph(qm, input_shape=shape)
        eng = CppEngine(ig)
        q = ig["output"].out_q
        fl_list, fk_list, it_list = [], [], []
        mism = 0; any_mism = 0; total = 0; max_level = 0
        for lr, hr in ds.test_batches(BS):
            with torch.no_grad():
                fk = qm(lr).numpy()
                if float_psnr is None:
                    fl_list.append(psnr(m(lr).numpy(), hr.numpy()))
            it = eng.predict(lr.numpy())
            fk_list.append(psnr(fk, hr.numpy())); it_list.append(psnr(it, hr.numpy()))
            cf, ci = out_codes(fk, q), out_codes(it, q)
            d = np.abs(cf - ci)
            mism += int((d != 0).sum()); any_mism += int((d != 0).any(axis=(1, 2, 3)).sum())
            total += cf.size; max_level = max(max_level, int(d.max()))
        if float_psnr is None:
            float_psnr = np.concatenate(fl_list)
        fake_psnr, int_psnr = np.concatenate(fk_list), np.concatenate(it_list)
        lr_agree, _ = next(ds.test_batches(AGREE_IMAGES))
        rows, summ = compare(qm, ig, lr_agree)
        rec = dict(model="espcn_x2", scheme=sname, n_test=int(len(int_psnr)), input_shape=list(shape),
                   float_psnr=float(float_psnr.mean()), fake_psnr=float(fake_psnr.mean()),
                   int_psnr=float(int_psnr.mean()),
                   int_vs_fake=paired_delta(fake_psnr, int_psnr), int_vs_float=paired_delta(float_psnr, int_psnr),
                   fake_vs_float=paired_delta(float_psnr, fake_psnr),
                   output_codes=dict(mismatch_frac=mism / total, images_with_any_mismatch=any_mism,
                                     images=int(len(int_psnr)), values_per_image=int(total // len(int_psnr)),
                                     max_abs_code_diff=max_level),
                   agreement_batch=dict(images=int(len(lr_agree)), **summ),
                   per_layer_agreement=[r.to_dict() for r in rows], seconds=time.time() - t)
        res.add(rec)
        p = rec["int_vs_fake"]
        log(f"espcn_x2 {sname:12s} float={rec['float_psnr']:.3f}dB fake={rec['fake_psnr']:.3f} int={rec['int_psnr']:.3f} "
            f"int-fake={p['delta']:+.4f}dB ± {p['se']:.4f} codes≠={mism/total*100:.1f}% max|Δcode|={max_level} "
            f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
