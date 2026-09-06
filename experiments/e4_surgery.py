"""E4: repairing quantization damage — CLE / bias correction / activation swap + healing / QAT.

(a) MobileNetV2 with per-tensor weights (NPU without per-channel support): PTQ -> +CLE -> +CLE+BC -> +QAT
(b) ResNet-20-SiLU: PTQ (LUT) vs swap SiLU->ReLU / SiLU->HardSwish + short healing fine-tune, with NPU cycles
    on the LUT-capable and strict presets
(c) QAT on every baseline with the default NPU scheme (3 epochs)
"""
import copy, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.npu import estimate
from npuloop.quant import (prepare, calibrate, evaluate, PRESET_SCHEMES, QScheme, fold_bn, equalize, bias_correction,
                           swap_activations, qat)
from npuloop.zoo import fit
from npuloop.intengine import export_int_graph, NumpyEngine

HEAL_EPOCHS = int(os.environ.get("NPULOOP_HEAL_EPOCHS", "4"))
QAT_EPOCHS = int(os.environ.get("NPULOOP_QAT_EPOCHS", "3"))
INT_EVAL = int(os.environ.get("NPULOOP_INT_EVAL", "5000"))


def int_acc(qm, ds):
    return NumpyEngine(export_int_graph(qm)).evaluate(ds, limit=INT_EVAL)


def part_a(ds, res):
    name = "mnv2_050_relu6"
    if name not in available_baselines():
        return
    m = load_model(name); float_acc = evaluate(m, ds)
    calib = calib_batches(ds, 512, seed=0)
    for sname in ["per-tensor", "npu-default"]:
        sch = PRESET_SCHEMES[sname]
        steps = []
        if not res.has(part="a", model=name, scheme=sname, step="ptq"):
            qm = prepare(m, sch); calibrate(qm, calib)
            res.add(dict(part="a", model=name, scheme=sname, step="ptq", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds)))
        if not res.has(part="a", model=name, scheme=sname, step="cle"):
            gm = fold_bn(m); st = equalize(gm)
            qm = prepare(gm, sch); calibrate(qm, calib)
            res.add(dict(part="a", model=name, scheme=sname, step="cle", float_acc=float_acc, cle=st, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds),
                         float_acc_after_cle=evaluate(gm, ds)))
        if not res.has(part="a", model=name, scheme=sname, step="cle+bc"):
            gm = fold_bn(m); equalize(gm)
            qm = prepare(gm, sch); calibrate(qm, calib); corr = bias_correction(qm, gm, calib)
            res.add(dict(part="a", model=name, scheme=sname, step="cle+bc", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds),
                         bias_shift_mean=float(np.mean(list(corr.values())))))
        if not res.has(part="a", model=name, scheme=sname, step="bc"):
            gm = fold_bn(m)
            qm = prepare(gm, sch); calibrate(qm, calib); corr = bias_correction(qm, gm, calib)
            res.add(dict(part="a", model=name, scheme=sname, step="bc", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds)))
        if not res.has(part="a", model=name, scheme=sname, step="cle+qat"):
            gm = fold_bn(m); equalize(gm)
            qm = prepare(gm, sch); calibrate(qm, calib)
            t = time.time(); lg = qat(qm, ds, epochs=QAT_EPOCHS, lr=0.005, seed=0)
            res.add(dict(part="a", model=name, scheme=sname, step="cle+qat", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds),
                         qat_epochs=QAT_EPOCHS, minutes=(time.time() - t) / 60, qat_log=lg["epochs"]))
        log(f"part a {sname} done")


def part_b(ds, res):
    name = "resnet20_silu"
    if name not in available_baselines():
        return
    m = load_model(name); float_acc = evaluate(m, ds)
    calib = calib_batches(ds, 512, seed=0)
    sch = PRESET_SCHEMES["npu-default"]

    def cost(model):
        g = trace(model)
        return {s: estimate(g, s).total_cycles for s in ["edge-10tops", "edge-10tops-strict", "pcie-80tops", "tiny-1tops"]}
    if not res.has(part="b", variant="silu-ptq"):
        qm = prepare(m, sch); calibrate(qm, calib)
        res.add(dict(part="b", variant="silu-ptq", act="silu", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds), cycles=cost(m), heal_epochs=0), provenance="measured+simulated")
    if not res.has(part="b", variant="silu-qat"):
        qm = prepare(m, sch); calibrate(qm, calib); lg = qat(qm, ds, epochs=QAT_EPOCHS, lr=0.005, seed=0)
        res.add(dict(part="b", variant="silu-qat", act="silu", float_acc=float_acc, fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds), cycles=cost(m), qat_epochs=QAT_EPOCHS), provenance="measured+simulated")
    for target in ["relu", "hswish"]:
        for heal in [0, HEAL_EPOCHS]:
            v = f"swap-{target}-heal{heal}"
            if res.has(part="b", variant=v):
                continue
            ms = swap_activations(m, {"silu": target})
            acc0 = evaluate(ms, ds)
            lg = None
            if heal:
                t = time.time(); lg = fit(ms, ds, epochs=heal, lr=0.02, seed=0, warmup_pct=0.2)
                ms.eval()
            acc1 = evaluate(ms, ds)
            qm = prepare(ms, sch); calibrate(qm, calib)
            res.add(dict(part="b", variant=v, act=target, float_acc=float_acc, float_acc_after_swap=acc0, float_acc_after_heal=acc1,
                         fake_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds), cycles=cost(ms), heal_epochs=heal,
                         heal_log=(lg["epochs"] if lg else None)), provenance="measured+simulated")
            log(f"part b {v}: swap {acc0:.4f} -> heal {acc1:.4f} -> int8 {res.data['records'][-1]['fake_acc']:.4f}")


def part_c(ds, res):
    calib = calib_batches(ds, 512, seed=0)
    for name in available_baselines():
        if res.has(part="c", model=name):
            continue
        m = load_model(name); float_acc = evaluate(m, ds)
        for sname in ["npu-default", "per-tensor"]:
            if res.has(part="c", model=name, scheme=sname):
                continue
            qm = prepare(m, PRESET_SCHEMES[sname]); calibrate(qm, calib)
            ptq = evaluate(qm, ds)
            t = time.time(); lg = qat(qm, ds, epochs=QAT_EPOCHS, lr=0.005, seed=0)
            res.add(dict(part="c", model=name, scheme=sname, float_acc=float_acc, ptq_acc=ptq, qat_acc=evaluate(qm, ds), int_acc=int_acc(qm, ds),
                         qat_epochs=QAT_EPOCHS, minutes=(time.time() - t) / 60, qat_log=lg["epochs"]))
            log(f"part c {name} {sname}: ptq {ptq:.4f} -> qat {res.data['records'][-1]['qat_acc']:.4f}")


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "4")))
    ds = dataset()
    res = Results("e4_surgery", meta=dict(heal_epochs=HEAL_EPOCHS, qat_epochs=QAT_EPOCHS, int_eval_images=INT_EVAL))
    parts = os.environ.get("NPULOOP_PARTS", "abc")
    if "a" in parts: part_a(ds, res)
    if "b" in parts: part_b(ds, res)
    if "c" in parts: part_c(ds, res)


if __name__ == "__main__":
    main()
