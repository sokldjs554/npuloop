"""E9: two fictional customer models arrive; take them through the whole loop and measure what changed.

Customer A ships a ViT (attention, LayerNorm, GELU, 12 activation-x-activation matmuls).
Customer B ships a concat-heavy Inception-style CNN (three branches per block, one shared output scale).

For each: intake on two NPU presets, PTQ with the bit-exact integer engine, then apply the intake's own
top prescription and re-measure — so the predicted cycle saving and the measured accuracy sit side by side.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.intake import intake_report, alternatives
from npuloop.npu import estimate
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES
from npuloop.quant.surgery import swap_activations
from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.verify import compare
from npuloop.zoo import fit, load_checkpoint

CUSTOMERS = {
    "cust_vit": dict(label="고객 A · ViT-128/6", spec="edge-10tops", strict="edge-10tops-strict",
                     swap={"gelu": "relu"}, heal_epochs=3, heal_lr=0.0005, optimizer="adamw"),
    "cust_inception": dict(label="고객 B · Inception-32", spec="edge-10tops", strict="edge-10tops-strict",
                           swap=None, heal_epochs=3, heal_lr=0.02, optimizer="sgd"),
}
INT_EVAL = int(os.environ.get("NPULOOP_E9_INT_EVAL", 2000))
# The models trained for E1 go through the same intake sheet (no surgery here — E4 already measured theirs).
WALK_INS = {"resnet20_relu": "기존 · ResNet-20 ReLU", "resnet20_silu": "기존 · ResNet-20 SiLU",
            "mnv2_050_relu6": "기존 · MobileNetV2-0.5"}


def measure(model, ds, calib, scheme="npu-default", int_images=INT_EVAL, verify_batch=None):
    qm = prepare(model, PRESET_SCHEMES[scheme]); calibrate(qm, calib)
    fake = evaluate(qm, ds)
    ig = export_int_graph(qm)
    eng = NumpyEngine(ig)
    integer = eng.evaluate(ds, limit=int_images)
    fake_sub = evaluate(qm, ds, limit=int_images)
    agree = None
    if verify_batch is not None:
        _, summ = compare(qm, ig, verify_batch)
        agree = dict(top1_agreement=summ["top1_agreement"], output_mismatch_frac=summ["output_mismatch_frac"],
                     first_divergence=summ["first_divergence"])
    return dict(fake_acc=fake, fake_acc_on_int_subset=fake_sub, int_acc=integer,
                int_eval_images=int_images, agreement=agree)


def main():
    ds = dataset()
    res = Results("e9_customer_intake", meta=dict(int_eval_images=INT_EVAL, scheme="npu-default"))
    calib = calib_batches(ds, 512, seed=0)
    calib_np = calib[0].numpy()
    verify_batch = ds.calib_batch(64, seed=1)
    available = available_baselines()
    only = os.environ.get("NPULOOP_E9_ONLY")
    for name, cfg in CUSTOMERS.items():
        if only and name != only:
            continue
        if name not in available:
            log(f"{name}: no checkpoint yet, skipping")
            continue
        if res.has(model=name):
            continue
        t0 = time.time()
        model = load_checkpoint(available[name]).eval()
        g = trace(model)
        before = intake_report(model, cfg["spec"], calib_np)
        before_strict = intake_report(model, cfg["strict"], calib_np)
        float_acc = evaluate(model, ds)
        ptq = measure(model, ds, calib, verify_batch=verify_batch)
        log(f"{name}: fp32={float_acc:.4f} fake={ptq['fake_acc']:.4f} int={ptq['int_acc']:.4f} "
            f"cycles={before['diagnose']['cycles']:,.0f} strict={before_strict['diagnose']['cycles']:,.0f}")

        after = None
        if cfg["swap"]:
            swapped = swap_activations(model, cfg["swap"])
            log(f"{name}: swapped {swapped.swapped} activations {cfg['swap']}, healing {cfg['heal_epochs']} epochs")
            fit(swapped, ds, epochs=cfg["heal_epochs"], lr=cfg["heal_lr"], seed=0, warmup_pct=0.2,
                optimizer=cfg["optimizer"])
            swapped.eval()
            healed_acc = evaluate(swapped, ds)
            rep = intake_report(swapped, cfg["spec"], calib_np, with_prescriptions=False)
            rep_strict = intake_report(swapped, cfg["strict"], calib_np, with_prescriptions=False)
            ptq2 = measure(swapped, ds, calib, verify_batch=verify_batch)
            after = dict(action=f"swap {cfg['swap']} + heal {cfg['heal_epochs']} epochs",
                         float_acc=healed_acc, cycles=rep["diagnose"]["cycles"],
                         cycles_strict=rep_strict["diagnose"]["cycles"],
                         array_utilization=rep["diagnose"]["array_utilization"],
                         lint=rep["diagnose"]["lint"]["scores"], **ptq2)
            log(f"{name}: after surgery fp32={healed_acc:.4f} int={ptq2['int_acc']:.4f} "
                f"cycles={rep['diagnose']['cycles']:,.0f} strict={rep_strict['diagnose']['cycles']:,.0f}")

        res.add(dict(model=name, label=cfg["label"], customer=True, float_acc=float_acc, macs=int(g.total_macs),
                     params=int(sum(p.numel() for p in model.parameters())),
                     intake=before, intake_strict=before_strict,
                     alternatives=before["alternatives"], ptq=ptq, after=after,
                     minutes=(time.time() - t0) / 60), provenance="measured+simulated")
    if not only:
        walk_ins(ds, calib_np)


def walk_ins(ds, calib_np):
    """Intake sheets for the models trained in E1, so the demo can compare all five side by side."""
    res = Results("e9_customer_intake")
    for name, label in WALK_INS.items():
        if name not in available_baselines() or res.has(model=name):
            continue
        model = load_checkpoint(available_baselines()[name]).eval()
        rep = intake_report(model, "edge-10tops", calib_np)
        strict = intake_report(model, "edge-10tops-strict", calib_np)
        res.add(dict(model=name, label=label, customer=False,
                     macs=int(trace(model).total_macs),
                     params=int(sum(p.numel() for p in model.parameters())),
                     intake=rep, intake_strict=strict,
                     alternatives=rep["alternatives"], ptq=None, after=None), provenance="measured+simulated")
        log(f"{name}: intake only, cycles={rep['diagnose']['cycles']:,.0f} "
            f"strict={strict['diagnose']['cycles']:,.0f} eff={rep['diagnose']['lint']['scores']['efficiency']:.0f}")


if __name__ == "__main__":
    main()
