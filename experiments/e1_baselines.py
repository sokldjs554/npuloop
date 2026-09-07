"""E1: collect FP32 baselines + static analysis (MACs, cost model on every preset, lint scores)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.npu import estimate, PRESETS
from npuloop.lint import lint
from npuloop.zoo import count_params
from npuloop.quant import evaluate


def main():
    ds = dataset()
    res = Results("e1_baselines", meta=dict(dataset="CIFAR-10", epochs=30, schedule="OneCycle SGD nesterov, lr 0.1, wd 5e-4, bs 128, seed 0"))
    calib = calib_batches(ds, 256, seed=0)[0].numpy()
    for name, ckpt in available_baselines().items():
        if res.has(model=name):
            continue
        log_ = json.load(open(os.path.join(os.path.dirname(ckpt), "log.json")))
        m = load_model(name)
        g = trace(m)
        rec = dict(model=name, config=m.config, params=count_params(m), macs=g.total_macs,
                   float_acc=evaluate(m, ds),   # re-evaluate the loaded checkpoint so E1 matches E2/E4/E6 exactly
                   final_test_acc=log_["final_test_acc"], best_acc=log_["best_test_acc"], train_minutes=log_["train_minutes"],
                   epochs=[dict(epoch=e["epoch"], test_acc=e["test_acc"], train_loss=e["train_loss"]) for e in log_["epochs"]],
                   cost={}, lint={})
        for spec in PRESETS:
            r = estimate(g, spec)
            rec["cost"][spec] = dict(total_cycles=r.total_cycles, latency_ms=r.latency_ms, array_utilization=r.array_utilization,
                                     dram_bytes=r.dram_bytes, breakdown=r.breakdown(),
                                     layers=[dict(name=l.name, kind=l.kind, macs=l.macs, cycles=l.cycles, bound=l.bound, util=l.array_util, m=l.m, k=l.k, n=l.n, split=l.split, dram_bytes=l.dram_bytes) for l in r.layers if l.op not in ("input", "output", "flatten")])
            lr = lint(g, spec, calib)
            rec["lint"][spec] = dict(scores=lr.scores, counts=lr.counts(), findings=[f.to_dict() for f in lr.findings],
                                     alignment_util_weighted=lr.stats["alignment_util_weighted"])
        rec["weight_range_ratio"] = {n.name: float(v["ratio"]) for n, v in zip(g.compute_nodes(), [lr.stats["weights"][n.name] for n in g.compute_nodes()])}
        res.add(rec, provenance="measured+simulated")
        log(f"{name}: acc={rec['float_acc']:.4f} MACs={g.total_macs/1e6:.1f}M " + " ".join(f"{s}:{rec['cost'][s]['total_cycles']:.0f}cyc" for s in PRESETS))


if __name__ == "__main__":
    main()
