"""E6: MAC-array-aligned structured pruning vs FLOP-driven pruning, judged by the NPU cost model.

For each strategy x target: prune -> fine-tune FT_EPOCHS -> PTQ (npu-default) -> report accuracy, MACs,
cycles on several virtual NPUs, and array utilization. FLOPs and cycles disagree; that is the point.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.npu import estimate
from npuloop.prune import prune, prune_cost_greedy, find_groups
from npuloop.quant import prepare, calibrate, evaluate, PRESET_SCHEMES
from npuloop.zoo import fit

FT_EPOCHS = int(os.environ.get("NPULOOP_FT_EPOCHS", "3"))
SPECS = ["tiny-1tops", "edge-10tops", "pcie-80tops"]
MODELS = os.environ.get("NPULOOP_MODELS", "resnet20_relu,mnv2_050_relu6").split(",")
CONFIGS = [  # (strategy, ratio/target, align)
    ("uniform", 0.75, 0), ("aligned", 0.75, 16), ("aligned", 0.75, 32),
    ("uniform", 0.5, 0), ("aligned", 0.5, 16), ("aligned", 0.5, 32),
    ("uniform", 0.25, 0), ("aligned", 0.25, 16),
    ("cost-greedy", 0.85, 8), ("cost-greedy", 0.7, 8), ("cost-greedy", 0.55, 8),   # 8-channel steps: finer K-tile staircase
]


def costs(model):
    g = trace(model)
    out = dict(macs=g.total_macs, params=g.total_params)
    for s in SPECS:
        r = estimate(g, s)
        out[s] = dict(cycles=r.total_cycles, util=r.array_utilization, latency_ms=r.latency_ms)
    return out


def main():
    torch.set_num_threads(int(os.environ.get("NPULOOP_THREADS", "4")))
    ds = dataset()
    res = Results("e6_pruning", meta=dict(ft_epochs=FT_EPOCHS, specs=SPECS, greedy_spec="edge-10tops"))
    calib = calib_batches(ds, 512, seed=0)
    for name in MODELS:
        if name not in available_baselines():
            continue
        m = load_model(name)
        if not res.has(model=name, strategy="none"):
            base_costs = costs(m)
            qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
            res.add(dict(model=name, strategy="none", ratio=1.0, align=0, keep=[g.channels for g in find_groups(m)],
                         float_acc=evaluate(m, ds), ft_acc=evaluate(m, ds), int8_acc=evaluate(qm, ds), **base_costs), provenance="measured+simulated")
        configs = CONFIGS if name.startswith("resnet") else [c for c in CONFIGS if c[1] in (0.5, 0.7) or c == ("uniform", 0.25, 0)]
        for strategy, ratio, align in configs:
            if res.has(model=name, strategy=strategy, ratio=ratio, align=align):
                continue
            t = time.time()
            if strategy == "cost-greedy":
                pm, groups, hist = prune_cost_greedy(m, "edge-10tops", target_ratio=ratio, align=align, min_channels=align)
            else:
                pm, groups = prune(m, ratio=ratio, strategy=strategy, align=align); hist = None
            acc_pruned = evaluate(pm, ds)
            lg = fit(pm, ds, epochs=FT_EPOCHS, lr=0.02, seed=0, warmup_pct=0.2); pm.eval()
            ft_acc = evaluate(pm, ds)
            qm = prepare(pm, PRESET_SCHEMES["npu-default"]); calibrate(qm, calib)
            rec = dict(model=name, strategy=strategy, ratio=ratio, align=align, keep=[g.channels for g in groups],
                       acc_after_prune=acc_pruned, ft_acc=ft_acc, int8_acc=evaluate(qm, ds), ft_epochs=FT_EPOCHS,
                       greedy_steps=(len(hist) - 1) if hist else None, target_reached=(hist[-1].get("target_reached") if hist else None),
                       achieved_ratio=(hist[-1].get("achieved_ratio") if hist else None), minutes=(time.time() - t) / 60, **costs(pm))
            res.add(rec, provenance="measured+simulated")
            log(f"{name} {strategy} r={ratio} a={align}: keep={rec['keep']} MACs={rec['macs']/1e6:.1f}M edge-cycles={rec['edge-10tops']['cycles']:.0f} ft_acc={ft_acc:.4f} int8={rec['int8_acc']:.4f}")


if __name__ == "__main__":
    main()
