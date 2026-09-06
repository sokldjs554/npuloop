"""E8: validate the analytical compute-cycle model against SCALE-Sim (cycle-accurate systolic simulator, WS dataflow).

For every conv/linear layer of a model, SCALE-Sim's 'Total Cycles' (with CALC bandwidth, i.e. no memory stalls)
is compared with npuloop's gemm_cycles() for a single-core array of the same size.
"""
import configparser, csv, os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from npuloop.graph import trace
from npuloop.npu import gemm_cycles, NPUSpec

ARRAYS = [(32, 32), (64, 64), (16, 16)]
MODELS = os.environ.get("NPULOOP_MODELS", "resnet20_relu,mnv2_050_relu6").split(",")


def write_config(path, rows, cols, run_name):
    cfg = configparser.ConfigParser()
    cfg["general"] = {"run_name": run_name}
    cfg["architecture_presets"] = {"ArrayHeight": str(rows), "ArrayWidth": str(cols), "IfmapSramSzkB": "2048", "FilterSramSzkB": "2048",
                                   "OfmapSramSzkB": "2048", "IfmapOffset": "0", "FilterOffset": "10000000", "OfmapOffset": "20000000",
                                   "Dataflow": "ws", "Bandwidth": "10", "MemoryBanks": "1", "ReadRequestBuffer": "32", "WriteRequestBuffer": "32"}
    cfg["layout"] = {"IfmapCustomLayout": "False", "FilterCustomLayout": "False", "IfmapSRAMBankBandwidth": "10", "IfmapSRAMBankNum": "1",
                     "IfmapSRAMBankPort": "1", "FilterSRAMBankBandwidth": "10", "FilterSRAMBankNum": "1", "FilterSRAMBankPort": "1"}
    cfg["sparsity"] = {"SparsitySupport": "false", "SparseRep": "csr", "OptimizedMapping": "false", "BlockSize": "1", "RandomNumberGeneratorSeed": "40"}
    cfg["run_presets"] = {"InterfaceBandwidth": "CALC", "UseRamulatorTrace": "False"}
    with open(path, "w") as f:
        cfg.write(f)


def layers_of(model):
    g = trace(model)
    out = []
    by = g.by_name
    for n in g.compute_nodes():
        if n.op == "conv":
            if n.attrs["groups"] != 1:
                continue                      # SCALE-Sim's conv format has no groups; depthwise is skipped
            cout, cin, kh, kw = n.weight.shape
            _, hin, win = by[n.inputs[0]].out_shape
            _, ho, wo = n.out_shape
            s = n.attrs["stride"][0]
            # SCALE-Sim has no padding parameter: give it the padded ifmap so the ofmap matches ours
            ph, pw = n.attrs["padding"]
            out.append(dict(name=n.name, ifh=hin + 2 * ph, ifw=win + 2 * pw, kh=kh, kw=kw, cin=cin, cout=cout, stride=s, m=ho * wo, k=cin * kh * kw, n=cout))
        else:
            cout, k = n.weight.shape
            out.append(dict(name=n.name, ifh=1, ifw=1, kh=1, kw=1, cin=k, cout=cout, stride=1, m=1, k=k, n=cout))
    return out


def run_scalesim(layers, rows, cols, workdir):
    from scalesim.scale_sim import scalesim
    run_name = f"r{rows}c{cols}"
    cfg = os.path.join(workdir, f"{run_name}.cfg"); topo = os.path.join(workdir, f"{run_name}.csv")
    write_config(cfg, rows, cols, run_name)
    with open(topo, "w") as f:
        f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n")
        for l in layers:
            f.write(f"{l['name']}, {l['ifh']}, {l['ifw']}, {l['kh']}, {l['kw']}, {l['cin']}, {l['cout']}, {l['stride']},\n")
    sim = scalesim(save_disk_space=True, verbose=False, config=cfg, topology=topo, input_type_gemm=False)
    sim.run_scale(top_path=workdir)
    rep = os.path.join(workdir, run_name, "COMPUTE_REPORT.csv")
    rows_ = list(csv.DictReader(open(rep)))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows_]


def main():
    res = Results("e8_scalesim", meta=dict(scalesim_version="3.0.0", dataflow="ws", note="Total Cycles with CALC bandwidth (compute only) vs npuloop gemm_cycles single core"))
    for name in MODELS:
        if name not in available_baselines():
            continue
        m = load_model(name)
        layers = layers_of(m)
        for rows, cols in ARRAYS:
            if res.has(model=name, rows=rows, cols=cols):
                continue
            t = time.time()
            with tempfile.TemporaryDirectory() as wd:
                rep = run_scalesim(layers, rows, cols, wd)
            spec = NPUSpec("val", pe_rows=rows, pe_cols=cols, cores=1, fill_drain=True)
            spec_nofd = NPUSpec("val", pe_rows=rows, pe_cols=cols, cores=1, fill_drain=False)
            per = []
            for l, r in zip(layers, rep):
                sc = float(r["Total Cycles"]); ours = gemm_cycles(l["m"], l["k"], l["n"], spec)[0]; ours_nofd = gemm_cycles(l["m"], l["k"], l["n"], spec_nofd)[0]
                per.append(dict(layer=l["name"], m=l["m"], k=l["k"], n=l["n"], scalesim_cycles=sc, scalesim_util=float(r["Overall Util %"]),
                                ours_cycles=ours, ours_nofd=ours_nofd, ratio=ours / sc, ratio_nofd=ours_nofd / sc))
            tot_sc = sum(p["scalesim_cycles"] for p in per); tot_ours = sum(p["ours_cycles"] for p in per); tot_nofd = sum(p["ours_nofd"] for p in per)
            rec = dict(model=name, rows=rows, cols=cols, n_layers=len(per), total_scalesim=tot_sc, total_ours=tot_ours, total_ours_nofd=tot_nofd,
                       ratio=tot_ours / tot_sc, ratio_nofd=tot_nofd / tot_sc, max_layer_abs_err=max(abs(p["ratio"] - 1) for p in per), layers=per, seconds=time.time() - t)
            res.add(rec, provenance="simulated")
            log(f"{name} {rows}x{cols}: scalesim={tot_sc:,.0f} ours={tot_ours:,.0f} (ratio {tot_ours/tot_sc:.3f}; no fill/drain {tot_nofd/tot_sc:.3f}) worst layer err {rec['max_layer_abs_err']*100:.1f}% ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
