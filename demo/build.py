"""Build the self-contained demo page: results/*.json + per-model layer descriptors -> demo/index.html.

The page ports the analytical cost model to JavaScript, so it needs, per model, the per-layer GEMM shapes
(m, k, n), MACs, element counts and activation kinds — exported here from the traced StaticGraph.
"""
from __future__ import annotations
import json, os, sys, glob, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import available_baselines, load_model, RESULTS
from npuloop.graph import trace
from npuloop.npu import PRESETS


def layer_descriptors(model):
    g = trace(model)
    out_bytes = {n.name: n.n_elements for n in g.nodes}
    layers = []
    for n in g.nodes:
        d = dict(name=n.name, op=n.op, inputs=n.inputs, elements=n.n_elements, in_bytes=[out_bytes[s] for s in n.inputs])
        if n.op == "conv":
            cout, cin_g, kh, kw = n.weight.shape; _, ho, wo = n.out_shape
            d.update(m=ho * wo, k=cin_g * kh * kw, n=cout, macs=n.macs, groups=n.attrs["groups"], depthwise=bool(n.attrs["depthwise"]),
                     kh=kh, kw=kw, weight_bytes=int(n.weight.size) + 4 * cout)
        elif n.op == "linear":
            d.update(m=1, k=int(n.weight.shape[1]), n=int(n.weight.shape[0]), macs=n.macs, groups=1, depthwise=False, kh=1, kw=1,
                     weight_bytes=int(n.weight.size) + 4 * int(n.weight.shape[0]))
        elif n.op == "act":
            d.update(kind=n.attrs["kind"])
        layers.append(d)
    return dict(layers=layers, total_macs=g.total_macs, total_params=g.total_params)


def main():
    data = dict(built=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), presets={k: v.to_dict() for k, v in PRESETS.items()}, models={}, results={})
    for name in available_baselines():
        m = load_model(name)
        data["models"][name] = dict(config=m.config, **layer_descriptors(m))
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        key = os.path.basename(path)[:-5]
        d = json.load(open(path))
        # strip the bulky per-layer agreement rows down to what the page plots
        if key == "e2_ptq_grid":
            for r in d["records"]:
                r.pop("sensitivity_full", None)
        data["results"][key] = d
    src = open(os.path.join(ROOT, "demo", "index.template.html"), encoding="utf-8").read()
    payload = json.dumps(data, separators=(",", ":"), default=float)
    html = src.replace("/*__DATA__*/null", payload)
    out = os.path.join(ROOT, "demo", "index.html")
    open(out, "w", encoding="utf-8").write(html)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8").write(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB), models={list(data['models'])}, results={list(data['results'])}")


if __name__ == "__main__":
    main()
