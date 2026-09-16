"""Build the self-contained demo page: results/*.json + per-model layer descriptors -> demo/index.html.

The page ports the analytical cost model to JavaScript, so it needs, per model, the per-layer GEMM shapes
(m, k, n), MACs, element counts and activation kinds — exported here from the traced StaticGraph.
"""
from __future__ import annotations
import argparse, json, os, sys, glob, time, shutil
from pathlib import Path
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
        d = dict(name=n.name, op=n.op, inputs=n.inputs, elements=n.n_elements, macs=n.macs, in_bytes=[out_bytes[s] for s in n.inputs])
        if n.op == "conv":
            cout, cin_g, kh, kw = n.weight.shape; _, ho, wo = n.out_shape
            d.update(m=ho * wo, k=cin_g * kh * kw, n=cout, macs=n.macs, groups=n.attrs["groups"], depthwise=bool(n.attrs["depthwise"]),
                     kh=kh, kw=kw, weight_bytes=int(n.weight.size) + 4 * cout)
        elif n.op == "linear":
            d.update(m=int(n.attrs.get("tokens", 1)), k=int(n.weight.shape[1]), n=int(n.weight.shape[0]), macs=n.macs, groups=1, depthwise=False, kh=1, kw=1,
                     weight_bytes=int(n.weight.size) + 4 * int(n.weight.shape[0]))
        elif n.op == "matmul":
            d.update(m=n.attrs["m"], k=n.attrs["k"], n=n.attrs["n"], batch=n.attrs["batch"])
        elif n.op in ("act", "mul"):
            d.update(kind=n.attrs.get("kind", ""))
        layers.append(d)
    return dict(layers=layers, total_macs=g.total_macs, total_params=g.total_params)


def build_from_configs(configs):
    """Trace architecture-only graphs for analytical costs. Never evaluate random weights.

    Stored accuracies still come from results/*.json; this path is not a checkpoint
    reproduction. Expected MAC/parameter counts detect stale or mismatched configs.
    """
    import torch
    from npuloop.zoo import build_model
    if not configs:
        raise ValueError("No architecture configurations")
    data = {"models": {}, "model_source": "architecture_config_only",
            "checkpoint_evaluation_performed": False}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for name, record in configs.items():
            model = build_model(record["config"]).eval()
            desc = layer_descriptors(model)
            for key in ("total_macs", "total_params"):
                if key in record and desc[key] != record[key]:
                    raise ValueError(f"{name}: {key} differs from recorded architecture")
            data["models"][name] = dict(config=model.config, **desc)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-configs", type=Path,
                        help="Explicit architecture-only build; no trained-weight evaluation")
    args = parser.parse_args()
    data = dict(built=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                presets={k: v.to_dict() for k, v in PRESETS.items()}, models={}, results={})
    if args.from_configs:
        data.update(build_from_configs(json.loads(args.from_configs.read_text(encoding="utf-8"))))
    else:
        names = available_baselines()
        if not names:
            sys.exit(f"no trained checkpoints under {os.environ.get('NPULOOP_RUNS', os.path.join(ROOT, 'runs'))} — "
                     "use --from-configs demo/model_configs.json for an explicit architecture-only build; "
                     "refusing to silently substitute random weights")
        data["model_source"] = "checkpoint_structure"
        data["checkpoint_evaluation_performed"] = False
        for name in names:
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
    # CSS and controller are inline so file:// works without a server or external assets.
    for marker, filename in [("/*__WORKBENCH_CSS__*/", "workbench.css"),
                             ("/*__COST_MODEL__*/", "cost-model.js"),
                             ("/*__GUIDED_DATA__*/", "guided-data.js"),
                             ("/*__WORKBENCH_MODEL__*/", "workbench-model.js"),
                             ("/*__WORKBENCH_UI__*/", "workbench-ui.js")]:
        if src.count(marker) != 1:
            raise ValueError(f"Expected one asset marker: {marker}")
        src = src.replace(marker, (Path(ROOT) / "demo" / filename).read_text(encoding="utf-8"))
    payload = json.dumps(data, separators=(",", ":"), default=float).replace("<", "\\u003c")
    html = src.replace("/*__DATA__*/null", payload)
    out = os.path.join(ROOT, "demo", "index.html")
    open(out, "w", encoding="utf-8").write(html)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8").write(html)
    # Same relative evidence links work in both the local demo and GitHub Pages' docs root.
    for folder in ("docs", "demo"):
        dest = Path(ROOT) / folder / "reference"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("USAGE.md", "VALIDATION_SCOPE.md", "EXPERIMENTS.md"):
            shutil.copyfile(Path(ROOT) / "docs" / name, dest / name)
        shutil.copyfile(Path(ROOT) / "results" / "e9_customer_intake.json", dest / "e9_customer_intake.json")
    print(f"wrote {out} ({len(html)/1024:.0f} KB), models={list(data['models'])}, results={list(data['results'])}")


if __name__ == "__main__":
    main()
