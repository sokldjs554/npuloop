"""The demo page ports npuloop/npu/cost.py to JavaScript; this test runs that JS (node) on exported layer
descriptors and checks per-layer cycles against the Python model for every preset."""
import json, os, re, shutil, subprocess, sys, tempfile
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "demo", "index.template.html")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_cost_model_matches_python(small_resnet, small_mobilenet):
    sys.path.insert(0, os.path.join(ROOT, "demo"))
    from build import layer_descriptors
    from npuloop.graph import trace
    from npuloop.npu import estimate, PRESETS
    src = open(TEMPLATE, encoding="utf-8").read()
    js = src[src.index("function gemmCycles"):src.index("// ---------- header meta")]
    models = {"r": layer_descriptors(small_resnet), "m": layer_descriptors(small_mobilenet)}
    presets = {k: v.to_dict() for k, v in PRESETS.items()}
    script = js + f"""
const M = {json.dumps(models)}; const P = {json.dumps(presets)};
const out = {{}};
for (const [mn, model] of Object.entries(M)) for (const [pn, spec] of Object.entries(P)) {{
  const r = estimate(model, spec); out[mn + '|' + pn] = {{ total: r.total, util: r.util, rows: r.rows.map(x => [x.name, x.cycles, x.bound]) }};
}}
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(script); path = f.name
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    js_out = json.loads(res.stdout)
    for mn, model in (("r", small_resnet), ("m", small_mobilenet)):
        g = trace(model)
        for pn in PRESETS:
            py = estimate(g, pn)
            j = js_out[f"{mn}|{pn}"]
            assert abs(j["total"] - py.total_cycles) < 1e-6 * max(1, py.total_cycles), (mn, pn, j["total"], py.total_cycles)
            assert abs(j["util"] - py.array_utilization) < 1e-9
            for (name, cyc, bound), l in zip(j["rows"], py.layers):
                assert name == l.name and abs(cyc - l.cycles) < 1e-6 and bound == l.bound, (mn, pn, name, cyc, l.cycles, bound, l.bound)
