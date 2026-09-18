"""Regression coverage for all advertised models, not only residual/depthwise CNNs."""
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from npuloop.graph import trace
from npuloop.npu import PRESETS, estimate
from npuloop.zoo import build_model
from demo.build import layer_descriptors

ROOT = Path(__file__).resolve().parents[1]
TAG = r'<script id="npuloop-data"[^>]*>(.*?)</script>'

def page_data(path):
    return json.loads(re.search(TAG, path.read_text(encoding='utf-8'), re.S).group(1))

CONFIGS = {k: v['config'] for k, v in page_data(ROOT/'docs/index.html')['models'].items()}


def js_result(model, spec):
    code = (ROOT/'demo/cost-model.js').read_text(encoding='utf-8')
    code += '\nprocess.stdout.write(JSON.stringify(estimate('+json.dumps(model)+','+json.dumps(spec)+')));'
    proc = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
    return proc

@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
@pytest.mark.parametrize('name', list(CONFIGS))
@pytest.mark.parametrize('preset', list(PRESETS))
def test_all_demo_models_match_python(name, preset):
    model = build_model(CONFIGS[name]).eval()
    graph = trace(model)
    py = estimate(graph, preset)
    proc = js_result(layer_descriptors(model), PRESETS[preset].to_dict())
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    assert js['totalMacs'] == py.total_macs
    assert js['total'] == pytest.approx(py.total_cycles, rel=1e-12)
    assert js['util'] == pytest.approx(py.array_utilization, rel=1e-12)
    assert js['dram'] == py.dram_bytes
    assert len(js['rows']) == len(py.layers)
    for j, p in zip(js['rows'], py.layers):
        assert (j['name'], j['kind'], j['bound']) == (p.name, p.kind, p.bound)
        assert j['cycles'] == pytest.approx(p.cycles, rel=1e-12), p.name

@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_unknown_browser_operator_fails_instead_of_costing_zero():
    model = {'layers':[{'name':'input','op':'input','inputs':[], 'elements':4,'in_bytes':[]},
                       {'name':'bad','op':'not_supported','inputs':['input'],'elements':4,'in_bytes':[4]}]}
    proc = js_result(model, PRESETS['edge-10tops'].to_dict())
    assert proc.returncode != 0, 'Unknown ops must not silently become zero-cost'
    assert 'unhandled op' in proc.stderr

@pytest.mark.parametrize('filename', ['docs/index.html', 'demo/index.html'])
def test_built_demo_contains_current_result_files(filename):
    data = page_data(ROOT/filename)
    for path in (ROOT/'results').glob('*.json'):
        stored = json.loads(path.read_text())
        if path.stem == 'e2_ptq_grid':
            for record in stored['records']:
                record.pop('sensitivity_full', None)
        assert data['results'].get(path.stem) == stored, path.stem


def test_structure_only_builder_reports_no_weight_evaluation():
    import demo.build as build
    assert callable(getattr(build, 'build_from_configs', None)), 'Need an explicit checkpoint-free architecture build'
    configs = {'tiny':{'config':{'arch':'resnet','depth':8,'width':8,'act':'relu'}}}
    data = build.build_from_configs(configs)
    assert data['model_source'] == 'architecture_config_only'
    assert data['checkpoint_evaluation_performed'] is False
    assert data['models']['tiny']['layers']


def test_structure_only_builder_checks_declared_geometry():
    import demo.build as build
    assert callable(getattr(build, 'build_from_configs', None))
    bad = {'tiny':{'config':{'arch':'resnet','depth':8,'width':8,'act':'relu'}, 'total_macs':1}}
    with pytest.raises(ValueError, match='total_macs'):
        build.build_from_configs(bad)


def test_demo_exposes_new_research_and_correct_measurement_scope():
    src = (ROOT/'demo/index.template.html').read_text(encoding='utf-8')
    for name in ('research-fidelity', 'research-layernorm', 'research-dense', 'research-other'):
        assert f'id="{name}"' in src
    assert '학습은 seed 0 한 번입니다' not in src
    assert '짝지은 비교라 잡음이 없습니다' not in src
    assert 'model_source' in (ROOT/'demo/workbench-ui.js').read_text(encoding='utf-8')

@pytest.mark.skipif(shutil.which('node') is None,reason='node not installed')
def test_preset_controls_preserve_nonactivation_restrictions():
    p = PRESETS['edge-10tops-strict'].to_dict()
    code = "const W = require('./demo/workbench-model.js'); const p = " + json.dumps(p)
    code += ";process.stdout.write(JSON.stringify(W.specWithOverrides(p, {array:64, activation:'fallback'})));"
    run = subprocess.run(['node', '-e', code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    spec = json.loads(run.stdout)
    assert {'softmax', 'layernorm'} <= set(spec['unsupported_ops'])


def test_every_registered_experiment_has_display_columns_and_survives_a_missing_result_file():
    """An experiment registered before its results file lands used to crash the search box.

    filteredCatalog() read D.results[key].records directly, so any query typed while an experiment was
    registered but unmeasured threw. It also silently rendered an empty column list for a key that nobody had
    added to COLUMNS.
    """
    model = (ROOT / 'demo/workbench-model.js').read_text(encoding='utf-8')
    ui = (ROOT / 'demo/workbench-ui.js').read_text(encoding='utf-8')
    keys = re.findall(r"\['(e\d+_[a-z0-9_]+)','E\d+'", model)
    assert len(keys) >= 19, keys
    for key in keys:
        assert re.search(rf"\b{key}:\[\[", ui), f'{key} has no column definition in workbench-ui.js COLUMNS'
    assert 'D.results[e.key].records' not in ui, 'filteredCatalog must go through A.rows() so a missing file is []'


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_catalog_counts_an_unmeasured_experiment_as_zero_rather_than_throwing(tmp_path):
    src = (ROOT / 'demo/workbench-model.js').read_text(encoding='utf-8')
    script = tmp_path / 'catalog.mjs'
    script.write_text(
        # workbench-model.js takes its helpers off the global in a browser; give it the same globals node lacks.
        "globalThis.NpuDemoData={rows:(d,k)=>Array.isArray(d?.results?.[k]?.records)?d.results[k].records:[]};\n"
        "globalThis.estimate=()=>{throw new Error('not needed');};\n"
        "const A=globalThis.NpuDemoData;\n"
        + src
        + "\nconst W=globalThis.NpuWorkbench, D={results:{}};\n"
          "const cat=W.catalog(D);\n"
          "const q='adaround';\n"
          "const hits=cat.filter(e=>(!q||(e.id+' '+e.title+' '+e.description+' '"
          "+JSON.stringify(A.rows(D,e.key))).toLowerCase().includes(q)));\n"
          "process.stdout.write(JSON.stringify({n:cat.length,zero:cat.every(e=>e.count===0),hits:hits.length}));\n",
        encoding='utf-8')
    proc = subprocess.run(['node', str(script)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out['n'] >= 19 and out['zero'] is True
