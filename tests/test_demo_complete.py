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
