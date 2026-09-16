"""The guided presentation must use the recorded experiment, not optimistic invented values."""
from pathlib import Path
import copy
import json
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'demo/guided-data.js'

@pytest.fixture
def data():
    return {'results': {p.stem: json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')}}


def call(data, fn, *args):
    assert ADAPTER.is_file(), 'The guided data adapter has not been implemented'
    if shutil.which('node') is None:
        pytest.skip('Node is needed to execute browser data transforms')
    code = "const fs=require('fs');const api=require(process.argv[1]);const x=JSON.parse(fs.readFileSync(0,'utf8'));try{const before=JSON.stringify(x.data);const value=api[x.fn](x.data,...x.args);process.stdout.write(JSON.stringify({value,unchanged:before===JSON.stringify(x.data)}));}catch(e){process.stdout.write(JSON.stringify({error:e.message}));}"
    run = subprocess.run(['node','-e',code,str(ADAPTER)],input=json.dumps({'data':data,'fn':fn,'args':args}),capture_output=True,text=True,timeout=20)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)

@pytest.mark.parametrize('preset,cycles,after',[
    ('edge-10tops-strict',2503878,1708230),('edge-10tops',52246,52054)])
def test_vit_case_uses_the_selected_e9_preset(data,preset,cycles,after):
    out=call(data,'getCase','cust_vit',preset)
    assert 'error' not in out, out
    r=out['value']
    assert r['cycles']==cycles and r['after']['cycles']==after
    assert r['after']['saving']==pytest.approx(1-after/cycles)
    assert r['intImages']==2000 and r['after']['intImages']==2000
    assert r['intAccuracy']==.808 and r['after']['intAccuracy']==.81
    assert r['source']=='e9_customer_intake'
    assert out['unchanged']


def test_strict_case_retains_layernorm_softmax_after_activation_swap(data):
    r=call(data,'getCase','cust_vit','edge-10tops-strict')['value']
    assert [(x['op'],x['count']) for x in r['hostOps']]==[('layernorm',13),('softmax',6),('act:gelu',6)]
    assert [(x['op'],x['count']) for x in r['after']['remainingHostOps']]==[('layernorm',13),('softmax',6)]
    assert r['hostShare']==pytest.approx(2439040/2503878)
    assert r['after']['fullySupported'] is False


def test_supported_preset_is_not_presented_as_an_unsupported_case(data):
    r=call(data,'getCase','cust_vit','edge-10tops')['value']
    assert r['hostOps']==[] and r['hostShare']==0 and r['fullySupported'] is True
    assert r['after']['fullySupported'] is True

@pytest.mark.parametrize('model',['cust_inception','resnet20_relu','resnet20_silu','mnv2_050_relu6'])
def test_absent_interventions_are_not_zero_cost_or_zero_loss(data,model):
    r=call(data,'getCase',model,'edge-10tops-strict')['value']
    assert r['after'] is None
    if model != 'cust_inception':
        assert r['intAccuracy'] is None and r['intImages'] is None

@pytest.mark.parametrize('args',[('missing','edge-10tops'),('cust_vit','not-a-preset')])
def test_unknown_selection_is_an_explicit_error(data,args):
    assert 'error' in call(data,'getCase',*args)

@pytest.mark.parametrize('value',[None,'missing'])
def test_missing_required_case_data_is_an_explicit_error(data,value):
    modified=copy.deepcopy(data)
    r=next(x for x in modified['results']['e9_customer_intake']['records'] if x['model']=='cust_vit')
    if value is None:r['intake_strict']['diagnose']['cycles']=None
    else:del r['intake_strict']['diagnose']['cycles']
    assert 'error' in call(modified,'getCase','cust_vit','edge-10tops-strict')


def test_e15_research_is_separate_from_e9_case(data):
    out=call(data,'getResearch','cust_vit','npu-default');r=out['value']
    assert out['unchanged']
    assert r['source']=='e15_fidelity' and r['images']==10000
    assert r['fakeAccuracy']==.8087 and r['intAccuracy']==.8097
    assert r['mismatch']==pytest.approx(.72655)
    assert r['delta']==pytest.approx(.001)


def test_layernorm_does_not_mix_local_and_output_sample_counts(data):
    r=call(data,'getLayerNorm','npu-default')['value']
    assert r['localImages']==250 and r['outputImages']==10000
    assert r['beforeLocal']==pytest.approx(.24772614933894233)
    assert r['afterLocal']==0
    assert r['beforeMismatch']>r['afterMismatch']>.65
    assert r['source']=='e16_ln_emulation'


def test_missing_research_is_not_replaced_with_another_model(data):
    assert 'error' in call(data,'getResearch','not-a-model','npu-default')
    data['results'].pop('e16_ln_emulation')
    assert call(data,'getLayerNorm','npu-default')['value'] is None


def test_template_has_workbench_views_without_a_tutorial():
    text=(ROOT/'demo/index.template.html').read_text(encoding='utf-8')
    for view in ['analysis','research','experiments','resources']:
        assert f'data-view="{view}"' in text
    assert 'id="model-select"' in text
    assert 'id="analysis-summary"' in text
    assert 'data-guide-step' not in text
    assert '<details' in text


def test_built_assets_are_inline_and_offline():
    text=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert 'NpuDemoData' in text and 'npuloop-workbench-ui' in text
    assert 'fonts.googleapis.com' not in text
    assert '/*__GUIDED_' not in text
    assert '/*__WORKBENCH_' not in text

@pytest.mark.parametrize('host_value',[None,-1,99999999])
def test_invalid_host_cost_cannot_become_an_optimistic_zero(data,host_value):
    record=next(x for x in data['results']['e9_customer_intake']['records'] if x['model']=='cust_vit')
    record['intake_strict']['diagnose']['by_unit']['host']=host_value
    assert 'error' in call(data,'getCase','cust_vit','edge-10tops-strict')


def test_unknown_intervention_has_no_invented_support_verdict(data):
    record=next(x for x in data['results']['e9_customer_intake']['records'] if x['model']=='cust_vit')
    record['after']['action']='an unrecorded future procedure'
    result=call(data,'getCase','cust_vit','edge-10tops-strict')['value']
    assert result['after']['fullySupported'] is None
    assert result['after']['remainingHostOps'] is None
