"""Workbench contract: numerical provenance, non-fabricated comparisons, and application UI."""
import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def payload():
    html=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    return json.loads(re.search(r'<script id="npuloop-data"[^>]*>(.*?)</script>', html,re.S).group(1))


def call(data, function, *args):
    module=ROOT/'demo/workbench-model.js'
    assert module.is_file(), 'Workbench model is not implemented'
    if not shutil.which('node'):pytest.skip('node not installed')
    script="""const fs=require('fs');const api=require(process.argv[1]);const x=JSON.parse(fs.readFileSync(0,'utf8'));const before=JSON.stringify(x.data);try{const value=api[x.fn](x.data,...x.args);console.log(JSON.stringify({value,unchanged:before===JSON.stringify(x.data)}));}catch(e){console.log(JSON.stringify({error:e.message}));}"""
    run=subprocess.run(['node','-e',script,str(module)],input=json.dumps({'data':data,'fn':function,'args':args}),capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr
    return json.loads(run.stdout)

@pytest.mark.parametrize('model',['cust_vit','cust_inception','resnet20_relu','resnet20_silu','mnv2_050_relu6'])
@pytest.mark.parametrize('preset',['tiny-1tops','edge-10tops','edge-10tops-strict','pcie-80tops'])
def test_analysis_is_complete_and_nonmutating(payload,model,preset):
    result=call(payload,'analysis',model,preset,{})
    assert 'error' not in result,result
    r=result['value']
    assert result['unchanged']
    assert r['estimate']['total']>0
    assert sum(g['cycles'] for g in r['units'])==pytest.approx(r['estimate']['total'])
    assert sum(g['count'] for g in r['operators'])==len(r['estimate']['rows'])
    assert r['hostCount']==sum(x['count'] for x in r['operators'] if x['host'])
    assert r['provenance']=='simulated'
    assert r['custom'] is False


def test_default_vit_matches_recorded_cost(payload):
    r=call(payload,'analysis','cust_vit','edge-10tops-strict',{})['value']
    assert r['hostCount']==25
    assert r['estimate']['total']==2503878
    assert r['hostCycles']==2439040
    assert r['supported'] is False

@pytest.mark.parametrize('value',[0,-1,256.5,'64',None])
def test_invalid_custom_array_is_rejected(payload,value):
    assert 'error' in call(payload,'analysis','cust_vit','edge-10tops',{'array':value})


def test_custom_settings_preserve_layernorm_and_softmax_restrictions(payload):
    r=call(payload,'analysis','cust_vit','edge-10tops-strict',{'activation':'lut','array':32})['value']
    assert r['hostCount']==19
    assert {'softmax','layernorm'}<=set(r['spec']['unsupported_ops'])
    assert r['custom'] is True
    assert r['recordedComparison'] is None

@pytest.mark.parametrize('preset,cycles',[('edge-10tops-strict',1708230),('edge-10tops',52054)])
def test_comparison_uses_e9_and_preserves_sample_size(payload,preset,cycles):
    r=call(payload,'analysis','cust_vit',preset,{})['value']['recordedComparison']
    assert r['source']=='e9_customer_intake'
    assert r['after']['cycles']==cycles
    assert r['intImages']==r['after']['intImages']==2000
    assert r['intAccuracy']==.808 and r['after']['intAccuracy']==.81

@pytest.mark.parametrize('model',['resnet20_relu','resnet20_silu','cust_inception','mnv2_050_relu6'])
def test_missing_after_record_remains_null(payload,model):
    r=call(payload,'analysis',model,'edge-10tops-strict',{})['value']
    assert r['recordedComparison'] is None


def test_experiments_preserve_all_records(payload):
    result=call(payload,'catalog')
    assert len(result['value'])==17
    assert result['unchanged']
    for e in result['value']:
        assert e['count']==len(payload['results'][e['key']]['records'])
        assert '?' not in e['title']


def test_research_does_not_reuse_e9_accuracy(payload):
    r=call(payload,'research','cust_vit','npu-default')['value']
    assert r['summary']['images']==10000
    assert r['summary']['intAccuracy']==.8097
    assert r['summary']['mismatch']==.72655
    assert r['localImages']==250
    assert len(r['layers'])>0


def test_missing_research_is_an_error(payload):
    assert 'error' in call(payload,'research','missing','npu-default')


def test_html_is_an_application_not_a_question_landing():
    src=(ROOT/'demo/index.template.html').read_text(encoding='utf-8')
    assert '<h1 id="page-title">모델 실험·경량화</h1>' in src
    for banned in ['무엇을 고쳐야','할까요','예제 모델 분석 보기','대표 모델 분석','data-guide-step','guide-hero']:
        assert banned not in src
    for view in ['studies','analysis','research','experiments','resources']:
        assert f'data-view="{view}"' in src
    assert 'id="analysis-summary"' in src
    assert 'id="operator-search"' in src


def test_public_ui_copy_has_no_question_or_lecture_headers():
    for name in ['index.template.html','workbench-ui.js','study-ui.js']:
        src=(ROOT/'demo'/name).read_text(encoding='utf-8')
        for banned in ['무엇을 고쳐야','할까요','따라가','눌러보','살펴보','이해할 수','고객이 체크포인트','처방']:
            assert banned not in src,(name,banned)

@pytest.mark.parametrize('preset',['tiny-1tops','edge-10tops','edge-10tops-strict','pcie-80tops'])
def test_applying_unchanged_settings_keeps_original_preset(payload,preset):
    s=payload['presets'][preset]
    changes=dict(array=s['pe_rows'],cores=s['cores'],dram=s['dram_gbps'],depthwise=s['dw_lanes']>0,
                 activation='lut' if 'gelu' in s['lut_acts'] else 'fallback')
    r=call(payload,'analysis','cust_vit',preset,changes)['value']
    assert r['spec']==s
    assert r['custom'] is False


def test_toggling_activation_keeps_unrelated_lut_and_unsupported_entries(payload):
    s=payload['presets']['edge-10tops-strict']
    r=call(payload,'analysis','cust_vit','edge-10tops-strict',{'activation':'lut'})['value']
    unrelated={'sigmoid','tanh','lrelu','hsigmoid','layernorm','softmax'}
    assert set(s['unsupported_ops'])&unrelated == set(r['spec']['unsupported_ops'])&unrelated
    s=payload['presets']['edge-10tops']
    r=call(payload,'analysis','cust_vit','edge-10tops',{'activation':'fallback'})['value']
    assert (set(s['lut_acts'])-{'silu','gelu','hswish'}) <= set(r['spec']['lut_acts'])

@pytest.mark.parametrize('index,expected',[(0,'silu-ptq'),(3,'swap-relu-heal3'),(6,'QAT · 2 epoch'),(8,'cle'),(11,'cle+qat')])
def test_surgery_table_uses_recorded_condition_fields(payload,index,expected):
    result=call(payload,'recordValue','e4_surgery',index,'condition')
    assert 'error' not in result,result
    assert result['value']==expected and result['unchanged']


def test_surgery_table_distinguishes_original_and_modified_fp32(payload):
    assert call(payload,'recordValue','e4_surgery',3,'original_fp32')['value']==.9034
    assert call(payload,'recordValue','e4_surgery',3,'modified_fp32')['value']==.8974
    assert call(payload,'recordValue','e4_surgery',0,'modified_fp32')['value'] is None
    assert call(payload,'recordValue','e4_surgery',6,'simulated_after')['value']==.8936

@pytest.mark.parametrize('index',range(4))
def test_imagenette_fp32_field_respects_record_schema(payload,index):
    assert call(payload,'recordValue','e12_imagenette',index,'fp32')['value']==pytest.approx(.8397452229299363)
