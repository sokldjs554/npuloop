"""Submission checks, independent of PyTorch and the experiment datasets."""
from pathlib import Path
import importlib.util, hashlib, json, sys
import pytest
ROOT=Path(__file__).resolve().parents[2]

def quality():
    path=ROOT/'tools/submission_quality.py'
    assert path.is_file(), 'submission_quality.py has not been implemented'
    spec=importlib.util.spec_from_file_location('submission_quality_under_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('bad',[
 '어텐션 블록\n12개', '무너지는 것은 Q7 이하다', '단조롭게 올라가',
 'a simulator cannot be repaired into a source of bit-accurate vectors one operator at a time',
 '[저자 성명]', '정확도 이외의 축에서 측정된 바가 없다',
])
def test_known_old_claims_are_rejected(bad):
    assert quality().claim_errors(bad)

def test_unfilled_manuscript_placeholders_are_rejected(tmp_path):
    q=quality();(tmp_path/'paper').mkdir()
    draft=tmp_path/'paper/npuloop_esl.tex'
    draft.write_text('\\author{\\textsc{[Author~Name]}}',encoding='utf-8')
    assert q.placeholder_errors(tmp_path)
    draft.write_text('\\author{%\n\\thanks{no name here}}',encoding='utf-8')
    assert q.placeholder_errors(tmp_path)==[]

def test_shipped_manuscripts_carry_no_placeholder():
    assert quality().placeholder_errors(ROOT)==[]

def test_qualified_claims_are_not_rejected():
    text='노드별 비율의 단순평균은 1% 미만. 풀링 개별 노드 최대 2.697%. 동등성 입증이 아니다. LayerNorm 단독 교체.'
    assert quality().claim_errors(text)==[]

def test_snapshot_checks_bytes_not_just_existence(tmp_path):
    q=quality(); f=tmp_path/'table.tex';f.write_bytes(b'correct')
    refs=[{'path':'table.tex','sha256':hashlib.sha256(b'correct').hexdigest()}]
    assert q.snapshot_errors(tmp_path,refs)==[]
    f.write_bytes(b'corrupt')
    assert q.snapshot_errors(tmp_path,refs)

def test_missing_snapshot_is_not_a_pass(tmp_path):
    assert quality().snapshot_errors(tmp_path,[{'path':'missing','sha256':'0'*64}])

def test_inventory_hashes_real_files(tmp_path):
    f=tmp_path/'weight.pt';f.write_bytes(b'original checkpoint bytes')
    got=quality().artifact_inventory(tmp_path,['weight.pt'])[0]
    assert got['status']=='present'
    assert got['sha256']==hashlib.sha256(f.read_bytes()).hexdigest()
    assert got['historical_identity_verified'] is False

def test_inventory_never_invents_missing_hashes(tmp_path):
    got=quality().artifact_inventory(tmp_path,['absent.pt'])[0]
    assert got['status']=='missing' and got['sha256'] is None

@pytest.mark.parametrize('unsafe',['../outside.pt','/etc/passwd','.git/config'])
def test_inventory_rejects_unsafe_paths(tmp_path,unsafe):
    with pytest.raises(ValueError): quality().artifact_inventory(tmp_path,[unsafe])

def test_inventory_rejects_external_symlink(tmp_path):
    outside=tmp_path.parent/'external-model';outside.write_bytes(b'x')
    (tmp_path/'linked.pt').symlink_to(outside)
    with pytest.raises(ValueError):quality().artifact_inventory(tmp_path,['linked.pt'])

def test_fidelity_summary_counts_nonzero_intervals():
    records=[{'int_vs_fake':{'delta':0.002,'se':0.0001}}, {'int_vs_fake':{'delta':0.,'se':0.001}}]
    s=quality().fidelity_summary(records)
    assert '1/2' in s and '동등성' in s
    assert '전부' not in s and '맞힙니다' not in s

def test_fidelity_summary_all_zero_se():
    s=quality().fidelity_summary([{'int_vs_fake':{'delta':0.,'se':0.}}])
    assert '1/1' in s and '0.00%p' in s

@pytest.mark.parametrize('records',[[],[{'int_vs_fake':{'delta':float('nan'),'se':0.1}}],[{'int_vs_fake':{'delta':0.,'se':-1.}}]])
def test_invalid_fidelity_records_fail(records):
    with pytest.raises(ValueError):quality().fidelity_summary(records)

def test_checked_tables_have_required_structure(tmp_path):
    (tmp_path/'paper').mkdir(); (tmp_path/'paper/npuloop_thesis.tex').write_text('')
    assert quality().paper_errors(tmp_path)

def generator_fixture(tmp_path, first_script='pass\n'):
    q=quality();(tmp_path/'paper').mkdir();(tmp_path/'results').mkdir()
    for name in q.TABLE_ROWS:(tmp_path/'paper'/name).write_text('unchanged table')
    for name in ['make_tables.py','make_thesis.py','make_figs.py']:(tmp_path/'paper'/name).write_text(first_script if name=='make_tables.py' else 'pass\n')
    return q

def test_regeneration_does_not_copy_secrets_or_checkpoints(tmp_path):
    q=generator_fixture(tmp_path,"from pathlib import Path\nassert not Path('.env').exists()\nassert not Path('runs').exists()\n")
    (tmp_path/'.env').write_text('must stay outside isolated generation copy')
    errors,runs=q.regeneration_errors(tmp_path)
    assert errors==[]
    assert len(runs)==3

def test_regeneration_detects_changed_table_without_editing_original(tmp_path):
    q=generator_fixture(tmp_path,"from pathlib import Path\nPath('paper/tab1_rows_ko.tex').write_text('changed')\n")
    errors,runs=q.regeneration_errors(tmp_path)
    assert 'Stale table: paper/tab1_rows_ko.tex' in errors
    assert (tmp_path/'paper/tab1_rows_ko.tex').read_text()=='unchanged table'

def test_failed_generator_is_not_reported_as_fresh(tmp_path):
    q=generator_fixture(tmp_path,'raise SystemExit(7)\n')
    errors,runs=q.regeneration_errors(tmp_path)
    assert errors and runs[0]['exit_code']==7 and len(runs)==1

def test_missing_regeneration_inputs_are_blocking(tmp_path):
    errors,runs=quality().regeneration_errors(tmp_path)
    assert errors and runs==[]
