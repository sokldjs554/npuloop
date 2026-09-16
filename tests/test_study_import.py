"""Imported learning results must retain the meaning of measured evidence."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report():
    return json.loads((ROOT / 'verification/model-study/run/study.json').read_text())


def validate(report):
    if not shutil.which('node'):
        pytest.skip('Node is needed for the browser report validator')
    script = """const fs=require('fs');const api=require('./demo/study-ui.js');
    try{api.validateReport(JSON.parse(fs.readFileSync(0,'utf8')));console.log('accepted');}
    catch(e){console.log(e.message);process.exitCode=1;}"""
    return subprocess.run(['node', '-e', script], cwd=ROOT, input=json.dumps(report),
                          capture_output=True, text=True, timeout=10)


def test_real_training_report_can_be_imported(report):
    assert report['training']['weights_changed']
    assert validate(report).returncode == 0


@pytest.mark.parametrize('stage', ['baseline_fp32', 'changed_fp32_before_training',
                                  'fine_tuned_fp32', 'fake_quant', 'integer'])
def test_simulated_observation_is_not_a_measured_result(report, stage):
    report['observed'][stage]['provenance'] = 'simulated'
    assert validate(report).returncode != 0


def test_failed_execution_cannot_be_imported_as_complete(report):
    report['status'] = 'failed'
    assert validate(report).returncode != 0


def test_different_evaluation_images_cannot_be_compared(report):
    report['observed']['integer']['subset_sha256'] = 'a' * 64
    assert validate(report).returncode != 0


def test_mismatched_accuracy_denominator_is_rejected(report):
    report['observed']['integer']['correct'] -= 1
    assert validate(report).returncode != 0


def test_zero_epoch_control_is_distinct_from_missing_training(report):
    report['training']['epochs'] = []
    assert validate(report).returncode != 0
    report['config']['epochs'] = 0
    report['training']['actual_steps'] = 0
    assert validate(report).returncode == 0


def test_missing_cost_provenance_cannot_imply_chip_measurement(report):
    report['simulated_costs']['provenance'] = 'measured'
    assert validate(report).returncode != 0
