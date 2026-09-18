"""Recorded training/compression studies: sample scopes, metrics, and provenance."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "demo" / "study-data.js"
PRESETS = ["tiny-1tops", "edge-10tops", "edge-10tops-strict", "pcie-80tops"]


@pytest.fixture(scope="module")
def payload():
    return {
        "presets": {key: {} for key in PRESETS},
        "results": {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("*.json")},
    }


def call(function, *args):
    assert MODULE.is_file(), "Recorded study evidence API is not implemented"
    if not shutil.which("node"):
        pytest.skip("node not installed")
    script = """
const fs=require('fs'), api=require(process.argv[1]);
const x=JSON.parse(fs.readFileSync(0,'utf8')), before=JSON.stringify(x.args);
try { const value=api[x.fn](...x.args);
  function finite(v) { if(typeof v==='number' && !Number.isFinite(v)) throw new Error('Non-finite evidence');
    if(v && typeof v==='object') Object.values(v).forEach(finite); }
  finite(value);
  process.stdout.write(JSON.stringify({value, unchanged:before===JSON.stringify(x.args)}));
} catch(e) { process.stdout.write(JSON.stringify({error:e.message})); }
"""
    result = subprocess.run(["node", "-e", script, str(MODULE)],
                            input=json.dumps({"fn": function, "args": args}),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def study(payload, study_id, preset="edge-10tops-strict"):
    result = call("study", payload, study_id, preset)
    assert "error" not in result, result
    assert result["unchanged"]
    return result["value"]


def variant(result, variant_id):
    return next(row for row in result["variants"] if row["id"] == variant_id)


def test_catalog_exposes_available_studies_and_preserves_data(payload):
    result = call("catalog", payload)
    assert result["unchanged"]
    ids = {row["id"] for row in result["value"]}
    assert {"e4-activation", "e1-training", "e6-resnet20_relu", "e6-cust_vit",
            "e9-vit", "e2-mnv2_050_relu6", "e12-imagenette", "e17-espcn_x2", "e17-espcn_x2_deep"} <= ids
    assert call("catalog", {"results": {}})["value"] == []


def test_activation_recovery_uses_modified_accuracy_and_real_validation_log(payload):
    result = study(payload, "e4-activation")
    row = variant(result, "swap-relu-heal3")
    assert result["baselineId"] == "silu-ptq"
    assert row["fp32Accuracy"] == .8974
    assert row["fakeAccuracy"] == .8967
    assert row["intAccuracy"] == .8981
    assert row["cycles"] == 34417
    assert row["epochs"] == 3
    assert row["fp32Images"] == row["intImages"] == 10000
    assert len(row["training"]) == 3
    assert row["training"][0] == {"epoch": 1, "trainLoss": .21462415871966598,
                                   "trainAccuracy": .9234330484330484, "valAccuracy": .8704}
    assert row["training"][-1]["valAccuracy"] == .8976
    assert any(p["sourceKey"] == "e4_surgery" and p["recordIndex"] == 3 for p in row["sourceRecords"])


def test_qat_does_not_relabel_original_accuracy_as_post_qat_fp32(payload):
    row = variant(study(payload, "e4-activation"), "silu-qat")
    assert row["fp32Accuracy"] is None
    assert row["fakeAccuracy"] == .8988
    assert row["intAccuracy"] == .9001
    assert row["epochs"] == 2 and row["training"] == []
    relu = variant(study(payload, "e4-qat-resnet20_relu"), "qat")
    assert relu["fp32Accuracy"] is None
    assert relu["fakeAccuracy"] == .8936
    assert relu["training"][-1]["valAccuracy"] == .898


def test_pruning_is_fake_quant_and_never_borrows_a_missing_preset(payload):
    result = study(payload, "e6-resnet20_relu", "edge-10tops")
    row = next(r for r in result["variants"] if r.get("strategy") == "uniform" and r.get("ratio") == .5)
    assert row["fp32Accuracy"] == .8722 and row["fakeAccuracy"] == .874
    assert row["intAccuracy"] is None and row["intImages"] is None
    assert row["params"] == 137890 and row["macs"] == 20759168
    assert row["cycles"] == 27739 and row["training"] == []
    assert variant(result, result["baselineId"])["params"] == 271690
    strict = study(payload, "e6-resnet20_relu", "edge-10tops-strict")
    assert all(r["cycles"] is None for r in strict["variants"])


def test_vit_surgery_keeps_full_test_and_integer_subset_separate(payload):
    result = study(payload, "e9-vit")
    before, after = variant(result, "before"), variant(result, "after")
    assert (before["fp32Accuracy"], after["fp32Accuracy"]) == (.8098, .8095)
    assert after["fp32Images"] == 10000 and after["intImages"] == 2000
    assert after["intAccuracy"] == .81 and after["cycles"] == 1708230
    assert after["training"] == [] and after["epochs"] == 3
    assert variant(study(payload, "e9-vit", "tiny-1tops"), "after")["cycles"] is None


def test_ptq_keeps_subset_baselines_and_calibration_metadata(payload):
    result = study(payload, "e2-resnet20_relu")
    default = variant(result, "npu-default")
    secondary = variant(result, "npu-percentile")
    assert default["intImages"] == 10000
    assert secondary["intImages"] == 2000 and secondary["fp32Images"] == 10000
    assert secondary["fp32OnIntSubset"] == .9 and secondary["fakeOnIntSubset"] == .9015
    assert secondary["calibration"] == "512 random train images (seed 0)"
    assert secondary["cycles"] is None
    assert secondary["training"] == []


def test_baseline_training_uses_recorded_epochs_without_invented_train_accuracy(payload):
    result = study(payload, "e1-training", "edge-10tops")
    row = variant(result, "resnet20_relu")
    assert len(row["training"]) == 30
    assert row["training"][0]["trainAccuracy"] is None
    assert row["training"][-1]["valAccuracy"] == .9024
    vit = variant(result, "cust_vit")
    assert vit["epochs"] == 40 and vit["selectedEpoch"] == 39
    assert vit["fp32Accuracy"] == .8098 and vit["finalTestAccuracy"] == .8092


def test_imagenette_and_super_resolution_use_their_own_metrics(payload):
    result = study(payload, "e12-imagenette")
    base = variant(result, "baseline")
    assert base["fp32Accuracy"] == .8397452229299363
    assert base["fp32Images"] == 3925 and base["macs"] == 163250816
    assert base["epochs"] == 30 and base["training"] == []
    sr = study(payload, "e17-espcn_x2")
    row = variant(sr, "npu-default")
    assert row["fp32Accuracy"] is None and row["intAccuracy"] is None
    assert row["metric"]["unit"] == "dB"
    assert row["metric"]["baseline"] == 30.234268209553726
    assert row["metric"]["value"] == 29.954936272901545
    assert row["cycles"] is None and row["training"] == []
    # The two depths are separate cards: the deep card must read the deep records, not the shallow ones.
    deep = variant(study(payload, "e17-espcn_x2_deep"), "npu-default")
    assert deep["metric"]["baseline"] == 30.11345206880338
    assert deep["metric"]["value"] != row["metric"]["value"]


def test_missing_numerical_evidence_remains_null(payload):
    data = copy.deepcopy(payload)
    row = data["results"]["e4_surgery"]["records"][3]
    for key in ["float_acc_after_heal", "float_acc_after_swap", "int_acc", "cycles", "heal_log"]:
        row.pop(key, None)
    recovered = variant(study(data, "e4-activation"), "swap-relu-heal3")
    assert recovered["fp32Accuracy"] is None
    assert recovered["intAccuracy"] is None and recovered["cycles"] is None
    assert recovered["training"] == []
    assert "error" in call("study", payload, "e4-activation", "unknown-npu")
    assert "error" in call("study", payload, "missing", "edge-10tops")


def test_assessment_uses_only_same_scope_fp32_and_recorded_cycles(payload):
    result = study(payload, "e4-activation")
    assessed = call("assess", result, {"maxDropPp": 1, "minSavingPercent": 50})
    assert assessed["unchanged"]
    row = next(r for r in assessed["value"] if r["variantId"] == "swap-relu-heal3")
    assert row["accuracyDropPp"] == pytest.approx(.6)
    assert row["cycleSavingPercent"] == pytest.approx(97.78649207486302)
    assert row["eligible"] is True
    qat = next(r for r in assessed["value"] if r["variantId"] == "silu-qat")
    assert qat["eligible"] is None and qat["accuracyDropPp"] is None
    modified = copy.deepcopy(result)
    variant(modified, "swap-relu-heal3")["accuracyScope"] = "different-samples"
    row = next(r for r in call("assess", modified, {})["value"] if r["variantId"] == "swap-relu-heal3")
    assert row["eligible"] is None and row["accuracyDropPp"] is None


def test_assessment_has_no_compression_claim_for_missing_cost_or_psnr(payload):
    for key in ["e6-resnet20_relu", "e17-espcn_x2", "e17-espcn_x2_deep"]:
        result = study(payload, key)
        rows = call("assess", result, {"maxDropPp": 1, "minSavingPercent": 10})["value"]
        assert all(r["eligible"] is None for r in rows)
    assert "error" in call("assess", study(payload, "e4-activation"), {"maxDropPp": "1"})


def test_assessment_requires_matching_sample_counts_even_if_scope_label_is_reused(payload):
    result = study(payload, "e4-activation")
    variant(result, "swap-relu-heal3")["fp32Images"] = 2000
    row = next(r for r in call("assess", result, {})["value"] if r["variantId"] == "swap-relu-heal3")
    assert row["eligible"] is None and row["accuracyDropPp"] is None


def test_missing_epoch_numbers_remain_unknown(payload):
    data = copy.deepcopy(payload)
    data["results"]["e1_baselines"]["records"][0]["epochs"] = [{"train_loss": .2}]
    result = study(data, "e1-training")
    assert variant(result, "resnet20_relu")["epochs"] is None


def test_scope_counts_include_their_actual_source_pointer(payload):
    recovered = variant(study(payload, "e4-activation"), "swap-relu-heal3")
    assert any(p["sourceKey"] == "e1_baselines" for p in recovered["sourceRecords"])
