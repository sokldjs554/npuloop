"""Real CPU regressions for exact scoring and isolated model-change studies."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from npuloop.intengine.graph import export_int_graph
from npuloop.intengine.numpy_engine import NumpyEngine
from npuloop.quant.prepare import calibrate, prepare
from npuloop.zoo.data import CIFAR10NPZ
from npuloop.zoo.models import build_model
from npuloop.zoo.train import fit, load_checkpoint


class _NineExamples:
    def batches(self, split, batch_size):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 1])
        for i in range(0, len(labels), batch_size):
            yield torch.zeros(len(labels[i:i + batch_size]), 3, 8, 8), labels[i:i + batch_size]


@pytest.mark.parametrize("limit,expected", [(1, 1.0), (7, 4 / 7), (100, 4 / 9)])
@pytest.mark.parametrize("backend", ["float", "fake", "integer"])
def test_public_evaluators_score_exactly_the_requested_prefix(backend, limit, expected):
    # Rounding the requested prefix up to a batch changes these independently known accuracies.
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)).eval()
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.copy_(torch.tensor([1.0, 0.0]))
    if backend == "integer":
        qm = calibrate(prepare(model), [torch.zeros(1, 3, 8, 8)])
        evaluate = NumpyEngine(export_int_graph(qm, input_shape=(3, 8, 8))).evaluate
        actual = evaluate(_NineExamples(), batch_size=8, limit=limit)
    else:
        module = importlib.import_module("npuloop.zoo.train" if backend == "float" else "npuloop.quant.prepare")
        actual = module.evaluate(model, _NineExamples(), batch_size=8, limit=limit)
    assert actual == pytest.approx(expected)


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(37)
    data = tmp_path / "sample.npz"
    np.savez(data, x_train=rng.integers(0, 256, (96, 16, 16, 3), dtype=np.uint8),
             y_train=np.arange(96) % 3,
             x_test=rng.integers(0, 256, (9, 16, 16, 3), dtype=np.uint8),
             y_test=np.arange(9) % 3, classes=np.array(["a", "b", "c"]),
             val_per_class=0, pad=2)
    torch.manual_seed(3)
    model = build_model(dict(arch="resnet", depth=8, width=8, act="relu", num_classes=3))
    checkpoint = tmp_path / "original.pt"
    torch.save(dict(config=model.config, state_dict=model.state_dict()), checkpoint)
    return checkpoint, data


def test_shared_trainer_runs_two_real_steps_and_reports_counts(inputs):
    checkpoint, path = inputs
    ds = CIFAR10NPZ(str(path), val_per_class=1)
    model = load_checkpoint(str(checkpoint))
    before = model.fc.weight.detach().clone()
    log = fit(model, ds, epochs=1, bs=16, steps_per_epoch=2, channels_last=False, eval_limit=1)
    assert not torch.equal(before, model.fc.weight)
    assert log["epochs"][0]["steps"] == 2
    assert log["epochs"][0]["samples"] == 32
    assert log["epochs"][0]["val_images"] == 1
    assert np.isfinite(log["epochs"][0]["train_loss"])


def test_shared_trainer_rejects_empty_training_instead_of_dividing_by_zero(inputs):
    checkpoint, path = inputs
    ds = CIFAR10NPZ(str(path), val_per_class=32)
    with pytest.raises(ValueError, match="training|batch"):
        fit(load_checkpoint(str(checkpoint)), ds, epochs=1, bs=16)


def _runner():
    assert importlib.util.find_spec("npuloop.study_runner") is not None, "The executable study runner is missing"
    return importlib.import_module("npuloop.study_runner")


def test_study_trains_exports_integer_and_preserves_original(inputs, tmp_path):
    runner = _runner()
    checkpoint, data = inputs
    original = checkpoint.read_bytes()
    report = runner.run_study(runner.StudyConfig(
        checkpoint=str(checkpoint), data=str(data), out=str(tmp_path / "study"),
        epochs=1, steps_per_epoch=2, eval_images=7, calibration_images=5, threads=1))
    assert checkpoint.read_bytes() == original
    assert report["schema_version"] == "npuloop.model-study.v1"
    assert report["status"] == "complete"
    assert isinstance(report["software"]["git_worktree_dirty"], bool)
    assert all(stage["n"] == 7 for stage in report["observed"].values())
    assert len({s["subset_sha256"] for s in report["observed"].values()}) == 1
    assert all(stage["accuracy"] == stage["correct"] / 7 for stage in report["observed"].values())
    assert report["training"]["epochs"][0]["steps"] == 2
    assert report["training"]["epochs"][0]["val_images"] == 7
    before = report["observed"]["changed_fp32_before_training"]["model_state_sha256"]
    after = report["observed"]["fine_tuned_fp32"]["model_state_sha256"]
    assert before != after
    assert report["operation"]["parameters_after"] < report["operation"]["parameters_before"]
    assert report["simulated_costs"]["provenance"] == "simulated"
    assert report["simulated_costs"]["changed"]["total_cycles"] > 0
    split = report["dataset"]
    assert split["calibration_split"] == "train"
    assert len(split["calibration_train_indices"]) == 5
    assert set(split["calibration_train_indices"]).isdisjoint(split["validation_train_indices"])
    assert len(split["test_indices"]) == 7
    assert split["benchmark_scope"] == "diagnostic_sample"
    out = tmp_path / "study"
    assert report["artifacts"]["training_log"]["path"] == "training_log.json"
    parsed = json.loads((out / "study.json").read_text(), parse_constant=lambda v: pytest.fail(v))
    assert parsed == report
    for artifact in report["artifacts"].values():
        payload = (out / artifact["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
    restored = load_checkpoint(str(out / report["artifacts"]["selected_checkpoint"]["path"]))
    with torch.no_grad():
        assert restored(torch.zeros(1, 3, 16, 16)).shape == (1, 3)


def test_zero_epoch_activation_study_has_a_real_untrained_control(inputs, tmp_path):
    runner = _runner()
    checkpoint, data = inputs
    report = runner.run_study(runner.StudyConfig(
        checkpoint=str(checkpoint), data=str(data), out=str(tmp_path / "control"),
        operation="activation", activation="relu6", epochs=0, steps_per_epoch=2,
        eval_images=100, calibration_images=1000, threads=1))
    assert report["training"]["epochs"] == []
    assert report["training"]["actual_steps"] == 0
    assert report["operation"]["activations_replaced"] > 0
    stages = report["observed"]
    assert stages["changed_fp32_before_training"]["model_state_sha256"] == stages["fine_tuned_fp32"]["model_state_sha256"]
    assert all(s["n"] == 9 for s in stages.values())
    assert report["dataset"]["calibration_images"] == report["dataset"]["train_images"]


def test_existing_output_is_never_overwritten(inputs, tmp_path):
    runner = _runner()
    checkpoint, data = inputs
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "study.json"
    sentinel.write_bytes(b"keep historical result")
    with pytest.raises((ValueError, FileExistsError), match="empty|exists"):
        runner.run_study(runner.StudyConfig(checkpoint=str(checkpoint), data=str(data), out=str(out)))
    assert sentinel.read_bytes() == b"keep historical result"
    assert len(list(out.iterdir())) == 1


def test_integer_npz_label_dtypes_are_accepted_for_real_training(inputs, tmp_path):
    runner = _runner()
    checkpoint, original_data = inputs
    with np.load(original_data, allow_pickle=False) as source:
        arrays = dict(source)
    arrays["y_train"] = arrays["y_train"].astype(np.int32)
    arrays["y_test"] = arrays["y_test"].astype(np.uint8)
    data = tmp_path / "int32-labels.npz"
    np.savez(data, **arrays)
    report = runner.run_study(runner.StudyConfig(
        checkpoint=str(checkpoint), data=str(data), out=str(tmp_path / "int32-study"),
        epochs=1, steps_per_epoch=1, eval_images=1, calibration_images=1, threads=1))
    assert report["training"]["actual_steps"] == 1
    assert report["observed"]["integer"]["n"] == 1


def test_pruned_vit_checkpoint_reloads_at_its_configured_image_size(inputs, tmp_path):
    runner = _runner()
    _, data = inputs  # 16px images, not the model zoo's default 32px.
    torch.manual_seed(9)
    model = build_model(dict(arch="vit", dim=16, depth=1, heads=2, patch=4,
                            mlp_ratio=2, num_classes=3, img=16, act="gelu"))
    checkpoint = tmp_path / "vit.pt"
    torch.save(dict(config=model.config, state_dict=model.state_dict()), checkpoint)
    out = tmp_path / "vit-study"
    report = runner.run_study(runner.StudyConfig(
        checkpoint=str(checkpoint), data=str(data), out=str(out),
        epochs=1, steps_per_epoch=1, eval_images=1, calibration_images=2, threads=1))
    assert report["training"]["actual_steps"] == 1
    restored = load_checkpoint(str(out / "fine_tuned.pt"))
    with torch.no_grad():
        assert restored(torch.zeros(1, 3, 16, 16)).shape == (1, 3)


@pytest.mark.parametrize("override", [dict(prune_ratio=0), dict(prune_ratio=1), dict(prune_ratio=float("nan")),
                                    dict(epochs=-1), dict(steps_per_epoch=0), dict(eval_images=0),
                                    dict(calibration_images=0), dict(threads=0)])
def test_invalid_study_configuration_creates_no_outputs(inputs, tmp_path, override):
    runner = _runner()
    checkpoint, data = inputs
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        runner.run_study(runner.StudyConfig(checkpoint=str(checkpoint), data=str(data), out=str(out), **override))
    assert not out.exists()
