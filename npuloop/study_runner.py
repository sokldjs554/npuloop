"""Run an isolated CPU model-change, recovery-training and INT8 study.

Usage: python -m npuloop.study_runner --help
The output is new evidence; it never updates the repository's historical results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from .graph.ir import trace
from .intengine.graph import export_int_graph
from .intengine.numpy_engine import NumpyEngine
from .intengine.serialize import save_int_graph
from .npu.cost import estimate
from .npu.spec import PRESETS, get_spec
from .prune.structured import find_groups, prune
from .quant.prepare import calibrate, prepare
from .quant.scheme import PRESET_SCHEMES
from .quant.surgery import swap_activations
from .zoo.data import CIFAR10NPZ, VAL_SEED
from .zoo.models import ACTS, count_params
from .zoo.train import fit, load_checkpoint

SCHEMA_VERSION = "npuloop.model-study.v1"


@dataclass(frozen=True)
class StudyConfig:
    checkpoint: str
    data: str
    out: str
    preset: str = "edge-10tops"
    operation: str = "prune"
    prune_ratio: float = 0.25       # fraction removed, unlike prune()'s keep ratio
    activation: str = "relu6"
    epochs: int = 1
    steps_per_epoch: int = 2
    eval_images: int = 32
    calibration_images: int = 32
    seed: int = 0
    threads: int = 2
    batch_size: int = 32
    lr: float = 0.01

    def validate(self):
        for name in ("steps_per_epoch", "eval_images", "calibration_images", "threads", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.epochs, int) or isinstance(self.epochs, bool) or self.epochs < 0:
            raise ValueError("epochs must be a nonnegative integer (0 is the no-training control)")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer between 0 and 2**32-1")
        if not math.isfinite(self.prune_ratio) or not 0 < self.prune_ratio < 1:
            raise ValueError("prune_ratio must be finite and between 0 and 1 (fraction removed)")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if self.operation not in ("prune", "activation"):
            raise ValueError("operation must be prune or activation")
        if self.activation not in ACTS:
            raise ValueError(f"activation must be one of {sorted(ACTS)}")
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        for name in ("checkpoint", "data"):
            if not Path(getattr(self, name)).is_file():
                raise ValueError(f"{name} must name an existing file")
        _check_output(Path(self.out))


def _check_output(out: Path):
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError(f"output must be a new or empty directory: {out}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value)
        digest.update(json.dumps([str(arr.dtype), arr.shape]).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(_array_hash(tensor.numpy()).encode())
    return digest.hexdigest()


def _json(path: Path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _dataset(path: Path):
    # A calibration-only bundle can declare val_per_class=0. Create a small,
    # deterministic holdout from x_train, never from x_test.
    with np.load(path, allow_pickle=False) as data:
        required = {"x_train", "y_train", "x_test", "y_test", "classes"}
        if not required.issubset(data.files):
            raise ValueError(f"data is missing keys: {sorted(required - set(data.files))}")
        classes = data["classes"]
        if classes.ndim != 1 or len(classes) < 2:
            raise ValueError("classes must contain at least two class names")
        for split in ("train", "test"):
            x, y = data[f"x_{split}"], data[f"y_{split}"]
            if x.ndim != 4 or x.shape[-1] != 3 or x.shape[1] != x.shape[2] or x.dtype != np.uint8:
                raise ValueError(f"x_{split} must contain square NHWC uint8 RGB images")
            if y.ndim != 1 or len(x) != len(y) or not len(y) or not np.issubdtype(y.dtype, np.integer):
                raise ValueError(f"y_{split} must contain one integer label per nonempty image split")
            if y.min() < 0 or y.max() >= len(classes):
                raise ValueError(f"y_{split} has labels outside classes")
        if data["x_train"].shape[1:] != data["x_test"].shape[1:]:
            raise ValueError("train and test image shapes must agree")
        labels, counts = np.unique(data["y_train"], return_counts=True)
        if len(labels) != len(classes) or counts.min() < 2:
            raise ValueError("training source needs at least two images of every class for a validation holdout")
        stored_holdout = int(data["val_per_class"]) if "val_per_class" in data else None
        if stored_holdout is not None and stored_holdout < 0:
            raise ValueError("val_per_class must be nonnegative")
        per_class = stored_holdout or min(500, max(1, int(counts.min()) // 10))
        if per_class >= counts.min():
            raise ValueError("validation holdout would leave an empty training class")
        raw_train_count, raw_test_count = len(data["y_train"]), len(data["y_test"])
    ds = CIFAR10NPZ(str(path), val_per_class=per_class, val_seed=VAL_SEED)
    # The NPZ contract allows any integer label dtype; cross_entropy requires
    # class-index tensors to be int64. Normalize in memory, preserving the file.
    for split in ("train", "val", "test"):
        setattr(ds, f"y_{split}", getattr(ds, f"y_{split}").astype(np.int64, copy=False))
    if not np.isfinite(ds.mean).all() or not np.isfinite(ds.std).all() or np.any(ds.std <= 0) or ds.pad < 0:
        raise ValueError("normalization mean/std and augmentation padding must be valid")
    keep = np.ones(raw_train_count, dtype=bool)
    keep[ds.val_idx] = False
    return ds, np.flatnonzero(keep), {
        "source_train_images": raw_train_count, "source_test_images": raw_test_count,
        "train_images": len(ds.x_train), "validation_images": len(ds.x_val), "test_images": len(ds.x_test),
        "validation_seed": VAL_SEED, "validation_per_class": per_class,
        "validation_policy": "provided_train_holdout" if stored_holdout else "adaptive_train_holdout",
        "validation_train_indices": ds.val_idx.tolist(), "classes": list(ds.classes),
        "normalization": {"mean": ds.mean.tolist(), "std": ds.std.tolist(), "augmentation_pad": ds.pad},
    }


@torch.no_grad()
def _measure(predict, ds, n: int, subset_hash: str, state_hash: str, execution: str):
    predictions, labels = [], []
    for offset in range(0, n, 32):
        stop = min(offset + 32, n)
        x = ds.to_tensor(ds.x_test[offset:stop])
        logits = predict(x)
        logits = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
        if logits.shape != (len(x), len(ds.classes)) or not np.isfinite(logits).all():
            raise ValueError("model must produce finite classification logits with the declared class count")
        predictions.extend(logits.argmax(1).tolist())
        labels.extend(ds.y_test[offset:stop].tolist())
    correct = int(np.equal(predictions, labels).sum())
    return dict(accuracy=correct / n, correct=correct, n=n, split="test", subset_sha256=subset_hash,
                model_state_sha256=state_hash, provenance="measured_cpu", execution=execution,
                predictions=predictions)


def _cost(model, input_shape, spec):
    graph = trace(model.eval(), input_shape)
    cost = estimate(graph, spec)
    return dict(total_cycles=cost.total_cycles, total_macs=cost.total_macs,
                parameters=count_params(model), estimated_latency_ms=cost.latency_ms,
                array_utilization=cost.array_utilization, dram_bytes=cost.dram_bytes,
                cycles_by_kind=cost.breakdown())


def _software():
    package = Path(__file__).resolve().parent
    files = {str(path.relative_to(package.parent)): _sha256(path) for path in sorted(package.rglob("*.py"))}
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package.parent, capture_output=True,
                                text=True, check=True, timeout=3).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=package.parent, capture_output=True,
                                text=True, check=True, timeout=3).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                device="cpu", platform=platform.platform(), git_commit=commit, git_worktree_dirty=dirty,
                source_files_sha256=files,
                source_tree_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def _artifact(path: Path, out: Path, kind: str):
    return dict(path=str(path.relative_to(out)), sha256=_sha256(path), bytes=path.stat().st_size, kind=kind)


def run_study(config: StudyConfig) -> dict:
    """Execute real model operations and write a portable, strictly finite study.json.

    Existing nonempty directories are always rejected. A failed run may leave
    diagnostic artifacts but never a completed study.json.
    """
    config.validate()
    out = Path(config.out).resolve()
    checkpoint, data = Path(config.checkpoint).resolve(), Path(config.data).resolve()
    checkpoint_hash, data_hash = _sha256(checkpoint), _sha256(data)
    ds, train_indices, dataset_info = _dataset(data)
    n_eval = min(config.eval_images, len(ds.x_test))
    n_val = min(config.eval_images, len(ds.x_val))
    n_calib = min(config.calibration_images, len(ds.x_train))
    batch_size = min(config.batch_size, len(ds.x_train))
    steps_per_epoch = min(config.steps_per_epoch, len(ds.x_train) // batch_size)
    rng = np.random.default_rng(config.seed)
    calibration_idx = rng.choice(len(ds.x_train), n_calib, replace=False)
    test_indices = ds.test_order[:n_eval]
    subset_hash = _array_hash(test_indices, ds.x_test[:n_eval], ds.y_test[:n_eval])
    # Source split membership is recorded; also catch accidentally duplicated
    # calibration/evaluation images in a user-supplied NPZ.
    eval_content = {_array_hash(x) for x in ds.x_test[:n_eval]}
    if any(_array_hash(ds.x_train[i]) in eval_content for i in calibration_idx):
        raise ValueError("calibration and evaluation contain duplicate image content across source splits")
    dataset_info.update(file_name=data.name, file_sha256=data_hash, input_shape=[3, ds.img_size, ds.img_size],
                        calibration_split="train", calibration_images=n_calib,
                        calibration_train_indices=train_indices[calibration_idx].tolist(),
                        calibration_subset_sha256=_array_hash(train_indices[calibration_idx], ds.x_train[calibration_idx]),
                        test_indices=test_indices.tolist(), evaluation_labels=ds.y_test[:n_eval].tolist(),
                        evaluation_images=n_eval, evaluation_subset_sha256=subset_hash,
                        validation_evaluated_images=n_val,
                        validation_evaluated_train_indices=ds.val_idx[:n_val].tolist(),
                        benchmark_scope="diagnostic_sample" if n_eval < len(ds.x_test) or len(ds.x_test) < 10000
                        else "provided_test_split", source_split_identity="user_supplied_npz")
    resolved_config = dict(asdict(config), actual_eval_images=n_eval, actual_calibration_images=n_calib,
                           actual_validation_images=n_val, actual_batch_size=batch_size,
                           actual_steps_per_epoch=steps_per_epoch if config.epochs else 0)
    old_threads = torch.get_num_threads()
    started = time.perf_counter()
    torch.set_num_threads(config.threads)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            baseline = load_checkpoint(str(checkpoint))  # weights_only=True; no unrestricted-pickle fallback
            if baseline.config.get("arch") not in {"resnet", "mobilenetv2", "vit", "inception"}:
                raise ValueError("this study runner supports zoo classification checkpoints only")
            if any(not torch.isfinite(t).all() for t in baseline.state_dict().values() if t.is_floating_point()):
                raise ValueError("checkpoint contains nonfinite tensors")
            input_shape = tuple(dataset_info["input_shape"])
            baseline_hash = _state_hash(baseline)
            observed = {"baseline_fp32": _measure(baseline, ds, n_eval, subset_hash, baseline_hash, "pytorch_fp32_cpu")}
            base_cost = _cost(baseline, input_shape, config.preset)
            if config.operation == "prune":
                original_groups = find_groups(baseline, input_shape)
                changed, groups = prune(baseline, ratio=1 - config.prune_ratio, strategy="uniform", input_shape=input_shape)
                operation = dict(kind="prune", strategy="uniform", requested_fraction_removed=config.prune_ratio,
                                 groups=[dict(name=g.name, channels_before=b.channels, channels_after=g.channels)
                                         for b, g in zip(original_groups, groups)])
                if count_params(changed) >= count_params(baseline):
                    raise ValueError("prune ratio removes no channels in this model; increase it or choose activation")
            else:
                source_act = baseline.config.get("act")
                if source_act == config.activation:
                    raise ValueError("activation must differ from the checkpoint activation")
                changed = swap_activations(baseline, {source_act: config.activation}).eval()
                if not changed.swapped:
                    raise ValueError("checkpoint has no safely replaceable configured activations")
                operation = dict(kind="activation", activation_before=source_act, activation_after=config.activation,
                                 activations_replaced=changed.swapped)
            operation.update(parameters_before=count_params(baseline), parameters_after=count_params(changed),
                             baseline_config=baseline.config, changed_config=changed.config)
            immediate_hash = _state_hash(changed)
            observed["changed_fp32_before_training"] = _measure(changed, ds, n_eval, subset_hash, immediate_hash, "pytorch_fp32_cpu")
            changed_cost = _cost(changed, input_shape, config.preset)
            # Reserve after loading and validating inputs, before any output write.
            _check_output(out)
            out.mkdir(parents=True, exist_ok=True)
            lock = out / ".study-running"
            with lock.open("x", encoding="utf-8") as stream:
                stream.write("npuloop model study in progress\n")
            _json(out / "config.json", resolved_config)
            immediate_path = out / "changed_before_training.pt"
            torch.save(dict(config=changed.config, state_dict=changed.state_dict()), immediate_path)
            if config.epochs:
                training = fit(changed, ds, epochs=config.epochs, lr=config.lr, bs=batch_size,
                               seed=config.seed, out=str(out / "training"), channels_last=False,
                               steps_per_epoch=steps_per_epoch, eval_limit=config.eval_images)
                changed = load_checkpoint(str(out / "training" / "best.pt"))
                training["selection"] = "highest validation accuracy; ties select the later epoch; test never selects"
            else:
                training = dict(epochs=[], selected_epoch=0, selection="zero-training control",
                                hparams=dict(epochs=0, lr=config.lr, bs=batch_size, seed=config.seed))
            training["actual_steps"] = sum(row["steps"] for row in training["epochs"])
            training["actual_samples"] = sum(row["samples"] for row in training["epochs"])
            selected_path = out / "fine_tuned.pt"
            torch.save(dict(config=changed.config, state_dict=changed.state_dict()), selected_path)
            selected_hash = _state_hash(changed)
            training["weights_changed"] = immediate_hash != selected_hash
            observed["fine_tuned_fp32"] = _measure(changed.eval(), ds, n_eval, subset_hash, selected_hash, "pytorch_fp32_cpu")
            qm = prepare(changed, PRESET_SCHEMES["npu-default"])
            calibration_batches = [ds.to_tensor(ds.x_train[calibration_idx[i:i + 32]]) for i in range(0, n_calib, 32)]
            calibrate(qm, calibration_batches)
            observed["fake_quant"] = _measure(qm, ds, n_eval, subset_hash, _state_hash(qm), "pytorch_fake_quant_cpu")
            graph = export_int_graph(qm, input_shape=input_shape)
            integer_path = out / "model.npuloop"
            save_int_graph(graph, str(integer_path), provenance=dict(schema_version=SCHEMA_VERSION,
                           source_checkpoint_sha256=checkpoint_hash, selected_state_sha256=selected_hash,
                           calibration_subset_sha256=dataset_info["calibration_subset_sha256"]))
            engine = NumpyEngine(graph)
            observed["integer"] = _measure(lambda x: engine.predict(x.numpy()), ds, n_eval, subset_hash,
                                            _sha256(integer_path), "numpy_integer_semantics_cpu")
            if _state_hash(baseline) != baseline_hash or _sha256(checkpoint) != checkpoint_hash or _sha256(data) != data_hash:
                raise RuntimeError("an input changed while the study was running")
            _json(out / "training_log.json", training)
            artifacts = {
                "configuration": _artifact(out / "config.json", out, "configuration"),
                "immediate_checkpoint": _artifact(immediate_path, out, "fp32_checkpoint"),
                "selected_checkpoint": _artifact(selected_path, out, "fp32_checkpoint"),
                "training_log": _artifact(out / "training_log.json", out, "training_log"),
                "integer_graph": _artifact(integer_path, out, "portable_integer_graph"),
            }
            for path in sorted((out / "training").glob("*")):
                if path.is_file():
                    artifacts[f"trainer_{path.stem}"] = _artifact(path, out, "trainer_output")
            comparison = dict(
                recovery_delta_pp=100 * (observed["fine_tuned_fp32"]["accuracy"] - observed["changed_fp32_before_training"]["accuracy"]),
                integer_vs_baseline_delta_pp=100 * (observed["integer"]["accuracy"] - observed["baseline_fp32"]["accuracy"]),
                fake_integer_top1_agreement=float(np.equal(observed["fake_quant"]["predictions"], observed["integer"]["predictions"]).mean()),
                simulated_cycle_reduction_fraction=1 - changed_cost["total_cycles"] / base_cost["total_cycles"],
                parameter_reduction_fraction=1 - count_params(changed) / count_params(baseline))
            report = dict(schema_version=SCHEMA_VERSION, status="complete", created_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.perf_counter() - started, config=resolved_config, software=_software(),
                          source_checkpoint=dict(file_name=checkpoint.name, sha256=checkpoint_hash, state_sha256=baseline_hash),
                          dataset=dataset_info, operation=operation, observed=observed, training=training,
                          quantization=dict(scheme="npu-default", **PRESET_SCHEMES["npu-default"].to_dict()),
                          simulated_costs=dict(provenance="simulated", preset=config.preset, spec=get_spec(config.preset).to_dict(),
                                               baseline=base_cost, changed=changed_cost, hardware_measurement=False),
                          comparison=comparison, artifacts=artifacts)
            _json(out / "study.json", report)
            lock.unlink()
            return report
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Real CPU model change → recovery training → PTQ → integer study")
    for name in ("checkpoint", "data", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="edge-10tops")
    parser.add_argument("--operation", choices=["prune", "activation"], default="prune")
    parser.add_argument("--prune-ratio", type=float, default=0.25, help="fraction of prunable channels REMOVED (0<r<1)")
    parser.add_argument("--activation", choices=sorted(ACTS), default="relu6")
    parser.add_argument("--epochs", type=int, default=1, help="0 saves a real no-training control")
    parser.add_argument("--steps-per-epoch", type=int, default=2, help="cap; actual available steps are recorded")
    parser.add_argument("--eval-images", type=int, default=32, help="exact test and validation caps")
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    config = StudyConfig(**vars(parser.parse_args(argv)))
    try:
        report = run_study(config)
    except (ValueError, OSError, RuntimeError, pickle.UnpicklingError) as exc:
        print(f"study failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(study=str(Path(config.out).resolve() / "study.json"),
                          scope=report["dataset"]["benchmark_scope"],
                          evaluated_images=report["dataset"]["evaluation_images"],
                          actual_training_steps=report["training"]["actual_steps"],
                          observed_accuracy={name: value["accuracy"] for name, value in report["observed"].items()},
                          simulated_cycles={name: report["simulated_costs"][name]["total_cycles"] for name in ("baseline", "changed")}),
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
