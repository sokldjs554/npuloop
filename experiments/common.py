"""Shared plumbing for the experiment scripts: paths, checkpoints, calibration sets, JSON results with provenance."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.environ.get("NPULOOP_DATA", "/home/user/data/cifar10.npz")
RUNS = os.environ.get("NPULOOP_RUNS", "/home/user/work/runs")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

BASELINES = {   # name -> run dir
    "resnet20_relu": "resnet20_relu",
    "resnet20_silu": "resnet20_silu",
    "resnet20_hswish": "resnet20_hswish",
    "resnet20_gelu": "resnet20_gelu",
    "mnv2_050_relu6": "mnv2_050_relu6",
}


def available_baselines() -> dict[str, str]:
    out = {}
    for name, d in BASELINES.items():
        p = os.path.join(RUNS, d, "log.json")
        if os.path.exists(p) and "final_test_acc" in json.load(open(p)):
            out[name] = os.path.join(RUNS, d, "best.pt")
    return out


def load_model(name: str):
    from npuloop.zoo import load_checkpoint
    return load_checkpoint(available_baselines()[name])


def dataset():
    from npuloop.zoo import CIFAR10NPZ
    return CIFAR10NPZ(DATA)


def calib_batches(ds, n: int = 512, seed: int = 0, per_batch: int = 256, balanced: bool = False) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    if balanced:
        idx = []
        extra = set(rng.permutation(10)[: n % 10].tolist())     # classes that get one extra image so the total is exactly n
        for c in range(10):
            cls = np.flatnonzero(ds.y_train == c)
            idx.extend(rng.choice(cls, size=n // 10 + (1 if c in extra else 0), replace=False))
        idx = np.array(idx)
    else:
        idx = rng.choice(len(ds.x_train), size=n, replace=False)
    x = ds.to_tensor(ds.x_train[idx])
    return [x[i:i + per_batch] for i in range(0, n, per_batch)]


class Results:
    """Append-only JSON store: results/<name>.json with {'meta':..., 'records': [...]}; skip-if-done support."""

    def __init__(self, name: str, meta: dict | None = None):
        self.path = os.path.join(RESULTS, f"{name}.json")
        if os.path.exists(self.path):
            self.data = json.load(open(self.path))
        else:
            self.data = {"meta": meta or {}, "records": []}
        self.data["meta"].update(meta or {})

    def has(self, **key) -> bool:
        return any(all(r.get(k) == v for k, v in key.items()) for r in self.data["records"])

    def get(self, **key):
        for r in self.data["records"]:
            if all(r.get(k) == v for k, v in key.items()):
                return r
        return None

    def add(self, record: dict, provenance: str = "measured"):
        record = dict(record, provenance=provenance, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.data["records"].append(record)
        self.save()

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=_default)
        os.replace(tmp, self.path)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
