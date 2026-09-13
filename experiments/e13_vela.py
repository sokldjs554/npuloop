"""E13 — cross-check the cost model against Arm's Vela estimator on the same INT8 graphs.

E8 compares `npu.estimate` with SCALE-Sim, which is another research model of an idealized
systolic array. Vela is a different kind of witness: it is the compiler Arm ships for real
Ethos-U silicon, and for an INT8 TFLite file it reports per-operator cycles, MAC utilization,
SRAM/flash traffic, and — the field no research simulator produces — how many operators it
could *not* put on the NPU.

So this experiment asks a narrower question than E8: given the same network and a spec built
from Vela's own published machine configuration, how far apart do two independent analytical
estimators land, and do they disagree structurally or only in magnitude?

Fair-comparison notes (also recorded in the result file):
  * Ethos-U55 is not the weight-stationary array `npu.cost` models. The spec below is taken
    from Vela's configuration dump (256 MACs/cycle, 500 MHz, 4 MiB arena, 0.466 GB/s off-chip
    flash for weights), NOT tuned to make the totals agree. Tuning it would make the
    comparison circular.
  * Vela costs a *compiled* schedule: it fuses, tiles, picks block configs and cascades.
    `npu.estimate` costs one layer at a time with no compiler. Vela should therefore be the
    faster of the two wherever fusion pays, and that gap is the interesting quantity.
  * Both are analytical estimators. Vela's own docs label its performance estimation
    EXPERIMENTAL and warn the numbers are not real measurements. Nothing here is silicon.

Needs `pip install ethos-u-vela tensorflow`. Writes results/e13_vela.json.
"""
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.common import Results                      # noqa: E402
from npuloop.graph import trace                             # noqa: E402
from npuloop.npu import estimate                            # noqa: E402
from npuloop.npu.spec import NPUSpec                        # noqa: E402

ACCEL = "ethos-u55-256"

# Built from Vela's own summary CSV for this accelerator, not fitted to its cycle counts:
#   accelerator_configuration=Ethos_U55_256  core_clock=500 MHz  arena_cache_size=4 MiB
#   off_chip_flash_bandwidth=0.4657 GB/s (weights_storage_area=Off-chip Flash)
# 256 MACs/cycle is expressed here as a 16x16 array because that is how `npu.cost` counts.
ETHOS_U55_256 = NPUSpec(
    "ethos-u55-256", pe_rows=16, pe_cols=16, cores=1, freq_mhz=500.0, sram_kb=4096.0,
    dram_gbps=0.4657, vector_lanes=64, dw_lanes=256,
)

# One architecture spec builds both a Keras model (-> TFLite -> Vela) and a torch model
# (-> npuloop), so the two estimators see the same shapes, strides and channel counts.
# ("conv", cout, k, stride) | ("dw", k, stride) | ("gap",) | ("fc", units)
NETS: dict[str, list[tuple]] = {
    "conv-stack": [("conv", 16, 3, 2), ("conv", 32, 3, 1), ("conv", 32, 3, 2), ("conv", 64, 3, 1), ("gap",), ("fc", 10)],
    "dw-separable": [("conv", 16, 3, 2), ("dw", 3, 1), ("conv", 32, 1, 1), ("dw", 3, 2), ("conv", 64, 1, 1), ("gap",), ("fc", 10)],
    "pointwise-heavy": [("conv", 32, 3, 2), ("conv", 64, 1, 1), ("conv", 32, 1, 1), ("conv", 64, 1, 1), ("gap",), ("fc", 10)],
    "wide-late": [("conv", 16, 3, 2), ("conv", 32, 3, 2), ("conv", 128, 3, 1), ("gap",), ("fc", 10)],
    "deep-narrow": [("conv", 8, 3, 2)] + [("conv", 8, 3, 1)] * 5 + [("gap",), ("fc", 10)],
}
IN_HW, IN_CH = 32, 3


# ----------------------------------------------------------------------------- torch side
class TorchNet(nn.Module):
    def __init__(self, layers: list[tuple], in_ch=IN_CH):
        super().__init__()
        seq, c = [], in_ch
        for spec in layers:
            if spec[0] == "conv":
                _, cout, k, s = spec
                seq += [nn.Conv2d(c, cout, k, stride=s, padding=k // 2, bias=True), nn.ReLU()]
                c = cout
            elif spec[0] == "dw":
                _, k, s = spec
                seq += [nn.Conv2d(c, c, k, stride=s, padding=k // 2, groups=c, bias=True), nn.ReLU()]
            elif spec[0] == "gap":
                seq += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            elif spec[0] == "fc":
                seq += [nn.Linear(c, spec[1])]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- keras side
def keras_int8_tflite(layers: list[tuple]) -> bytes:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(0)
    i = tf.keras.Input((IN_HW, IN_HW, IN_CH))
    x = i
    for spec in layers:
        if spec[0] == "conv":
            _, cout, k, s = spec
            x = tf.keras.layers.Conv2D(cout, k, strides=s, padding="same", activation="relu")(x)
        elif spec[0] == "dw":
            _, k, s = spec
            x = tf.keras.layers.DepthwiseConv2D(k, strides=s, padding="same", activation="relu")(x)
        elif spec[0] == "gap":
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif spec[0] == "fc":
            x = tf.keras.layers.Dense(spec[1])(x)
    rng = np.random.default_rng(0)

    def rep():
        for _ in range(32):
            yield [rng.standard_normal((1, IN_HW, IN_HW, IN_CH), dtype=np.float32)]

    c = tf.lite.TFLiteConverter.from_keras_model(tf.keras.Model(i, x))
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = c.inference_output_type = tf.int8
    return c.convert()


# ----------------------------------------------------------------------------- vela side
def run_vela(tflite_bytes: bytes, workdir: str) -> dict:
    """Compile with Vela and read back its per-layer report. Returns totals + per-op rows."""
    src = os.path.join(workdir, "net.tflite")
    with open(src, "wb") as f:
        f.write(tflite_bytes)
    out = subprocess.run(
        ["vela", "--accelerator-config", ACCEL, "--output-dir", workdir, "--verbose-performance", src],
        capture_output=True, text=True, timeout=900,
    )
    if out.returncode != 0:
        raise RuntimeError(f"vela failed: {out.stderr[-800:]}")

    per_layer = os.path.join(workdir, "net_per-layer.csv")
    rows = []
    with open(per_layer) as f:
        for r in csv.DictReader(f):
            rows.append(dict(op=r["Original Operator"], nng=r["NNG Operator"], target=r["Target"],
                             cycles=float(r["Op Cycles"]), macs=int(float(r["MAC Count"])),
                             util=float(r["Util% (MAC)"]) / 100.0, name=r["Name"]))
    summary = next(p for p in os.listdir(workdir) if p.startswith("net_summary"))
    with open(os.path.join(workdir, summary)) as f:
        s = next(csv.DictReader(f))
    # Vela emits a row per fused sub-operator (a Relu row carries 0 cycles); group by source op.
    grouped: list[dict] = []
    for r in rows:
        if grouped and r["name"] == grouped[-1]["name"]:
            grouped[-1]["cycles"] += r["cycles"]
            grouped[-1]["macs"] += r["macs"]
        else:
            grouped.append(dict(r))
    return dict(
        rows=grouped,
        total_cycles=sum(r["cycles"] for r in rows),
        total_macs=sum(r["macs"] for r in rows),
        npu_ops=int(float(s["passes_after_fusing"])),
        inference_time_us=float(s["inference_time"]) * 1e6,
        core_clock_mhz=float(s["core_clock"]) / 1e6,
        cpu_operator_pct=_cpu_pct(out.stdout),
    )


def _cpu_pct(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.strip().startswith("CPU operators"):
            return float(line.split("(")[1].split("%")[0])
    return float("nan")


# ----------------------------------------------------------------------------- comparison
KIND_TO_VELA = {"conv": "Conv2D", "dwconv": "DepthwiseConv2D", "linear": "FullyConnected", "pool": "Mean"}


def npuloop_cost(layers: list[tuple]) -> dict:
    m = TorchNet(layers).eval()
    g = trace(m, (IN_CH, IN_HW, IN_HW))
    rep = estimate(g, ETHOS_U55_256)
    rows = [dict(name=l.name, kind=l.kind, cycles=float(l.cycles), macs=int(l.macs),
                 util=float(l.array_util or l.engine_util), bound=l.bound)
            for l in rep.layers if l.macs or l.kind in ("pool", "dwconv")]
    return dict(rows=rows, total_cycles=float(rep.total_cycles), total_macs=int(rep.total_macs),
                latency_us=float(rep.latency_ms) * 1e3)


def pair_by_kind(v_rows: list[dict], n_rows: list[dict]) -> list[dict]:
    """Match the two reports positionally within each operator kind (both keep source order)."""
    pairs = []
    for kind, vela_op in KIND_TO_VELA.items():
        vs = [r for r in v_rows if r["op"] == vela_op]
        ns = [r for r in n_rows if r["kind"] == kind]
        for i in range(min(len(vs), len(ns))):
            v, n = vs[i], ns[i]
            pairs.append(dict(kind=kind, index=i, name=n["name"],
                              vela_cycles=v["cycles"], npuloop_cycles=n["cycles"],
                              vela_macs=v["macs"], npuloop_macs=n["macs"],
                              vela_util=v["util"], npuloop_util=n["util"],
                              ratio=(n["cycles"] / v["cycles"]) if v["cycles"] else None))
        if len(vs) != len(ns):
            pairs.append(dict(kind=kind, index=-1, name="(count mismatch)",
                              vela_count=len(vs), npuloop_count=len(ns)))
    return pairs


def main():
    if shutil.which("vela") is None:
        print("vela not found; pip install ethos-u-vela", file=sys.stderr)
        return 1
    ver = subprocess.run(["vela", "--version"], capture_output=True, text=True).stdout.strip()
    res = Results("e13_vela", meta=dict(
        vela_version=ver, accelerator=ACCEL,
        spec=ETHOS_U55_256.to_dict(),
        note=("Two analytical estimators on the same INT8 graph. The npuloop spec is taken from Vela's "
              "configuration dump, not fitted to its cycle counts. Vela costs a compiled, fused, cascaded "
              "schedule; npu.estimate costs one layer at a time with no compiler. Vela's own docs mark its "
              "performance estimation EXPERIMENTAL — neither side is a hardware measurement."),
    ))
    for name, layers in NETS.items():
        if res.has(net=name):
            print(f"{name}: already done", flush=True)
            continue
        tfl = keras_int8_tflite(layers)
        with tempfile.TemporaryDirectory() as wd:
            v = run_vela(tfl, wd)
        n = npuloop_cost(layers)
        rec = dict(net=name, layers=[list(s) for s in layers],
                   vela=dict(total_cycles=v["total_cycles"], total_macs=v["total_macs"],
                             inference_time_us=v["inference_time_us"], npu_passes=v["npu_ops"],
                             cpu_operator_pct=v["cpu_operator_pct"], rows=v["rows"]),
                   npuloop=dict(total_cycles=n["total_cycles"], total_macs=n["total_macs"],
                                latency_us=n["latency_us"], rows=n["rows"]),
                   mac_match=(v["total_macs"] == n["total_macs"]),
                   cycle_ratio=n["total_cycles"] / v["total_cycles"] if v["total_cycles"] else None,
                   per_layer=pair_by_kind(v["rows"], n["rows"]))
        res.add(rec, provenance="simulated")
        print(f"{name}: vela {v['total_cycles']:.0f} cyc / npuloop {n['total_cycles']:.0f} cyc "
              f"(x{rec['cycle_ratio']:.2f}) · MACs {v['total_macs']} vs {n['total_macs']} "
              f"({'match' if rec['mac_match'] else 'DIFFER'}) · CPU ops {v['cpu_operator_pct']:.0f}%", flush=True)
    print(json.dumps({r["net"]: round(r["cycle_ratio"], 3) for r in res.data["records"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
