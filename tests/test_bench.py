"""The engine benchmark: per-node wall-clock for both engines, joined with the cost model by node name."""
import numpy as np
import torch


def _tiny_int_graph():
    from npuloop.zoo import build_model
    from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
    from npuloop.intengine import export_int_graph
    torch.manual_seed(0)
    m = build_model(dict(arch="resnet", depth=8, width=4, act="relu")).eval()
    qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, [torch.randn(16, 3, 32, 32)])
    return m, export_int_graph(qm)


def test_bench_times_every_node_for_both_engines():
    from npuloop.graph import trace
    from npuloop.npu import estimate
    from npuloop.intengine import bench_engines, layer_table, render_bench
    m, ig = _tiny_int_graph()
    x = np.random.default_rng(0).standard_normal((4, 3, 32, 32)).astype(np.float32)
    b = bench_engines(ig, x, repeats=2, warmup=1)
    assert set(b["engines"]) == {"numpy", "cpp"} and b["batch"] == 4 and b["speedup"] > 0
    for e in b["engines"].values():
        assert set(e["nodes"]) == {n.name for n in ig.nodes}
        assert all(v >= 0 for v in e["nodes"].values())
        assert abs(e["per_image_ms"] * 4 - e["total_ms"]) < 1e-6
    rows = layer_table(ig, b, estimate(trace(m), "edge-10tops"))
    names = {r["name"] for r in rows}
    assert "input" not in names and "output" not in names
    conv = next(r for r in rows if r["op"] == "conv")
    assert conv["cycles"] > 0 and 0 < conv["cycle_share"] <= 1 and conv["kind"] == "conv"
    assert rows == sorted(rows, key=lambda r: -r["cpp_ms"])             # sorted by C++ time
    txt = render_bench(b, rows[:3])
    assert "images/s" in txt and "modelled cycles" in txt


def test_intake_report_measured_block_is_optional():
    import os, pytest
    from npuloop.intake import intake_report, render
    from npuloop.zoo import build_model
    m = build_model(dict(arch="resnet", depth=8, width=4, act="relu")).eval()
    rep = intake_report(m, "edge-10tops", with_prescriptions=False)
    assert rep["measured"] is None and "Measured on this host" not in render(rep)
    data = os.environ.get("NPULOOP_DATA", "/home/user/data/cifar10.npz")
    if not os.path.exists(data):
        pytest.skip("CIFAR-10 npz not available")
    from npuloop.zoo import CIFAR10NPZ
    rep = intake_report(m, "edge-10tops", with_prescriptions=False, bench=8, bench_data=CIFAR10NPZ(data))
    mb = rep["measured"]
    assert mb["batch"] == 8 and set(mb["engines"]) == {"numpy", "cpp"} and mb["top_nodes"]
    assert "Measured on this host" in render(rep)
