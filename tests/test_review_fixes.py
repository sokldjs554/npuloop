"""Regression tests for defects found in the adversarial code review."""
import numpy as np
import torch
import torch.nn as nn
import pytest
from npuloop.quant import prepare, calibrate, QScheme, weight_qparams
from npuloop.intengine import export_int_graph, NumpyEngine, quantize_multiplier, multiply_by_quantized_multiplier
from npuloop.intengine.verify import compare
from npuloop.intengine.cpp_engine import CppEngine, load_library
from npuloop.intengine.numpy_engine import quantize_input


class SharedProducer(nn.Module):
    """conv output used by BOTH a ReLU branch and a raw branch: the ReLU must not be fused into the conv."""
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1); self.act = nn.ReLU(); self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.drop = nn.Dropout(0.1); self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(8, 10)
    def forward(self, x):
        y = self.c(x)
        z = self.c2(self.act(y)) + y            # y consumed raw AND through ReLU
        z = self.drop(z)
        return self.fc(self.pool(z).flatten(1))  # call_method flatten


def test_relu_with_shared_producer_becomes_lut(calib_batches, batch):
    torch.manual_seed(0)
    m = SharedProducer().eval()
    qm = prepare(m, QScheme()); calibrate(qm, calib_batches)
    ig = export_int_graph(qm)
    assert any(n.op == "lut" and n.attrs["kind"] == "relu" for n in ig.nodes)
    conv = ig["c"]
    assert "fused_act" not in conv.attrs
    rows, summ = compare(qm, ig, batch)
    assert all(r.local_max_abs <= 1 for r in rows) and summ["logit_max_abs_diff"] < 0.5


def test_zero_weight_channel_keeps_its_bias(calib_batches, batch):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(4, 10)).eval()
    with torch.no_grad():
        m[0].weight[1].zero_(); m[0].bias[1] = 0.7            # dead channel with a constant output of 0.7
    s, _ = weight_qparams(m[0].weight)
    assert s[1] > 1e-6
    qm = prepare(m, QScheme()); calibrate(qm, calib_batches)
    ig = export_int_graph(qm)
    conv = next(n for n in ig.nodes if n.op == "conv")
    out = NumpyEngine(ig).run(quantize_input(batch.numpy(), ig.input_q), return_all=True)[conv.name]
    q = conv.out_q
    val = (out[:, 1] - q.zero_point) * q.scale
    assert np.allclose(val, 0.7, atol=2 * q.scale)


def test_load_checkpoint_roundtrip_pruned(small_resnet, tmp_path):
    from npuloop.prune import prune
    from npuloop.zoo import load_checkpoint
    pm, _ = prune(small_resnet, ratio=0.5, strategy="aligned", align=8)
    torch.save({"config": pm.config, "state_dict": pm.state_dict()}, tmp_path / "p.pt")
    rb = load_checkpoint(str(tmp_path / "p.pt"))
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(rb(x), pm(x))


def test_fallback_activation_charges_one_write_plus_consumer_read(small_silu_resnet):
    from npuloop.graph import trace
    from npuloop.npu import estimate
    r = estimate(trace(small_silu_resnet), "edge-10tops-strict")
    fb = [l for l in r.layers if l.kind == "act-fallback"][0]
    assert fb.dram_bytes == [n for n in trace(small_silu_resnet).nodes if n.name == fb.name][0].n_elements


def test_mbqm_positive_shift_engines_agree():
    lib = load_library()
    rng = np.random.default_rng(5)
    x = rng.integers(-2 ** 30, 2 ** 30, 2000)
    for m in [1.7, 5.0, 40.0]:
        q, s = quantize_multiplier(m)
        y = multiply_by_quantized_multiplier(x, q, s)
        for v, yy in zip(x[:200], y[:200]):
            assert lib.test_mbqm(int(v), q, s, 0) == yy
    # half_even at exponent 31 must not overflow in C++
    for v in [1, -1, 2 ** 31 - 1, -(2 ** 31)]:
        assert lib.test_rdbpot(v, 31, 1) == int(__import__("npuloop.intengine.requant", fromlist=["_rdbpot"])._rdbpot(np.array([v]), np.array([31]), "half_even")[0])


def test_qat_unfrozen_minmax_ranges_move(small_resnet, calib_batches):
    from npuloop.quant import set_mode, quantizers
    qm = prepare(small_resnet, QScheme()); calibrate(qm, calib_batches)
    fq = quantizers(qm)[1]
    before = float(fq.scale)
    set_mode(qm, enabled=True, w_enabled=True, calibrating=True)
    qm(calib_batches[0] * 10.0)                     # much larger activations -> min/max range must grow
    assert float(fq.scale) > before * 2
    set_mode(qm, calibrating=False)


def test_mobilenet_importance_uses_dw_bn(small_mobilenet):
    from npuloop.prune import find_groups, importance
    g = find_groups(small_mobilenet)[0]
    m = small_mobilenet
    dw_bn = m.get_submodule(g.dw_bn)
    with torch.no_grad():
        dw_bn.weight[0] = 0.0                        # channel 0 dead after the depthwise BN
    imp = importance(m, g)
    assert imp[0] == 0 and imp[1:].min() > 0
