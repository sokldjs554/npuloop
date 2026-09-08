"""Concat and attention: tracing, quantization, and bit-exact integer execution of the new ops."""
import numpy as np
import pytest
import torch

from npuloop.graph.ir import trace, run_reference
from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.cpp_engine import CppEngine
from npuloop.intengine.numpy_engine import quantize_input
from npuloop.intengine.requant import isqrt64, round_div
from npuloop.lint import lint
from npuloop.npu.cost import estimate
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.zoo.models import build_model

VIT = dict(arch="vit", dim=64, depth=2, heads=2, patch=8, mlp_ratio=2)
INCEPTION = dict(arch="inception", width=16)


@pytest.fixture(params=[VIT, INCEPTION], ids=["vit", "inception"])
def model(request):
    torch.manual_seed(0)
    return build_model(request.param).eval()


def _quantized(model, scheme="npu-default", batches=3):
    torch.manual_seed(1)
    qm = prepare(model, PRESET_SCHEMES[scheme])
    calibrate(qm, [torch.randn(8, 3, 32, 32) for _ in range(batches)])
    return qm


def test_reference_matches_torch(model):
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = model(x).numpy()
    got = run_reference(trace(model), x.numpy())["output"]
    assert np.abs(ref - got).max() < 2e-4


def test_integer_engine_tracks_fake_quant(model):
    qm = _quantized(model)
    ig = export_int_graph(qm)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        fake = qm(x).numpy()
    got = NumpyEngine(ig).predict(x.numpy())
    assert np.abs(fake - got).max() < 0.15
    assert (fake.argmax(1) == got.argmax(1)).all()


def test_cpp_engine_is_bit_exact(model):
    ig = export_int_graph(_quantized(model))
    codes = quantize_input(torch.randn(3, 3, 32, 32).numpy(), ig.input_q)
    a, b = NumpyEngine(ig).run(codes, True), CppEngine(ig).run(codes, True)
    assert set(a) == set(b)
    for k in a:
        assert a[k].shape == b[k].shape, k
        assert np.array_equal(a[k], b[k]), k


def test_softmax_rows_sum_to_one(model):
    ig = export_int_graph(_quantized(model))
    sm = [n for n in ig.nodes if n.op == "softmax"]
    if not sm:
        pytest.skip("no softmax in this model")
    codes = quantize_input(torch.randn(2, 3, 32, 32).numpy(), ig.input_q)
    vals = NumpyEngine(ig).run(codes, True)
    for n in sm:
        probs = (vals[n.name].astype(np.float64) - n.out_q.zero_point) * n.out_q.scale
        total = probs.sum(axis=n.attrs["dim"])
        assert np.abs(total - 1.0).max() < 0.02          # int8 probabilities, 1 LSB per element


def test_attention_costs_more_without_a_vector_unit():
    g = trace(build_model(VIT).eval())
    relaxed = estimate(g, "edge-10tops").total_cycles
    strict = estimate(g, "edge-10tops-strict").total_cycles
    assert strict > 5 * relaxed                          # gelu/softmax/layernorm fall back to the host
    r = lint(g, "edge-10tops-strict")
    assert any(f.check == "dynamic-op-support" and f.severity == "high" for f in r.findings)
    assert any(f.check == "dynamic-matmul" for f in r.findings)


def test_matmul_pays_fill_drain_per_head():
    """A head-group matmul cannot amortize the weight load, so cycles scale with the head count."""
    g = trace(build_model(VIT).eval())
    mm = [l for l in estimate(g, "edge-10tops").layers if l.kind == "matmul"]
    assert mm and all(l.weight_tiles >= VIT["heads"] for l in mm)


def test_concat_requantizes_every_input():
    ig = export_int_graph(_quantized(build_model(INCEPTION).eval()))
    cat = [n for n in ig.nodes if n.op == "concat"]
    assert cat
    for n in cat:
        assert len(n.in_mult) == len(n.inputs) and len(n.in_shift) == len(n.inputs)
        assert all(m > 0 for m in n.in_mult)


def test_isqrt_and_round_div_are_exact():
    xs = np.array([0, 1, 2, 3, 4, 99, 100, 101, 10 ** 6, 10 ** 12 + 7, (1 << 40) - 1], dtype=np.int64)
    assert np.array_equal(isqrt64(xs), np.array([int(np.floor(np.sqrt(float(v)))) if v < 2 ** 40 else 1048575 for v in xs]))
    assert np.array_equal(isqrt64(xs) ** 2 <= xs, np.ones(len(xs), bool))
    assert np.array_equal((isqrt64(xs) + 1) ** 2 > xs, np.ones(len(xs), bool))
    num = np.array([7, -7, 5, -5, 0], dtype=np.int64)
    assert np.array_equal(round_div(num, np.int64(2)), np.array([4, -4, 3, -3, 0]))
