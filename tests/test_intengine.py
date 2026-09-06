import math
import numpy as np
import torch
import pytest
from npuloop.quant import prepare, calibrate, QScheme
from npuloop.intengine import export_int_graph, NumpyEngine, RequantConfig, quantize_multiplier, multiply_by_quantized_multiplier
from npuloop.intengine.requant import _srdhm, _rdbpot
from npuloop.intengine.numpy_engine import quantize_input, dequantize
from npuloop.intengine.verify import compare
from npuloop.intengine.cpp_engine import CppEngine, load_library


def ref_srdhm(a, b):
    if a == b == -2 ** 31:
        return 2 ** 31 - 1
    ab = a * b; nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    v = ab + nudge
    return int(v / (1 << 31)) if v >= 0 else -int((-v) / (1 << 31))


def ref_rdbpot(x, e):
    mask = (1 << e) - 1; rem = x & mask; thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def test_fixed_point_primitives_match_gemmlowp_reference():
    rng = np.random.default_rng(0)
    a = rng.integers(-2 ** 31, 2 ** 31, 2000); b = rng.integers(2 ** 30, 2 ** 31, 2000)
    for x, y in zip(a, b):
        assert _srdhm(np.array([x]), np.array([y]))[0] == ref_srdhm(int(x), int(y))
    x = rng.integers(-2 ** 31, 2 ** 31, 2000); e = rng.integers(0, 31, 2000)
    for v, k in zip(x, e):
        assert _rdbpot(np.array([v]), np.array([k]))[0] == ref_rdbpot(int(v), int(k))
    assert _srdhm(np.array([-2 ** 31]), np.array([-2 ** 31]))[0] == 2 ** 31 - 1


def test_quantize_multiplier_roundtrip():
    for m in [1e-4, 0.00073, 0.3, 0.5, 0.999, 1.0, 1.7, 5.0, 123.4]:
        q, s = quantize_multiplier(m)
        assert 2 ** 30 <= q < 2 ** 31
        assert abs(q * 2.0 ** s / 2 ** 31 - m) / m < 1e-8
    assert quantize_multiplier(0.0) == (0, 0)
    q7, _ = quantize_multiplier(0.3, mult_bits=7)
    assert q7 % (1 << 24) == 0


def round_half_away(v):
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def test_requant_double_vs_single_rounding():
    """gemmlowp's legacy path rounds twice; TFLITE_SINGLE_ROUNDING rounds once. Both stay within 1 LSB of the
    exactly rounded product, but double rounding is off by one LSB far more often when the right shift is small."""
    rng = np.random.default_rng(1)
    acc = rng.integers(-3_000_000, 3_000_000, 100000)
    for m, expect_double in [(0.00073, 0.002), (0.31415926, 0.30), (1.7320508, 0.30), (0.0012345, 0.005)]:
        q, s = quantize_multiplier(m)
        ref = round_half_away(acc * m)
        y_double = multiply_by_quantized_multiplier(acc, q, s, "tflite")
        y_single = multiply_by_quantized_multiplier(acc, q, s, "single")
        assert np.abs(y_double - ref).max() <= 1 and np.abs(y_single - ref).max() <= 1
        assert (y_double != ref).mean() < expect_double
        assert (y_single != ref).mean() < 0.005        # only the q31 approximation of m can flip a near-tie
    q, s = quantize_multiplier(0.31415926)             # shift = -1: double rounding hurts most
    assert (multiply_by_quantized_multiplier(acc, q, s, "tflite") != round_half_away(acc * 0.31415926)).mean() > 0.1
    # rational multipliers create exact ties where half-even (fake-quant, torch.round) legitimately differs
    q, s = quantize_multiplier(0.3)
    y = multiply_by_quantized_multiplier(acc, q, s, "single")
    assert 0.02 < (y != np.rint(acc * 0.3)).mean() < 0.3


def test_rounding_modes_differ_only_at_ties():
    x = np.array([5, -5, 6, -6, 7, -7, 3, -3])   # /4 : 1.25, -1.25, 1.5, -1.5, 1.75, -1.75, .75, -.75
    assert _rdbpot(x, 2, "tflite").tolist() == [1, -1, 2, -2, 2, -2, 1, -1]
    assert _rdbpot(x, 2, "half_even").tolist() == [1, -1, 2, -2, 2, -2, 1, -1]
    assert _rdbpot(np.array([2, -2, 10, -10]), 2, "half_even").tolist() == [0, 0, 2, -2]   # ties -> even
    assert _rdbpot(np.array([2, -2, 10, -10]), 2, "tflite").tolist() == [1, -1, 3, -3]      # ties -> away
    assert _rdbpot(x, 2, "truncate").tolist() == [1, -1, 1, -1, 1, -1, 0, 0]
    assert _rdbpot(x, 2, "floor").tolist() == [1, -2, 1, -2, 1, -2, 0, -1]
    assert _rdbpot(x, 0, "half_even").tolist() == x.tolist()


@pytest.mark.parametrize("fixture", ["small_resnet", "small_silu_resnet", "small_mobilenet"])
def test_int_engine_agrees_with_fake_quant(fixture, request, calib_batches, batch):
    m = request.getfixturevalue(fixture)
    qm = prepare(m, QScheme()); calibrate(qm, calib_batches)
    ig = export_int_graph(qm)
    rows, summ = compare(qm, ig, batch)
    # local (teacher-forced) mismatch must be tiny and at most 1 LSB per op
    for r in rows:
        assert r.local_max_abs <= 1, r
        assert r.local_mismatch_frac < 0.02, r
    assert summ["logit_max_abs_diff"] < 0.5
    eng = NumpyEngine(ig)
    logits = eng.predict(batch.numpy())
    assert np.abs(logits - qm(batch).detach().numpy()).max() < 0.5


def test_saturation_counter_and_narrow_accumulator(small_resnet, calib_batches, batch):
    qm = prepare(small_resnet); calibrate(qm, calib_batches)
    ig32 = export_int_graph(qm, RequantConfig(acc_bits=32))
    e32 = NumpyEngine(ig32); e32.predict(batch.numpy())
    assert sum(e32.saturations.values()) == 0
    ig12 = export_int_graph(qm, RequantConfig(acc_bits=12))
    e12 = NumpyEngine(ig12); e12.predict(batch.numpy())
    assert sum(e12.saturations.values()) > 0


def test_cpp_engine_bit_exact(small_resnet, small_silu_resnet, small_mobilenet, calib_batches, batch):
    load_library()
    for m in (small_resnet, small_silu_resnet, small_mobilenet):
        for cfg in (RequantConfig(), RequantConfig(rounding="single"), RequantConfig(rounding="half_even", mult_bits=15, acc_bits=16)):
            qm = prepare(m); calibrate(qm, calib_batches)
            ig = export_int_graph(qm, cfg)
            codes = quantize_input(batch.numpy(), ig.input_q)
            vn = NumpyEngine(ig).run(codes, return_all=True)
            vc = CppEngine(ig).run(codes, return_all=True)
            for k in vn:
                assert np.array_equal(vn[k], vc[k]), (type(m).__name__, cfg, k)


def test_cpp_primitives_match_numpy():
    lib = load_library()
    rng = np.random.default_rng(2)
    for _ in range(500):
        a, b = int(rng.integers(-2 ** 31, 2 ** 31)), int(rng.integers(2 ** 30, 2 ** 31))
        assert lib.test_srdhm(a, b) == _srdhm(np.array([a]), np.array([b]))[0]
        x, e = int(rng.integers(-2 ** 31, 2 ** 31)), int(rng.integers(0, 31))
        for name, code in [("tflite", 0), ("half_even", 1), ("truncate", 2), ("floor", 3)]:
            assert lib.test_rdbpot(x, e, code) == _rdbpot(np.array([x]), np.array([e]), name)[0], (x, e, name)
        v, mult, sh = int(rng.integers(-2 ** 20, 2 ** 20)), int(rng.integers(2 ** 30, 2 ** 31)), int(rng.integers(-20, 3))
        for name, code in [("tflite", 0), ("single", 4)]:
            assert lib.test_mbqm(v, mult, sh, code) == multiply_by_quantized_multiplier(np.array([v]), mult, sh, name)[0], (v, mult, sh, name)


def test_float32_chunked_mac_is_exact_at_worst_case():
    """Adversarial: all inputs at the extreme code, weights at +-127, K=576 (the layer where a naive float32
    conv would exceed 2^24) -> must equal exact int64 arithmetic."""
    from npuloop.intengine.graph import IntNode, QParams
    torch.manual_seed(0)
    x = np.full((2, 64, 8, 8), 255, dtype=np.int64)
    w = np.full((32, 64, 3, 3), 127, dtype=np.int64)          # no cancellation: sum = 576*255*127 = 18.65M > 2^24
    w[1] = -127
    n = IntNode("c", "conv", ["x"], attrs=dict(stride=(1, 1), padding=(1, 1), groups=1))
    n.w_int = w
    class G:  # minimal stand-in for IntGraph
        requant = RequantConfig()
        def __getitem__(self, k): return IntNode("x", "input", [], out_q=QParams(1.0, 0, 0, 255))
    eng = NumpyEngine.__new__(NumpyEngine); eng.g = G(); eng.cfg = RequantConfig(); eng.saturations = {}
    acc = eng._mac(n, x)
    ref = torch.nn.functional.conv2d(torch.from_numpy(x.astype(np.float64)), torch.from_numpy(w.astype(np.float64)), None, 1, 1).numpy().astype(np.int64)
    assert np.array_equal(acc, ref)
    assert np.abs(ref).max() > 2 ** 24      # the case really is beyond naive float32 range
