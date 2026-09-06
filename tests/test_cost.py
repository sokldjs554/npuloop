import math
import pytest
from npuloop.graph import trace
from npuloop.npu import estimate, gemm_cycles, NPUSpec, PRESETS, get_spec


def test_gemm_cycles_single_core_formula():
    spec = NPUSpec("t", pe_rows=32, pe_cols=32, cores=1, fill_drain=True)
    cyc, split, tiles = gemm_cycles(m=100, k=64, n=48, spec=spec)
    assert tiles == 2 * 2
    assert cyc == 4 * (100 + 64)
    spec2 = NPUSpec("t", pe_rows=32, pe_cols=32, cores=1, fill_drain=False)
    assert gemm_cycles(100, 64, 48, spec2)[0] == 4 * 100


def test_multicore_picks_best_split():
    spec = NPUSpec("t", pe_rows=64, pe_cols=64, cores=4, fill_drain=False)
    # small Cout: splitting N gives 1 tile per core anyway -> M split must win
    cyc, split, _ = gemm_cycles(m=1024, k=144, n=16, spec=spec)
    assert split == "M" and cyc == 3 * 256
    # big Cout: N split wins or ties
    cyc2, split2, _ = gemm_cycles(m=64, k=576, n=512, spec=spec)
    assert cyc2 <= 9 * 2 * 64 + 1


def test_utilization_bounds_and_totals(small_resnet):
    g = trace(small_resnet)
    r = estimate(g, "edge-10tops")
    assert 0 < r.array_utilization <= 1
    assert math.isclose(r.total_cycles, sum(l.cycles for l in r.layers))
    assert r.total_macs == g.total_macs
    for l in r.layers:
        assert l.cycles >= 0 and l.array_util <= 1.0 + 1e-9


def test_aligned_channels_use_array_better():
    from npuloop.zoo import ResNetCIFAR
    spec = "edge-10tops"
    r16 = estimate(trace(ResNetCIFAR(widths=(16, 32, 64)).eval()), spec)
    r64 = estimate(trace(ResNetCIFAR(widths=(64, 64, 64)).eval()), spec)
    assert r64.array_utilization > r16.array_utilization


def test_strict_npu_penalises_silu(small_silu_resnet):
    g = trace(small_silu_resnet)
    lut = estimate(g, "edge-10tops"); strict = estimate(g, "edge-10tops-strict")
    assert strict.total_cycles > 5 * lut.total_cycles
    assert strict.breakdown().get("act-fallback", 0) > 0 and lut.breakdown().get("act-fallback", 0) == 0


def test_depthwise_engine_vs_array(small_mobilenet):
    g = trace(small_mobilenet)
    with_engine = estimate(g, "edge-10tops"); no_engine = estimate(g, "edge-10tops-strict")
    dw_e = sum(l.cycles for l in with_engine.layers if l.kind == "dwconv")
    dw_a = sum(l.cycles for l in no_engine.layers if l.kind == "dwconv")
    assert dw_a > dw_e


def test_presets_have_sane_peak_tops():
    assert 9 < PRESETS["edge-10tops"].peak_tops < 11
    assert 75 < PRESETS["pcie-80tops"].peak_tops < 85
    with pytest.raises(KeyError):
        get_spec("nope")
