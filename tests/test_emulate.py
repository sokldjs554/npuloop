"""Integer-faithful LayerNorm emulation inside the fake-quant graph (npuloop.quant.emulate)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from npuloop.intengine import export_int_graph, NumpyEngine
from npuloop.intengine.verify import compare
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.quant.emulate import IntLayerNormEmu, emulate_integer_layernorm, set_layernorm_emulation
from npuloop.zoo.models import build_model

VIT = dict(arch="vit", dim=64, depth=2, heads=2, patch=8, mlp_ratio=2)


@pytest.fixture
def vit_q():
    torch.manual_seed(0)
    m = build_model(VIT).eval()
    torch.manual_seed(1)
    qm = prepare(m, PRESET_SCHEMES["npu-default"])
    calibrate(qm, [torch.randn(8, 3, 32, 32) for _ in range(3)])
    return qm


def test_emulated_layernorm_is_bit_exact_with_the_engine(vit_q):
    ig = export_int_graph(vit_q)
    x = torch.randn(6, 3, 32, 32)
    rows_before, _ = compare(vit_q, ig, x)
    assert any(r.op == "layernorm" and r.local_mismatch_frac > 0 for r in rows_before)   # float LN does differ
    assert emulate_integer_layernorm(vit_q) == 5                                           # 2 blocks x 2 + final norm
    rows_after, _ = compare(vit_q, export_int_graph(vit_q), x)
    ln = [r for r in rows_after if r.op == "layernorm"]
    assert len(ln) == 5 and all(r.local_mismatch_frac == 0.0 and r.local_max_abs == 0 for r in ln)


def test_export_is_unchanged_by_emulation(vit_q):
    a = export_int_graph(vit_q)
    emulate_integer_layernorm(vit_q)
    b = export_int_graph(vit_q)
    for na, nb in zip(a.nodes, b.nodes):
        assert (na.name, na.op) == (nb.name, nb.op)
        if na.op == "layernorm":
            assert np.array_equal(na.mult, nb.mult) and np.array_equal(na.shift, nb.shift)
            assert np.array_equal(na.bias_int, nb.bias_int) and na.attrs["eps_int"] == nb.attrs["eps_int"]
    codes = np.random.default_rng(0).integers(0, 256, size=(2, 3, 32, 32))
    assert np.array_equal(NumpyEngine(a).run(codes), NumpyEngine(b).run(codes))


def test_following_quantizer_is_idempotent_and_float_path_survives(vit_q):
    emulate_integer_layernorm(vit_q)
    emu = next(m for m in vit_q.modules() if isinstance(m, IntLayerNormEmu))
    s_in, zp_in = emu.in_fq.qparams()
    q = torch.randint(emu.in_fq.qmin, emu.in_fq.qmax + 1, (3, 16, 64))
    x = ((q - zp_in) * s_in).float()                    # what the preceding quantizer emits
    with torch.no_grad():
        y = emu(x)
        assert torch.equal(emu.out_fq(y), y)             # re-quantizing the emulated output changes nothing
        emu.active = False
        assert torch.allclose(emu(x), nn.functional.layer_norm(x, emu.normalized_shape, emu.weight, emu.bias, emu.eps))
    set_layernorm_emulation(vit_q, True)
    assert emu.active


def test_training_and_calibration_fall_back_to_float(vit_q):
    """The integer path is a NumPy round trip with no gradient, so it must not be taken while training."""
    emulate_integer_layernorm(vit_q)
    emu = next(m for m in vit_q.modules() if isinstance(m, IntLayerNormEmu))
    s_in, zp_in = emu.in_fq.qparams()
    x = ((torch.randint(emu.in_fq.qmin, emu.in_fq.qmax + 1, (2, 16, 64)) - zp_in) * s_in).float()
    vit_q.eval()
    with torch.no_grad():
        integer_path = emu(x)
    vit_q.train()
    with torch.no_grad():
        assert not torch.equal(emu(x), integer_path)                       # train() -> float LayerNorm
        emu.in_fq.calibrating = emu.out_fq.calibrating = True
        vit_q.eval()
        assert not torch.equal(emu(x), integer_path)                       # calibrating -> float LayerNorm
        emu.in_fq.calibrating = emu.out_fq.calibrating = False
        assert torch.equal(emu(x), integer_path)
    vit_q.train()
    xr = x.clone().requires_grad_(True)
    emu(xr).sum().backward()
    assert xr.grad is not None and float(xr.grad.abs().sum()) > 0          # gradients survive the fallback


def test_cached_integer_parameters_follow_gamma_beta_and_eps(vit_q):
    emulate_integer_layernorm(vit_q)
    emu = next(m for m in vit_q.modules() if isinstance(m, IntLayerNormEmu))
    s_in, zp_in = emu.in_fq.qparams()
    x = ((torch.randint(emu.in_fq.qmin, emu.in_fq.qmax + 1, (2, 16, 64)) - zp_in) * s_in).float()
    vit_q.eval()
    with torch.no_grad():
        before = emu(x).clone()
        emu.weight.mul_(1.7)
        assert not torch.equal(emu(x), before)        # gamma feeds the fixed-point multipliers
        emu.weight.div_(1.7)
        assert torch.equal(emu(x), before)
        emu.bias.add_(0.5)
        assert not torch.equal(emu(x), before)        # beta feeds bias_int


def test_emulation_needs_quantizers_around_the_layernorm():
    torch.manual_seed(0)
    m = build_model(VIT).eval()
    with pytest.raises(ValueError):
        emulate_integer_layernorm(torch.fx.symbolic_trace(m))      # no quantizers at all
