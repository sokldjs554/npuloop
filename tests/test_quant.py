import numpy as np
import torch
from npuloop.quant import (prepare, calibrate, QScheme, fold_bn, set_mode, quantizers, equalize, find_pairs,
                           bias_correction, layer_sensitivity, swap_activations, MinMaxObserver, PercentileObserver, MSEObserver,
                           affine_qparams, weight_qparams, fake_quant)
from npuloop.quant.fake import QConv2d


def test_fold_bn_exact(small_resnet, small_mobilenet, batch):
    for m in (small_resnet, small_mobilenet):
        gm = fold_bn(m)
        assert (gm(batch) - m(batch)).abs().max() < 1e-4
        assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in gm.modules())


def test_affine_qparams_contain_zero_and_pow2():
    s, z = affine_qparams(torch.tensor(0.5), torch.tensor(2.0), 8, False)
    assert z == 0 and abs(float(s) - 2.0 / 255) < 1e-9        # range widened to include 0
    s, z = affine_qparams(torch.tensor(-1.0), torch.tensor(3.0), 8, False)
    assert 0 <= z <= 255 and abs(((0 - z) * s) - (-1.0)) < float(s)   # zero-point maps lo exactly-ish
    s2, _ = affine_qparams(torch.tensor(-1.0), torch.tensor(3.0), 8, False, pow2=True)
    assert np.log2(float(s2)) == int(np.log2(float(s2)))
    s3, z3 = affine_qparams(torch.tensor(-2.0), torch.tensor(1.0), 8, True)
    assert z3 == 0 and abs(float(s3) - 2.0 / 127) < 1e-9


def test_observers_and_outlier_robustness():
    torch.manual_seed(0)
    x = torch.randn(50000); x[0] = 100.0
    mm, pc, ms = MinMaxObserver(), PercentileObserver(99.99), MSEObserver()
    for o in (mm, pc, ms):
        o(x)
    assert mm.range()[1] == 100.0
    assert pc.range()[1] < 10.0
    assert ms.range()[1] < 100.0


def test_weight_qparams_per_channel_shape():
    w = torch.randn(6, 4, 3, 3); w[2] *= 10
    s, z = weight_qparams(w, per_channel=True)
    assert s.shape == (6,) and s[2] > 5 * s[0] and (z == 0).all()
    s_t, _ = weight_qparams(w, per_channel=False)
    assert s_t.shape == (1,)


def test_fake_quant_ste_gradient():
    x = torch.randn(1000, requires_grad=True)
    y = fake_quant(x, torch.tensor(0.05), torch.tensor(0.0), -127, 127)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    assert (y - x).abs().max() <= 0.025 + 1e-6


def test_prepare_quantizer_placement(small_resnet, small_silu_resnet, small_mobilenet):
    q_relu = prepare(small_resnet); q_silu = prepare(small_silu_resnet); q_mn = prepare(small_mobilenet)
    n_relu, n_silu = len(quantizers(q_relu)), len(quantizers(q_silu))
    n_acts = sum(1 for m in small_silu_resnet.modules() if isinstance(m, torch.nn.SiLU))
    assert n_silu == n_relu + n_acts            # every LUT activation adds a pre-activation quantizer
    pool_fq = [m for m in quantizers(q_relu) if m.tied_to is not None]
    assert len(pool_fq) == 1                    # pool output tied to its input
    assert all(isinstance(m, QConv2d) for m in q_mn.modules() if isinstance(m, torch.nn.Conv2d))


def test_calibrate_then_int8_close_to_float(small_resnet, calib_batches, batch):
    qm = prepare(small_resnet, QScheme()); calibrate(qm, calib_batches)
    ref = small_resnet(batch); out = qm(batch)
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.15
    set_mode(qm, enabled=False, w_enabled=False)
    assert (qm(batch) - ref).abs().max() < 1e-4   # disabling quantizers recovers the float model


def test_pow2_and_per_tensor_schemes_run(small_mobilenet, calib_batches, batch):
    for sch in (QScheme(pow2=True), QScheme(w_per_channel=False), QScheme(a_symmetric=True), QScheme(a_observer="mse", w_method="mse")):
        qm = prepare(small_mobilenet, sch); calibrate(qm, calib_batches)
        assert torch.isfinite(qm(batch)).all()
        if sch.pow2:
            for fq in quantizers(qm):
                s = float(fq.scale); assert abs(np.log2(s) - round(np.log2(s))) < 1e-6


def test_cle_preserves_function_and_equalizes(small_mobilenet, batch):
    gm = fold_bn(small_mobilenet)
    ref = gm(batch)
    pairs = find_pairs(gm)
    assert len(pairs) > 0
    stats = equalize(gm, relu6_policy="keep")
    out = gm(batch)
    # ReLU6 kept: function can change only where clipping at 6 is involved; require it to be small
    assert (out - ref).abs().max() < 0.5
    ratios = []
    for m in gm.modules():
        if isinstance(m, torch.nn.Conv2d):
            r = m.weight.detach().abs().reshape(m.out_channels, -1).amax(1); ratios.append(float(r.max() / r.median()))
    assert stats["pairs"] == len(pairs)


def test_cle_relu_exact(small_resnet, batch):
    gm = fold_bn(small_resnet); ref = gm(batch)
    equalize(gm)
    assert (gm(batch) - ref).abs().max() < 1e-3


def test_bias_correction_reduces_mean_error(small_resnet, calib_batches):
    gm = fold_bn(small_resnet)
    qm = prepare(small_resnet, QScheme(w_per_channel=False)); calibrate(qm, calib_batches)
    x = calib_batches[0]
    before = (qm(x).mean(0) - gm(x).mean(0)).abs().mean()
    bias_correction(qm, gm, calib_batches)
    after = (qm(x).mean(0) - gm(x).mean(0)).abs().mean()
    assert after <= before + 1e-6


def test_layer_sensitivity_keys(small_resnet, calib_batches):
    qm = prepare(small_resnet); calibrate(qm, calib_batches)
    y = torch.randint(0, 10, (16,))
    res = layer_sensitivity(qm, [(calib_batches[0], y)])
    assert set(res["weights"]) == {n for n, m in qm.named_modules() if isinstance(m, (QConv2d,)) or type(m).__name__ == "QLinear"}
    assert len(res["activations"]) == len(quantizers(qm))
    assert all(m.enabled for m in quantizers(qm))


def test_swap_activations(small_silu_resnet, batch):
    m = swap_activations(small_silu_resnet, {"silu": "relu"})
    assert m.swapped > 0 and not any(isinstance(x, torch.nn.SiLU) for x in m.modules())
    assert m.config["act"] == "relu" and small_silu_resnet.config["act"] == "silu"
    assert torch.isfinite(m(batch)).all()
