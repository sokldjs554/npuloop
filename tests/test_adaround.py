import numpy as np
import torch
import torch.nn as nn
from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
from npuloop.quant.adaround import adaround
from npuloop.quant.fake import QConv2d, QLinear
from npuloop.intengine import export_int_graph


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(8 * 8 * 8, 10)).eval()


def _calib(n=4, bs=16):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(bs, 3, 8, 8, generator=g) for _ in range(n)]


def test_adaround_lowers_layer_error_and_flips_weights():
    """A learned rounding that never differs from round-to-nearest would make the rest of this file vacuous."""
    qm = prepare(_net(), PRESET_SCHEMES["npu-default"]); calibrate(qm, _calib())
    report = adaround(qm, _calib(), iters=600)
    flipped = [L["flipped_frac"] for L in report["layers"]]
    assert max(flipped) > 0.01, f"AdaRound changed almost nothing: {flipped}"
    # the later layers, where the quantization error actually lives, must improve
    improved = [L["mse_after"] <= L["mse_before"] * 1.01 for L in report["layers"][1:]]
    assert all(improved), [(L["layer"], L["mse_before"], L["mse_after"]) for L in report["layers"]]


def test_learned_rounding_reaches_the_integer_program():
    """The point of writing the rounding to w_round: what was optimized is what the device executes.

    The fake-quantization weight must be exactly int_weight() * scale, and the integer graph exported from the
    model must carry those same codes -- otherwise AdaRound would improve the simulator and leave the integer
    program on round-to-nearest, and E19's question would be unanswerable.
    """
    qm = prepare(_net(), PRESET_SCHEMES["npu-default"]); calibrate(qm, _calib())
    layers = [m for m in qm.modules() if isinstance(m, (QConv2d, QLinear))]
    rtn = [m.int_weight().clone() for m in layers]
    adaround(qm, _calib(), iters=600)

    changed = 0
    for m, before in zip(layers, rtn):
        codes = m.int_weight()
        changed += int((codes != before).sum())
        s = (m.w_scale if m.per_channel else m.w_scale.expand(m.weight.shape[0]))
        s = s.reshape((-1,) + (1,) * (m.weight.dim() - 1))
        assert torch.allclose(m.quantized_weight(), codes.to(torch.float32) * s, atol=1e-6)
        assert int(codes.min()) >= m.qmin and int(codes.max()) <= m.qmax
    assert changed > 0, "no weight code changed, so the test proves nothing"

    # The exported graph must carry those codes. Its compute nodes keep no module name, so match them to the
    # quantized layers by execution order -- the export walks the traced graph in the same order.
    ig = export_int_graph(qm, input_shape=(3, 8, 8))
    compute = [n for n in ig.nodes if n.op in ("conv", "linear")]
    assert len(compute) == len(layers), (len(compute), len(layers))
    for node, m in zip(compute, layers):
        assert np.array_equal(node.w_int, m.int_weight().numpy()), node.name
