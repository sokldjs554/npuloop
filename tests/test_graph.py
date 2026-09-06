import numpy as np
import torch
import pytest
from npuloop.graph import trace, run_reference, UnsupportedOpError


def test_trace_matches_pytorch_after_bn_fold(small_resnet, small_silu_resnet, small_mobilenet, batch):
    for m in (small_resnet, small_silu_resnet, small_mobilenet):
        g = trace(m)
        ref = m(batch).detach().numpy()
        out = run_reference(g, batch.numpy())["output"]
        assert np.abs(out - ref).max() < 1e-4
        assert all(n.attrs.get("bn_folded") for n in g.nodes if n.op == "conv")


def test_mac_count_resnet20():
    from npuloop.zoo import ResNetCIFAR
    g = trace(ResNetCIFAR().eval())
    assert g.total_macs == 40_813_184          # well-known ~41M MACs for ResNet-20 on 32x32
    assert g.total_params == 271_690           # 272,474 params minus BN params folded away


def test_depthwise_flag(small_mobilenet):
    g = trace(small_mobilenet)
    dw = [n for n in g.nodes if n.op == "conv" and n.attrs["depthwise"]]
    assert len(dw) == 5
    assert all(n.weight.shape[1] == 1 for n in dw)


def test_unsupported_op_is_reported():
    class Bad(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.c = torch.nn.Conv2d(3, 4, 3, padding=1)
        def forward(self, x):
            return torch.sigmoid(self.c(x)) * 2
    with pytest.raises(UnsupportedOpError):
        trace(Bad().eval())
