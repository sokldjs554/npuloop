import numpy as np
import torch
from npuloop.graph import trace
from npuloop.lint import lint


def test_scores_in_range_and_silu_flagged(small_resnet, small_silu_resnet, batch):
    r_relu = lint(trace(small_resnet), "edge-10tops", batch.numpy())
    r_silu = lint(trace(small_silu_resnet), "edge-10tops", batch.numpy())
    for r in (r_relu, r_silu):
        for v in r.scores.values():
            assert 0 <= v <= 100
    assert "activation-support" not in r_relu.by_check()
    assert len(r_silu.by_check()["activation-support"]) > 0
    assert r_silu.scores["quant_robustness"] < r_relu.scores["quant_robustness"]


def test_strict_spec_makes_silu_high_severity(small_silu_resnet):
    r = lint(trace(small_silu_resnet), "edge-10tops-strict")
    sev = {f.severity for f in r.by_check()["activation-support"]}
    assert sev == {"high"}
    assert r.scores["efficiency"] < lint(trace(small_silu_resnet), "edge-10tops").scores["efficiency"]


def test_weight_range_disparity_detected():
    from npuloop.zoo import ResNetCIFAR
    m = ResNetCIFAR(depth=8, width=8).eval()
    conv = m.blocks[0].conv1
    with torch.no_grad():
        conv.weight[0] *= 200.0     # one channel dominates -> per-tensor weights would crush the rest
    r = lint(trace(m), "edge-10tops")
    hits = [f for f in r.by_check().get("weight-range-disparity", []) if f.layer == "blocks_0_conv1"]
    assert hits and hits[0].severity == "high"


def test_markdown_and_json(small_mobilenet, batch):
    r = lint(trace(small_mobilenet), "edge-10tops", batch.numpy())
    md = r.markdown(); d = r.to_dict()
    assert "depthwise" in md and d["scores"]["overall"] == r.score
