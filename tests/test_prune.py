import torch
from npuloop.prune import prune, prune_cost_greedy, find_groups, keep_counts, rebuild_from_config
from npuloop.graph import trace
from npuloop.npu import estimate


def test_groups_and_aligned_counts(small_resnet, small_mobilenet):
    g_r = find_groups(small_resnet); g_m = find_groups(small_mobilenet)
    assert len(g_r) == 3 and len(g_m) == 4          # depth-8 resnet: 3 blocks; mobilenet: expand blocks only
    assert [g.channels for g in g_r] == [8, 16, 32]
    assert keep_counts(g_r, 0.5, "aligned", align=16) == [8, 16, 16]     # never above the group's size
    assert keep_counts(g_r, 0.5, "uniform") == [8, 8, 16]               # min_channels=8 floor


def test_prune_runs_and_reduces_macs(small_resnet, small_mobilenet, batch):
    for m in (small_resnet, small_mobilenet):
        base = estimate(trace(m), "edge-10tops").total_cycles
        pm, groups = prune(m, ratio=0.5, strategy="uniform")
        assert torch.isfinite(pm(batch)).all()
        assert not pm.training and all(not x.training for x in pm.modules())
        # MACs always drop; cycles can only drop or stay (channels below the array width save no cycles)
        assert estimate(trace(pm), "edge-10tops").total_cycles <= base
        assert trace(pm).total_macs < trace(m).total_macs
        assert pm.config["pruned_channels"] == [g.channels for g in groups]


def test_pruning_below_array_width_saves_no_cycles_on_wide_array():
    """FLOPs vs cycles: on a 64x64 array a 1x1 conv with 48 or 24 output channels costs the same."""
    from npuloop.npu import gemm_cycles, NPUSpec
    spec = NPUSpec("t", pe_rows=64, pe_cols=64, cores=1)
    assert gemm_cycles(256, 64, 48, spec)[0] == gemm_cycles(256, 64, 24, spec)[0]
    assert gemm_cycles(256, 64, 128, spec)[0] == 2 * gemm_cycles(256, 64, 64, spec)[0]
    assert gemm_cycles(256, 64, 96, spec)[0] == gemm_cycles(256, 64, 128, spec)[0]


def test_rebuild_from_config_roundtrip(small_resnet, batch):
    pm, _ = prune(small_resnet, ratio=0.5, strategy="aligned", align=8)
    rb = rebuild_from_config(pm.config); rb.load_state_dict(pm.state_dict()); rb.eval()
    assert torch.allclose(rb(batch), pm(batch))


def test_cost_greedy_hits_target(small_resnet):
    pm, groups, hist = prune_cost_greedy(small_resnet, "edge-10tops", target_ratio=0.8, align=8, min_channels=8)
    assert hist[-1]["cycles"] <= hist[0]["cycles"] * 0.8 + 1 or all(g.channels == 8 for g in groups)
    assert all(g.channels % 8 == 0 for g in groups)
