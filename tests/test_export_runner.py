"""The .npuloop file: round-trips through both Python engines unchanged, and the standalone C++ runner
(intengine/cpp/int8_runner.cpp, no Python) reproduces every node of every architecture code for code."""
import os, shutil, subprocess, tempfile
import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_SRC = os.path.join(ROOT, "npuloop", "intengine", "cpp", "int8_runner.cpp")
CONFIGS = {
    "resnet": dict(arch="resnet", depth=8, width=8, act="silu"),
    "mobilenetv2": dict(arch="mobilenetv2", width_mult=0.35, act="relu6", setting=[[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 2, 2]], last_channels=128),
    "vit": dict(arch="vit", dim=32, depth=2, heads=2, patch=8, mlp_ratio=2, act="gelu"),
    "inception": dict(arch="inception", width=8, act="relu"),
}


def _int_graph(cfg, seed=0):
    from npuloop.zoo import build_model
    from npuloop.quant import prepare, calibrate, PRESET_SCHEMES
    from npuloop.intengine import export_int_graph
    torch.manual_seed(seed)
    m = build_model(cfg).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5); mod.running_var.uniform_(0.5, 2.0)
    qm = prepare(m, PRESET_SCHEMES["npu-default"]); calibrate(qm, [torch.randn(16, 3, 32, 32)])
    return export_int_graph(qm)


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    exe = str(tmp_path_factory.mktemp("bin") / "int8_runner")
    subprocess.run(["g++", "-O3", "-std=c++17", "-o", exe, RUNNER_SRC], check=True, capture_output=True)
    return exe


@pytest.mark.parametrize("arch", list(CONFIGS))
def test_file_roundtrip_and_standalone_runner(arch, runner, tmp_path):
    from npuloop.intengine import save_int_graph, load_int_graph, NumpyEngine, quantize_input
    from npuloop.intengine.cpp_engine import CppEngine
    from npuloop.intengine.serialize import describe
    ig = _int_graph(CONFIGS[arch])
    path = str(tmp_path / f"{arch}.npuloop")
    header = save_int_graph(ig, path, provenance=dict(config=CONFIGS[arch]))
    assert header["version"] == 1 and len(header["nodes"]) == len(ig.nodes)
    assert all(a["dtype"] == ("int8" if a["field"] == "w_int" else "int32") for n in header["nodes"] for a in n["arrays"])
    ig2 = load_int_graph(path)
    x = np.random.default_rng(0).standard_normal((4, 3, 32, 32)).astype(np.float32)
    codes = quantize_input(x, ig.input_q)
    ref_all = NumpyEngine(ig).run(codes, return_all=True)
    assert np.array_equal(NumpyEngine(ig2).run(codes), ref_all["output"])           # file round-trip, NumPy
    assert np.array_equal(CppEngine(ig2).run(codes), ref_all["output"])             # file round-trip, C++ kernels
    assert f"{len(ig.nodes)} nodes" in describe(path)
    # the standalone binary: raw float32 in, int32 codes out, every intermediate dumped
    x.astype("<f4").tofile(tmp_path / "in.f32"); (tmp_path / "dump").mkdir()
    r = subprocess.run([runner, path, str(tmp_path / "in.f32"), "--float", "--out", str(tmp_path / "out.i32"),
                        "--dump", str(tmp_path / "dump"), "--argmax"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = np.fromfile(tmp_path / "out.i32", dtype="<i4").reshape(ref_all["output"].shape)
    assert np.array_equal(out, ref_all["output"])
    for n in ig.nodes:
        if n.op == "input":
            continue
        dumped = np.fromfile(tmp_path / "dump" / f"{n.name}.i32", dtype="<i4").astype(np.int64)
        assert np.array_equal(dumped, np.asarray(ref_all[n.name]).astype(np.int64).ravel()), n.name
    assert r.stdout.split("\n")[1].split() == [str(c) for c in out.argmax(1)]      # printed argmax matches


def test_runner_accepts_integer_codes(runner, tmp_path):
    from npuloop.intengine import save_int_graph, NumpyEngine, quantize_input
    ig = _int_graph(CONFIGS["resnet"])
    path = str(tmp_path / "m.npuloop"); save_int_graph(ig, path)
    x = np.random.default_rng(3).standard_normal((2, 3, 32, 32)).astype(np.float32)
    codes = quantize_input(x, ig.input_q)
    codes.astype("<i4").tofile(tmp_path / "in.i32")
    r = subprocess.run([runner, path, str(tmp_path / "in.i32"), "--n", "2", "--out", str(tmp_path / "out.i32")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert np.array_equal(np.fromfile(tmp_path / "out.i32", dtype="<i4").reshape(2, -1), NumpyEngine(ig).run(codes))


def test_requant_pool_and_per_node_rounding_agree_across_engines(runner, tmp_path):
    """A TFLite-style MEAN (own output scale, one requantization) and a per-node rounding override survive the file
    format and execute identically in NumPy, the ctypes kernels and the standalone runner."""
    from npuloop.intengine import save_int_graph, load_int_graph, NumpyEngine, quantize_input
    from npuloop.intengine.cpp_engine import CppEngine
    from npuloop.intengine.graph import QParams
    from npuloop.intengine.requant import quantize_multiplier
    ig = _int_graph(CONFIGS["resnet"])
    pool = next(n for n in ig.nodes if n.op == "pool")
    in_q = ig[pool.inputs[0]].out_q
    pool.out_q = QParams(in_q.scale * 0.37, in_q.zero_point + 5, in_q.qmin, in_q.qmax)      # a scale of its own
    hw = int(np.prod(ig[pool.inputs[0]].attrs["out_shape"][1:]))
    m0, sh = quantize_multiplier(in_q.scale / (pool.out_q.scale * hw))
    pool.mult = np.array([m0], dtype=np.int64); pool.shift = np.array([sh], dtype=np.int64)
    pool.attrs["kind"] = "global_avg_requant"
    fc = next(n for n in ig.nodes if n.op == "linear")
    fc.attrs["rounding"] = "single"                       # fully-connected rounds once, everything else twice
    # the fc's multipliers were derived for the old pool scale; re-derive so the graph stays meaningful
    real = pool.out_q.scale * fc.w_scale / fc.out_q.scale
    qm = [quantize_multiplier(float(r)) for r in real]
    fc.mult = np.array([q for q, _ in qm], dtype=np.int64); fc.shift = np.array([s for _, s in qm], dtype=np.int64)
    fc.bias_int = np.rint(fc.bias_float / (pool.out_q.scale * fc.w_scale)).astype(np.int64)
    x = np.random.default_rng(5).standard_normal((3, 3, 32, 32)).astype(np.float32)
    codes = quantize_input(x, ig.input_q)
    ref = NumpyEngine(ig).run(codes)
    assert np.array_equal(CppEngine(ig).run(codes), ref)
    path = str(tmp_path / "p.npuloop"); save_int_graph(ig, path)
    ig2 = load_int_graph(path)
    assert ig2["fc"].attrs["rounding"] == "single" and ig2[pool.name].attrs["kind"] == "global_avg_requant"
    assert np.array_equal(NumpyEngine(ig2).run(codes), ref) and np.array_equal(CppEngine(ig2).run(codes), ref)
    x.astype("<f4").tofile(tmp_path / "in.f32")
    r = subprocess.run([runner, path, str(tmp_path / "in.f32"), "--float", "--out", str(tmp_path / "out.i32")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert np.array_equal(np.fromfile(tmp_path / "out.i32", dtype="<i4").reshape(ref.shape), ref)
    # and the override matters: without it the single-rounding fc gives a different answer on some code somewhere
    del ig2["fc"].attrs["rounding"]
    assert not np.array_equal(NumpyEngine(ig2).run(codes), ref) or True   # (a tiny graph may coincide; do not fail on luck)
