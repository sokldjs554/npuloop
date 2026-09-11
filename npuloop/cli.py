"""npuloop command line: lint / cost / quantize / verify a checkpoint against a virtual NPU."""
from __future__ import annotations
import argparse, json
import torch


def _load(path):
    from .zoo.train import load_checkpoint
    from .prune import rebuild_from_config
    ck = torch.load(path, weights_only=False)
    if ck.get("config", {}) and "pruned_channels" in ck["config"]:
        m = rebuild_from_config(ck["config"]); m.load_state_dict(ck["state_dict"]); return m.eval()
    return load_checkpoint(path)


def _shape(a, ds=None):
    size = a.input_size or (ds.img_size if ds is not None else 32)
    return (3, size, size)


def cmd_cost(a):
    from .graph import trace
    from .npu import estimate
    m = _load(a.ckpt)
    r = estimate(trace(m, _shape(a)), a.spec)
    print(r.table())
    if a.json:
        json.dump(r.to_dict(), open(a.json, "w"), indent=1, default=float)


def cmd_lint(a):
    from .graph import trace
    from .lint import lint
    from .zoo import CIFAR10NPZ
    m = _load(a.ckpt)
    ds = CIFAR10NPZ(a.data) if a.data else None
    calib = ds.calib_batch(a.calib).numpy() if ds else None
    r = lint(trace(m, _shape(a, ds)), a.spec, calib)
    print(r.markdown())
    if a.json:
        json.dump(r.to_dict(), open(a.json, "w"), indent=1, default=float)


def cmd_intake(a):
    from .intake import intake_report, render
    from .zoo import CIFAR10NPZ
    m = _load(a.ckpt)
    ds = CIFAR10NPZ(a.data) if a.data else None
    calib = ds.calib_batch(a.calib).numpy() if ds else None
    rep = intake_report(m, a.spec, calib, input_shape=_shape(a, ds), bench=a.bench, bench_data=ds)
    print(render(rep))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, default=float)


def cmd_export(a):
    """PTQ + export the integer graph to a portable .npuloop file (runnable by intengine/cpp/int8_runner)."""
    from .zoo import CIFAR10NPZ
    from .quant import prepare, calibrate, evaluate, PRESET_SCHEMES
    from .intengine import export_int_graph, save_int_graph
    from .intengine.serialize import describe
    ds = CIFAR10NPZ(a.data)
    m = _load(a.ckpt)
    scheme = PRESET_SCHEMES[a.scheme]
    qm = prepare(m, scheme); calibrate(qm, [ds.calib_batch(a.calib // 2, seed=s) for s in range(2)])
    ig = export_int_graph(qm, input_shape=(3, ds.img_size, ds.img_size))
    prov = dict(checkpoint=a.ckpt, config=getattr(m, "config", None), scheme=scheme.tag, calib_images=a.calib, data=a.data)
    if a.eval:
        prov["fake_quant_test_acc"] = evaluate(qm, ds, limit=a.eval)
    save_int_graph(ig, a.out, provenance=prov)
    print(describe(a.out))
    if a.eval:
        print(f"fake-quant accuracy on {a.eval} test images: {prov['fake_quant_test_acc']:.4f}")
    if a.sample:
        x, y = next(ds.test_batches(a.sample))
        x.numpy().astype("<f4").tofile(a.sample_path); y.numpy().astype("<i4").tofile(a.sample_path + ".labels")
        print(f"wrote {a.sample} test images as float32 NCHW to {a.sample_path} (labels: {a.sample_path}.labels)")


def cmd_bench(a):
    """Wall-clock of the NumPy and C++ reference engines per node, next to the cost model's cycles (host CPU, not NPU)."""
    from .zoo import CIFAR10NPZ
    from .graph import trace
    from .npu import estimate
    from .quant import prepare, calibrate, PRESET_SCHEMES
    from .intengine import export_int_graph, bench_engines, layer_table, render_bench
    torch.set_num_threads(1)
    ds = CIFAR10NPZ(a.data)
    m = _load(a.ckpt)
    qm = prepare(m, PRESET_SCHEMES[a.scheme]); calibrate(qm, [ds.calib_batch(a.calib // 2, seed=s) for s in range(2)])
    ig = export_int_graph(qm, input_shape=(3, ds.img_size, ds.img_size))
    x, _ = next(ds.test_batches(a.batch))
    engines = tuple(a.engines.split(","))
    b = bench_engines(ig, x.numpy(), repeats=a.repeats, engines=engines)
    rows = layer_table(ig, b, estimate(trace(m, (3, ds.img_size, ds.img_size)), a.spec), top=a.top)
    print(render_bench(b, rows))
    if a.json:
        json.dump(dict(bench=b, rows=rows, spec=a.spec, scheme=a.scheme), open(a.json, "w"), indent=1, default=float)


def cmd_quantize(a):
    from .zoo import CIFAR10NPZ
    from .quant import prepare, calibrate, evaluate, PRESET_SCHEMES
    from .intengine import export_int_graph, NumpyEngine
    from .intengine.verify import compare, agreement_table
    torch.set_num_threads(a.threads)
    ds = CIFAR10NPZ(a.data)
    m = _load(a.ckpt)
    scheme = PRESET_SCHEMES[a.scheme]
    calib = [ds.calib_batch(a.calib // 2, seed=s) for s in range(2)]
    qm = prepare(m, scheme); calibrate(qm, calib)
    fp = evaluate(m, ds); fq = evaluate(qm, ds)
    print(f"float acc {fp:.4f}  fake-quant int8 acc {fq:.4f}  (scheme {scheme.tag})")
    ig = export_int_graph(qm, input_shape=(3, ds.img_size, ds.img_size))
    if a.verify:
        x, _ = next(ds.test_batches(a.verify))
        rows, summ = compare(qm, ig, x)
        print(agreement_table(rows)); print(json.dumps(summ, indent=1))
    if a.int_eval:
        acc = NumpyEngine(ig).evaluate(ds, limit=a.int_eval)
        print(f"bit-exact integer engine acc ({a.int_eval} images): {acc:.4f}")
    if a.out:
        # float weights (loadable by cost/lint/quantize) + the calibrated fake-quant state for reproducibility
        torch.save({"config": m.config, "state_dict": m.state_dict(), "scheme": scheme.to_dict(), "qstate": qm.state_dict()}, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="npuloop")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost", help="analytical cycle/utilization estimate on a virtual NPU"); c.add_argument("ckpt"); c.add_argument("--spec", default="edge-10tops"); c.add_argument("--json"); c.add_argument("--input-size", type=int, default=None); c.set_defaults(fn=cmd_cost)
    l = sub.add_parser("lint", help="NPU readiness report"); l.add_argument("ckpt"); l.add_argument("--spec", default="edge-10tops"); l.add_argument("--data"); l.add_argument("--calib", type=int, default=64); l.add_argument("--json"); l.add_argument("--input-size", type=int, default=None); l.set_defaults(fn=cmd_lint)
    i = sub.add_parser("intake", help="customer intake: can this model run here, what does it cost, what should change")
    i.add_argument("ckpt"); i.add_argument("--spec", default="edge-10tops"); i.add_argument("--data")
    i.add_argument("--calib", type=int, default=64); i.add_argument("--json")
    i.add_argument("--bench", type=int, default=0, help="also time the NumPy and C++ engines on this many test images (needs --data)")
    i.add_argument("--input-size", type=int, default=None, help="input resolution (default: from --data, else 32)")
    i.set_defaults(fn=cmd_intake)
    e = sub.add_parser("export", help="PTQ + write the integer graph as a portable .npuloop file (see cpp/int8_runner)")
    e.add_argument("ckpt"); e.add_argument("--data", required=True); e.add_argument("--out", required=True)
    e.add_argument("--scheme", default="npu-default"); e.add_argument("--calib", type=int, default=512)
    e.add_argument("--eval", type=int, default=0, help="also record fake-quant accuracy on this many test images")
    e.add_argument("--sample", type=int, default=0, help="also write this many test images as raw float32 for the runner")
    e.add_argument("--sample-path", default="sample_input.f32"); e.set_defaults(fn=cmd_export)
    b = sub.add_parser("bench", help="wall-clock of the reference engines per node (host CPU), next to modelled cycles")
    b.add_argument("ckpt"); b.add_argument("--data", required=True); b.add_argument("--spec", default="edge-10tops")
    b.add_argument("--scheme", default="npu-default"); b.add_argument("--calib", type=int, default=512)
    b.add_argument("--batch", type=int, default=64); b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--engines", default="numpy,cpp"); b.add_argument("--top", type=int, default=12); b.add_argument("--json")
    b.set_defaults(fn=cmd_bench)
    q = sub.add_parser("quantize", help="PTQ + export + bit-exact verification"); q.add_argument("ckpt"); q.add_argument("--data", required=True)
    q.add_argument("--scheme", default="npu-default"); q.add_argument("--calib", type=int, default=512); q.add_argument("--verify", type=int, default=0)
    q.add_argument("--int-eval", type=int, default=0); q.add_argument("--out"); q.add_argument("--threads", type=int, default=4); q.set_defaults(fn=cmd_quantize)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
