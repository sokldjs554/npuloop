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


def cmd_cost(a):
    from .graph import trace
    from .npu import estimate
    m = _load(a.ckpt)
    r = estimate(trace(m), a.spec)
    print(r.table())
    if a.json:
        json.dump(r.to_dict(), open(a.json, "w"), indent=1, default=float)


def cmd_lint(a):
    from .graph import trace
    from .lint import lint
    from .zoo import CIFAR10NPZ
    m = _load(a.ckpt)
    calib = CIFAR10NPZ(a.data).calib_batch(a.calib).numpy() if a.data else None
    r = lint(trace(m), a.spec, calib)
    print(r.markdown())
    if a.json:
        json.dump(r.to_dict(), open(a.json, "w"), indent=1, default=float)


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
    ig = export_int_graph(qm)
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
    c = sub.add_parser("cost", help="analytical cycle/utilization estimate on a virtual NPU"); c.add_argument("ckpt"); c.add_argument("--spec", default="edge-10tops"); c.add_argument("--json"); c.set_defaults(fn=cmd_cost)
    l = sub.add_parser("lint", help="NPU readiness report"); l.add_argument("ckpt"); l.add_argument("--spec", default="edge-10tops"); l.add_argument("--data"); l.add_argument("--calib", type=int, default=64); l.add_argument("--json"); l.set_defaults(fn=cmd_lint)
    q = sub.add_parser("quantize", help="PTQ + export + bit-exact verification"); q.add_argument("ckpt"); q.add_argument("--data", required=True)
    q.add_argument("--scheme", default="npu-default"); q.add_argument("--calib", type=int, default=512); q.add_argument("--verify", type=int, default=0)
    q.add_argument("--int-eval", type=int, default=0); q.add_argument("--out"); q.add_argument("--threads", type=int, default=4); q.set_defaults(fn=cmd_quantize)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
