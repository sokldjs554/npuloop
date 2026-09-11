"""E11: does npuloop's integer engine reproduce a real deployed runtime? Cross-check against TensorFlow Lite.

A small Keras CNN (conv-relu, conv-relu, global average pool, dense) is converted with TFLite's full-integer
post-training quantization. The quantization parameters, int8 weights and int32 biases are then read back out
of the .tflite model and used to build an npuloop IntGraph *with TFLite's own parameters* — nothing is
re-calibrated on our side. Both runtimes get the same int8 input codes; every output code (and, with
`experimental_preserve_all_tensors`, every intermediate tensor) is compared. The RequantConfig rounding mode
that reproduces TFLite code for code is reported; the others show how far a "slightly different" integer
implementation drifts. This is the external oracle for the claim "gemmlowp/TFLite-identical requantization".
"""
import os, sys, time
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # deterministic Keras training -> reproducible model
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
from npuloop.intengine.graph import IntGraph, IntNode, QParams
from npuloop.intengine.requant import RequantConfig, quantize_multiplier
from npuloop.intengine import NumpyEngine
from npuloop.intengine.cpp_engine import CppEngine

N_IMAGES = int(os.environ.get("NPULOOP_E11_IMAGES", 1000))
# (graph-wide rounding, fully-connected override): TFLite's reference CONV_2D/MEAN round twice (gemmlowp), its
# FULLY_CONNECTED rounds once (ruy-style) — found empirically here, see the README.
VARIANTS = [("tflite+fc-single", "tflite", "single"), ("tflite", "tflite", None), ("single", "single", None),
            ("half_even", "half_even", None), ("truncate", "truncate", None), ("floor", "floor", None)]


def build_and_convert(x_rep, seed=0):
    import tensorflow as tf
    tf.random.set_seed(seed); tf.keras.utils.set_random_seed(seed); tf.config.experimental.enable_op_determinism()
    inp = tf.keras.Input((32, 32, 3))
    h = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(inp)
    # stride 2 with an explicit symmetric pad: TFLite's SAME padding is asymmetric for stride 2, npuloop's is symmetric
    h = tf.keras.layers.ZeroPadding2D(1, name="pad2")(h)
    h = tf.keras.layers.Conv2D(16, 3, strides=2, padding="valid", activation="relu", name="conv2")(h)
    h = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv3")(h)
    h = tf.keras.layers.GlobalAveragePooling2D(name="pool")(h)
    out = tf.keras.layers.Dense(10, name="fc")(h)
    model = tf.keras.Model(inp, out)
    # deterministic, non-trivial weights (random kernels, non-zero biases) instead of a training run, so the
    # converted model — and therefore every number below — is reproducible bit for bit
    rng = np.random.default_rng(seed)
    model.set_weights([rng.standard_normal(w.shape).astype(np.float32) / np.sqrt(np.prod(w.shape[:-1])) if w.ndim > 1
                       else rng.uniform(-0.2, 0.2, w.shape).astype(np.float32) for w in model.get_weights()])
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x_rep[i:i + 1]] for i in range(len(x_rep)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def read_int_graph(tflite_bytes, rounding="tflite", fc_rounding=None) -> tuple[IntGraph, dict]:
    """Rebuild TFLite's quantized graph as an npuloop IntGraph from the interpreter's tensor and op details."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite_bytes, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    tens = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()

    def q_of(idx):
        t = tens[idx]; scale, zp = t["quantization"]
        return QParams(float(scale), int(zp), -128, 127)

    def per_channel(idx):
        qp = tens[idx]["quantization_parameters"]
        return np.asarray(qp["scales"], dtype=np.float64), qp["quantized_dimension"]

    nodes = []
    in_idx = it.get_input_details()[0]["index"]
    nodes.append(IntNode(tens[in_idx]["name"], "input", [], out_q=q_of(in_idx), attrs=dict(out_shape=(3, 32, 32))))
    name_of = {in_idx: nodes[0].name}
    pads = {}                      # PAD output tensor -> (source tensor, pad)
    for op in ops:
        kind = op["op_name"]; ins, outs = op["inputs"], op["outputs"]
        o = outs[0]; oname = tens[o]["name"]
        if kind == "PAD":
            padding = it.get_tensor(ins[1])           # (4, 2) [[0,0],[p,p],[p,p],[0,0]]
            assert padding[1][0] == padding[1][1] == padding[2][0] == padding[2][1]
            pads[o] = (ins[0], int(padding[1][0]))
            continue
        if kind == "CONV_2D":
            x, w, b = ins[0], ins[1], ins[2]
            pad = 0
            if x in pads:
                x, pad = pads[x]
            w_int = it.get_tensor(w).astype(np.int64)                    # OHWI -> OIHW
            w_int = np.transpose(w_int, (0, 3, 1, 2))
            bias_int = it.get_tensor(b).astype(np.int64)
            w_scales, qdim = per_channel(w)
            assert qdim == 0 and len(w_scales) == w_int.shape[0]
            in_h, out_h = int(tens[x]["shape"][1]), int(tens[o]["shape"][1])
            stride = in_h // out_h                                        # builtin options are not exposed: infer
            if pad == 0:                                                  # Keras 'same', stride 1: symmetric (k-1)/2
                assert stride == 1, "stride-2 SAME padding is asymmetric in TFLite; use an explicit ZeroPadding2D"
                pad = (w_int.shape[2] - 1) // 2
            n = IntNode(oname, "conv", [name_of[x]], out_q=q_of(o), w_int=w_int, w_scale=w_scales, bias_int=bias_int,
                        attrs=dict(stride=(stride, stride), padding=(pad, pad), groups=1, depthwise=False, fused_act="relu",
                                   out_shape=tuple(int(v) for v in tens[o]["shape"][1:][[2, 0, 1]])))
        elif kind == "MEAN":
            c = int(tens[o]["shape"][-1]); hw = int(tens[ins[0]]["shape"][1] * tens[ins[0]]["shape"][2])
            n = IntNode(oname, "pool", [name_of[ins[0]]], out_q=q_of(o), attrs=dict(kind="global_avg_requant", out_shape=(c, 1, 1), hw=hw))
            nodes.append(n)
            n = IntNode(oname + "_flat", "flatten", [oname], out_q=q_of(o), attrs=dict(out_shape=(c,)))   # TFLite's MEAN already emits (N, C)
        elif kind == "FULLY_CONNECTED":
            x, w, b = ins[0], ins[1], ins[2]
            w_int = it.get_tensor(w).astype(np.int64)
            bias_int = it.get_tensor(b).astype(np.int64)
            w_scales, qdim = per_channel(w)
            if len(w_scales) == 1:
                w_scales = np.full(w_int.shape[0], w_scales[0])
            n = IntNode(oname, "linear", [name_of[x]], out_q=q_of(o), w_int=w_int, w_scale=w_scales, bias_int=bias_int,
                        attrs=dict(out_shape=(int(tens[o]["shape"][-1]),)))
        elif kind in ("RESHAPE", "SQUEEZE"):
            n = IntNode(oname, "flatten", [name_of[ins[0]]], out_q=q_of(o), attrs=dict(out_shape=(int(tens[o]["shape"][-1]),)))
        else:
            raise ValueError(f"unexpected TFLite op {kind}")
        nodes.append(n); name_of[o] = n.name          # (after MEAN this is the synthetic flatten)
    out_idx = it.get_output_details()[0]["index"]
    nodes.append(IntNode("output", "output", [name_of[out_idx]], out_q=q_of(out_idx), attrs=dict(out_shape=(10,))))
    # second pass: the same requantization arithmetic npuloop applies to its own graphs
    cfg = RequantConfig(rounding=rounding)
    by = {n.name: n for n in nodes}
    for n in nodes:
        if n.op == "pool":                   # TFLite MEAN: multiplier s_in / (s_out * HW) on the centred sum
            s_in = by[n.inputs[0]].out_q.scale
            m0, sh = quantize_multiplier(float(s_in / (n.out_q.scale * n.attrs["hw"])), cfg.mult_bits)
            n.mult = np.array([m0], dtype=np.int64); n.shift = np.array([sh], dtype=np.int64)
        if n.op in ("conv", "linear"):
            s_in = by[n.inputs[0]].out_q.scale
            real = s_in * n.w_scale / n.out_q.scale
            qm = [quantize_multiplier(float(m), cfg.mult_bits) for m in real]
            n.mult = np.array([q for q, _ in qm], dtype=np.int64); n.shift = np.array([s for _, s in qm], dtype=np.int64)
            n.attrs["real_multiplier"] = real
            lo, hi = n.out_q.qmin, n.out_q.qmax
            if n.attrs.get("fused_act") == "relu":
                lo = max(lo, n.out_q.zero_point)
            n.attrs["clamp"] = (lo, hi)
            if n.op == "linear" and fc_rounding:
                n.attrs["rounding"] = fc_rounding
    return IntGraph(nodes, nodes[0].out_q, cfg), dict(interpreter=it, tensors=tens, ops=ops)


def tflite_run(tflite_bytes, x_int8: np.ndarray, preserve=False, backend="reference"):
    """Run the interpreter image by image on NHWC int8 codes; return output codes (and named intermediates).

    backend="reference" uses TFLite's reference kernels (the specification); "xnnpack" lets the default
    XNNPACK delegate take the ops (the optimized path a phone actually runs)."""
    import tensorflow as tf
    kw = dict(model_content=tflite_bytes, experimental_preserve_all_tensors=preserve, num_threads=1)
    if backend == "reference":
        kw["experimental_op_resolver_type"] = tf.lite.experimental.OpResolverType.BUILTIN_REF
    it = tf.lite.Interpreter(**kw)
    it.allocate_tensors()
    i_det, o_det = it.get_input_details()[0], it.get_output_details()[0]
    outs, inter = [], {}
    for i in range(len(x_int8)):
        it.set_tensor(i_det["index"], x_int8[i:i + 1])
        it.invoke()
        outs.append(it.get_tensor(o_det["index"]).copy())
        if preserve:
            for t in it.get_tensor_details():
                if t["dtype"] == np.int8 and t["index"] not in (i_det["index"],) and t["shape"].size and t["shape"][0] == 1:
                    inter.setdefault(t["name"], []).append(it.get_tensor(t["index"]).copy())
    return np.concatenate(outs), {k: np.concatenate(v) for k, v in inter.items()}


def to_nchw(a):
    return np.transpose(a, (0, 3, 1, 2)) if a.ndim == 4 else a


def main():
    import tensorflow as tf
    ds = dataset()
    res = Results("e11_tflite_crosscheck", meta=dict(tensorflow=tf.__version__, images=N_IMAGES,
                                                     note="TFLite full-integer PTQ; npuloop IntGraph built from TFLite's own qparams/weights/biases"))
    x_rep = np.transpose(ds.calib_batch(256, seed=0).numpy(), (0, 2, 3, 1)).astype(np.float32)   # NHWC float
    t0 = time.time()
    tfl = build_and_convert(x_rep)
    log(f"converted TFLite model: {len(tfl)} bytes in {time.time() - t0:.1f}s")
    x_f, _ = next(ds.test_batches(N_IMAGES))
    x_nhwc = np.transpose(x_f.numpy(), (0, 2, 3, 1))
    ig0, info = read_int_graph(tfl)
    q_in = ig0.input_q
    x_int8 = np.clip(np.rint(x_nhwc / q_in.scale) + q_in.zero_point, -128, 127).astype(np.int8)
    ref_out, ref_inter = tflite_run(tfl, x_int8, preserve=True)
    log(f"TFLite reference kernels ran {len(x_int8)} images; ops: {[o['op_name'] for o in info['ops']]}")
    xnn_out, _ = tflite_run(tfl, x_int8, backend="xnnpack")
    xnn_diff = int((xnn_out.astype(np.int64) != ref_out.astype(np.int64)).any(axis=1).sum())
    res.data["meta"]["xnnpack_vs_reference"] = dict(images=int(len(x_int8)), output_mismatch_images=xnn_diff,
                                                    output_mismatch_elems=int((xnn_out != ref_out).sum()))
    log(f"TFLite XNNPACK delegate vs reference kernels: outputs differ on {xnn_diff}/{len(x_int8)} images")
    codes_nchw = np.transpose(x_int8.astype(np.int64), (0, 3, 1, 2))
    for label, rounding, fc_rounding in VARIANTS:
        ig, _ = read_int_graph(tfl, rounding, fc_rounding)
        for engine_name, Eng in (("numpy", NumpyEngine), ("cpp", CppEngine)):
            eng = Eng(ig)
            all_vals = eng.run(codes_nchw, return_all=True)
            out = np.asarray(all_vals["output"]).astype(np.int64)
            out_mismatch_elems = int((out != ref_out.astype(np.int64)).sum())
            out_mismatch_images = int((out != ref_out.astype(np.int64)).any(axis=1).sum())
            top1_agree = float((out.argmax(1) == ref_out.astype(np.int64).argmax(1)).mean())
            per_node = {}
            for n in ig.nodes:
                if n.name in ref_inter and n.op in ("conv", "linear", "pool"):
                    ours = np.asarray(all_vals[n.name]).astype(np.int64)
                    theirs = to_nchw(ref_inter[n.name].astype(np.int64)).reshape(ours.shape)
                    per_node[n.name] = dict(op=n.op, mismatch_frac=float((ours != theirs).mean()), max_abs=int(np.abs(ours - theirs).max()))
            rec = dict(rounding=label, graph_rounding=rounding, fc_rounding=fc_rounding or rounding, engine=engine_name,
                       images=int(len(x_int8)), output_mismatch_elems=out_mismatch_elems,
                       output_mismatch_images=out_mismatch_images, output_elems=int(out.size), top1_agreement=top1_agree,
                       per_node=per_node, exact=bool(out_mismatch_elems == 0 and all(v["mismatch_frac"] == 0 for v in per_node.values())))
            res.add(rec, provenance="measured")
            log(f"{label:17s} {engine_name:5s}: output codes differ on {out_mismatch_images}/{len(x_int8)} images "
                f"({out_mismatch_elems}/{out.size} elements), top-1 agreement {top1_agree:.4f}, "
                + " ".join(f"{k}:{v['mismatch_frac'] * 100:.2f}%" for k, v in per_node.items()))
    res.data["meta"]["tflite_bytes"] = len(tfl)
    res.data["meta"]["ops"] = [o["op_name"] for o in info["ops"]]
    res.save()


if __name__ == "__main__":
    main()
