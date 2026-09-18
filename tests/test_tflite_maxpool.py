"""Max pooling against TensorFlow Lite's own reference kernel.

E11 is the external oracle for convolution, MEAN and fully-connected. Max pooling arrived later, and its one
semantic question -- what happens at a padded cell -- cannot be settled by comparing our two engines to each
other. TFLite's reference MAX_POOL_2D clamps the window to the input instead of filling it; npuloop fills with
qmin. Those agree only because qmin can never win a max, which is exactly the kind of claim that deserves an
external check rather than an argument.

Skipped without TensorFlow, which CI does not install.
"""
import numpy as np
import pytest

from npuloop.graph.ir import max_pool2d


@pytest.mark.slow
def test_max_pool_matches_the_tflite_reference_kernel():
    tf = pytest.importorskip("tensorflow")
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((32, 16, 16, 3)).astype(np.float32)

    inp = tf.keras.Input((16, 16, 3))
    h = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv")(inp)
    h = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same", name="mp_same")(h)    # symmetric at stride 1
    h = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="mp_valid")(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    model = tf.keras.Model(inp, tf.keras.layers.Dense(10)(h))
    model.set_weights([rng.standard_normal(w.shape).astype(np.float32) / np.sqrt(np.prod(w.shape[:-1]))
                       if w.ndim > 1 else rng.uniform(-0.2, 0.2, w.shape).astype(np.float32)
                       for w in model.get_weights()])

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([rep[i:i + 1]] for i in range(len(rep)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    blob = conv.convert()

    it = tf.lite.Interpreter(model_content=blob, experimental_preserve_all_tensors=True,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.resize_tensor_input(it.get_input_details()[0]["index"], [4, 16, 16, 3])
    it.allocate_tensors()
    tens = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    pools = [op for op in ops if op["op_name"] == "MAX_POOL_2D"]
    assert len(pools) == 2, [op["op_name"] for op in ops]

    x = (rng.standard_normal((4, 16, 16, 3)).astype(np.float32) * 40).astype(np.int8)
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()

    geometry = {}
    for op in pools:
        src, dst = op["inputs"][0], op["outputs"][0]
        a = it.get_tensor(src).transpose(0, 3, 1, 2).astype(np.int64)        # NHWC -> NCHW
        want = it.get_tensor(dst).transpose(0, 3, 1, 2).astype(np.int64)
        # TFLite gives max pooling the input's quantization; that is the tie npuloop's exporter asserts.
        assert tens[src]["quantization"] == tens[dst]["quantization"], tens[dst]["name"]
        ih, oh = a.shape[2], want.shape[2]
        k, stride, pad = ((3, 3), (1, 1), (1, 1)) if oh == ih else ((2, 2), (2, 2), (0, 0))
        got = max_pool2d(a, k, stride, pad, fill=-128)
        assert got.shape == want.shape, (tens[dst]["name"], got.shape, want.shape)
        assert np.array_equal(got, want), tens[dst]["name"]
        geometry[tens[dst]["name"]] = (k, stride, pad)
    assert len(geometry) == 2
    # A padded window must actually have been exercised, or the SAME case proves nothing about padding.
    assert any(p != (0, 0) for _, _, p in geometry.values())
