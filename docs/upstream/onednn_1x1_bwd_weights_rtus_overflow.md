# oneDNN: 1x1 convolution backward_weights writes past the rtus workspace (nhwc, IC < ic_block)

Bug report for https://github.com/uxlfoundation/oneDNN/issues (template sections below). Found while tracing a stalled CI run of this repository, then reproduced and verified against upstream `main` (960edf5) with benchdnn in a separate session. Companion files: `rtus_ws_nxc.patch` (the two-line fix, validated with benchdnn) and `tools/onednn_1x1_repro.py` (a PyTorch-level reproducer for convenience; the report below relies on benchdnn only).

---

# Summary

A strided 1x1 f32 convolution with an nhwc (`acdb` / benchdnn `axb`) source whose input-channel count is smaller than the implementation's channel block writes outside its scratchpad in `backward_weights`:

* `jit_avx2_1x1_convolution_bwd_weights_t` for IC < 8,
* `jit_avx512_common_1x1_convolution_bwd_weights_t` for IC < 16.

Both implementations address the reduce-to-unit-stride (rtus) workspace with `sp * jcp.ic_block` per spatial position, but for nspc sources `rtus_prepare_space_info()` books only `jcp.is * jcp.ic` elements per thread, and the rtus driver and the kernel themselves use a compact `[os][ic]` layout (`ic` elements per position; the non-rtus branch of the same function, a few lines below, indexes the source with `(is_src_layout_nxc ? jcp.ic : jcp.ic_block)`). Every spatial chunk that starts beyond position `is * ic / ic_block` is therefore written past the thread's workspace: into the next thread's workspace, then into the reducer scratch, then past the scratchpad allocation.

The primary symptom is **silent wrong results**: `diff_weights` comes back wrong without any error (in the canonical case below every one of the 32 weight elements is wrong, with a maximum relative error of 615x against the reference). Depending on what happens to sit after the overflowed region, the same call can instead die with SIGSEGV. In another environment the overflow also landed on the reducer's spin-barrier context and the primitive never returned; that symptom is environment-dependent, the root cause is the same.

Stride 1 (no rtus), a blocked / nchw source, or IC being a multiple of the channel block are not affected, and the official 1x1 benchdnn suites do not contain any IC < ic_block shape, so they pass on the unpatched library (see "Test coverage gap").

# Version

Reproduced on `main` at `960edf5`. The three statements involved are identical in v3.12.0 (`80afa71049cd69a3df32adcccb623b12cd7baa22`, the version bundled in current PyTorch wheels):

* `src/cpu/x64/jit_avx2_1x1_convolution.cpp`, `execute_backward_weights` (line 644 on `main`): `rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_ + (ic_b * jcp.is + sp) * jcp.ic_block;`
* `src/cpu/x64/jit_avx512_common_1x1_convolution.cpp`, `execute_backward_weights` (line 845 on `main`): `rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_ + sp * jcp.ic_block;`
* `src/cpu/x64/jit_uni_1x1_conv_utils.hpp`, `rtus_prepare_space_info`: `self->rtus_.space_per_thread_ = is_nspc ? jcp.is * jcp.ic : factor * jcp.is * jcp.ic_block;` (with `jcp.is` already reduced to the output spatial size by `rtus_prepare`).

# Environment

* CPU: Intel Xeon, 4 cores, AVX-512 and AVX2 available. The AVX-512 implementation (`jit_1x1:avx512_core`) runs natively; the AVX2 implementation (`jit_1x1:avx2`) is selected with `ONEDNN_MAX_CPU_ISA=AVX2`.
* OS: Linux x86_64
* Compiler: GCC 13.3.0
* Build: `main` at `960edf5`, `-DDNNL_CPU_RUNTIME=OMP`, benchdnn built with `-DDNNL_BUILD_TESTS=ON` (Graph API and examples off to shorten the build; primitive selection `CONVOLUTION;REORDER` is enough for benchdnn's conv driver).

# Steps to reproduce

Canonical case (nhwc source and destination, stride 2, IC = 4 < 8):

```
$ ONEDNN_MAX_CPU_ISA=AVX2 ./benchdnn --conv --dir=BWD_W --dt=f32 --stag=axb --wtag=any --dtag=axb mb32ic4ih32oc8oh16kh1sh2ph0
```

Unpatched: `FAILED` with every diff_weights element wrong (32/32, max relative error 615x), or SIGSEGV. Patched: `0:PASSED`.

The same shape family on the AVX-512 implementation needs IC below 16, e.g. `mb32ic8ih128oc16oh64kh1sh2ph0` (run without the ISA override).

Across 12 benchdnn cases with IC < ic_block (nhwc, stride 2, both implementations), the unpatched library fails 8 (silent wrong results or SIGSEGV; which of the two depends on what lies after the overflowed region) and the patched library passes all 12.

Regression check with the patch applied: 16 large-IC strided runs (IC a multiple of the block, e.g. `mb8ic256ih16oc64oh8kh1sh2ph0`), 4 additional IC < ic_block runs, and 8 runs of the official suites (`--batch=inputs/conv/shapes_1x1`, 27 shapes, and `--batch=inputs/conv/shapes_regression_1x1`, 21 shapes, each in `--dir=BWD_W` with `--stag=axb --dtag=axb` and with `--stag=abx --dtag=abx`) all pass.

## Test coverage gap

The official 1x1 suites do not catch this bug: on the unpatched library, `shapes_1x1` (27) and `shapes_regression_1x1` (21) pass in `--dir=BWD_W` for both the nhwc and the blocked layouts. The smallest IC in either list is 16, so no shape exercises IC < ic_block on the strided nhwc path. Adding a few strided nhwc 1x1 shapes with IC in 1..15 (e.g. `mb32ic4ih32oc8oh16kh1sh2ph0`, `mb32ic8ih128oc16oh64kh1sh2ph0`) to the regression list would cover it.

# Observed behavior

`backward_weights` returns without error but `diff_weights` is wrong (all 32 elements of the canonical case, max relative error 615x), or the process dies with SIGSEGV. The primitive descriptor is accepted and the implementation reports success, so nothing tells the caller that the workspace was overrun.

# Expected behavior

`backward_weights` produces correct `diff_weights` for any input-channel count, or the two implementations reject nhwc sources with `ic < ic_block` when rtus is needed, so that another implementation is picked.

# Analysis and fix

For nspc sources the rtus driver copies `icb = ic` elements per output position contiguously (`rtus_driver_t::loop_is_nspc`), `rtus_prepare_space_info` books `jcp.is * jcp.ic` per thread accordingly, and the bwd-weights kernel steps through the reduced source with `reduce_loop_bcast_step = reduce_loop_unroll * jcp.ic`: the layout is `[os][ic]`. Only the chunk base in `execute_backward_weights` still uses the blocked stride `jcp.ic_block`, so for `ic < ic_block` the base of chunk `sp` is `sp * ic_block` in a buffer of `is * ic` elements: with `ic = 4`, `ic_block = 8`, `is = 256` the last chunks start at ~2040 elements in a 1024-element workspace, i.e. inside the next thread's workspace and, for the last thread, past the rtus region (the reducer scratch is booked right after `key_conv_rtus_space` in `pd_t::init`). The kernel then reads from the same (wrong) base, so the values stay right until the overflow hits something that matters. `ic >= ic_block` is safe because the base `sp * ic_block` never exceeds `sp * ic`.

The non-rtus branch of the same function, 14 lines below, already indexes the source with `(is_src_layout_nxc ? jcp.ic : jcp.ic_block)`; the rtus branch simply kept the blocked stride when nspc support was added. Using the same per-position stride there (matching `rtus_prepare_space_info`) restores the intended layout:

```diff
diff --git a/src/cpu/x64/jit_avx2_1x1_convolution.cpp b/src/cpu/x64/jit_avx2_1x1_convolution.cpp
--- a/src/cpu/x64/jit_avx2_1x1_convolution.cpp
+++ b/src/cpu/x64/jit_avx2_1x1_convolution.cpp
@@ -642,7 +642,9 @@ void jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights(
 
                         rp.ws = rtus_space
                                 + ithr * pd()->rtus_.space_per_thread_
-                                + (ic_b * jcp.is + sp) * jcp.ic_block;
+                                + (ic_b * jcp.is + sp)
+                                        * (is_src_layout_nxc ? jcp.ic
+                                                             : jcp.ic_block);
                         size_t src_offset
                                 = iw * src_d.blocking_desc().strides[ndims - 1];
                         if (ndims > 3)
diff --git a/src/cpu/x64/jit_avx512_common_1x1_convolution.cpp b/src/cpu/x64/jit_avx512_common_1x1_convolution.cpp
--- a/src/cpu/x64/jit_avx512_common_1x1_convolution.cpp
+++ b/src/cpu/x64/jit_avx512_common_1x1_convolution.cpp
@@ -843,7 +843,8 @@ void jit_avx512_common_1x1_convolution_bwd_weights_t::execute_backward_weights(
 
                             rp.ws = rtus_space
                                     + ithr * pd()->rtus_.space_per_thread_
-                                    + sp * jcp.ic_block;
+                                    + sp * (is_src_layout_nxc ? jcp.ic
+                                                              : jcp.ic_block);
 
                             if (ndims == 3)
                                 rp.src = local_src
```

(Line numbers refer to `main` at `960edf5`; the hunks apply to v3.12.0 as well.)

Validation: with the patch, all 12 IC < ic_block cases pass on both implementations, the 16 large-IC strided runs and the 4 extra IC < ic_block runs pass, and the two official 1x1 suites pass in both layouts (8 runs).

I have not looked at whether the forward / backward-data rtus paths or the `ic_b > 0` case for nxc sources have a related problem; the `ic_b * jcp.is` term is kept as it was.
