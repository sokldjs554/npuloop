# oneDNN: 1x1 convolution backward_weights writes past the rtus workspace (nhwc, IC < ic_block)

Bug report for https://github.com/uxlfoundation/oneDNN/issues (template sections below). Found while tracing the CI stall of this repository (runs 12-16: the trainer test hung in `loss.backward()` on AMD EPYC runners), then reproduced with benchdnn on v3.12.0 and verified again in a separate session against upstream `main` (960edf5) and PyTorch 2.14.0. Companion files: `rtus_ws_nxc.patch` (the two-line fix, validated with benchdnn on both implementations) and `tools/onednn_1x1_repro.py` (pure PyTorch reproducer).

---

# Summary

A strided 1x1 f32 convolution with an nhwc (`acdb` / benchdnn `axb`) source whose input-channel count is smaller than the implementation's channel block writes outside its scratchpad in `backward_weights`:

* `jit_avx2_1x1_convolution_bwd_weights_t` for IC < 8,
* `jit_avx512_common_1x1_convolution_bwd_weights_t` for IC < 16.

Both implementations address the reduce-to-unit-stride (rtus) workspace with `sp * jcp.ic_block` per spatial position, but for nspc sources `rtus_prepare_space_info()` books only `jcp.is * jcp.ic` elements per thread, and the rtus driver and the kernel themselves use a compact `[os][ic]` layout (`ic` elements per position, exactly what the non-rtus branch of the same function uses: `p.bcast_data = src + (ic_b * jcp.reduce_dim + sp) * (is_src_layout_nxc ? jcp.ic : jcp.ic_block)`). Every spatial chunk that starts beyond position `is * ic / ic_block` is therefore written past the thread's workspace: into the next thread's workspace, then into the reducer scratch and its `simple_barrier` contexts, then past the scratchpad allocation.

What that looks like from the outside depends on what happens to sit after the overflowed region, and it differs between machines:

* **silent wrong results**: the primitive returns normally and `diff_weights` is wrong. On an Intel Xeon with the AVX2 implementation and 2 or 4 OpenMP threads, benchdnn reports `FAILED` with all 32 of 32 weight elements wrong and a maximum relative error of 615x, with no crash and no hang. This is the most dangerous form, because nothing tells the caller;
* **SIGSEGV**: with 1 thread (AVX2 implementation), and with the AVX-512 implementation once the input is 64x64 or larger, the overflow reaches a pointer and the process dies in the `execute_backward_weights` lambda right before it calls the kernel;
* **endless spin**: with 2 threads on AMD EPYC 7763 / 9V74 GitHub Actions runners, and on an Intel machine with `ONEDNN_MAX_CPU_ISA=AVX2`, the overflow lands on the weights reducer's barrier context: both threads sit at ~100 % CPU in the JIT `simple_barrier` wait loop and the barrier word holds float bit patterns instead of a sense flag, so the barrier never releases. Which iteration hangs is not deterministic (see the tables below).

Stride 1 (no rtus), a blocked / nchw source, or IC being a multiple of the channel block all run fine, and `oc < oc_block` selects another implementation. The official 1x1 benchdnn suites contain no IC < ic_block shape and therefore pass on the unpatched library (see "Test coverage gap").

Found through PyTorch 2.14.0: `torch.nn.Conv2d(4, 8, kernel_size=1, stride=2)` in `channels_last` hangs in `loss.backward()` on AMD EPYC GitHub Actions runners (AVX2 implementation) and segfaults on Intel Xeon (AVX-512 implementation) as soon as the input is 64x64 or larger (32x32 happens to stay inside allocation slack).

# Version

oneDNN v3.12.0, git hash `80afa71049cd69a3df32adcccb623b12cd7baa22`, the copy bundled in the PyTorch 2.14.0 wheels (`torch.__config__.show()` prints exactly `Intel(R) MKL-DNN v3.12.0 (Git Hash 80afa71049cd69a3df32adcccb623b12cd7baa22)`). Also reproduced on `main` at `960edf5`, where the three statements involved are unchanged:

* `src/cpu/x64/jit_avx2_1x1_convolution.cpp`, `execute_backward_weights` (line 644 on `main`): `rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_ + (ic_b * jcp.is + sp) * jcp.ic_block;`
* `src/cpu/x64/jit_avx512_common_1x1_convolution.cpp`, `execute_backward_weights` (line 845 on `main`): `rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_ + sp * jcp.ic_block;`
* `src/cpu/x64/jit_uni_1x1_conv_utils.hpp`, `rtus_prepare_space_info`: `self->rtus_.space_per_thread_ = is_nspc ? jcp.is * jcp.ic : factor * jcp.is * jcp.ic_block;` (with `jcp.is` already reduced to the output spatial size by `rtus_prepare`).

# Environment

* CPU: AMD EPYC 7763 and AMD EPYC 9V74 (GitHub Actions `ubuntu-latest` runners, 4 vCPUs) for the AVX2 implementation; Intel Xeon machines (4 cores, avx512f/bw/dq/vl and AVX2) for the AVX-512 implementation natively and for the AVX2 implementation through `ONEDNN_MAX_CPU_ISA=AVX2`.
* OS: Ubuntu 24.04.5 LTS / Linux 6.x, x86_64, glibc 2.39; Python 3.11.
* Library binaries: the prebuilt PyTorch 2.14.0+cpu and 2.14.0+cu130 wheels (OpenMP runtime, libgomp), 1 to 4 threads.
* Source builds for benchdnn: tag `v3.12` (`80afa71`) and `main` (`960edf5`), GCC 13.3.0, CMake 3.28.3, OpenMP runtime:
  `cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DDNNL_CPU_RUNTIME=OMP -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=ON -DONEDNN_BUILD_GRAPH=OFF -DDNNL_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DDNNL_ENABLE_PRIMITIVE_CPU_ISA=AVX2` (and `=AVX512` for the second implementation).
* How the CI catches it now: the repository's `ci.yml` runs pytest on `ubuntu-latest` with a thread-based watchdog (py-spy native stack dump after 90 s, faulthandler after 150 s, kill at 180 s) in a matrix over `ONEDNN_MAX_CPU_ISA` in `{ALL, AVX2}`, so the AVX2 kernel family is exercised on every run regardless of which CPU the runner pool hands out.

# Steps to reproduce

## 1. benchdnn

AVX2 implementation (`jit_1x1:avx2`, AVX2-only build). The problem descriptor is the one from the PyTorch verbose line below; `--stag=axb --dtag=axb` is the nhwc case.

Unpatched:

```
$ OMP_NUM_THREADS=2 timeout 90 ./benchdnn --conv --dir=BWD_W --dt=f32 --stag=axb --wtag=any --dtag=axb mb32ic4ih32oc8oh16kh1sh2ph0
  machine A (Intel Xeon dev VM, ONEDNN_MAX_CPU_ISA=AVX2 build): no output, killed by the timeout after 90 s (both threads spinning)
  machine B (Intel Xeon, 4 cores):                              0:FAILED  -- all 32/32 diff_weights elements wrong, max relative error 615x, no crash
$ OMP_NUM_THREADS=4 ... mb32ic4ih32oc8oh16kh1sh2ph0        -> hang (A) / FAILED with wrong diff_weights (B)
$ OMP_NUM_THREADS=1 ... mb32ic4ih32oc8oh16kh1sh2ph0        -> Segmentation fault (general protection fault)
$ OMP_NUM_THREADS=2 ... mb32ic4ih64oc8oh32kh1sh2ph0        -> hang / wrong results
$ OMP_NUM_THREADS=2 ... mb32ic8ih32oc8oh16kh1sh2ph0        -> 0:PASSED   (IC = ic_block)
$ OMP_NUM_THREADS=2 ... mb32ic4ih32oc8oh32kh1sh1ph0        -> 0:PASSED   (stride 1: no rtus)
$ OMP_NUM_THREADS=2 ... --stag=abx --dtag=abx mb32ic4ih32oc8oh16kh1sh2ph0 -> 0:PASSED (blocked source)
```

With the patch from the last section applied and benchdnn rebuilt, every one of these passes (`0:PASSED`), including `mb32ic4ih32oc8oh16kh1sh2ph0` with 1, 2 and 4 threads, `mb32ic4ih64oc8oh32kh1sh2ph0`, `mb32ic2ih32oc8oh16kh1sh2ph0`, `mb32ic7ih32oc16oh16kh1sh2ph0`, `mb32ic16ih64oc32oh32kh1sh2ph0` and `mb16ic24ih48oc8oh24kh1sh2ph0`.

AVX-512 implementation (`jit_1x1:avx512_core`, `ic_block = 16`, AVX-512 build on the Intel machine), unpatched:

```
$ OMP_NUM_THREADS=2 ./benchdnn --conv --dir=BWD_W --dt=f32 --stag=axb --wtag=any --dtag=axb mb32ic4ih128oc8oh64kh1sh2ph0   -> Segmentation fault
$ OMP_NUM_THREADS=1 ... mb32ic4ih128oc8oh64kh1sh2ph0     -> Segmentation fault
$ OMP_NUM_THREADS=4 ... mb32ic4ih128oc8oh64kh1sh2ph0     -> Segmentation fault
$ OMP_NUM_THREADS=2 ... mb32ic8ih128oc16oh64kh1sh2ph0    -> Segmentation fault   (IC = 8 < 16)
$ OMP_NUM_THREADS=2 ... mb32ic16ih128oc16oh64kh1sh2ph0   -> 0:PASSED             (IC = ic_block)
     (the kernel log reports the faults as user-mode write faults)
```

With the patch: `mb32ic4ih128oc8oh64kh1sh2ph0` with 1, 2 and 4 threads, `mb32ic8ih128oc16oh64kh1sh2ph0`, `mb32ic12ih128oc16oh64kh1sh2ph0`, `mb32ic15ih128oc16oh64kh1sh2ph0`, `mb32ic16ih128oc16oh64kh1sh2ph0`, `mb32ic4ih32oc8oh16kh1sh2ph0` and the stride-1 control all pass.

Overall, across 12 benchdnn cases with IC < ic_block (nhwc, stride 2, both implementations) the patched library passes all 12, while the unpatched one fails them with one of the three symptoms above depending on the machine. Regression check with the patch: 16 large-IC strided runs (IC a multiple of the block: `mb8ic256ih16oc64oh8kh1sh2ph0`, `mb4ic512ih14oc128oh7kh1sh2ph0`, `mb2ic1024ih14oc256oh7kh1sh2ph0`, `mb8ic136ih16oc16oh8kh1sh2ph0`, `mb8ic144ih16oc16oh8kh1sh2ph0`, 1 and 4 threads), 4 additional IC < ic_block runs, and 8 runs of the official suites (`--batch=inputs/conv/shapes_1x1`, 27 shapes, and `--batch=inputs/conv/shapes_regression_1x1`, 21 shapes, each in `--dir=BWD_W` with `--stag=axb --dtag=axb` and with `--stag=abx --dtag=abx`) all pass.

### Test coverage gap

The official 1x1 suites do not catch this bug: on the unpatched library, `shapes_1x1` (27) and `shapes_regression_1x1` (21) pass in `--dir=BWD_W` for both the nhwc and the blocked layouts. The smallest IC in either list is 16, so no shape exercises IC < ic_block on the strided nhwc path, and upstream CI misses the defect structurally. Adding a few strided nhwc 1x1 shapes with IC in 1..15 (e.g. `mb32ic4ih32oc8oh16kh1sh2ph0`, `mb32ic8ih128oc16oh64kh1sh2ph0`) to the regression list would cover it.

## 2. Through PyTorch 2.14.0 (any x86 CPU)

The failing primitive as reported by `ONEDNN_VERBOSE=1` (PyTorch 2.14.0+cpu, `ONEDNN_MAX_CPU_ISA=AVX2`):

```
onednn_verbose,primitive,exec,cpu,convolution,jit_1x1:avx2,backward_weights,src:f32::blocked:acdb::f0 wei:f32:ap:blocked:ABcd8b8a::f0 bia:undef::undef::: dst:f32::blocked:acdb::f0,attr-scratchpad:user,alg:convolution_direct,mb32_ic4oc8_ih32oh16kh1sh2dh0ph0_iw32ow16kw1sw2dw0pw0
```

(the forward pass of the same layer goes through `brgconv_1x1:avx2` and is fine; with the default ISA on the Intel machine the backward-weights primitive is `jit_1x1:avx512_core` with `wei:f32:ap:blocked:ABcd16b16a`.)

```python
# onednn_1x1_repro.py  --  python onednn_1x1_repro.py [ic] [oc] [stride] [batch] [threads] [cl|nchw] [spatial] [iters]
import sys, time, torch
ic, oc, stride, bs, threads = (int(a) for a in (sys.argv[1:6] + ["4", "8", "2", "32", "2"][len(sys.argv) - 1:]))
layout = sys.argv[6] if len(sys.argv) > 6 else "cl"
hw = int(sys.argv[7]) if len(sys.argv) > 7 else 32
iters = int(sys.argv[8]) if len(sys.argv) > 8 else 20
torch.set_num_threads(threads); torch.manual_seed(0)
conv = torch.nn.Conv2d(ic, oc, kernel_size=1, stride=stride, bias=False)
if layout == "cl":
    conv = conv.to(memory_format=torch.channels_last)
t0 = time.time()
for it in range(iters):
    x = torch.randn(bs, ic, hw, hw)
    if layout == "cl":
        x = x.to(memory_format=torch.channels_last)
    conv.weight.grad = None
    conv(x).sum().backward()
    print(f"iter {it} ok ({time.time() - t0:.2f}s)", flush=True)
print("FINISHED", flush=True)
```

AVX2 implementation (`ONEDNN_MAX_CPU_ISA=AVX2` on the Intel machines; the same happens natively on the AMD EPYC 7763 runners), 32x32 input, batch 32, 20 iterations, `timeout 20`. The iteration at which a run hangs differs from run to run (the same shape stopped at iteration 0 in one run and after several in another, and `ic=7` hung in 3 of 5 runs), so only the outcome is listed:

| ic | oc | stride | threads | layout | result |
|---|---|---|---|---|---|
| 4 | 8 | 2 | 2 | nhwc | **hang** (timeout) |
| 4 | 8 | 2 | 1 | nhwc | **SIGSEGV** |
| 2 | 8 | 2 | 2 | nhwc | **hang** (iteration varies per run) |
| 6 | 8 | 2 | 2 | nhwc | **hang** (iteration varies per run) |
| 7 | 8 | 2 | 2 | nhwc | **hang** in 3 of 5 runs |
| 4 | 16 | 2 | 2 | nhwc | **hang** (iteration varies per run) |
| 8 | 8 | 2 | 2 | nhwc | ok |
| 4 | 8 | 1 | 2 | nhwc | ok (no rtus) |
| 4 | 8 | 2 | 2 | nchw | ok (blocked path) |
| 4 | 8 | 2 | 2 | nhwc, batch 1 | ok (overflow stays inside slack) |
| 4 | 4 | 2 | 2 | nhwc | ok (other implementation) |

AVX-512 implementation (default ISA on the Intel machines, `jit_1x1:avx512_core`), 128x128 input, batch 32, 2 threads, 30 iterations:

| ic | oc | stride | layout | result |
|---|---|---|---|---|
| 4 | 8 | 2 | nhwc | **SIGSEGV** (also with 1 and 4 threads, and at 64x64) |
| 8 | 16 | 2 | nhwc | **SIGSEGV** |
| 12 | 16 | 2 | nhwc | **SIGSEGV** |
| 15 | 16 | 2 | nhwc | **SIGSEGV** |
| 16 | 16 | 2 | nhwc | ok |
| 24 | 16 | 2 | nhwc | ok |
| 4 | 8 | 1 | nhwc | ok |
| 4 | 8 | 2 | nchw | ok |

# Observed behavior

Silent wrong results: on the Intel Xeon with the AVX2 implementation and 2 or 4 threads, benchdnn's correctness check fails on the canonical case with every `diff_weights` element wrong (32 of 32, maximum relative error 615x) while the primitive reports success.

Native stack of the hung process (AMD EPYC 7763 runner, PyTorch 2.14.0+cpu, 2 threads; `py-spy dump --native` after 90 s, both threads at ~94 % CPU; addresses omitted):

```
Thread (active): "MainThread"
    <JIT code: simple_barrier wait loop>
    dnnl::impl::parallel (libtorch_cpu.so)
    GOMP_parallel (libgomp.so.1)
    dnnl::impl::parallel (libtorch_cpu.so)
    dnnl::impl::cpu::x64::jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights (libtorch_cpu.so)
    dnnl::impl::cpu::x64::jit_avx2_1x1_convolution_bwd_weights_t::execute (libtorch_cpu.so)
    dnnl::impl::primitive_execute (libtorch_cpu.so)
    dnnl_primitive_execute (libtorch_cpu.so)
    dnnl::primitive::execute (libtorch_cpu.so)
    at::native::(anonymous namespace)::mkldnn_convolution_backward_weights (libtorch_cpu.so)
    ...
Thread (OpenMP worker):
    <JIT code: simple_barrier wait loop>
    std::_Function_handler<..., execute_backward_weights(...)::{lambda(int, int)#3}>::_M_invoke
    dnnl::impl::parallel [_omp_fn.0]
    gomp_thread_start
```

gdb attached to the same hang on the Intel machine (`ONEDNN_MAX_CPU_ISA=AVX2`) shows both threads in the JIT `simple_barrier` wait loop (`pause`, compare the barrier's `sense` word, loop), and the value the loop waits for is not a sense flag but a pair of float32 bit patterns (small values of the order of 0.05 to 0.4, i.e. weight / gradient data): the reducer's barrier context was overwritten by the rtus overflow.

With one thread the same case dies instead (gdb, `ONEDNN_MAX_CPU_ISA=AVX2`):

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.
#0  std::_Function_handler<void (int, int), dnnl::impl::cpu::x64::jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights(dnnl::impl::exec_ctx_t const&) const::{lambda(int, int)#3}>::_M_invoke(...) ()
#1  dnnl::impl::parallel(int, std::function<void (int, int)> const&) ()
#2  dnnl::impl::cpu::x64::jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights(dnnl::impl::exec_ctx_t const&) const ()
#3  dnnl::impl::cpu::x64::jit_avx2_1x1_convolution_bwd_weights_t::execute(dnnl::impl::exec_ctx_t const&) const ()
#4  dnnl::impl::primitive_execute(dnnl_primitive const*, dnnl::impl::exec_ctx_t&) ()
```

The faulting instruction is the load of the kernel pointer just before `(*kernel_)(&p)`, through a pointer that also holds float bit patterns.

# Expected behavior

`backward_weights` returns and produces correct `diff_weights` for any input-channel count, or the two implementations reject nhwc sources with `ic < ic_block` when rtus is needed, so that another implementation is picked.

# Analysis and a possible fix

For nspc sources the rtus driver copies `icb = ic` elements per output position contiguously (`rtus_driver_t::loop_is_nspc`), `rtus_prepare_space_info` books `jcp.is * jcp.ic` per thread accordingly, and the bwd-weights kernel steps through the reduced source with `reduce_loop_bcast_step = reduce_loop_unroll * jcp.ic`: the layout is `[os][ic]`. Only the chunk base in `execute_backward_weights` still uses the blocked stride `jcp.ic_block`, so for `ic < ic_block` the base of chunk `sp` is `sp * ic_block` in a buffer of `is * ic` elements: with `ic = 4`, `ic_block = 8`, `is = 256` the last chunks start at ~2040 elements in a 1024-element workspace, i.e. inside the next thread's workspace and, for the last thread, in the reducer scratch and its barrier contexts (booked right after `key_conv_rtus_space` in `pd_t::init`). The kernel then reads from the same (wrong) base, so the values stay right until the overflow hits something that matters. `ic >= ic_block` is safe because the base `sp * ic_block` never exceeds `sp * ic`.

The non-rtus branch of the same function, 14 lines below, already indexes the source with `(is_src_layout_nxc ? jcp.ic : jcp.ic_block)`; the rtus branch simply kept the blocked stride when nspc support was added. Using the same per-position stride there is not new behaviour, it restores the intended layout (and matches `rtus_prepare_space_info`):

```diff
diff --git a/src/cpu/x64/jit_avx2_1x1_convolution.cpp b/src/cpu/x64/jit_avx2_1x1_convolution.cpp
--- a/src/cpu/x64/jit_avx2_1x1_convolution.cpp
+++ b/src/cpu/x64/jit_avx2_1x1_convolution.cpp
@@ -635,7 +635,9 @@ void jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights(
 
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
@@ -829,7 +829,8 @@ void jit_avx512_common_1x1_convolution_bwd_weights_t::execute_backward_weights(
 
                             rp.ws = rtus_space
                                     + ithr * pd()->rtus_.space_per_thread_
-                                    + sp * jcp.ic_block;
+                                    + sp * (is_src_layout_nxc ? jcp.ic
+                                                              : jcp.ic_block);
 
                             if (ndims == 3)
                                 rp.src = local_src
```

(Hunk line numbers are those of v3.12.0; on `main` at `960edf5` the statements are at lines 644 and 845 and the patch applies unchanged.)

Validation: see the benchdnn section. With the patch, every failing case passes on both the AVX2 build (`jit_1x1:avx2`) and the AVX-512 build (`jit_1x1:avx512_core`), the large-IC strided cases and the `shapes_1x1` / `shapes_regression_1x1` suites still pass in both layouts, and reverting the patch on the same build trees brings the failures back.

I have not looked at whether the forward / backward-data rtus paths or the `ic_b > 0` case for nxc sources have a related problem; the `ic_b * jcp.is` term is kept as it was.
