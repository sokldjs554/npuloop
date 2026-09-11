"""Pure-PyTorch reproducer: 1x1 convolution backward-weights with in_channels < 8 in channels_last on the AVX2 JIT path.

usage: python onednn_1x1_repro.py [ic] [oc] [stride] [batch] [threads] [layout: cl|nchw] [spatial] [iters]
Run with ONEDNN_MAX_CPU_ISA=AVX2 on any x86 CPU (or natively on a CPU without AVX-512, e.g. AMD EPYC 7763).
"""
import sys
import time

import torch

ic = int(sys.argv[1]) if len(sys.argv) > 1 else 4
oc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 32
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 2
layout = sys.argv[6] if len(sys.argv) > 6 else "cl"
hw = int(sys.argv[7]) if len(sys.argv) > 7 else 32
iters = int(sys.argv[8]) if len(sys.argv) > 8 else 20
torch.set_num_threads(threads)
torch.manual_seed(0)
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
    print(f"ic={ic} oc={oc} s={stride} bs={bs} thr={threads} {layout} hw={hw}: iter {it} ok ({time.time() - t0:.2f}s)", flush=True)
print("FINISHED", flush=True)
