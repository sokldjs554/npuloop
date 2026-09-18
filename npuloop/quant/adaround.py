"""AdaRound: layer-wise adaptive rounding for post-training quantization (Nagel et al., ICML 2020).

Round-to-nearest is not the rounding that minimizes a layer's output error -- whether a weight should go up or
down depends on the other weights it is summed with. AdaRound makes the choice per weight, by optimizing a
continuous relaxation of it against the layer's own output, then hardening it.

What makes this worth having here rather than in any PTQ library: the learned rounding is written to the
layer's `w_round` buffer, which both the fake-quantization path and `int_weight()` read. So the rounding that
was optimized is the rounding the exported integer program executes, and E19 can ask whether the accuracy
AdaRound buys in the simulator survives bit-exact integer execution -- a question the simulator alone cannot
answer.

    qm = prepare(model); calibrate(qm, batches)
    adaround(qm, batches)                       # per-layer, in graph order
"""
from __future__ import annotations
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fake import QConv2d, QLinear

GAMMA, ZETA = -0.1, 1.1          # rectified-sigmoid support, so h() can reach exactly 0 and 1


def _h(v: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def _init_v(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """V such that h(V) equals the weight's fractional part, i.e. AdaRound starts at round-to-floor+frac."""
    rest = (w / s - torch.floor(w / s)).clamp(1e-4, 1 - 1e-4)
    return -torch.log((ZETA - GAMMA) / (rest - GAMMA) - 1)


def _scale_for(layer, dims: int) -> torch.Tensor:
    s = layer.w_scale if layer.per_channel else layer.w_scale.expand(layer.weight.shape[0])
    return s.detach().reshape((-1,) + (1,) * (dims - 1))


def _apply(layer, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, QConv2d):
        return F.conv2d(x, w, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
    return F.linear(x, w, layer.bias)


SAMPLES = 256          # feature maps kept per layer; a wide layer's activations are tens of MB per 100


@torch.no_grad()
def _capture(qm: nn.Module, target: nn.Module, batches, limit: int = SAMPLES) -> torch.Tensor:
    """Inputs seen by `target` when the (partly AdaRounded) quantized model runs the calibration batches.

    Stops once `limit` samples are in: a wide layer's activations over the whole calibration set are hundreds
    of megabytes, and the optimizer only ever draws `batch_size` of them at a time.
    """
    seen: list[torch.Tensor] = []
    got = 0

    def grab(_m, inp):
        nonlocal got
        x = inp[0].detach()
        if got < limit:
            seen.append(x[: limit - got].clone()); got += len(seen[-1])

    h = target.register_forward_pre_hook(grab)
    try:
        for b in batches:
            qm(b)
            if got >= limit:
                break
    finally:
        h.remove()
    return torch.cat(seen)


def adaround(qm: nn.Module, batches, iters: int = 1500, lr: float = 1e-2, lam: float = 0.01,
             beta_range: tuple[float, float] = (20.0, 2.0), warmup: float = 0.2, batch_size: int = 32,
             seed: int = 0, log=None) -> dict:
    """Optimize the rounding of every QConv2d/QLinear in `qm`, in graph order. Returns a per-layer report."""
    batches = list(batches)
    layers = [(n, m) for n, m in qm.named_modules() if isinstance(m, (QConv2d, QLinear))]
    was_training = qm.training
    qm.eval()
    g = torch.Generator().manual_seed(seed)
    report = {"layers": [], "iters": iters, "lam": lam, "lr": lr, "beta_range": list(beta_range)}
    for name, layer in layers:
        t0 = time.time()
        x_all = _capture(qm, layer, batches)
        dims = layer.weight.dim()
        s = _scale_for(layer, dims)
        with torch.no_grad():
            y_fp = torch.cat([_apply(layer, x_all[i:i + batch_size], layer.weight)
                              for i in range(0, len(x_all), batch_size)])
            # Layer outputs differ in scale by orders of magnitude across a network, so the reconstruction term
            # is normalized by the layer's own output power. Without this, a fixed `lam` means something
            # different in every layer and the rounding regularizer simply drowns the reconstruction.
            power = float(y_fp.pow(2).mean().clamp_min(1e-12))
            before = float(((_apply(layer, x_all[:batch_size], layer.quantized_weight())
                             - y_fp[:batch_size]) ** 2).mean())
        v = _init_v(layer.weight.detach(), s).requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        n_warm = int(iters * warmup)
        for it in range(iters):
            idx = torch.randint(0, len(x_all), (min(batch_size, len(x_all)),), generator=g)
            xb, yb = x_all[idx], y_fp[idx]
            h = _h(v)
            w_soft = torch.clamp(torch.floor(layer.weight.detach() / s) + h, layer.qmin, layer.qmax) * s
            rec = ((_apply(layer, xb, w_soft) - yb) ** 2).mean() / power
            if it < n_warm:
                loss = rec                                       # warm-up: reconstruction only, h stays soft
            else:                                                # then anneal beta so h is pushed to 0 or 1
                p = (it - n_warm) / max(1, iters - n_warm)
                beta = beta_range[1] + 0.5 * (beta_range[0] - beta_range[1]) * (1 + math.cos(p * math.pi))
                loss = rec + lam * (1 - (2 * h - 1).abs().pow(beta)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            layer.w_round = (_h(v) >= 0.5).to(layer.weight.dtype)
            after = float(((_apply(layer, x_all[:batch_size], layer.quantized_weight()) - y_fp[:batch_size]) ** 2).mean())
            frac = float((layer.w_round != (torch.round(layer.weight / s) - torch.floor(layer.weight / s))).float().mean())
        rec_layer = dict(layer=name, kind=type(layer).__name__, weights=int(layer.weight.numel()),
                         mse_before=before, mse_after=after, rel_before=before / power,
                         rel_after=after / power, flipped_frac=frac, seconds=time.time() - t0)
        report["layers"].append(rec_layer)
        if log:
            log(f"  {name:28s} mse {before:.3e} -> {after:.3e}  flipped {frac*100:5.2f}%  ({time.time()-t0:.0f}s)")
    if was_training:
        qm.train()
    return report
