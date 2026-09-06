"""Cross-layer equalization (Nagel et al., ICCV 2019) on a BN-folded fx graph.

For every pair conv_A -> [ReLU-family act] -> conv_B where the tensor between them has a single consumer,
per-channel scales s_i are chosen so that the output-channel ranges of A and the input-channel ranges
of B become equal: W_A[i] /= s_i, b_A[i] /= s_i, W_B[:, i] *= s_i (depthwise B: W_B[i] *= s_i).
ReLU is positively homogeneous so the function is unchanged (ReLU6 is not exactly: see `relu6_policy`).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.fx as fx
from ..graph.ir import ACT_MODULE_KINDS, RELU_FAMILY


def _is_dw(conv: nn.Conv2d) -> bool:
    return conv.groups == conv.in_channels == conv.out_channels and conv.groups > 1


def find_pairs(gm: fx.GraphModule) -> list[tuple[str, str, str | None]]:
    """Return (conv_A target, conv_B target, act target or None) triples eligible for equalization."""
    modules = dict(gm.named_modules())
    pairs = []
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(modules[node.target], nn.Conv2d):
            continue
        if len(node.users) != 1:
            continue
        nxt = next(iter(node.users))
        act_target = None
        if nxt.op == "call_module" and type(modules[nxt.target]) in ACT_MODULE_KINDS:
            if ACT_MODULE_KINDS[type(modules[nxt.target])] not in RELU_FAMILY:
                continue
            if len(nxt.users) != 1:
                continue
            act_target = nxt.target
            nxt = next(iter(nxt.users))
        if nxt.op == "call_module" and isinstance(modules[nxt.target], nn.Conv2d):
            a, b = modules[node.target], modules[nxt.target]
            if a.groups not in (1, a.out_channels) or b.groups not in (1, b.in_channels):
                continue
            if b.groups == 1 or _is_dw(b):
                pairs.append((node.target, nxt.target, act_target))
    return pairs


@torch.no_grad()
def equalize(gm: fx.GraphModule, iterations: int = 20, relu6_policy: str = "relu", tol: float = 1e-4) -> dict:
    """In-place CLE on a BN-folded GraphModule. Returns stats (pairs, iterations, final max |log s|)."""
    modules = dict(gm.named_modules())
    pairs = find_pairs(gm)
    if relu6_policy == "relu":
        for _, _, act in pairs:
            if act is not None and isinstance(modules[act], nn.ReLU6):
                _replace(gm, act, nn.ReLU())
        modules = dict(gm.named_modules())
    max_log_s = 0.0
    it = 0
    for it in range(1, iterations + 1):
        max_log_s = 0.0
        for a_t, b_t, _ in pairs:
            A, B = modules[a_t], modules[b_t]
            wa = A.weight.data                       # (Ca_out, ., kh, kw)
            ra = wa.abs().reshape(wa.shape[0], -1).amax(dim=1)
            wb = B.weight.data
            if _is_dw(B):
                rb = wb.abs().reshape(wb.shape[0], -1).amax(dim=1)          # channel i = group i
            else:
                rb = wb.abs().permute(1, 0, 2, 3).reshape(wb.shape[1], -1).amax(dim=1)
            s = torch.sqrt(ra * rb) / rb.clamp_min(1e-12)
            s = torch.where((ra > 0) & (rb > 0), s, torch.ones_like(s))
            A.weight.data = wa / s.reshape(-1, 1, 1, 1)
            if A.bias is not None:
                A.bias.data = A.bias.data / s
            if _is_dw(B):
                B.weight.data = wb * s.reshape(-1, 1, 1, 1)
            else:
                B.weight.data = wb * s.reshape(1, -1, 1, 1)
            max_log_s = max(max_log_s, float(s.log().abs().max()))
        if max_log_s < tol:
            break
    return dict(pairs=len(pairs), iterations=it, max_log_scale=max_log_s)


def _replace(gm: fx.GraphModule, target: str, module: nn.Module):
    *path, name = target.split(".")
    parent = gm
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, name, module)


@torch.no_grad()
def bias_correction(qm: fx.GraphModule, float_gm: fx.GraphModule, batches: list[torch.Tensor]) -> dict:
    """Empirical, sequential bias correction (DFQ): shift each quantized layer's bias by the mean
    per-channel output error w.r.t. the float (BN-folded) model, layer by layer in graph order."""
    from .fake import QConv2d, QLinear
    qmods = {n.target: qm.get_submodule(n.target) for n in qm.graph.nodes
             if n.op == "call_module" and isinstance(qm.get_submodule(n.target), (QConv2d, QLinear))}
    fmods = dict(float_gm.named_modules())
    corrections = {}
    for target, qmod in qmods.items():
        fmod = fmods[target]
        cache = {}

        def mk(key):
            def hook(mod, inp, out):
                dims = (0, 2, 3) if out.ndim == 4 else (0,)
                cache.setdefault(key, []).append(out.detach().mean(dim=dims))
            return hook
        h1 = qmod.register_forward_hook(mk("q")); h2 = fmod.register_forward_hook(mk("f"))
        for xb in batches:
            qm(xb); float_gm(xb)
        h1.remove(); h2.remove()
        err = torch.stack(cache["q"]).mean(0) - torch.stack(cache["f"]).mean(0)
        qmod.bias.data -= err
        corrections[target] = float(err.abs().mean())
    return corrections
