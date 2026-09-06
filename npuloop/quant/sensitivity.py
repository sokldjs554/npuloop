"""Per-layer quantization sensitivity: quantize ONE tensor at a time and measure the damage."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .fake import FakeQuantAct, QConv2d, QLinear
from .prepare import set_mode


@torch.no_grad()
def _loss_acc(gm, batches):
    gm.eval(); tot = 0; correct = 0; loss = 0.0
    for xb, yb in batches:
        out = gm(xb); loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item(); tot += len(yb)
    return loss / tot, correct / tot


@torch.no_grad()
def sqnr_db(x: torch.Tensor, xq: torch.Tensor) -> float:
    num = x.pow(2).mean().item(); den = (x - xq).pow(2).mean().item()
    return 10 * math.log10(num / den) if den > 0 else float("inf")


@torch.no_grad()
def layer_sensitivity(gm, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Returns {'baseline': {...}, 'weights': {layer: {...}}, 'activations': {fq: {...}}} with loss/acc when
    only that tensor is quantized (everything else float). Also per-tensor SQNR on the first batch."""
    set_mode(gm, enabled=False, w_enabled=False, calibrating=False)
    base_loss, base_acc = _loss_acc(gm, batches)
    res = {"baseline": dict(loss=base_loss, acc=base_acc), "weights": {}, "activations": {}}
    x0 = batches[0][0]
    for name, m in gm.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            m.w_enabled = True
            loss, acc = _loss_acc(gm, batches)
            m.w_enabled = False
            w = m.weight.detach(); m.w_enabled = True; wq = m.quantized_weight().detach(); m.w_enabled = False
            res["weights"][name] = dict(loss=loss, acc=acc, dloss=loss - base_loss, dacc=acc - base_acc, sqnr_db=sqnr_db(w, wq))
    for name, m in gm.named_modules():
        if isinstance(m, FakeQuantAct):
            m.enabled = True
            loss, acc = _loss_acc(gm, batches)
            # SQNR of this tensor
            cache = {}
            def hook(mod, inp, out): cache["x"] = inp[0].detach(); cache["q"] = out.detach()
            h = m.register_forward_hook(hook); gm(x0); h.remove()
            m.enabled = False
            res["activations"][name] = dict(loss=loss, acc=acc, dloss=loss - base_loss, dacc=acc - base_acc,
                                            sqnr_db=sqnr_db(cache["x"], cache["q"]))
    set_mode(gm, enabled=True, w_enabled=True)
    return res
