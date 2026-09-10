"""Reusable trainer (FP32 baselines, QAT fine-tuning, pruning recovery, activation-surgery healing).

Deterministic given the seed; resumable from <out>/state.pt; logs JSON per epoch.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import CIFAR10NPZ
from .models import ResNetCIFAR, MobileNetV2CIFAR, build_model, count_params


@torch.no_grad()
def evaluate(model: nn.Module, ds: CIFAR10NPZ, batch_size: int = 500, channels_last: bool = True, limit: int | None = None,
             split: str = "test") -> float:
    """Top-1 accuracy on the 'test' (default) or 'val' split, optionally only its first `limit` images."""
    model.eval()
    correct = 0; total = 0
    for xb, yb in ds.batches(split, batch_size):
        if channels_last:
            xb = xb.to(memory_format=torch.channels_last)
        correct += (model(xb).argmax(1) == yb).sum().item(); total += len(yb)
        if limit and total >= limit:
            break
    return correct / total


def fit(model: nn.Module, ds: CIFAR10NPZ, epochs: int, lr: float = 0.1, wd: float = 5e-4, bs: int = 128, seed: int = 0,
        out: str | None = None, resume: bool = False, channels_last: bool = True, warmup_pct: float = 0.15,
        div_factor: float = 10.0, final_div: float = 100.0, steps_per_epoch: int | None = None, log_prefix: str = "",
        param_filter=None, eval_limit: int | None = None, optimizer: str = "sgd") -> dict:
    """OneCycle training loop (SGD, or AdamW for transformers). Returns the log dict.

    Model selection never sees the test split: every epoch is scored on the validation split, `best.pt` is the
    epoch with the highest validation accuracy, and the test split is evaluated once at the end (for the selected
    checkpoint, `test_acc`, and for the last epoch, `final_test_acc`). When `out` is None nothing is selected: the
    caller gets the final-epoch model, and the log still carries the per-epoch validation curve.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    if param_filter is not None:
        params = [p for p in params if param_filter(p)]
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    opt = (torch.optim.AdamW(groups, lr=lr) if optimizer == "adamw"
           else torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True))
    spe = steps_per_epoch or len(ds.x_train) // bs
    total = spe * epochs
    warmup_pct = max(warmup_pct, 2.0 / total) if total > 4 else 0.5   # OneCycle needs >= 2 warm-up steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=warmup_pct,
                                                anneal_strategy="cos", div_factor=div_factor, final_div_factor=final_div)
    log = {"config": getattr(model, "config", None), "epochs": [],
           "hparams": dict(epochs=epochs, lr=lr, wd=wd, bs=bs, seed=seed, optimizer=optimizer),
           "splits": dict(train=len(ds.x_train), val=len(ds.x_val), test=len(ds.x_test)),
           "selection": "best.pt = highest val_acc epoch; test split evaluated once at the end"}
    best = -1.0; best_epoch = 0; start_ep = 0
    state_path = os.path.join(out, "state.pt") if out else None
    if out:
        os.makedirs(out, exist_ok=True)
    if resume and state_path and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        log, best, best_epoch, start_ep = st["log"], st["best"], st.get("best_epoch", 0), st["epoch"]
        print(f"{log_prefix}resumed from epoch {start_ep}", flush=True)
    t0 = time.time()
    val_acc = (log["epochs"][-1]["val_acc"] if log["epochs"]
               else evaluate(model, ds, channels_last=channels_last, limit=eval_limit, split="val"))
    for ep in range(start_ep, epochs):
        model.train()
        tl = tc = tn = 0
        te = time.time()
        for it, (xb, yb) in enumerate(ds.train_batches(bs, rng)):
            if it >= spe:
                break
            if channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            out_ = model(xb)
            loss = F.cross_entropy(out_, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(yb); tc += (out_.argmax(1) == yb).sum().item(); tn += len(yb)
        val_acc = evaluate(model, ds, channels_last=channels_last, limit=eval_limit, split="val")
        if val_acc >= best:                      # ties go to the later (more trained) epoch
            best, best_epoch = val_acc, ep + 1
            if out:
                torch.save({"config": getattr(model, "config", None), "state_dict": model.state_dict()}, os.path.join(out, "best.pt"))
        rec = dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn, val_acc=val_acc, lr=sched.get_last_lr()[0], epoch_sec=time.time() - te)
        log["epochs"].append(rec)
        print(log_prefix + json.dumps(rec), flush=True)
        if out:
            torch.save({"config": getattr(model, "config", None), "state_dict": model.state_dict()}, os.path.join(out, "last.pt"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "log": log, "best": best, "best_epoch": best_epoch, "epoch": ep + 1}, state_path)
    log["final_val_acc"] = val_acc; log["best_val_acc"] = best; log["selected_epoch"] = best_epoch
    log["train_minutes"] = (time.time() - t0) / 60
    # The test split is touched only here: once for the last epoch and once for the selected checkpoint.
    log["final_test_acc"] = evaluate(model, ds, channels_last=channels_last, limit=eval_limit)
    if out and best_epoch:
        selected = torch.load(os.path.join(out, "best.pt"), weights_only=False)["state_dict"]
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(selected)
        log["test_acc"] = evaluate(model, ds, channels_last=channels_last, limit=eval_limit)
        model.load_state_dict(final_state)       # leave the caller with the final-epoch weights, as before
    else:
        log["test_acc"] = log["final_test_acc"]
    if out:
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    return log


def load_checkpoint(path: str) -> nn.Module:
    """Load a checkpoint saved by fit() (also for pruned models, whose config carries `pruned_channels`)."""
    from .models import build_model
    ck = torch.load(path, weights_only=False)
    if ck["config"] and ck["config"].get("pruned_channels"):
        from ..prune.structured import rebuild_from_config
        m = rebuild_from_config(ck["config"])
    else:
        m = build_model(ck["config"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    p = argparse.ArgumentParser(description="train an FP32 CIFAR-10 baseline")
    p.add_argument("--data", default=os.environ.get("NPULOOP_DATA", "data/cifar10.npz"))
    p.add_argument("--arch", default="resnet", choices=["resnet", "mobilenetv2", "vit", "inception"])
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--width-mult", type=float, default=0.5)
    p.add_argument("--act", default="relu")
    p.add_argument("--dim", type=int, default=128)          # vit
    p.add_argument("--heads", type=int, default=4)          # vit
    p.add_argument("--patch", type=int, default=4)          # vit
    p.add_argument("--mlp-ratio", type=int, default=2)      # vit
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    ds = CIFAR10NPZ(a.data)
    cfg = dict(arch=a.arch, act=a.act)
    if a.arch == "resnet":
        cfg.update(depth=a.depth, width=a.width)
    elif a.arch == "mobilenetv2":
        cfg.update(width_mult=a.width_mult)
    elif a.arch == "vit":
        cfg.update(dim=a.dim, depth=a.depth, heads=a.heads, patch=a.patch, mlp_ratio=a.mlp_ratio)
    else:
        cfg.update(width=a.width)
    model = build_model(cfg)
    print(f"model {model.config} params={count_params(model):,}", flush=True)
    log = fit(model, ds, epochs=1 if a.smoke else a.epochs, lr=a.lr, wd=a.wd, bs=a.bs, seed=a.seed, out=a.out, resume=a.resume,
              steps_per_epoch=10 if a.smoke else None, optimizer=a.optimizer)
    print(f"done: selected epoch {log['selected_epoch']} (val {log['best_val_acc']:.4f}) test={log['test_acc']:.4f} "
          f"final-epoch test={log['final_test_acc']:.4f} minutes={log['train_minutes']:.1f}")


if __name__ == "__main__":
    main()
