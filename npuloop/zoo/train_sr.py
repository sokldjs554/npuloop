"""Trainer for the dense-output baseline (x2 super-resolution).

Separate from train.py because the objective and the selection metric differ: L1 on pixels and validation
PSNR, not cross-entropy and top-1. Everything else follows the same conventions -- OneCycle schedule,
best.pt is the highest-validation epoch, the test split is touched once at the end, resumable from state.pt,
and the checkpoint layout is the one load_checkpoint() already reads.

    python -m npuloop.zoo.train_sr --data data/imagenette128.npz --out runs/espcn_x2 --epochs 25
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import SRPairs, psnr
from .models import build_model, count_params


@torch.no_grad()
def evaluate_psnr(model: nn.Module, ds: SRPairs, split: str = "test", batch_size: int = 100,
                  limit: int | None = None) -> float:
    """Mean per-image PSNR (dB) over a split."""
    model.eval()
    vals = []
    for lr, hr in ds.batches(split, batch_size):
        vals.append(psnr(model(lr).numpy(), hr.numpy()))
        if limit and sum(len(v) for v in vals) >= limit:
            break
    return float(np.concatenate(vals).mean())


@torch.no_grad()
def baseline_psnr(ds: SRPairs, split: str = "test", batch_size: int = 100) -> float:
    """PSNR of the trivial upsampler (nearest-neighbour block repeat), for context."""
    vals = []
    for lr, hr in ds.batches(split, batch_size):
        up = lr.repeat_interleave(ds.scale, dim=2).repeat_interleave(ds.scale, dim=3)
        vals.append(psnr(up.numpy(), hr.numpy()))
    return float(np.concatenate(vals).mean())


def fit_sr(model: nn.Module, ds: SRPairs, epochs: int, lr: float = 1e-3, wd: float = 0.0, bs: int = 32,
           seed: int = 0, out: str | None = None, resume: bool = False, steps_per_epoch: int | None = None,
           log_prefix: str = "") -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    spe = steps_per_epoch or len(ds.x_train) // bs
    total = max(spe * epochs, 4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.15,
                                                anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0)
    log = {"config": getattr(model, "config", None), "epochs": [],
           "hparams": dict(epochs=epochs, lr=lr, wd=wd, bs=bs, seed=seed, optimizer="adamw", loss="l1"),
           "splits": dict(train=len(ds.x_train), val=len(ds.x_val), test=len(ds.x_test)),
           "selection": "best.pt = highest val_psnr epoch; test split evaluated once at the end"}
    best = -1.0; best_epoch = 0; start_ep = 0
    state_path = os.path.join(out, "state.pt") if out else None
    if out:
        os.makedirs(out, exist_ok=True)
    if resume and state_path and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=True, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        log, best, best_epoch, start_ep = st["log"], st["best"], st.get("best_epoch", 0), st["epoch"]
        print(f"{log_prefix}resumed from epoch {start_ep}", flush=True)
    t0 = time.time()
    val = log["epochs"][-1]["val_psnr"] if log["epochs"] else evaluate_psnr(model, ds, "val")
    for ep in range(start_ep, epochs):
        model.train()
        tl = tn = 0
        te = time.time()
        for it, (xb, yb) in enumerate(ds.train_batches(bs, rng)):
            if it >= spe:
                break
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(xb); tn += len(xb)
        val = evaluate_psnr(model, ds, "val")
        if val >= best:                          # ties go to the later (more trained) epoch
            best, best_epoch = val, ep + 1
            if out:
                torch.save({"config": getattr(model, "config", None), "state_dict": model.state_dict()},
                           os.path.join(out, "best.pt"))
        rec = dict(epoch=ep + 1, train_l1=tl / tn, val_psnr=val, lr=sched.get_last_lr()[0],
                   epoch_sec=time.time() - te)
        log["epochs"].append(rec)
        print(log_prefix + json.dumps(rec), flush=True)
        if out:
            torch.save({"config": getattr(model, "config", None), "state_dict": model.state_dict()},
                       os.path.join(out, "last.pt"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "log": log,
                        "best": best, "best_epoch": best_epoch, "epoch": ep + 1}, state_path)
    log["final_val_psnr"] = val; log["best_val_psnr"] = best; log["selected_epoch"] = best_epoch
    log["train_minutes"] = (time.time() - t0) / 60
    log["final_test_psnr"] = evaluate_psnr(model, ds, "test")
    if out and best_epoch:
        selected = torch.load(os.path.join(out, "best.pt"), weights_only=True, map_location="cpu")["state_dict"]
        final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(selected)
        log["test_psnr"] = evaluate_psnr(model, ds, "test")
        model.load_state_dict(final_state)
    else:
        log["test_psnr"] = log["final_test_psnr"]
    log["baseline_test_psnr"] = baseline_psnr(ds, "test")
    if out:
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    return log


def main():
    p = argparse.ArgumentParser(description="train the x2 super-resolution baseline (dense output)")
    p.add_argument("--data", default=os.environ.get("NPULOOP_DATA", "data/imagenette128.npz"))
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--feat", type=int, default=32)
    p.add_argument("--act", default="relu")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    ds = SRPairs(a.data, scale=a.scale)
    torch.manual_seed(a.seed)                    # weight init follows --seed, as in train.py
    model = build_model(dict(arch="espcn", lr_size=ds.lr_size, scale=a.scale, feat=a.feat, act=a.act))
    print(f"espcn x{a.scale} {ds.lr_size}->{ds.hr_size} params={count_params(model)}", flush=True)
    log = fit_sr(model, ds, epochs=1 if a.smoke else a.epochs, lr=a.lr, wd=a.wd, bs=a.bs, seed=a.seed,
                 out=a.out, resume=a.resume, steps_per_epoch=4 if a.smoke else None)
    print(json.dumps({k: v for k, v in log.items() if not isinstance(v, (list, dict))}, indent=1), flush=True)


if __name__ == "__main__":
    main()
