"""Training loop (AdamW, cross-entropy). Deliberately has no evaluation code."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from tor.models.base import Recipe


@dataclass
class TrainResult:
    losses: list[float]
    epoch_times_s: list[float]
    diverged: bool


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _freeze_eval(m: torch.nn.Module) -> None:
    """Put fully frozen submodules in eval() (no dropout, no BN stat updates)."""
    ps = list(m.parameters())
    if ps and not any(p.requires_grad for p in ps):
        m.eval()
        return
    for c in m.children():
        _freeze_eval(c)


def train_epochs(model, X, y, recipe: Recipe, epochs: int, device: str, amp: bool = True) -> TrainResult:
    """Train for up to `epochs`. bf16 autocast only if `amp` and device is CUDA (ignored on CPU).

    Stops at the first non-finite loss: that epoch's loss is recorded as nan, diverged=True.
    """
    dev = torch.device(device)
    use_amp = amp and dev.type == "cuda"
    model.to(dev).train()
    _freeze_eval(model)
    X, y = X.to(dev), y.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    n = len(X)
    steps_per_epoch = max(1, -(-n // recipe.batch_size))
    warmup = int(recipe.warmup_frac * epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) if warmup > 0 else 1.0
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    res = TrainResult([], [], False)
    for _ in range(epochs):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=dev)
        total, bad = 0.0, False
        for i in range(0, n, recipe.batch_size):
            idx = perm[i : i + recipe.batch_size]
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(X[idx]).float(), y[idx])
            if not torch.isfinite(loss):
                bad = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        res.epoch_times_s.append(time.perf_counter() - t0)
        if bad:
            res.losses.append(float("nan"))
            res.diverged = True
            break
        res.losses.append(total / n)
    return res
