"""End-to-end smoke run: train each (model, mode) for a few epochs on the FULL TRAINING split of
one seeded UCR dataset, recording losses and cost. No evaluation of any kind, no predictions
stored, and the test split is never read.

The chosen dataset (see `choose_smoke_dataset`) is DECLARED PILOT-EXPOSED: it is used only for
pipeline checking and must be treated as seen in any later selection or reporting.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import math
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tor.config import get_settings, load_secrets
from tor.cost import limit_threads, measure
from tor.data import load_metadata, load_train
from tor.gpu import acquire_gpu
from tor.models import available_adapters, get_adapter
from tor.models.base import ALL_MODES
from tor.probes.memtime import DONE_STATUSES, Runner, _is_oom, _not_run, _peaks, isolated_runner
from tor.runlog import RunRecord, append, config_hash, new_record, read_all
from tor.splits import support_draw
from tor.train import count_params, train_epochs

SEED = 20261120
BUDGET = ("fraction", 1.0)
SMOKE_EPOCHS = 5


def choose_smoke_dataset(meta) -> str:
    """Seeded pick among UCR rows: equal_length & ~missing & n_train<=500 & length<=512 & classes<=10."""
    ok = ((meta["archive"] == "UCR") & meta["equal_length"].fillna(False).astype(bool)
          & ~meta["missing"].fillna(True).astype(bool) & (meta["n_train"] <= 500)
          & (meta["length"] <= 512) & (meta["classes"] <= 10))
    names = sorted(meta.loc[ok, "name"])
    if not names:
        raise ValueError("no eligible smoke dataset")
    return str(names[int(np.random.default_rng(SEED).choice(len(names)))])


def smoke_config(model: str, mode: str, epochs: int, dataset: str) -> dict:
    return {"model": model, "mode": mode, "dataset": dataset, "budget": list(BUDGET),
            "epochs": epochs, "draw": 0}


def run_smoke(adapter_name: str, mode: str, epochs: int = SMOKE_EPOCHS, device: str | None = None,
              gpu_index: int | None = None, progress: Callable[[dict], None] | None = None,
              dataset: str | None = None) -> RunRecord:
    """Smoke one (model, mode). device None = acquire a GPU via acquire_gpu (released on return)."""
    lease = None
    if device is None:
        lease = acquire_gpu()
        device, gpu_index = "cuda", lease.index
    try:
        return _smoke(adapter_name, mode, epochs, device, gpu_index, progress, dataset)
    finally:
        if lease is not None:
            lease.release()


def _smoke(adapter_name, mode, epochs, device, gpu_index, progress, dataset) -> RunRecord:
    import os
    dev = torch.device(device)
    if dev.type == "cuda" and gpu_index is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_index = int(cvd) if cvd.isdigit() else (dev.index or 0)
    dataset = dataset or choose_smoke_dataset(load_metadata())
    cfg = smoke_config(adapter_name, mode, epochs, dataset)
    extra: dict = {"dataset": dataset, "epochs": epochs, "mode": mode, "losses": [],
                   "device": "cpu"}
    status = "ok"
    with measure(gpu_index if dev.type == "cuda" else None) as cost:
        try:
            if dev.type == "cuda":
                extra["device"] = torch.cuda.get_device_name(dev)
            status = _body(adapter_name, mode, epochs, dev, dataset, extra, progress)
        except Exception as e:  # noqa: BLE001 - every failure becomes a record
            status = "oom" if _is_oom(e) else "error"
            extra["error"] = f"{type(e).__name__}: {str(e)[:1000]}"
            if status == "error":
                extra["traceback_tail"] = traceback.format_exc()[-1500:]
        finally:
            try:
                extra["peak_gpu_mem_reserved_mib"] = _peaks(dev)["peak_gpu_mem_reserved_mib"]
            except Exception as e:  # noqa: BLE001
                extra["peak_reserved_error"] = f"{type(e).__name__}: {e}"
            gc.collect()
            torch.cuda.empty_cache()
    extra["cost"] = cost.to_dict()
    return new_record(f"smoke-{adapter_name}-{mode}-{int(time.time() * 1000)}", "smoke", cfg,
                      status, cost=cost.to_dict(), extra=extra)


def _body(adapter_name, mode, epochs, dev, dataset, extra, progress) -> str:
    X, y_raw = load_train(dataset)  # training split only
    idx = support_draw(y_raw, BUDGET, 0, dataset)
    X, y_raw = X[idx], y_raw[idx]
    labels = sorted(np.unique(y_raw).tolist())
    y = torch.as_tensor(np.searchsorted(labels, y_raw), dtype=torch.long)
    n_train, channels, raw_len = X.shape[0], X.shape[1], X.shape[-1]
    adapter = get_adapter(adapter_name)()
    adapter.load(("random", 0) if mode == "scratch" else "pretrained")
    Xp = adapter.preprocess(X)
    seq_len = int(Xp.shape[-1])
    recipe = adapter.default_recipe()
    recipe = dataclasses.replace(recipe, epochs=epochs, batch_size=min(recipe.batch_size, n_train))
    extra.update(n_train=n_train, n_classes=len(labels), raw_len=raw_len,
                 preprocessed_len=seq_len, channels=channels, batch_size=recipe.batch_size,
                 lr=recipe.lr, recipe_batch_size_default=adapter.default_recipe().batch_size)
    model = adapter.classifier(len(labels), mode, channels, seq_len, seed=0)
    total, trainable = count_params(model)
    extra.update(params_total=total, params_trainable=trainable)

    def report(res):
        extra["losses"] = list(res.losses)
        if progress is not None:
            progress({"epoch_times_s": list(res.epoch_times_s), "losses": list(res.losses),
                      **_peaks(dev)})

    res = train_epochs(model, Xp, y, recipe, epochs, str(dev), on_epoch=report)
    extra["losses"] = [float(v) for v in res.losses]
    extra["epoch_times_s"] = list(res.epoch_times_s)
    if res.diverged or not all(math.isfinite(v) for v in res.losses):
        return "diverged"
    if not res.losses[-1] < res.losses[0]:
        extra["reason"] = "loss did not decrease"
        return "error"
    return "ok"


def _run_cfg(cfg: dict, device: str | None, gpu_index: int | None = None,
             progress=None) -> RunRecord:
    return run_smoke(cfg["model"], cfg["mode"], cfg["epochs"], device=device,
                     gpu_index=gpu_index, progress=progress, dataset=cfg["dataset"])


def run_plan(models: list[str], modes: list[str], dataset: str, runner: Runner,
             runlog: Path | None = None, epochs: int = SMOKE_EPOCHS, jobs: int = 2,
             force: bool = False) -> list[RunRecord]:
    """Smoke every supported (model, mode) via `runner`; skip ones already ok/oom/diverged in
    the log (unless force) and record unsupported modes as not_run. Returns new records."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = max(1, min(jobs, 2))
    runlog = Path(runlog) if runlog is not None else get_settings().runlog_path
    existing = read_all(runlog)
    done = {r.config_hash for r in existing if r.status in DONE_STATUSES}
    static_done = {r.config_hash for r in existing if r.status == "not_run: mode unsupported"}
    new: list[RunRecord] = []
    tasks = []
    for m in models:
        supported = get_adapter(m).supported_modes
        for mode in modes:
            cfg = smoke_config(m, mode, epochs, dataset)
            h = config_hash(cfg)
            if mode not in supported:
                if h not in static_done:
                    rec = _not_run(cfg, "mode unsupported", kind="smoke")
                    append(rec, runlog)
                    new.append(rec)
            elif force or h not in done:
                tasks.append(cfg)

    def one(cfg):
        rec = runner(cfg)
        append(rec, runlog)
        new.append(rec)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, tasks))
    return new


def main(argv: list[str] | None = None) -> int:
    limit_threads(16)
    load_secrets()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="adapter name or 'all'")
    ap.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=list(ALL_MODES))
    ap.add_argument("--epochs", type=int, default=SMOKE_EPOCHS)
    ap.add_argument("--jobs", type=int, default=1, help="concurrent children (max 2)")
    ap.add_argument("--timeout", type=float, default=1800, help="per-run seconds (default 1800)")
    ap.add_argument("--force", action="store_true", help="re-run regardless of the log")
    ap.add_argument("--runlog", type=Path, default=None)
    args = ap.parse_args(argv)
    models = available_adapters() if args.model == "all" else [args.model]
    jobs = max(1, min(args.jobs, 2))
    dataset = choose_smoke_dataset(load_metadata())
    print(f"smoke dataset (declared pilot-exposed): {dataset}")
    runner = isolated_runner(args.timeout, threads=16 // jobs, run_cfg=_run_cfg, kind="smoke")
    recs = run_plan(models, args.modes, dataset, runner, args.runlog, args.epochs, jobs,
                    args.force)
    for r in recs:
        print(r.config["model"], r.config["mode"], r.config["dataset"], r.status,
              r.extra.get("losses"))
    print(dict(Counter(r.status for r in recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
