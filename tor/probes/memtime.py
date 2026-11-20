"""Memory and time probe runner (synthetic data only; no accuracy, no real data).

`run_probe` measures one (model, mode, shape, grad_ckpt) cell in-process. The CLI runs
every cell in a child process (spawn) so a hang, crash or hard OOM cannot kill the grid.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from tor.config import get_settings, load_secrets
from tor.cost import limit_threads, measure
from tor.data.metadata import load_metadata
from tor.gpu import NoGpuAvailable, acquire_gpu
from tor.models import available_adapters, get_adapter
from tor.models.base import ALL_MODES
from tor.probes.shapes import FITS_SHAPE, ProbeShape, probe_grid
from tor.runlog import RunRecord, append, config_hash, new_record, read_all
from tor.train import count_params, train_epochs

N_CLASSES = 5
N_INFERENCE = 1000
DONE_STATUSES = ("ok", "oom", "diverged")
EXTRA_MODULES_ENV = "TOR_PROBE_EXTRA_MODULES"  # comma-separated modules imported in children (tests)

Runner = Callable[[dict], RunRecord]


def probe_config(model: str, mode: str, shape: ProbeShape, n_series: int = 512,
                 warmup_epochs: int = 1, timed_epochs: int = 3, grad_ckpt: bool = False) -> dict:
    return {
        "model": model, "mode": mode, "length": shape.length, "channels": shape.channels,
        "batch": shape.batch, "n_series": n_series, "warmup_epochs": warmup_epochs,
        "timed_epochs": timed_epochs, "grad_ckpt": grad_ckpt,
    }


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def run_probe(adapter_name: str, mode: str, shape: ProbeShape, n_series: int = 512,
              warmup_epochs: int = 1, timed_epochs: int = 3, grad_ckpt: bool = False,
              device: str | None = None, gpu_index: int | None = None) -> RunRecord:
    """Probe one cell. device None = acquire a GPU via acquire_gpu (released on return)."""
    lease = None
    if device is None:
        lease = acquire_gpu()
        device, gpu_index = "cuda", lease.index
    try:
        return _probe(adapter_name, mode, shape, n_series, warmup_epochs, timed_epochs,
                      grad_ckpt, device, gpu_index)
    finally:
        if lease is not None:
            lease.release()


def _probe(adapter_name, mode, shape, n_series, warmup_epochs, timed_epochs, grad_ckpt,
           device, gpu_index) -> RunRecord:
    dev = torch.device(device)
    if dev.type == "cuda" and gpu_index is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_index = int(cvd) if cvd.isdigit() else (dev.index or 0)
    meter_idx = gpu_index if dev.type == "cuda" else None
    cfg = probe_config(adapter_name, mode, shape, n_series, warmup_epochs, timed_epochs, grad_ckpt)
    extra: dict = {
        "grad_ckpt": grad_ckpt, "raw_len": shape.length, "channels": shape.channels,
        "batch": shape.batch, "n_series": n_series, "warmup_epochs": warmup_epochs,
        "timed_epochs": timed_epochs,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
    }
    status = "ok"
    with measure(meter_idx) as cost:
        try:
            _body(adapter_name, mode, shape, n_series, warmup_epochs, timed_epochs, grad_ckpt,
                  dev, extra)
            if extra.pop("_diverged", False):
                status = "diverged"
        except Exception as e:  # noqa: BLE001 - every failure becomes a record
            if _is_oom(e):
                status = "oom"
                extra["error"] = f"{type(e).__name__}: {str(e)[:500]}"
            else:
                status = "error"
                extra["error"] = f"{type(e).__name__}: {str(e)[:1000]}"
                extra["traceback_tail"] = traceback.format_exc()[-1500:]
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return new_record(f"probe-{adapter_name}-{mode}-{int(time.time() * 1000)}", "probe", cfg,
                      status, cost=cost.to_dict(), extra=extra)


def _body(adapter_name, mode, shape, n_series, warmup_epochs, timed_epochs, grad_ckpt, dev, extra):
    adapter = get_adapter(adapter_name)()
    adapter.load(("random", 0) if mode == "scratch" else "pretrained")
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n_series, shape.channels, shape.length, generator=g)
    y = torch.randint(0, N_CLASSES, (n_series,), generator=g)
    # Adapters resize/pad/truncate differently: classifier needs the post-preprocess length.
    seq_len = int(adapter.preprocess(X[:1].numpy()).shape[-1])
    extra["preprocessed_len"] = seq_len
    model = adapter.classifier(N_CLASSES, mode, shape.channels, seq_len, grad_ckpt=grad_ckpt, seed=0)
    total, trainable = count_params(model)
    unused = getattr(adapter, "unused_param_count", None)
    extra.update(params_total=total, params_trainable=trainable,
                 unused_params=int(unused()) if callable(unused) else 0)
    Xp = adapter.preprocess(X.numpy())
    epochs = warmup_epochs + timed_epochs
    recipe = dataclasses.replace(adapter.default_recipe(), batch_size=shape.batch, epochs=epochs)
    res = train_epochs(model, Xp, y, recipe, epochs, str(dev))
    extra["epoch_times_s"] = list(res.epoch_times_s)
    timed = res.epoch_times_s[warmup_epochs:]
    if timed:
        s_ep = sum(timed) / len(timed)
        extra["s_per_epoch"] = s_ep
        extra["s_per_1000_series"] = s_ep / n_series * 1000
    if res.diverged:
        extra["_diverged"] = True
        return
    del Xp
    g2 = torch.Generator().manual_seed(1)
    Xi = adapter.preprocess(torch.randn(N_INFERENCE, shape.channels, shape.length, generator=g2).numpy())
    Xi = Xi.to(dev)
    model.eval()
    use_amp = dev.type == "cuda"
    _sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, N_INFERENCE, shape.batch):
            model(Xi[i:i + shape.batch])
    _sync(dev)
    extra["inference_s_per_1000"] = time.perf_counter() - t0


# ---------------------------------------------------------------- isolation

def _rid(cfg: dict) -> str:
    return f"probe-{cfg['model']}-{cfg['mode']}-{int(time.time() * 1000)}"


def _not_run(cfg: dict, reason: str, extra: dict | None = None) -> RunRecord:
    return new_record(_rid(cfg), "probe", cfg, "not_run: " + reason, extra=extra or {})


def _run_cfg(cfg: dict, device: str | None, gpu_index: int | None = None) -> RunRecord:
    return run_probe(cfg["model"], cfg["mode"],
                     ProbeShape(cfg["length"], cfg["channels"], cfg["batch"]),
                     cfg["n_series"], cfg["warmup_epochs"], cfg["timed_epochs"], cfg["grad_ckpt"],
                     device=device, gpu_index=gpu_index)


def _child_main(cfg: dict, out_path: str, marker_path: str, device: str | None) -> None:
    limit_threads(16)
    load_secrets()
    for mod in filter(None, os.environ.get(EXTRA_MODULES_ENV, "").split(",")):
        importlib.import_module(mod)
    lease = None
    try:
        if device is None:
            try:
                lease = acquire_gpu()
            except NoGpuAvailable as e:
                rec = _not_run(cfg, f"no idle GPU ({e})")
                Path(out_path).write_text(json.dumps(asdict(rec), default=str))
                return
            Path(marker_path).touch()  # GPU held: the timeout clock starts now
            rec = _run_cfg(cfg, "cuda", lease.index)
        else:
            Path(marker_path).touch()
            rec = _run_cfg(cfg, device)
        Path(out_path).write_text(json.dumps(asdict(rec), default=str))
    finally:
        if lease is not None:
            lease.release()


def isolated_runner(timeout_s: float = 1800, device: str | None = None) -> Runner:
    """Runner executing each cell in a spawned child, killed after timeout_s of run time
    (time waiting for a GPU lease does not count). device None = child leases a GPU."""

    def run(cfg: dict) -> RunRecord:
        tmp = Path(tempfile.mkdtemp(prefix="tor-probe-"))
        out, marker = tmp / "out.json", tmp / "started"
        p = mp.get_context("spawn").Process(
            target=_child_main, args=(cfg, str(out), str(marker), device))
        p.start()
        t_start = None
        try:
            while p.is_alive():
                p.join(0.2)
                if t_start is None and marker.exists():
                    t_start = time.monotonic()
                if t_start is not None and time.monotonic() - t_start > timeout_s:
                    p.kill()
                    p.join()
                    return _not_run(cfg, f"timeout ({timeout_s:g} s)")
            if out.exists():
                return RunRecord(**json.loads(out.read_text()))
            return new_record(_rid(cfg), "probe", cfg, "error",
                              extra={"error": f"child exited with code {p.exitcode} without a result"})
        finally:
            if p.is_alive():
                p.kill()
                p.join()
            shutil.rmtree(tmp, ignore_errors=True)

    return run


# ---------------------------------------------------------------- orchestration

def run_plan(models: list[str], modes: list[str], meta, runner: Runner, runlog: Path | None = None,
             fits_only: bool = False, jobs: int = 2, n_series: int = 512, warmup_epochs: int = 1,
             timed_epochs: int = 3) -> list[RunRecord]:
    """Run FITS_SHAPE for every (model, mode) first (with ckpt retry on oom), then the rest of
    the grid. Cells with an ok/oom/diverged record already in the log are skipped.
    Returns the records produced by this call (all are also appended to the run log)."""
    jobs = max(1, min(jobs, 2))
    runlog = Path(runlog) if runlog is not None else get_settings().runlog_path
    existing = read_all(runlog)
    done = {r.config_hash: r.status for r in existing if r.status in DONE_STATUSES}
    static_done = {r.config_hash for r in existing
                   if r.status in ("not_run: mode unsupported", "not_run: no grad ckpt support")}
    new: list[RunRecord] = []
    fits = FITS_SHAPE(meta)

    def emit(rec: RunRecord) -> RunRecord:
        append(rec, runlog)
        new.append(rec)
        if rec.status in DONE_STATUSES:
            done[rec.config_hash] = rec.status
        return rec

    def static(cfg: dict, reason: str) -> None:
        if config_hash(cfg) not in static_done:
            emit(_not_run(cfg, reason))

    def cell(model, mode, shape, ckpt) -> str:
        cfg = probe_config(model, mode, shape, n_series, warmup_epochs, timed_epochs, ckpt)
        h = config_hash(cfg)
        return done[h] if h in done else emit(runner(cfg)).status

    adapters = {m: get_adapter(m) for m in models}
    pairs = []
    for m in models:
        for mode in modes:
            if mode in adapters[m].supported_modes:
                pairs.append((m, mode))
            else:
                static({"model": m, "mode": mode}, "mode unsupported")

    def fits_task(pair):
        m, mode = pair
        if cell(m, mode, fits, False) == "oom":
            if adapters[m].supports_grad_ckpt:
                cell(m, mode, fits, True)
            else:
                static(probe_config(m, mode, fits, n_series, warmup_epochs, timed_epochs, True),
                       "no grad ckpt support")

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(fits_task, pairs))
        if not fits_only:
            grid = [s for s in probe_grid(meta) if s != fits]
            list(ex.map(lambda t: cell(*t, False),
                        [(m, mode, s) for m, mode in pairs for s in grid]))
    return new


def main(argv: list[str] | None = None) -> int:
    limit_threads(16)
    load_secrets()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="adapter name or 'all'")
    ap.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=list(ALL_MODES))
    ap.add_argument("--fits-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent children (max 2)")
    ap.add_argument("--timeout", type=float, default=1800, help="per-run seconds (default 1800)")
    ap.add_argument("--runlog", type=Path, default=None)
    args = ap.parse_args(argv)
    models = available_adapters() if args.model == "all" else [args.model]
    recs = run_plan(models, args.modes, load_metadata(), isolated_runner(args.timeout),
                    args.runlog, args.fits_only, args.jobs)
    print(dict(Counter(r.status for r in recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
