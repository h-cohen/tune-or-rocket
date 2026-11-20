"""Training-free arms (MiniRocket, MR-Hydra, QUANT), timing, and the baselines CLI.

Verified against the installed aeon 1.6.0 sources (site-packages/aeon):
- MiniRocket: classification/convolution_based/_minirocket.py (MiniRocketClassifier._fit):
  MiniRocket(n_kernels=10000, max_dilations_per_kernel=32) -> StandardScaler(with_mean=False)
  -> RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).
- MR-Hydra: classification/convolution_based/_mr_hydra.py (MultiRocketHydraClassifier._fit):
  HydraTransformer(n_kernels=8, n_groups=64) + _SparseScaler (from _hydra.py) concatenated with
  MultiRocket() (n_kernels=6250) + StandardScaler, then RidgeClassifierCV(logspace(-3,3,10)).
- QUANT: classification/interval_based/_quant.py (QUANTClassifier._fit): QUANTTransformer(
  interval_depth=6, quantile_divisor=4) + ExtraTreesClassifier(n_estimators=200,
  max_features=0.1, criterion="entropy", class_weight=None unless balanced requested).
Deviations: n_jobs=threads (aeon's classifier does not set ET n_jobs; this does not change the
model) and random_state=0 everywhere.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tor.config import get_settings, load_secrets
from tor.cost import limit_threads
from tor.data.metadata import length_percentiles, load_metadata
from tor.probes.memtime import isolated_runner
from tor.probes.shapes import FITS_SHAPE, ProbeShape
from tor.runlog import RunRecord, append, config_hash, new_record, read_all
from tor.tabpfn_arms import (PipelineArm, time_arm, time_tabpfn, tabpfn_cfg, tabpfn_limits)

N_SERIES = 512
SEED = 0
DONE = ("ok",)


class MRHydraFeatures(BaseEstimator, TransformerMixin):
    """MultiRocket features (StandardScaler) concatenated with Hydra features (_SparseScaler),
    exactly as aeon's MultiRocketHydraClassifier does (hydra block first)."""

    def __init__(self, threads: int = 16, random_state: int = SEED):
        self.threads = threads
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        from aeon.classification.convolution_based._hydra import _SparseScaler
        from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket
        self.hydra_ = HydraTransformer(n_kernels=8, n_groups=64, n_jobs=self.threads,
                                       random_state=self.random_state)
        self.hydra_scaler_ = _SparseScaler()
        h = self.hydra_scaler_.fit_transform(self.hydra_.fit_transform(X))
        self.mr_ = MultiRocket(n_jobs=self.threads, random_state=self.random_state)
        self.mr_scaler_ = StandardScaler()
        m = self.mr_scaler_.fit_transform(self.mr_.fit_transform(X))
        return np.concatenate((np.asarray(h), np.asarray(m)), axis=1)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        h = self.hydra_scaler_.transform(self.hydra_.transform(X))
        m = self.mr_scaler_.transform(self.mr_.transform(X))
        return np.concatenate((np.asarray(h), np.asarray(m)), axis=1)


def _ridge():
    return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))


def minirocket(threads: int = 16):
    from aeon.transformations.collection.convolution_based import MiniRocket
    return make_pipeline(MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=threads,
                                    random_state=SEED),
                         StandardScaler(with_mean=False), _ridge())


def mr_hydra(threads: int = 16):
    return make_pipeline(MRHydraFeatures(threads=threads), _ridge())


def quant(threads: int = 16):
    from aeon.transformations.collection.interval_based import QUANTTransformer
    return make_pipeline(
        QUANTTransformer(interval_depth=6, quantile_divisor=4),
        ExtraTreesClassifier(n_estimators=200, max_features=0.1, criterion="entropy",
                             n_jobs=threads, random_state=SEED))


TRAINING_FREE = {"minirocket": minirocket, "mr_hydra": mr_hydra, "quant": quant}


def baseline_cfg(name: str, shape: ProbeShape, n_series: int, threads: int) -> dict:
    return {"model": name, "mode": "cpu", "length": shape.length, "channels": shape.channels,
            "n_series": n_series, "threads": threads, "hosted": False}


def time_baseline(name: str, shape: ProbeShape, n_series: int = N_SERIES,
                  threads: int = 16) -> RunRecord:
    """Time fit / transform / predict of a training-free arm on synthetic data (CPU only)."""
    arm = PipelineArm(TRAINING_FREE[name](threads=threads))
    return time_arm(arm, name, shape, n_series, baseline_cfg(name, shape, n_series, threads),
                    threads, device="cpu")


def datasets_exceeding_limits(meta: pd.DataFrame, limits: dict) -> list[str]:
    """Names of datasets whose class count or train size exceeds the limits (None limits and
    NaN metadata are not counted). Feature width is recipe-controlled, so not checked here."""
    out = []
    for _, r in meta.iterrows():
        over = False
        mc, ms = limits.get("max_classes"), limits.get("max_samples")
        if mc is not None and pd.notna(r.get("classes")) and r["classes"] > mc:
            over = True
        if ms is not None and pd.notna(r.get("n_train")) and r["n_train"] > ms:
            over = True
        if over:
            out.append(r["name"])
    return out


# ------------------------------------------------------------------ orchestration

def _run_cfg(cfg: dict, device: str | None, gpu_index: int | None = None,
             progress=None) -> RunRecord:
    """Module-level (picklable) cell runner for isolated_runner."""
    shape = ProbeShape(cfg["length"], cfg["channels"], 8)
    if cfg["model"] in TRAINING_FREE:
        return time_baseline(cfg["model"], shape, cfg["n_series"], cfg["threads"])
    return time_tabpfn(cfg["model"], shape, cfg["n_series"], hosted=cfg["hosted"],
                       threads=cfg["threads"], device=device or "cpu", gpu_index=gpu_index,
                       masht_budget_override=cfg.get("budget_override"))


def run_arms(models: list[str], shapes: dict[str, ProbeShape], runner, runlog: Path | None = None,
             n_series: int = N_SERIES, threads: int = 16, hosted: bool = False,
             masht_budget_override: int | None = None, force: bool = False) -> list[RunRecord]:
    """Run every (model, shape) cell not already `ok` in the log (config_hash resumability)."""
    runlog = Path(runlog) if runlog is not None else get_settings().runlog_path
    done = {r.config_hash for r in read_all(runlog) if r.status in DONE}
    new: list[RunRecord] = []
    for m in models:
        for shape in shapes.values():
            if m in TRAINING_FREE:
                cfg = baseline_cfg(m, shape, n_series, threads)
            else:
                cfg = tabpfn_cfg(m, shape, n_series, hosted, threads, "cuda",
                                 masht_budget_override)
            h = config_hash(cfg)
            if h in done and not force:
                continue
            if masht_budget_override is not None:
                cfg = {**cfg, "budget_override": masht_budget_override}
            rec = runner(cfg)
            append(rec, runlog)
            new.append(rec)
            if rec.status in DONE:
                done.add(h)
    return new


def _limits_record(query_hosted: bool) -> RunRecord:
    lim = tabpfn_limits(query_hosted)
    return new_record(f"tabpfn_limits-{int(time.time() * 1000)}", "tabpfn_limits",
                      {"package_version": lim["package_version"], "query_hosted": query_hosted},
                      "ok", extra=lim)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all",
                    choices=["all", "minirocket", "mr_hydra", "quant", "rocketpfn", "masht"])
    ap.add_argument("--shapes", default="fits,p50", help="comma list of fits,p50")
    ap.add_argument("--hosted", action="store_true", help="TabPFN recipes via tabpfn-client")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--n-series", type=int, default=N_SERIES)
    ap.add_argument("--masht-budget", type=int, default=None,
                    help="override MASHT feature budget (default: tier from 2*n_series)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--runlog", type=Path, default=None)
    args = ap.parse_args(argv)
    threads = max(1, min(args.threads, 16))
    limit_threads(threads)
    load_secrets()
    meta = load_metadata()
    avail = {"fits": FITS_SHAPE(meta),
             "p50": ProbeShape(length_percentiles(meta, require_no_missing=False)["p50"], 1, 8)}
    shapes = {s: avail[s] for s in args.shapes.split(",")}
    models = (list(TRAINING_FREE) + ["rocketpfn", "masht"]) if args.which == "all" else [args.which]
    runlog = args.runlog or get_settings().runlog_path
    if any(m in ("rocketpfn", "masht") for m in models):
        append(_limits_record(args.hosted), runlog)
    recs = []
    for m in models:
        tab = m in ("rocketpfn", "masht")
        # Local TabPFN: the child leases a GPU (device None). Hosted and CPU arms: no GPU.
        runner = isolated_runner(args.timeout, device=None if (tab and not args.hosted) else "cpu",
                                 threads=threads, run_cfg=_run_cfg, kind="baseline")
        recs += run_arms([m], shapes, runner, runlog, args.n_series, threads, args.hosted,
                         args.masht_budget, args.force)
    print(dict(Counter(r.status for r in recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
