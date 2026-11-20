"""TabPFN recipes (RocketPFN, MASHT), TabPFN limits and the shared timing core.

Synthetic data only; no accuracy is computed anywhere. Every arm exposes the same three
timed stages: `fit(X, y)`, `transform(X) -> features` and `predict(features) -> labels`.

Sources (read for this task; section numbers refer to the arXiv HTML versions):
- RocketPFN, arXiv 2606.21786 (cited as [RocketPFN]).
- MASHT, arXiv 2607.19234v1 (cited as [MASHT]).
- Installed `tabpfn` 10.2.0 and `tabpfn-client` 0.10.1 sources (cited by module/function).
"""
from __future__ import annotations

import importlib.metadata
import os
import time
import traceback
from typing import Callable, Literal

import numpy as np

from tor.cost import measure
from tor.probes.shapes import ProbeShape
from tor.runlog import RunRecord, new_record

N_CLASSES = 5
SEED = 0
Recipe = Literal["rocketpfn", "masht"]
TOKEN_ENV = "TABPFN_TOKEN"
NO_TOKEN = "not_run: no TabPFN token"
NO_WEIGHTS = "not_run: tabpfn weights unavailable"


# ------------------------------------------------------------------ timing core

def synth(n: int, shape: ProbeShape, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian noise series (n, channels, length) and balanced labels over N_CLASSES."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, shape.channels, shape.length))
    y = rng.permutation(np.arange(n) % N_CLASSES)
    return X, y


class PipelineArm:
    """Adapter giving an sklearn Pipeline (features..., final estimator) the arm interface."""

    def __init__(self, pipe, details: dict | None = None):
        self.pipe = pipe
        self.details = details or {}

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def transform(self, X):
        return self.pipe[:-1].transform(X)

    def predict(self, F):
        return self.pipe[-1].predict(F)

    @staticmethod
    def n_features(F) -> int:
        return int(np.shape(F)[1])


def _n_features(arm, F) -> int:
    if hasattr(arm, "n_features"):
        return int(arm.n_features(F))
    return int(np.shape(F)[1])


def time_arm(arm, name: str, shape: ProbeShape, n_series: int, cfg: dict, threads: int,
             hosted: bool = False, gpu_index: int | None = None, device: str = "cpu",
             extra_fixed: dict | None = None) -> RunRecord:
    """Fit on n_series synthetic series (seed 0), then transform and predict n_series fresh
    ones (seed 1), timing the three stages separately. Returns a kind="baseline" record."""
    extra: dict = {"n_series": n_series, "threads": threads, "device": device,
                   "raw_len": shape.length, "channels": shape.channels,
                   "n_classes": N_CLASSES, **(extra_fixed or {})}
    status = "ok"
    with measure(gpu_index, hosted=hosted) as cost:
        try:
            X, y = synth(n_series, shape, SEED)
            Xe, _ = synth(n_series, shape, SEED + 1)
            t0 = time.perf_counter()
            arm.fit(X, y)
            extra["fit_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            F = arm.transform(Xe)
            extra["transform_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = arm.predict(F)
            extra["predict_s"] = time.perf_counter() - t0
            assert len(pred) == n_series
            extra["n_features"] = _n_features(arm, F)
            for s in ("fit", "transform", "predict"):
                extra[f"{s}_s_per_1000"] = extra[f"{s}_s"] / n_series * 1000
            extra.update(getattr(arm, "details", {}))
        except Exception as e:  # noqa: BLE001 - every failure becomes a record
            msg = f"{type(e).__name__}: {str(e)[:1000]}"
            low = msg.lower()
            if any(m in low for m in ("license", "gated", "failed to download", "huggingface")):
                status = NO_WEIGHTS
                extra["reason"] = msg
            else:
                status = "error"
                extra["error"] = msg
                extra["traceback_tail"] = traceback.format_exc()[-1500:]
            extra.update(getattr(arm, "details", {}))
    return new_record(f"baseline-{name}-{int(time.time() * 1000)}", "baseline", cfg, status,
                      cost=cost.to_dict(), extra=extra)


# ------------------------------------------------------------------ RocketPFN

ROCKETPFN_CITATIONS = [
    "[RocketPFN] Sec. 3: G groups of k=1,000 standard-Rocket kernels, each group -> 2k=2,000 "
    "features (max, PPV); no truncation/compression/feature selection, features go straight "
    "to TabPFN; per-group classifiers are ensembled",
    "[RocketPFN] Sec. 3 (G=10 default: 10x1,000 = 10,000 kernels, same as default Rocket); "
    "Sec. 3 per-instance z-normalisation along time (aeon Rocket normalise=True)",
    "[RocketPFN] Sec. 2: TabPFN v2.5 internal ensemble of e=8 estimators (fixed in Sec. 4.7)",
    "[RocketPFN] Sec. 2: TabPFN receives the entire labelled training set as context "
    "(no row subsampling in the main method)",
    "[RocketPFN] Eq. 1: average the G probability matrices, argmax over classes",
    "[RocketPFN] Sec. 4.1/App. B/Sec. 5: >10 classes not handled (TabPFN v2.5 supports "
    "C<=10; such datasets are excluded; OvO/OvR/ECOC named but not used)",
]
ROCKETPFN_UNSPECIFIED = [
    "TabPFN checkpoint, device and hyperparameters beyond e=8 (we use package defaults)",
    "random seeds for kernels and TabPFN (we use seed 0 for TabPFN and seed+group for Rocket)",
    "how probabilities are combined if a group's classifier omits a class (we pass all "
    "classes in every group, so not triggered)",
    "whether the G per-group TabPFN calls run sequentially or in parallel (we run sequentially)",
    "exact aeon version and Rocket options (we use aeon 1.6.0 Rocket defaults except n_kernels "
    "and n_jobs)",
    "Rocket 'normalise': paper says z-normalise per instance (Sec. 3); realised with aeon "
    "Rocket(normalise=True)",
]


class RocketPFNArm:
    """RocketPFN [RocketPFN Sec. 3]: G independent Rocket groups -> TabPFN each -> mean proba."""

    def __init__(self, make_classifier: Callable[[], object], n_groups: int = 10,
                 kernels_per_group: int = 1000, threads: int = 16, seed: int = SEED):
        self.make_classifier = make_classifier
        self.n_groups, self.k, self.threads, self.seed = n_groups, kernels_per_group, threads, seed
        self.details = {"n_groups": n_groups, "kernels_per_group": kernels_per_group,
                        "citations": ROCKETPFN_CITATIONS,
                        "unspecified": list(ROCKETPFN_UNSPECIFIED)}

    def fit(self, X, y):
        from aeon.transformations.collection.convolution_based import Rocket
        self.rockets, self.clfs = [], []
        for g in range(self.n_groups):
            r = Rocket(n_kernels=self.k, normalise=True, n_jobs=self.threads,
                       random_state=self.seed + g)
            F = r.fit_transform(X)
            clf = self.make_classifier()
            clf.fit(F, y)
            self.rockets.append(r)
            self.clfs.append(clf)
        self.classes_ = np.unique(y)
        return self

    def transform(self, X) -> list[np.ndarray]:
        return [r.transform(X) for r in self.rockets]

    def predict(self, F) -> np.ndarray:
        P = np.mean([clf.predict_proba(f) for clf, f in zip(self.clfs, F)], axis=0)
        return self.classes_[P.argmax(axis=1)]

    @staticmethod
    def n_features(F) -> int:
        return int(sum(f.shape[1] for f in F))


def rocketpfn_pipeline(make_classifier: Callable[[], object], n_groups: int = 10,
                       threads: int = 16) -> RocketPFNArm:
    """RocketPFN arm; see RocketPFNArm and ROCKETPFN_CITATIONS for the sourced steps."""
    return RocketPFNArm(make_classifier, n_groups=n_groups, threads=threads)


# ------------------------------------------------------------------ MASHT

MASHT_CITATIONS = [
    "[MASHT] Sec. 3 'Feature Extraction': MultiRocket (raw + first-difference, 4 pooling "
    "features per kernel; requested budget split across raw/diff) and Hydra (k=8 kernels per "
    "group, kernel length 9, g = floor(B / (2*k*d)) groups for d dilations)",
    "[MASHT] Sec. 3: feature budget by total (train+test) instances: <1,000 -> 10,000; "
    "<100,000 -> 2,000; else 200 (the paper itself looks odd at the 200 tier; unverified)",
    "[MASHT] Sec. 3: budget divided equally between Hydra and MultiRocket, features "
    "concatenated [phi_MR(x); phi_H(x)]; no PCA/selection/feature subsampling/ensembling "
    "over feature subsets",
    "[MASHT] App. A: TabPFN-3 via tabpfn TabPFNClassifier, auto estimator scaling, "
    "fit_mode=low_memory, CUDA, auto inference precision, no tuning config, hidden progress "
    "bars, ignore_pretraining_limits=True",
]
MASHT_UNSPECIFIED = [
    "MultiRocket kernel count/lengths/dilations/bias sampling/seeds (we use aeon "
    "MultiRocket with n_kernels = budget_MR // 8, since aeon yields ~2*4*n_kernels features)",
    "Hydra number of groups B-derivation uses d; the paper's d formula (log2(L-1)/(9-1)) "
    "differs from aeon's number of dilations (int(log2((L-1)/8))+1); we use aeon's d",
    "Hydra max_num_channels and sampling beyond 'competing groups' (aeon default 8)",
    "whether features are standardised before TabPFN (we do not scale)",
    "multivariate handling in the feature extractors (we use aeon's native multichannel)",
    "handling of >10 classes (nothing done; local TabPFN raises beyond its class limit)",
    "train-row subsampling (none)",
    "random seeds and aggregation of TabPFN estimators",
    "TabPFN-3 n_estimators: App. A as extracted lists '88' (flagged uncertain; hosted server "
    "accepts at most 8); we use n_estimators=8 and record this as a deviation",
    "whether the budget is adjusted when realised dimensionality is below target (no)",
]


def masht_budget(n_total: int) -> int:
    """[MASHT Sec. 3] feature budget by total (train+test) instances."""
    return 10_000 if n_total < 1000 else (2_000 if n_total < 100_000 else 200)


class MashtArm:
    """MASHT [MASHT Sec. 3]: MultiRocket (+diff) ⊕ Hydra features (equal budget split) -> TabPFN."""

    def __init__(self, make_classifier: Callable[[], object], n_total: int,
                 budget: int | None = None, threads: int = 16, seed: int = SEED):
        self.make_classifier, self.threads, self.seed = make_classifier, threads, seed
        self.budget = masht_budget(n_total) if budget is None else budget
        self.details = {"budget": self.budget, "n_total_for_budget": n_total,
                        "citations": MASHT_CITATIONS, "unspecified": list(MASHT_UNSPECIFIED)}

    def fit(self, X, y):
        from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket
        half = self.budget // 2
        self.mr = MultiRocket(n_kernels=max(42, half // 8), n_jobs=self.threads,
                              random_state=self.seed)
        L = X.shape[-1]
        d = int(np.log2((L - 1) / 8)) + 1  # aeon's dilation count (see MASHT_UNSPECIFIED)
        k = 8
        g = max(1, half // (2 * k * max(d, 1)))
        self.hydra = HydraTransformer(n_kernels=k, n_groups=g, n_jobs=self.threads,
                                      random_state=self.seed, output_type="numpy")
        F = self._features(X, fit=True)
        self.details.update(hydra_groups=g, hydra_dilations=d, realised_features=F.shape[1])
        self.clf = self.make_classifier()
        self.clf.fit(F, y)
        self.classes_ = np.unique(y)
        return self

    def _features(self, X, fit: bool = False) -> np.ndarray:
        if fit:
            mr, hy = self.mr.fit_transform(X), self.hydra.fit_transform(X)
        else:
            mr, hy = self.mr.transform(X), self.hydra.transform(X)
        return np.concatenate([np.asarray(mr), np.asarray(hy)], axis=1)  # [phi_MR; phi_H]

    def transform(self, X) -> np.ndarray:
        return self._features(X)

    def predict(self, F) -> np.ndarray:
        return self.clf.predict(F)


def masht_pipeline(n_total: int, make_classifier: Callable[[], object], budget: int | None = None,
                   threads: int = 16) -> MashtArm:
    """MASHT arm; see MashtArm and MASHT_CITATIONS for the sourced steps."""
    return MashtArm(make_classifier, n_total, budget, threads)


# ------------------------------------------------------------------ classifiers

# Paper-pinned TabPFN versions: RocketPFN Sec. 2/3 = v2.5; MASHT App. A = TabPFN-3.
LOCAL_VERSION = {"rocketpfn": "V2_5", "masht": "V3"}
HOSTED_VERSION = {"rocketpfn": "v2.5", "masht": "v3"}
N_ESTIMATORS = 8


def _set_hosted_token(token: str) -> None:
    """Set the hosted-API token in-process only (tabpfn_client.set_access_token)."""
    import tabpfn_client
    tabpfn_client.set_access_token(token)


def _make_classifier(recipe: str, hosted: bool, device: str, threads: int):
    """One TabPFN classifier configured per the recipe (module-level so tests can mock it)."""
    extra: dict = {}
    if recipe == "masht":
        extra["ignore_pretraining_limits"] = True  # [MASHT App. A]
    if hosted:
        from tabpfn_client import TabPFNClassifier
        return TabPFNClassifier.create_default_for_version(
            HOSTED_VERSION[recipe], n_estimators=N_ESTIMATORS, random_state=SEED, **extra)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if recipe == "masht":
        extra.update(fit_mode="low_memory", show_progress_bar=False)  # [MASHT App. A]
    return TabPFNClassifier.create_default_for_version(
        getattr(ModelVersion, LOCAL_VERSION[recipe]), n_estimators=N_ESTIMATORS,
        device=device, random_state=SEED, n_preprocessing_jobs=threads, **extra)


def tabpfn_cfg(recipe: str, shape: ProbeShape, n_series: int, hosted: bool, threads: int,
               device: str, masht_budget_override: int | None = None) -> dict:
    cfg = {"model": recipe, "mode": "hosted" if hosted else f"local-{device}",
           "length": shape.length, "channels": shape.channels, "n_series": n_series,
           "threads": threads, "hosted": hosted, "n_estimators": N_ESTIMATORS}
    if recipe == "masht":
        cfg["budget"] = (masht_budget(2 * n_series) if masht_budget_override is None
                         else masht_budget_override)
    return cfg


def time_tabpfn(recipe: Recipe, shape: ProbeShape, n_series: int = 512, hosted: bool = False,
                threads: int = 16, device: str = "cpu", gpu_index: int | None = None,
                masht_budget_override: int | None = None) -> RunRecord:
    """Time a TabPFN recipe on synthetic data. Hosted runs need TABPFN_TOKEN in the environment
    (set in-process only, never logged) and are flagged cost.hosted=True. Without a token the
    record is not_run. Weights that cannot be downloaded (gated repo/licence) give a
    not_run record, not an error."""
    cfg = tabpfn_cfg(recipe, shape, n_series, hosted, threads, device, masht_budget_override)
    if hosted:
        token = os.environ.get(TOKEN_ENV)
        if not token:
            return new_record(f"baseline-{recipe}-{int(time.time() * 1000)}", "baseline", cfg,
                              NO_TOKEN, cost={"hosted": True}, extra={"hosted": True})
        _set_hosted_token(token)
    make = lambda: _make_classifier(recipe, hosted, device, threads)  # noqa: E731
    if recipe == "rocketpfn":
        arm = rocketpfn_pipeline(make, threads=threads)
    elif recipe == "masht":
        arm = masht_pipeline(2 * n_series, make, cfg["budget"], threads)
    else:
        raise ValueError(f"unknown recipe {recipe!r}")
    return time_arm(arm, recipe, shape, n_series, cfg, threads, hosted=hosted,
                    gpu_index=None if hosted else gpu_index, device="hosted" if hosted else device,
                    extra_fixed={"hosted": hosted})


# ------------------------------------------------------------------ limits

def _local_limits(version_name: str) -> dict:
    from tabpfn.constants import ModelVersion
    from tabpfn.inference_config import InferenceConfig, cpu_sample_limit
    v = getattr(ModelVersion, version_name)
    try:
        c = InferenceConfig.get_default("multiclass", v)
    except ValueError as e:
        return {"max_samples": None, "max_features": None, "max_classes": None,
                "max_cpu_samples": cpu_sample_limit(v),
                "reason": f"inference config lives in the checkpoint (weights not read): {e}"}
    return {"max_samples": c.MAX_NUMBER_OF_SAMPLES, "max_features": c.MAX_NUMBER_OF_FEATURES,
            "max_classes": c.MAX_NUMBER_OF_CLASSES, "max_cpu_samples": cpu_sample_limit(v),
            "reason": None}


def _hosted_limits(query: bool) -> dict:
    none = {"max_samples": None, "max_features": None, "max_classes": None}
    if not query:
        return {**none, "reason": "hosted limits are served by the API at runtime "
                "(tabpfn_client.client.ServiceClient.get_settings -> ModelLimit); not queried"}
    try:
        from tabpfn_client.client import ServiceClient
        if os.environ.get(TOKEN_ENV):
            _set_hosted_token(os.environ[TOKEN_ENV])
        s = ServiceClient.get_settings()
        if s is None:
            return {**none, "reason": "ServiceClient.get_settings() returned None (offline?)"}
        out = {}
        for ver, lim in s.model_limits.items():
            out[str(getattr(ver, "value", ver))] = {
                "max_samples": lim.train_set_max_rows, "max_features": lim.max_cols,
                "max_classes": lim.max_classes, "test_set_max_rows": lim.test_set_max_rows,
                "train_set_max_upload_cells": lim.train_set_max_upload_cells}
        return {**none, "by_version": out, "default_version": str(getattr(s.default_model_version, "value", s.default_model_version)),
                "reason": "per-version server limits in by_version"}
    except Exception as e:  # noqa: BLE001
        return {**none, "reason": f"hosted query failed: {type(e).__name__}: {str(e)[:200]}"}


def tabpfn_limits(query_hosted: bool = False) -> dict:
    """Limits read from the installed packages (never hard-coded). The flat max_* keys are
    the local v2.5 limits (the RocketPFN recipe's model); local["v3"] (MASHT) is None with a
    reason because TabPFN-3 stores its config in the checkpoint. Neither local model
    handles more than max_classes classes (tabpfn.validation.validate_num_classes)."""
    local = {"v2.5": _local_limits("V2_5"), "v2": _local_limits("V2"), "v3": _local_limits("V3")}
    ver = lambda p: importlib.metadata.version(p)  # noqa: E731
    return {
        "package_version": ver("tabpfn"), "client_version": ver("tabpfn-client"),
        "source": "tabpfn.inference_config.InferenceConfig.get_default / cpu_sample_limit; "
                  "tabpfn.validation.validate_num_classes; tabpfn_client.estimator.validate_train_set",
        "max_samples": local["v2.5"]["max_samples"], "max_features": local["v2.5"]["max_features"],
        "max_classes": local["v2.5"]["max_classes"],
        "local": local, "hosted": _hosted_limits(query_hosted),
        "gpu_used": {"local": "yes when device=cuda (CPU inference capped at max_cpu_samples "
                              "unless ignore_pretraining_limits)", "hosted": "none (remote)"},
        "modes": ["local", "hosted"],
        "weights_gated": "v2.5/v2.6/v3/v3.5 HF repos (Prior-Labs/tabpfn_*) are gated; "
                         "tabpfn.model_loading._download_model requires licence acceptance",
    }
