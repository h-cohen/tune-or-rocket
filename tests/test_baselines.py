import numpy as np
import pandas as pd
import pytest

import tor.baselines as bl
import tor.tabpfn_arms as ta
from tor.baselines import TRAINING_FREE, datasets_exceeding_limits, run_arms, time_baseline
from tor.probes.shapes import ProbeShape
from tor.runlog import read_all
from tor.tabpfn_arms import (masht_budget, masht_pipeline, rocketpfn_pipeline, tabpfn_limits,
                             time_tabpfn)

SHAPE = ProbeShape(48, 1, 8)


class FakeTabPFN:
    fits = 0

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        FakeTabPFN.fits += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


@pytest.fixture
def fake_tabpfn(monkeypatch):
    FakeTabPFN.fits = 0
    monkeypatch.setattr(ta, "_make_classifier", lambda recipe, hosted, device, threads: FakeTabPFN())
    monkeypatch.setattr(ta, "_set_hosted_token", lambda token: None)


@pytest.mark.parametrize("name", ["minirocket", "mr_hydra", "quant"])
def test_baselines_fit_on_synthetic(name):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 1, 64))
    y = np.arange(40) % 2
    pipe = TRAINING_FREE[name](threads=2)
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (40,)


def test_exceeding_limits():
    meta = pd.DataFrame({"name": ["a", "b"], "classes": [3, 12], "n_train": [100, 100]})
    assert datasets_exceeding_limits(meta, {"max_classes": 10, "max_samples": None}) == ["b"]
    meta2 = pd.DataFrame({"name": ["a", "b"], "classes": [3, 4], "n_train": [100, 60000]})
    assert datasets_exceeding_limits(meta2, {"max_classes": 10, "max_samples": 50000}) == ["b"]


def test_hosted_run_flags_and_needs_token(monkeypatch, fake_tabpfn):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    rec = time_tabpfn("masht", SHAPE, n_series=40, hosted=True, threads=2)
    assert rec.status == "not_run: no TabPFN token"
    assert rec.kind == "baseline"
    monkeypatch.setenv("TABPFN_TOKEN", "dummy-token-value")
    rec = time_tabpfn("masht", SHAPE, n_series=40, hosted=True, threads=2)
    assert rec.status == "ok", (rec.extra.get("error"), rec.extra.get("traceback_tail"))
    assert rec.cost["hosted"] is True
    assert "dummy-token-value" not in str(rec)
    for k in ("fit_s", "transform_s", "predict_s", "n_features", "threads", "unspecified"):
        assert k in rec.extra


def test_rocketpfn_groups_and_features(fake_tabpfn):
    rec = time_tabpfn("rocketpfn", SHAPE, n_series=40, hosted=False, threads=2, device="cpu")
    assert rec.status == "ok", (rec.extra.get("error"), rec.extra.get("traceback_tail"))
    assert rec.extra["n_groups"] == 10 and FakeTabPFN.fits == 10
    assert rec.extra["n_features"] == 20000
    assert rec.extra["unspecified"] and rec.extra["citations"]
    assert rec.cost["hosted"] is False


def test_masht_budget_tiers_and_features():
    assert masht_budget(999) == 10_000
    assert masht_budget(1000) == 2_000
    assert masht_budget(100_000) == 200
    X = np.random.default_rng(0).standard_normal((20, 2, 64))
    arm = masht_pipeline(n_total=20, budget=2000, threads=2, make_classifier=lambda: FakeTabPFN())
    arm.fit(X, np.arange(20) % 2)
    F = arm.transform(X)
    assert F.shape[0] == 20 and 0 < F.shape[1] <= 10_000
    assert arm.predict(F).shape == (20,)
    assert arm.details["unspecified"]


def test_tabpfn_limits_from_installed_package():
    lim = tabpfn_limits()
    assert lim["max_classes"] == 10 and lim["max_features"] == 2000 and lim["max_samples"] == 50_000
    assert lim["local"]["v3"]["max_classes"] is None and lim["local"]["v3"]["reason"]
    assert lim["hosted"]["max_classes"] is None and lim["hosted"]["reason"]
    assert lim["package_version"]


def test_time_baseline_record():
    rec = time_baseline("minirocket", SHAPE, n_series=40, threads=2)
    assert rec.kind == "baseline" and rec.status == "ok", rec.extra
    for k in ("fit_s_per_1000", "transform_s_per_1000", "predict_s_per_1000", "n_features",
              "threads"):
        assert rec.extra[k] is not None
    assert rec.cost["hosted"] is False


def test_run_arms_resumable(tmp_path, fake_tabpfn):
    log = tmp_path / "log.jsonl"
    calls = []

    def runner(cfg):
        calls.append(cfg)
        return bl.time_baseline(cfg["model"], ProbeShape(cfg["length"], cfg["channels"], 8),
                                n_series=cfg["n_series"], threads=2)

    shapes = {"fits": ProbeShape(40, 1, 8)}
    run_arms(["minirocket"], shapes, runner, log, n_series=30, threads=2)
    run_arms(["minirocket"], shapes, runner, log, n_series=30, threads=2)
    assert len(calls) == 1 and len(read_all(log)) == 1


def _failing_arm(exc):
    class A:
        details = {}

        def fit(self, X, y):
            raise exc

    return A()


def _run_failing(exc):
    return ta.time_arm(_failing_arm(exc), "x", SHAPE, 10, {"model": "x"}, 1)


def test_weights_classification_is_typed():
    from huggingface_hub.errors import GatedRepoError
    rec = _run_failing(RuntimeError("something about huggingface license gated failed to download"))
    assert rec.status == "error" and rec.extra["traceback_tail"]
    try:
        import httpx
        resp = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
        raise GatedRepoError("gated", response=resp)
    except GatedRepoError as g:
        wrapped = RuntimeError("Failed to download")
        wrapped.__cause__ = g
    rec = _run_failing(wrapped)
    assert rec.status == ta.NO_WEIGHTS and rec.extra["traceback_tail"]
    from tabpfn.errors import TabPFNLicenseError
    assert _run_failing(TabPFNLicenseError("x")).status == ta.NO_WEIGHTS


def test_secrets_scrubbed_from_errors(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "tpfn-secret-123")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_456")
    rec = _run_failing(RuntimeError("boom tpfn-secret-123 and hf_secret_456"))
    blob = str(rec)
    assert "tpfn-secret-123" not in blob and "hf_secret_456" not in blob


def test_masht_estimator_deviation_recorded(fake_tabpfn, monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "dummy")
    rec = time_tabpfn("masht", SHAPE, n_series=40, hosted=True, threads=2)
    assert rec.config["n_estimators_paper"] == 88 and rec.config["n_estimators_used"] == 8
    assert rec.extra["n_estimators_paper"] == 88 and rec.extra["n_estimators_used"] == 8
    assert "typo" in rec.extra["n_estimators_paper_note"]
