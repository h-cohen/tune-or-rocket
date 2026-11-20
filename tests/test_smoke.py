import numpy as np
import pytest

import tor.data.load
import tor.probes.smoke as sm
from tor.data import load_metadata
from tor.runlog import read_all


def test_smoke_choice_is_deterministic_and_eligible():
    meta = load_metadata()
    a, b = sm.choose_smoke_dataset(meta), sm.choose_smoke_dataset(meta)
    assert a == b
    r = meta[meta["name"] == a].iloc[0]
    assert r["archive"] == "UCR" and bool(r["equal_length"]) and not bool(r["missing"])
    assert r["n_train"] <= 500 and r["length"] <= 512 and r["classes"] <= 10


def _synth(monkeypatch, n=24, L=40, classes=("b", "a", "c")):
    rng = np.random.default_rng(0)
    y = np.array([classes[i % len(classes)] for i in range(n)])
    X = rng.normal(size=(n, 1, L)) + np.array([ord(c) - 96 for c in y])[:, None, None]
    monkeypatch.setattr(sm, "load_train", lambda name: (X.copy(), y.copy()))

    def boom(name):
        raise AssertionError("test split read")
    monkeypatch.setattr(tor.data.load, "load_test", boom)
    return X, y


def test_smoke_never_reads_test(monkeypatch):
    _synth(monkeypatch)
    rec = sm.run_smoke("inceptiontime", "scratch", epochs=1, device="cpu")
    assert "test split" not in str(rec.extra.get("error", ""))
    assert rec.kind == "smoke" and rec.config["budget"] == ["fraction", 1.0]
    assert rec.extra["dataset"] and len(rec.extra["losses"]) == 1
    # one epoch cannot show a decrease
    assert rec.status == "error" and rec.extra["reason"] == "loss did not decrease"


def test_smoke_ok_and_record_fields(monkeypatch):
    _synth(monkeypatch)
    rec = sm.run_smoke("inceptiontime", "scratch", epochs=4, device="cpu")
    e = rec.extra
    assert rec.status == "ok", e
    assert e["losses"][-1] < e["losses"][0]
    assert e["n_train"] == 24 and e["n_classes"] == 3 and e["raw_len"] == 40
    assert e["channels"] == 1 and e["epochs"] == 4 and e["mode"] == "scratch"
    assert e["batch_size"] <= 24 and "lr" in e and "preprocessed_len" in e
    assert rec.cost["wall_s"] > 0


def test_smoke_diverged(monkeypatch):
    _synth(monkeypatch)
    import dataclasses
    from tor.models import get_adapter
    cls = get_adapter("inceptiontime")
    orig = cls.default_recipe
    monkeypatch.setattr(cls, "default_recipe",
                        lambda self: dataclasses.replace(orig(self), lr=1e30))
    rec = sm.run_smoke("inceptiontime", "scratch", epochs=3, device="cpu")
    assert rec.status in ("diverged", "error")  # lr 1e30 may or may not overflow; never "ok"
    assert rec.status != "ok"


def test_plan_skips_unsupported_resumes_and_forces(monkeypatch, tmp_path):
    _synth(monkeypatch)
    log = tmp_path / "log.jsonl"
    calls = []

    def runner(cfg):
        calls.append(cfg)
        return sm._run_cfg(cfg, "cpu")

    from tor.models import get_adapter
    sup = get_adapter("inceptiontime").supported_modes
    modes = ["scratch", "head"]
    new = sm.run_plan(["inceptiontime"], modes, "DS", runner, log, epochs=4, jobs=1)
    assert len(calls) == 1 and calls[0]["mode"] == "scratch"
    statuses = {r.config["mode"]: r.status for r in new}
    if "head" not in sup:
        assert statuses["head"] == "not_run: mode unsupported"
    assert statuses["scratch"] == "ok"
    sm.run_plan(["inceptiontime"], modes, "DS", runner, log, epochs=4, jobs=1)
    assert len(calls) == 1  # resumed
    sm.run_plan(["inceptiontime"], modes, "DS", runner, log, epochs=4, jobs=1, force=True)
    assert len(calls) == 2
    assert len(read_all(log)) == 3 if "head" not in sup else True
