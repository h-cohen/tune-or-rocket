import aeon.datasets
import numpy as np
import pandas as pd
import pytest

import tor.data as data
from tor.data import PhaseGuardError, build_metadata, length_percentiles, load_test, load_train


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("TOR_CACHE", str(tmp_path))
    monkeypatch.setenv("TOR_PHASE", "0")


def test_load_test_blocked_in_phase0():
    with pytest.raises(PhaseGuardError, match="Phase 0"):
        load_test("GunPoint")


def test_load_test_guard_runs_before_any_io(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("I/O attempted")

    monkeypatch.setattr(aeon.datasets, "load_classification", boom)
    with pytest.raises(PhaseGuardError):
        load_test("GunPoint")


def test_load_train_only_requests_train(monkeypatch):
    calls = []

    def fake(name, split=None, **k):
        calls.append(split)
        return np.zeros((4, 1, 7)), np.array([0, 1, 0, 1])

    monkeypatch.setattr(aeon.datasets, "load_classification", fake)
    X, y = load_train("Whatever")
    assert calls == ["train"] and X.shape == (4, 1, 7) and y.dtype.kind == "U"
    with pytest.raises(TypeError):
        load_train("Whatever", split="test")


@pytest.mark.network
def test_load_train_shape():
    X, y = load_train("GunPoint")
    assert X.shape == (50, 1, 150)
    assert len(set(y)) == 2


def test_length_percentiles_on_fixture():
    meta = pd.DataFrame({
        "length": [10, 20, 30, 40, 1000],
        "equal_length": True,
        "missing": False,
    })
    p = length_percentiles(meta)
    assert p["max"] == 1000 and p["p50"] == 30


@pytest.mark.network
def test_metadata_has_no_test_derived_reads(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("test split read")

    monkeypatch.setattr(data, "load_test", boom)
    monkeypatch.setattr("tor.data.load.load_test", boom)
    meta = build_metadata()
    assert len(meta) == 158
