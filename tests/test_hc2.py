import pandas as pd
import pytest

from tor.hc2 import published_results_coverage


def _fake(tables):
    return lambda url: tables[url]


def test_coverage_columns():
    meta = pd.DataFrame({"name": ["A", "B", "C"], "archive": ["UCR"] * 3})
    csv = pd.DataFrame({"Resamples:": ["A", "C"], "0": [0.9, 0.8], "1": [0.9, None]})
    out = published_results_coverage(meta, "HC2", sources=["u1"], fetch=_fake({"u1": csv}))
    assert list(out.columns) == ["name", "available", "resamples", "source_url"]
    assert len(out) == 3 and out["available"].sum() == 2
    r = out.set_index("name")
    assert r.loc["A", "resamples"] == 2 and r.loc["C", "resamples"] == 1
    assert pd.isna(r.loc["B", "resamples"]) and r.loc["B", "source_url"] is None
    assert r.loc["A", "source_url"] == "u1"


def test_first_source_wins_and_no_aliasing():
    meta = pd.DataFrame({"name": ["MixedShapesRegularTrain", "X"]})
    t1 = pd.DataFrame({"n": ["MixedShapes"], "0": [0.5]})  # different name: must not match
    t2 = pd.DataFrame({"n": ["X"], "0": [0.5]})
    out = published_results_coverage(meta, "MR-Hydra", sources=["a", "b"], fetch=_fake({"a": t1, "b": t2}))
    r = out.set_index("name")
    assert not r.loc["MixedShapesRegularTrain", "available"]
    assert r.loc["X", "source_url"] == "b"


@pytest.mark.network
def test_live_files_have_names():
    from tor.data.metadata import load_metadata

    out = published_results_coverage(load_metadata(), "HC2")
    assert out["available"].sum() > 100
