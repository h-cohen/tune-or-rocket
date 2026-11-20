import json

import numpy as np
import pytest

from tor import splits
from tor.splits import InsufficientClassError, draw_seed, save_draws, support_draw


def _y(counts):
    return np.array([f"c{i}" for i, n in enumerate(counts) for _ in range(n)])


def test_per_class_exact_counts():
    y = _y([30, 30, 30])
    idx = support_draw(y, ("per_class", 5), 0, "ds")
    assert len(idx) == 15
    assert all((y[idx] == c).sum() == 5 for c in ["c0", "c1", "c2"])
    assert list(idx) == sorted(idx)


def test_deterministic():
    y = _y([30, 30, 30])
    a = support_draw(y, ("per_class", 5), 0, "ds")
    b = support_draw(y, ("per_class", 5), 0, "ds")
    c = support_draw(y, ("per_class", 5), 1, "ds")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_label_order_independent():
    y = _y([30, 30])
    perm = np.random.default_rng(0).permutation(len(y))
    a = support_draw(y, ("per_class", 5), 0, "ds")
    b = support_draw(y[perm], ("per_class", 5), 0, "ds")
    assert sorted(y[a]) == sorted(y[perm][b])


def test_fraction_stratified_at_least_one_per_class():
    y = _y([100, 3])
    idx = support_draw(y, ("fraction", 0.1), 0, "ds")
    assert len(idx) == 11
    assert (y[idx] == "c0").sum() == 10 and (y[idx] == "c1").sum() == 1


def test_full_budget_is_all_indices():
    y = _y([7, 5])
    assert np.array_equal(support_draw(y, ("fraction", 1.0), 3, "ds"), np.arange(12))


def test_insufficient_class_raises():
    y = _y([30, 4])
    with pytest.raises(InsufficientClassError, match="c1"):
        support_draw(y, ("per_class", 5), 0, "ds")


def test_draw_seed_pinned():
    import hashlib

    h = hashlib.sha256(b"ds|per_class|5|0").digest()[:8]
    assert draw_seed("ds", ("per_class", 5), 0) == int.from_bytes(h, "big")
    h = hashlib.sha256(b"ds|fraction|0.1|2").digest()[:8]
    assert draw_seed("ds", ("fraction", 0.1), 2) == int.from_bytes(h, "big")


def test_save_draws(tmp_path, monkeypatch):
    class S:
        repo_root = tmp_path

    monkeypatch.setattr(splits, "get_settings", lambda: S)
    y = _y([30, 4])
    p = save_draws("ds", y)
    assert p == tmp_path / "splits" / "ds.json"
    d = json.loads(p.read_text())
    assert set(d) == {"per_class=5", "per_class=20", "fraction=0.1", "fraction=1.0"}
    assert "c1" in d["per_class=5"]["error"]
    assert "error" in d["per_class=20"]
    assert set(d["fraction=0.1"]) == {str(i) for i in range(10)}
    assert d["fraction=1.0"]["0"] == list(range(34))
