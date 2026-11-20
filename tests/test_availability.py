from types import SimpleNamespace

import pandas as pd
import pytest
from huggingface_hub.errors import GatedRepoError

from tor import availability as av
from tor.availability import Candidate


def _info(license="apache-2.0", gated=False, sizes=(("config.json", 10), ("model.safetensors", 2_000_000_000))):
    return SimpleNamespace(
        gated=gated, card_data=SimpleNamespace(license=license),
        siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in sizes])


def _gated_exc():
    import httpx2 as httpx
    resp = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError("gated", response=resp)


C = Candidate("m", "general", "org/m", None, "paper")


@pytest.fixture(autouse=True)
def _cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TOR_CACHE", str(tmp_path))


def test_gated_without_acceptance_recorded(monkeypatch):
    monkeypatch.setattr(av, "model_info", lambda *a, **k: _info(gated="manual"))

    def boom(*a, **k):
        raise _gated_exc()
    monkeypatch.setattr(av, "hf_hub_download", boom)
    r = av.check(C)
    assert r["status"] == "not_run: gated licence"
    assert r["available"] is False and r["gated"] is True


def test_missing_repo_recorded():
    r = av.check(Candidate("x", "classification", None, None, "p"))
    assert r["status"] == "not_run: no public weights found"
    assert r["available"] is False


def test_licence_read_from_card(monkeypatch):
    monkeypatch.setattr(av, "model_info", lambda *a, **k: _info())
    monkeypatch.setattr(av, "hf_hub_download", lambda *a, **k: "/x/config.json")
    r = av.check(C)
    assert r["licence"] == "apache-2.0"
    assert r["available"] is True and r["status"] == "ok" and r["gated"] is False
    assert r["files_gb"] == pytest.approx(2.0, abs=1e-3)


def test_weights_url_reachable(monkeypatch):
    calls = []

    def fake_http(method, url, headers=None):
        calls.append(url)
        if url.endswith("LICENSE"):
            return 200, {}, b"MIT License\n\nCopyright"
        return 206, {"Content-Range": "bytes 0-0/90000000"}, b"x"
    monkeypatch.setattr(av, "_http", fake_http)
    c = Candidate("n", "classification", None, None, "p",
                  weights_url="https://raw.githubusercontent.com/o/r/main/ckpt/a.pth")
    r = av.check(c)
    assert r["available"] is True and r["licence"] == "mit"
    assert r["files_gb"] == pytest.approx(0.09)


def test_weights_url_unreachable(monkeypatch):
    monkeypatch.setattr(av, "_http", lambda *a, **k: (404, {}, b""))
    c = Candidate("n", "classification", None, None, "p", weights_url="https://raw.githubusercontent.com/o/r/main/a")
    r = av.check(c)
    assert r["available"] is False and r["status"] == "error"


def test_inceptiontime_from_scratch():
    c = next(c for c in av.CANDIDATES if c.name == "InceptionTime")
    r = av.check(c)
    assert r["available"] is True and r["status"] == "ok"
    assert "trained from scratch" in c.notes


def test_candidate_list_complete():
    names = [c.name for c in av.CANDIDATES]
    assert len(names) == 15 and len(set(names)) == 15
    assert {"reverso", "reverso-small"} <= {n.lower() for n in names}


def test_survey_writes_csv_and_records(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "check", lambda c: dict(
        available=False, licence=None, gated=False, status=av.NO_WEIGHTS, files_gb=None))
    monkeypatch.setattr(av, "load_secrets", lambda: None)
    log = tmp_path / "log.jsonl"
    df = av.survey(tmp_path / "o.csv", log)
    assert list(df.columns) == av.CSV_COLUMNS and len(df) == 15
    assert list(pd.read_csv(tmp_path / "o.csv").columns) == av.CSV_COLUMNS
    from tor import runlog
    recs = runlog.read_all(log)
    assert len(recs) == 15 and all(r.kind == "availability" for r in recs)
