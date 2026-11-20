import dataclasses

import pytest

from tor import config
from tor.runlog import RunRecord, append, config_hash, read_all


def make(run_id="r1", status="ok", extra=None):
    cfg = {"a": 1}
    return RunRecord(
        run_id=run_id, kind="probe", config=cfg, config_hash=config_hash(cfg),
        status=status, cost={"s": 1.5}, extra=extra or {}, git_commit="g1",
        versions={"torch": "1"}, started="2026-01-01T00:00:00+00:00",
    )


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert len(config_hash({"a": 1})) == 16


def test_append_then_read_roundtrip(tmp_path):
    p = tmp_path / "sub" / "log.jsonl"
    recs = [make("r1"), make("r2", status="oom")]
    for r in recs:
        append(r, p)
    out = read_all(p)
    assert len(out) == 2
    assert [dataclasses.asdict(r) for r in out] == [dataclasses.asdict(r) for r in recs]


def test_append_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "abc")
    monkeypatch.setattr(config, "_SECRETS", set())
    p = tmp_path / "log.jsonl"
    append(make(extra={"msg": "abc"}), p)
    text = p.read_text()
    assert "abc" not in text
    assert "***" in text


def test_status_must_be_valid():
    with pytest.raises(ValueError):
        make(status="great")
    make(status="not_run: no gpu")
